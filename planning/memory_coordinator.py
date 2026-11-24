"""Retrieve reusable work and index completed-task handoffs."""

import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Literal, NotRequired, TypedDict

from config import RuntimeConfig, SearchOptions, settings
from embedding.loaders import FileLoader
from embedding.pipeline import EmbeddingPipeline
from planning.memory_inputs import MemoryLookup, TaskChanges
from store.hybrid_store import HybridRetriever, RetrievalHit
from utils import sha256_hex

TASK_MEMORY_DIRECTORY = "task_memory"


class MemoryMatch(TypedDict):
    """A ranked source excerpt and its optional handoff location."""

    collection: str
    chunk_id: str
    source: str | None
    symbol: str | None
    excerpt: str
    score: float
    via_graph_relation: str | None
    handoff_id: NotRequired[str]
    project_path: NotRequired[str]


class MemorySearchResult(TypedDict):
    """Result of a lookup across the configured collections."""

    status: Literal["found", "no_results"]
    results: list[MemoryMatch]


class TaskChangesResult(TypedDict):
    """Ingestion outcome for a recorded handoff."""

    status: Literal["recorded"]
    handoff_id: str
    ingestion: dict[str, int | bool]


def describe_match(
    hit: RetrievalHit, collection: str, memory_root: Path
) -> MemoryMatch:
    """Expose an excerpt and recover the caller's path from a stored snapshot."""
    metadata = hit.metadata
    file_path = metadata.get("file_path")
    match: MemoryMatch = {
        "collection": collection,
        "chunk_id": metadata["chunk_id"],
        "source": metadata.get("source_url") or file_path,
        "symbol": metadata.get("symbol_name"),
        "excerpt": hit.document.page_content,
        "score": hit.rrf_score,
        "via_graph_relation": hit.via_graph_relation,
    }
    if not file_path:
        return match

    path = Path(file_path)
    if path.is_relative_to(memory_root):
        parts = path.relative_to(memory_root).parts
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            match["handoff_id"] = parts[0]
            if parts[1] == "files":
                match["project_path"] = "/".join(parts[2:])
    return match


def search_memory(lookup: MemoryLookup) -> MemorySearchResult:
    """Search both collections and return matching indexed work."""
    retriever = HybridRetriever()
    memory_root = settings.storage.markdown_vault_dir / TASK_MEMORY_DIRECTORY
    results: list[MemoryMatch] = []
    collections = dict.fromkeys((settings.collections.code, settings.collections.docs))
    for collection in collections:
        options = SearchOptions(
            collection_name=collection,
            mode="hybrid",
            keywords=lookup.keywords,
        )
        hits = retriever.search(lookup.query, options=options)
        results.extend(describe_match(hit, collection, memory_root) for hit in hits)

    return {"status": "found" if results else "no_results", "results": results}


def prepare_handoff_sources(
    changes: TaskChanges, loader: FileLoader, limits: RuntimeConfig
) -> dict[PurePosixPath, bytes]:
    """Render the summary and validate all attachments before staging the handoff."""
    snapshots: dict[PurePosixPath, bytes] = {}
    for file in changes.files:
        if file.content is None:
            continue
        location = PurePosixPath(file.path)
        if not loader.supports_file(location) or loader.ignored_dirs.intersection(
            location.parts[:-1]
        ):
            raise ValueError(
                f"Use a path-only entry for an unsupported source: {file.path}"
            )

        content = file.content.encode("utf-8")
        if len(content) > limits.max_source_bytes:
            raise ValueError(f"Source exceeds configured byte limit: {file.path}")
        snapshots[PurePosixPath("files") / location] = content

    file_descriptions = (f"- `{file.path}` — {file.change}" for file in changes.files)
    summary = "\n\n".join(
        [
            f"# {changes.task}",
            f"Project: {changes.project}",
            f"## Implementation\n\n{changes.implementation_summary}",
            f"## Verification\n\n{changes.verification}",
            "## Changed files\n\n"
            + ("\n".join(file_descriptions) or "No files reported."),
        ]
    )
    summary_bytes = (summary + "\n").encode("utf-8")
    if len(summary_bytes) > limits.max_source_bytes:
        raise ValueError("Handoff summary exceeds the configured source byte limit")

    sources = {PurePosixPath("implementation-summary.md"): summary_bytes, **snapshots}
    total_bytes = sum(len(content) for content in sources.values())
    if len(sources) > limits.max_scan_files or total_bytes > limits.max_scan_bytes:
        raise ValueError("Handoff exceeds configured file or byte limits")
    return sources


def save_handoff(root: Path, sources: dict[PurePosixPath, bytes]) -> None:
    """Publish the entire snapshot directory under the worker's index lock."""
    if root.exists():
        return

    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix=".handoff-", dir=root.parent) as temporary:
        staged = Path(temporary) / "handoff"
        for relative, content in sources.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(content)
        staged.rename(root)


def record_task_changes(changes: TaskChanges) -> TaskChangesResult:
    """Persist an immutable handoff and index its sources incrementally."""
    loader = FileLoader()
    sources = prepare_handoff_sources(changes, loader, settings.runtime)
    handoff_id = sha256_hex(changes.model_dump_json())
    root = settings.storage.markdown_vault_dir / TASK_MEMORY_DIRECTORY / handoff_id
    save_handoff(root, sources)

    stats = EmbeddingPipeline(loader=loader).ingest(root)
    return {
        "status": "recorded",
        "handoff_id": handoff_id,
        "ingestion": stats.model_dump(),
    }
