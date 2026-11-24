"""
Unit & Integration Tests for the Embedding Subsystem
====================================================
Purpose:
    Validates `FileLoader`, `PythonAstChunker`, `MarkdownChunker`, and
    `EmbeddingPipeline` end-to-end using deterministic offline embeddings and
    isolated `tmp_path` storage directories.
"""

import hashlib
from pathlib import Path
import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from config import settings
from embedding.code_chunker import PythonAstChunker
from embedding.loaders import FileLoader
from embedding.markdown_chunker import MarkdownChunker
from embedding.pipeline import EmbeddingPipeline
from planning.memory_coordinator import record_task_changes, search_memory
from planning.memory_inputs import ChangedFile, MemoryLookup, TaskChanges
from store.chroma_store import ChromaStoreManager
from store.graph_store import CodeGraphStore
from store.hybrid_store import HybridRetriever
from store.manifest_store import ManifestStore


class DeterministicFakeEmbeddings(Embeddings):
    """Fast, offline 16-dim bag-of-tokens embedding for unit testing ChromaDB."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * 16
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 16
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def isolated_pipeline(tmp_path: Path):
    """Builds an isolated `EmbeddingPipeline` + `HybridRetriever` inside `tmp_path`."""
    db_path = tmp_path / "state.sqlite3"
    chroma_dir = tmp_path / "chroma_db"
    fake_embed = DeterministicFakeEmbeddings()

    manifest = ManifestStore(db_path=db_path)
    graph = CodeGraphStore(db_path=db_path)
    chroma = ChromaStoreManager(
        persist_dir=chroma_dir,
        db_path=db_path,
        embeddings=fake_embed,
        batch_size=8,
    )
    pipeline = EmbeddingPipeline(chroma=chroma, manifest=manifest, graph=graph)
    retriever = HybridRetriever(chroma=chroma, graph=graph, db_path=db_path)

    return {
        "pipeline": pipeline,
        "retriever": retriever,
        "manifest": manifest,
        "graph": graph,
        "chroma": chroma,
    }


def test_file_loader_frontmatter_and_ignored_dirs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "__init__.py").write_text("# ignored", encoding="utf-8")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "ignored.py").write_text("def skip_me(): pass", encoding="utf-8")

    md_file = repo / "crawl_guide.md"
    md_file.write_text(
        "---\nsource_url: https://crawl4ai.com\ntopic: scraping\n---\n\n# Crawl4AI\nClean markdown.",
        encoding="utf-8",
    )

    loader = FileLoader()
    records = loader.scan(repo)

    assert len(records) == 1
    rec = records[0]
    assert rec.filename == "crawl_guide.md"
    assert rec.frontmatter == {"source_url": "https://crawl4ai.com", "topic": "scraping"}
    assert rec.content.startswith("# Crawl4AI")
    assert len(rec.file_hash) == 64


def test_python_ast_chunker_extracts_symbols_imports_and_edges(tmp_path: Path):
    py_file = tmp_path / "network_client.py"
    py_file.write_text(
        '''import asyncio
import httpx

class BaseClient:
    """Base HTTP client."""
    pass

async def retry_backoff(attempt: int) -> None:
    """Sleep with exponential backoff."""
    await asyncio.sleep(0.1 * (2 ** attempt))

class ResilientClient(BaseClient):
    """HTTP client with automatic backoff retries."""

    async def get_text(self, url: str) -> str:
        await retry_backoff(1)
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            return resp.text

def _private_helper() -> None:
    pass
''',
        encoding="utf-8",
    )

    record = FileLoader().load_file(py_file)
    assert record is not None

    chunks = PythonAstChunker(max_chars=2400).chunk(record)
    symbols = {c.metadata["symbol_name"]: c for c in chunks}

    # Public symbols extracted; private `_private_helper` skipped
    assert set(symbols.keys()) == {"BaseClient", "retry_backoff", "ResilientClient"}

    # Module imports attached to function chunk
    retry_chunk = symbols["retry_backoff"]
    assert "import asyncio" in retry_chunk.page_content
    assert "# Intent: Sleep with exponential backoff." in retry_chunk.page_content

    # AST edges extracted for `ResilientClient` (`inherits` BaseClient, `calls` retry_backoff, `imports` httpx)
    edges = set(symbols["ResilientClient"].metadata["graph_edges"])
    assert ("BaseClient", "inherits") in edges
    assert ("retry_backoff", "calls") in edges
    assert ("httpx", "imports") in edges


def test_markdown_chunker_preserves_breadcrumbs_and_frontmatter(tmp_path: Path):
    md_file = tmp_path / "rag_architecture.md"
    md_file.write_text(
        """---
source_url: https://example.com
topic: architecture
---

# RAG System

## Hybrid Search
Combines dense vectors with BM25 keyword search using Reciprocal Rank Fusion.
""",
        encoding="utf-8",
    )

    record = FileLoader().load_file(md_file)
    assert record is not None

    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).chunk(record)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["section"] == "RAG System > Hybrid Search"
    assert chunk.metadata["source_url"] == "https://example.com"
    assert "[Document: rag_architecture.md | Section: RAG System > Hybrid Search]" in chunk.page_content


def test_pipeline_incremental_update_stale_cleanup_and_graph_search(tmp_path: Path, isolated_pipeline):
    pipeline: EmbeddingPipeline = isolated_pipeline["pipeline"]
    retriever: HybridRetriever = isolated_pipeline["retriever"]

    project = tmp_path / "project"
    project.mkdir()
    helper_py = project / "retry_utils.py"
    helper_py.write_text(
        '''async def retry_async(fn) -> None:
    """Retry wrapper."""
    await fn()

async def fetch_json(url: str) -> str:
    """Fetch JSON calling retry_async."""
    await retry_async(lambda: None)
    return url
''',
        encoding="utf-8",
    )

    # 1. Initial ingest with dict override
    stats1 = pipeline.ingest(project, {"reset": True, "code_max_chars": 2000})
    assert stats1.scanned_files == 1
    assert stats1.indexed_files == 1
    assert stats1.total_chunks_upserted == 2
    assert stats1.total_graph_edges_upserted >= 1
    assert stats1.has_changes is True

    # 2. Re-run without edits -> skipped via SHA-256 manifest
    stats2 = pipeline.ingest(project, {"code_max_chars": 2000})
    assert stats2.indexed_files == 0
    assert stats2.skipped_unchanged == 1
    assert stats2.has_changes is False

    # 3. Hybrid search for `fetch_json` pulls 1-hop AST dependency `retry_async`
    hits = retriever.search(
        "fetch_json",
        {"collection_name": "reusable_code", "mode": "hybrid", "k": 1, "max_distance": 1.5},
    )
    symbols = [h.document.metadata.get("symbol_name") for h in hits]
    assert "fetch_json" in symbols
    assert "retry_async" in symbols

    # 4. Edit file to remove `fetch_json` -> verify stale chunk & graph node are purged
    helper_py.write_text(
        '''async def retry_async(fn) -> None:
    """Updated retry wrapper."""
    await fn()
''',
        encoding="utf-8",
    )
    stats3 = pipeline.ingest(project)
    assert stats3.indexed_files == 1
    assert stats3.total_chunks_upserted == 1

    hits_after_delete = retriever.search(
        "fetch_json",
        {"collection_name": "reusable_code", "mode": "keyword", "k": 5},
    )
    remaining_symbols = [h.document.metadata.get("symbol_name") for h in hits_after_delete]
    assert "fetch_json" not in remaining_symbols


def test_same_named_markdown_files_keep_distinct_chunks(tmp_path: Path, isolated_pipeline):
    pipeline = isolated_pipeline["pipeline"]
    retriever = isolated_pipeline["retriever"]

    files = [tmp_path / name / "guide.md" for name in ("first", "second")]
    for path in files:
        path.parent.mkdir()
        path.write_text(
            "---\nchunk_id: overwritten\nfile_path: incorrect\n---\n# Shared\nCommon searchable text.\n",
            encoding="utf-8",
        )

    assert pipeline.ingest(tmp_path).total_chunks_upserted == 2
    hits = retriever.search("searchable", "research_docs", mode="keyword")
    assert {hit.metadata["file_path"] for hit in hits} == {str(path.resolve()) for path in files}
    assert len({hit.metadata["chunk_id"] for hit in hits}) == 2

    files[0].write_text("# Revised\nDifferent content.\n", encoding="utf-8")
    assert pipeline.ingest(files[0]).indexed_files == 1
    hits = retriever.search("searchable", "research_docs", mode="keyword")
    assert [hit.metadata["file_path"] for hit in hits] == [str(files[1].resolve())]


def test_repeated_python_definitions_get_distinct_chunk_ids(tmp_path: Path, isolated_pipeline):
    source = tmp_path / "repeated.py"
    source.write_text("def repeated():\n    return 1\n\n" * 2, encoding="utf-8")
    pipeline = isolated_pipeline["pipeline"]
    assert pipeline.ingest(source).total_chunks_upserted == 2
    hits = isolated_pipeline["retriever"].search("repeated", mode="keyword")
    assert len({hit.metadata["chunk_id"] for hit in hits}) == 2


def test_embedding_failure_keeps_previous_index(tmp_path: Path, isolated_pipeline, monkeypatch):
    pipeline = isolated_pipeline["pipeline"]
    retriever = isolated_pipeline["retriever"]
    manifest = isolated_pipeline["manifest"]
    model = isolated_pipeline["chroma"].embeddings
    source = tmp_path / "module.py"
    source.write_text("def original_symbol():\n    return 1\n", encoding="utf-8")
    pipeline.ingest(source)
    entry = manifest.get_entry(str(source.resolve()))

    source.write_text("def replacement_symbol():\n    return 2\n", encoding="utf-8")

    def fail_embedding(_texts):
        raise RuntimeError("test embedding failure")

    with monkeypatch.context() as patch:
        patch.setattr(model, "embed_documents", fail_embedding)
        with pytest.raises(RuntimeError, match="test embedding failure"):
            pipeline.ingest(source)
    assert manifest.get_entry(str(source.resolve())) == entry
    assert retriever.search("original_symbol", mode="keyword")
    assert isolated_pipeline["chroma"].get_by_ids("reusable_code", entry[2])

    assert pipeline.ingest(source).indexed_files == 1
    assert retriever.search("replacement_symbol", mode="keyword")
    assert retriever.search("original_symbol", mode="keyword") == []


def test_reset_preserves_other_collections_and_requires_valid_input(tmp_path: Path, isolated_pipeline):
    pipeline = isolated_pipeline["pipeline"]
    chroma = isolated_pipeline["chroma"]
    retriever = isolated_pipeline["retriever"]
    manifest = isolated_pipeline["manifest"]
    source = tmp_path / "custom.py"
    source.write_text("def helper():\n    return 1\n\ndef custom_symbol():\n    return helper()\n", encoding="utf-8")
    assert pipeline.ingest(source, {"code_collection": "custom_code"}).indexed_files == 1
    entry = manifest.get_entry(str(source.resolve()))

    other = tmp_path / "default.py"
    other.write_text("def default_symbol():\n    return 2\n", encoding="utf-8")
    pipeline.ingest(other)
    with pytest.raises(FileNotFoundError):
        pipeline.ingest(tmp_path / "missing", {"reset": True})
    assert retriever.search("default_symbol", mode="keyword")
    assert pipeline.ingest(other, {"reset": True}).indexed_files == 1
    assert manifest.get_entry(str(source.resolve())) == entry
    assert len(chroma.get_by_ids("custom_code", entry[2])) == 2
    assert retriever.search("custom_symbol", "custom_code", mode="keyword", k=1)[1].via_graph_relation == "calls:helper"

    assert pipeline.ingest(source).indexed_files == 1
    assert chroma.get_by_ids("custom_code", entry[2]) == []
    assert retriever.search("custom_symbol", mode="keyword", k=1)[1].via_graph_relation == "calls:helper"


def test_completed_task_memory_is_searchable_and_retryable(tmp_path, monkeypatch, isolated_pipeline):
    monkeypatch.setattr(settings.storage, "markdown_vault_dir", tmp_path / "vault")
    monkeypatch.setattr(settings.retrieval, "code_max_distance", -1)
    monkeypatch.setattr(settings.retrieval, "docs_max_distance", -1)
    pipeline = isolated_pipeline["pipeline"]
    monkeypatch.setattr("planning.memory_coordinator.EmbeddingPipeline", lambda loader: pipeline)
    monkeypatch.setattr(
        "planning.memory_coordinator.HybridRetriever",
        lambda: isolated_pipeline["retriever"],
    )

    lookup = MemoryLookup(query="reuse prior behavior", keywords=["retry_delay", "retry timing"])
    changes = TaskChanges(
        project="http-client",
        task="Retry timing",
        implementation_summary="Cap the exponential delay.",
        verification="Checked the cap on the fourth attempt.",
        files=[
            ChangedFile(
                path="transport/retry.py", change="Added the delay helper.",
                content="def retry_delay(attempt):\n    return min(8, 2 ** attempt)\n",
            ),
            ChangedFile(
                path="docs/retries.md", change="Documented retry timing.",
                content="# Retry timing\n\nDouble the wait until it reaches eight seconds.",
            ),
            ChangedFile(path="old.cfg", change="Removed an obsolete toggle."),
        ],
    )
    original_encode = pipeline.chroma.encode_chunks
    attempts = 0

    def fail_second_source(chunks):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("Embedding interrupted")
        return original_encode(chunks)

    with monkeypatch.context() as interruption:
        interruption.setattr(pipeline.chroma, "encode_chunks", fail_second_source)
        with pytest.raises(RuntimeError, match="Embedding interrupted"):
            record_task_changes(changes)

    recorded = record_task_changes(changes)
    assert recorded["ingestion"]["indexed_files"] == 2
    assert recorded["ingestion"]["skipped_unchanged"] == 1

    hits = search_memory(lookup)
    assert hits["status"] == "found"
    assert {hit.get("project_path") for hit in hits["results"]} == {
        "transport/retry.py", "docs/retries.md", None,
    }
    assert any(hit["source"].endswith("implementation-summary.md") for hit in hits["results"])
    assert search_memory(MemoryLookup(
        query="retry_delay", keywords=["unrelated timing"],
    )) == {"status": "no_results", "results": []}

    follow_up = changes.model_copy(update={
        "task": "Adjust retry timing",
        "implementation_summary": "Extend the documented retry delay limit.",
        "files": [],
    })
    assert record_task_changes(follow_up)["ingestion"]["indexed_files"] == 1

    repeated = record_task_changes(changes)
    assert repeated["handoff_id"] == recorded["handoff_id"]
    assert repeated["ingestion"]["skipped_unchanged"] == 3


def test_memory_handoff_rejects_outside_path():
    with pytest.raises(ValidationError, match="project-relative"):
        ChangedFile(path="../outside.py", change="Outside workspace.", content="pass")


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        (["docs/caf\u00e9.md", "docs/cafe\u0301.md"], "Duplicate changed file"),
        (["Example.py", "example.py/nested.md"], "cannot also be a directory"),
    ],
)
def test_memory_handoff_rejects_colliding_snapshot_paths(paths, message):
    files = [ChangedFile(path=path, change="Changed file.", content="text") for path in paths]
    with pytest.raises(ValidationError, match=message):
        TaskChanges(
            project="test-project",
            task="Validate snapshot paths",
            implementation_summary="Prepare the handoff.",
            verification="Tested input validation.",
            files=files,
        )
