"""Prepare, journal, and apply incremental index changes."""

import json
import logging
import re
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from pydantic import BaseModel, Field, computed_field
from config import IngestOptions, merge_options, settings
from embedding.code_chunker import PythonAstChunker
from embedding.loaders import FileLoader
from embedding.markdown_chunker import MarkdownChunker
from store.chroma_store import ChromaStoreManager
from store.graph_store import CodeGraphStore
from store.index_state import IndexState
from store.manifest_store import ManifestStore
from utils import sha256_hex

log = logging.getLogger("rag_memory")
CHUNK_FORMAT_VERSION = 1


class IngestionStats(BaseModel):
    """Validated summary metrics returned after an `EmbeddingPipeline.ingest()` run."""

    scanned_files: int = Field(default=0, ge=0)
    indexed_files: int = Field(default=0, ge=0)
    removed_files: int = Field(default=0, ge=0)
    skipped_unchanged: int = Field(default=0, ge=0)
    total_chunks_upserted: int = Field(default=0, ge=0)
    total_graph_edges_upserted: int = Field(default=0, ge=0)

    @computed_field
    @property
    def has_changes(self) -> bool:
        return self.indexed_files > 0 or self.removed_files > 0


class EmbeddingPipeline:
    """Orchestrates incremental chunking, AST graph registration, and ChromaDB indexing."""

    def __init__(
        self,
        chroma: ChromaStoreManager | None = None,
        manifest: ManifestStore | None = None,
        graph: CodeGraphStore | None = None,
        loader: FileLoader | None = None,
        code_chunker: PythonAstChunker | None = None,
        md_chunker: MarkdownChunker | None = None,
    ):
        """
        Initialize the pipeline with injectable storage and chunking components.

        All arguments default to standard instances configured via `default_config.yaml`.
        """
        supplied_store = chroma or manifest or graph
        db_path = supplied_store.db_path if supplied_store is not None else None
        self.manifest = manifest or ManifestStore(db_path=db_path)
        self.chroma = chroma or ChromaStoreManager(db_path=db_path)
        self.graph = graph or CodeGraphStore(db_path=db_path)
        self.loader = loader or FileLoader()
        self.code_chunker = code_chunker or PythonAstChunker()
        self.md_chunker = md_chunker or MarkdownChunker()
        self.index_state = IndexState(db_path)

    def register_code_graph(
        self,
        chunks: list[Document],
        collection_name: str,
    ) -> int:
        """
        Persist AST symbol nodes and dependency edges into `CodeGraphStore`.

        Read dependency edges from the original documents.
        """
        edge_count = 0

        for doc in chunks:
            meta = doc.metadata
            edges: list[tuple[str, str]] = meta.get("graph_edges", [])
            self.graph.upsert_node_with_edges(
                chunk_id=meta["chunk_id"],
                symbol_name=meta.get("symbol_name", "module_block"),
                symbol_type=meta.get("symbol_type", "module"),
                file_path=meta["file_path"],
                source=meta["source"],
                collection_name=collection_name,
                edges=edges,
            )
            edge_count += len(edges)
        return edge_count

    def recover(self) -> int:
        pending = self.index_state.pending()
        for op_id, operation in pending:
            self.apply_operation(operation)
            self.index_state.finish(op_id)
            log.info("index_recovered", extra={"operation_id": op_id})
        return len(pending)

    def apply_operation(self, operation: dict) -> None:
        if operation.get("version") != 1:
            raise ValueError("Unsupported index journal version")
        staged_path = operation.get("chroma_dir")
        if staged_path is not None and Path(staged_path) != Path(self.chroma.persist_dir).resolve():
            raise ValueError("Pending journal belongs to a different Chroma directory. Select the original configuration.")
        reset = operation.get("reset", [])
        if reset:
            self.chroma.reset_collections(reset)
            self.manifest.clear_collections(reset)
            self.graph.clear_collections(reset)
            self.index_state.clear_contracts(reset)
        for change in operation["changes"]:
            path = change["path"]
            old = change.get("old")
            if change["kind"] == "delete":
                self.chroma.delete_ids(old[0], old[1])
                self.graph.delete_by_chunk_ids(old[1])
                self.manifest.delete_entry(path)
                continue

            collection = change["collection"]
            chunks = [Document.model_validate(item) for item in change["chunks"]]
            vectors = change["vectors"]
            clean = [
                Document(page_content=doc.page_content, metadata={
                    key: value for key, value in doc.metadata.items() if key != "graph_edges"
                })
                for doc in chunks
            ]
            ids = self.chroma.upsert_chunks(collection, clean, vectors=vectors)
            if change["code"]:
                self.register_code_graph(chunks, collection)
            if old:
                old_col, old_ids = old
                current = set(ids)
                stale_ids = [cid for cid in old_ids if cid not in current]
                self.chroma.delete_ids(old_col, old_ids if old_col != collection else stale_ids)
                self.graph.delete_by_chunk_ids(stale_ids)
            self.manifest.upsert_entry(path, change["hash"], collection, ids)
            self.index_state.set_fingerprint(path, change["fingerprint"])
            if vectors:
                self.index_state.set_contract(collection, change["model_id"], len(vectors[0]))

    def commit(self, changes: list[dict], reset: list[str] | None = None) -> None:
        operation = {
            "version": 1, "chroma_dir": str(Path(self.chroma.persist_dir).resolve()),
            "changes": changes, "reset": reset or [],
        }
        size = len(json.dumps(operation, ensure_ascii=True))
        limit = settings.runtime.max_reset_journal_bytes
        if size > limit:
            raise ValueError(f"Prepared index operation exceeds {limit} bytes")
        op_id = self.index_state.stage(operation)
        self.apply_operation(operation)
        self.index_state.finish(op_id)

    def remove_tracked(self, source: Path) -> bool:
        path = str(source.resolve())
        existing = self.manifest.get_entry(path)
        if existing is None:
            return False
        self.commit([{"kind": "delete", "path": path, "old": [existing[1], existing[2]]}])
        return True

    def ingest(
        self,
        target_path: str | Path,
        options: IngestOptions | dict[str, Any] | None = None,
    ) -> IngestionStats:
        opts = merge_options(IngestOptions(), options)
        code_col = opts.code_collection or settings.collections.code
        docs_col = opts.docs_collection or settings.collections.docs
        if opts.reset and opts.reconcile:
            raise ValueError("Choose reset or reconcile for one operation")
        names = list(dict.fromkeys([code_col, docs_col]))
        for name in names:
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]", name)
                or ".." in name or re.fullmatch(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", name)
            ):
                raise ValueError(f"Invalid collection name: {name}")
        md_size = opts.md_chunk_size if opts.md_chunk_size is not None else self.md_chunker.chunk_size
        md_overlap = opts.md_chunk_overlap if opts.md_chunk_overlap is not None else self.md_chunker.chunk_overlap
        code_size = opts.code_max_chars if opts.code_max_chars is not None else self.code_chunker.max_chars
        if md_overlap >= md_size:
            raise ValueError("Markdown chunk overlap must be smaller than the chunk size")
        target = Path(target_path).expanduser()
        if opts.reconcile and not target.is_dir():
            raise ValueError("Reconciliation requires an existing directory")
        records = self.loader.scan(target)
        if opts.reset and not records:
            raise ValueError("Reset requires at least one supported nonempty source")
        self.recover()

        stats = IngestionStats(scanned_files=len(records))
        prepared = []
        prepared_size = 0
        reset_dimensions: dict[str, int] = {}
        for record in records:
            path = record.resolved_path
            code = record.extension == ".py"
            collection = code_col if code else docs_col
            chunk_config = (code_size, self.code_chunker.overlap_chars) if code else (md_size, md_overlap)
            model_id = self.chroma.model_identity
            fingerprint = sha256_hex(json.dumps([CHUNK_FORMAT_VERSION, model_id, chunk_config]))
            existing = None if opts.reset else self.manifest.get_entry(path)
            if not opts.reset:
                self.chroma.check_contract(collection, model_id)
            if existing and existing[:2] == (record.file_hash, collection):
                if self.index_state.fingerprint(path) == fingerprint:
                    stats.skipped_unchanged += 1
                    continue
            chunks = (
                self.code_chunker.chunk(record, max_chars=code_size) if code
                else self.md_chunker.chunk(record, chunk_size=md_size, chunk_overlap=md_overlap)
            )
            vectors = self.chroma.encode_chunks(chunks)
            if vectors:
                dimension = len(vectors[0])
                if opts.reset and reset_dimensions.setdefault(collection, dimension) != dimension:
                    raise ValueError(f"Embedding dimensions changed during reset for {collection}")
                if not opts.reset:
                    self.chroma.check_contract(collection, model_id, dimension)
            change = {
                "kind": "upsert", "path": path, "hash": record.file_hash, "collection": collection,
                "fingerprint": fingerprint, "chunks": [doc.model_dump() for doc in chunks],
                "vectors": vectors, "model_id": model_id, "code": code,
                "old": [existing[1], existing[2]] if existing else None,
            }
            if opts.reset:
                prepared_size += len(json.dumps(change, ensure_ascii=True))
                if prepared_size > settings.runtime.max_reset_journal_bytes:
                    raise ValueError("Prepared reset exceeds the journal size limit")
                prepared.append(change)
            else:
                self.commit([change])
            stats.indexed_files += 1
            stats.total_chunks_upserted += len(chunks)
            stats.total_graph_edges_upserted += sum(len(doc.metadata.get("graph_edges", [])) for doc in chunks)

        if opts.reset:
            if not any(change["chunks"] for change in prepared):
                raise ValueError("Reset requires at least one embeddable chunk")
            self.commit(prepared, reset=names)
        elif opts.reconcile:
            seen = {record.resolved_path for record in records}
            root = target.resolve()
            tracked = self.index_state.paths_under(root, set(names))
            for path in sorted(tracked - seen):
                source = Path(path)
                if self.loader.ignored_dirs.intersection(source.relative_to(root).parts[:-1]):
                    continue
                stats.removed_files += self.remove_tracked(source)
        return stats
