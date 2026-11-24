# crawler/fleet.py (public helper methods: default_crawler_factory, normalize_targets, extract_title, crawl_single)
def default_crawler_factory() -> AsyncWebCrawler:
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    return AsyncWebCrawler(config=browser_cfg)


class CrawlFleet:
    def __init__(
        self,
        vault: MarkdownVaultStore | None = None,
        jobs: JobStore | None = None,
        pipeline: EmbeddingPipeline | None = None,
        crawler_factory: Callable[[], Any] | None = None,
    ):
        self.vault = vault or MarkdownVaultStore()
        self.jobs = jobs or JobStore()
        self.pipeline = pipeline or EmbeddingPipeline()
        self.crawler_factory = crawler_factory or default_crawler_factory

    @staticmethod
    def normalize_targets(
        targets: list[CrawlTarget | dict[str, Any] | str],
        default_topic: str = "general",
    ) -> list[CrawlTarget]:
        seen: set[str] = set()
        normalized: list[CrawlTarget] = []
        for item in targets:
            target = (
                CrawlTarget(url=item, topic=default_topic)  # type: ignore[arg-type]
                if isinstance(item, str)
                else CrawlTarget.model_validate(item)
            )
            if target.url_str not in seen:
                seen.add(target.url_str)
                normalized.append(target)
        return normalized

    @staticmethod
    def extract_title(markdown_text: str, fallback_url: str) -> str:
        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()[:120]
        return fallback_url.rstrip("/").split("/")[-1] or "web-page"

    async def crawl_single(
        self,
        crawler: Any,
        target: CrawlTarget,
        semaphore: asyncio.Semaphore,
        run_cfg: CrawlerRunConfig,
    ) -> PageOutcome:
        async with semaphore:
            try:
                result = await crawler.arun(url=target.url_str, config=run_cfg)
                if not result.success or not result.markdown:
                    err = getattr(result, "error_message", None) or "Empty crawl result"
                    return PageOutcome(
                        url=target.url_str, topic=target.topic, status="failed", error=str(err)
                    )

                fit_md = getattr(result.markdown, "fit_markdown", None)
                raw_md = getattr(result.markdown, "raw_markdown", str(result.markdown))
                clean_md = (fit_md or raw_md or "").strip()

                if len(clean_md) < 50:
                    return PageOutcome(
                        url=target.url_str,
                        topic=target.topic,
                        status="failed",
                        error="Extracted content too short (<50 chars)",
                    )

                title = self.extract_title(clean_md, target.url_str)
                extra_meta = {"tags": ",".join(target.tags)} if target.tags else None

                saved_path = await asyncio.to_thread(
                    self.vault.save_markdown,
                    content=clean_md,
                    title=title,
                    source_url=target.url_str,
                    topic=target.topic,
                    extra_meta=extra_meta,
                )
                await asyncio.to_thread(
                    self.jobs.record_crawled_url,
                    url=target.url_str,
                    topic=target.topic,
                    content_hash=sha256_hex(clean_md),
                    markdown_path=str(saved_path),
                    status="indexed",
                )
                return PageOutcome(
                    url=target.url_str,
                    topic=target.topic,
                    status="indexed",
                    markdown_path=str(saved_path),
                    title=title,
                    char_count=len(clean_md),
                )
            except Exception as exc:
                return PageOutcome(
                    url=target.url_str, topic=target.topic, status="failed", error=str(exc)
                )

    async def crawl_urls(
        self,
        targets: list[CrawlTarget | dict[str, Any] | str],
        options: CrawlOptions | dict[str, Any] | None = None,
        default_topic: str = "general",
    ) -> BatchCrawlReport:
        opts = merge_options(CrawlOptions(), options)
        concurrency = opts.max_concurrency or settings.crawler.max_concurrency
        pruning = (
            opts.pruning_threshold
            if opts.pruning_threshold is not None
            else settings.crawler.pruning_threshold
        )
        ttl_hours = (
            opts.cache_ttl_hours
            if opts.cache_ttl_hours is not None
            else settings.crawler.cache_ttl_hours
        )

        normalized = self.normalize_targets(targets, default_topic=default_topic)
        if not normalized:
            return BatchCrawlReport()

        outcomes: list[PageOutcome] = []
        to_crawl: list[CrawlTarget] = []

        for target in normalized:
            if not opts.force_refresh and self.jobs.is_url_fresh(target.url_str, ttl_hours=ttl_hours):
                outcomes.append(
                    PageOutcome(url=target.url_str, topic=target.topic, status="skipped_fresh")
                )
            else:
                to_crawl.append(target)

        if to_crawl:
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=pruning, threshold_type="fixed")
            )
            run_cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=md_generator,
                word_count_threshold=15,
                wait_until=opts.wait_until,
                page_timeout=opts.page_timeout_ms,
            )
            semaphore = asyncio.Semaphore(concurrency)

            async with self.crawler_factory() as crawler:
                tasks = [
                    self.crawl_single(crawler, target, semaphore, run_cfg)
                    for target in to_crawl
                ]
                outcomes.extend(await asyncio.gather(*tasks))

        chunks_indexed = 0
        if opts.auto_embed and any(o.status == "indexed" for o in outcomes):
            ingest_stats = await asyncio.to_thread(self.pipeline.ingest, self.vault.vault_dir)
            chunks_indexed = ingest_stats.total_chunks_upserted

        return BatchCrawlReport(outcomes=outcomes, chunks_indexed=chunks_indexed)

---

# crawler/models.py (normalize_topic validator without leading underscore)
class CrawlTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    topic: str = Field(default="general", min_length=1, max_length=128)
    tags: list[str] = Field(default_factory=list)

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def url_str(self) -> str:
        return str(self.url)

---

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
from store.manifest_store import ManifestStore


class IngestionStats(BaseModel):
    scanned_files: int = Field(default=0, ge=0)
    indexed_files: int = Field(default=0, ge=0)
    skipped_unchanged: int = Field(default=0, ge=0)
    total_chunks_upserted: int = Field(default=0, ge=0)
    total_graph_edges_upserted: int = Field(default=0, ge=0)

    @computed_field
    @property
    def has_changes(self) -> bool:
        return self.indexed_files > 0


class EmbeddingPipeline:
    def __init__(
        self,
        chroma: ChromaStoreManager | None = None,
        manifest: ManifestStore | None = None,
        graph: CodeGraphStore | None = None,
        loader: FileLoader | None = None,
        code_chunker: PythonAstChunker | None = None,
        md_chunker: MarkdownChunker | None = None,
    ):
        self.manifest = manifest or ManifestStore()
        self.chroma = chroma or ChromaStoreManager()
        self.graph = graph or CodeGraphStore()
        self.loader = loader or FileLoader()
        self.code_chunker = code_chunker or PythonAstChunker()
        self.md_chunker = md_chunker or MarkdownChunker()

    def register_code_graph(
        self,
        chunks: list[Document],
        collection_name: str,
    ) -> tuple[list[Document], int]:
        sanitized: list[Document] = []
        edge_count = 0

        for doc in chunks:
            meta = dict(doc.metadata)
            edges: list[tuple[str, str]] = meta.pop("graph_edges", [])
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
            sanitized.append(Document(page_content=doc.page_content, metadata=meta))

        return sanitized, edge_count

    def ingest(
        self,
        target_path: str | Path,
        options: IngestOptions | dict[str, Any] | None = None,
    ) -> IngestionStats:
        opts = merge_options(IngestOptions(), options)
        code_col = opts.code_collection or settings.collections.code
        docs_col = opts.docs_collection or settings.collections.docs

        if opts.reset:
            self.chroma.reset_collections([code_col, docs_col])
            self.manifest.clear_all()
            self.graph.clear_all()

        records = self.loader.scan(target_path)
        stats = IngestionStats(scanned_files=len(records))

        for rec in records:
            is_python = rec.extension == ".py"
            collection = code_col if is_python else docs_col

            if existing := self.manifest.get_entry(rec.resolved_path):
                old_hash, old_col, old_ids = existing
                if old_hash == rec.file_hash:
                    stats.skipped_unchanged += 1
                    continue
                self.chroma.delete_ids(old_col, old_ids)
                if old_col == code_col:
                    self.graph.delete_by_chunk_ids(old_ids)

            if is_python:
                raw_chunks = self.code_chunker.chunk(rec, max_chars=opts.code_max_chars)
                chunks, edges_added = self.register_code_graph(raw_chunks, collection)
                stats.total_graph_edges_upserted += edges_added
            else:
                chunks = self.md_chunker.chunk(
                    rec,
                    chunk_size=opts.md_chunk_size,
                    chunk_overlap=opts.md_chunk_overlap,
                )

            chunk_ids = self.chroma.upsert_chunks(collection, chunks)
            self.manifest.upsert_entry(rec.resolved_path, rec.file_hash, collection, chunk_ids)
            stats.indexed_files += 1
            stats.total_chunks_upserted += len(chunk_ids)

        return stats

---

import ast
import builtins
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from config import settings
from embedding.loaders import RawFileRecord
from utils import sha256_hex

BUILTINS = set(dir(builtins))


class PythonAstChunker:
    def __init__(self, max_chars: int | None = None, overlap_chars: int = 150):
        self.max_chars = max_chars if max_chars is not None else settings.chunking.code_max_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def extract_imports(tree: ast.Module, lines: list[str]) -> tuple[str, list[str]]:
        import_lines, symbols = [], []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.end_lineno:
                import_lines.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
                for alias in node.names:
                    if alias.name != "*":
                        symbols.append(alias.asname or alias.name.split(".")[-1])
        return "\n".join(import_lines), symbols

    @staticmethod
    def extract_edges(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        imported_symbols: list[str],
    ) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if name and name not in BUILTINS:
                    edges.append((name, "inherits"))

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = (
                    child.func.id
                    if isinstance(child.func, ast.Name)
                    else getattr(child.func, "attr", None)
                )
                if callee and callee not in {node.name, *BUILTINS}:
                    edges.append((callee, "calls"))

        referenced = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        edges.extend((imp, "imports") for imp in imported_symbols if imp in referenced)
        return list(dict.fromkeys(edges))

    def build_documents(
        self,
        record: RawFileRecord,
        text: str,
        symbol_name: str,
        symbol_type: str,
        max_chars: int,
        edges: list[tuple[str, str]] | None = None,
    ) -> list[Document]:
        parts = (
            [text]
            if len(text) <= max_chars
            else RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=max_chars,
                chunk_overlap=self.overlap_chars,
            ).split_text(text)
        )
        return [
            Document(
                page_content=part,
                metadata={
                    "chunk_id": f"{record.filename}::{symbol_name}::{idx}::{sha256_hex(part, 16)}",
                    "source": record.filename,
                    "file_path": record.resolved_path,
                    "type": "python_code",
                    "language": "python",
                    "symbol_name": symbol_name,
                    "symbol_type": symbol_type,
                    "graph_edges": (edges or []) if idx == 0 else [],
                },
            )
            for idx, part in enumerate(parts)
        ]

    def chunk(self, record: RawFileRecord, max_chars: int | None = None) -> list[Document]:
        limit = max_chars if max_chars is not None else self.max_chars
        lines = record.content.splitlines()
        try:
            tree = ast.parse(record.content)
        except SyntaxError:
            return self.build_documents(record, record.content, "module_block", "module", limit)

        imports_block, imported_symbols = self.extract_imports(tree, lines)
        chunks: list[Document] = []

        for node in tree.body:
            if (
                not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                or node.name.startswith("_")
                or not node.end_lineno
            ):
                continue

            start = node.decorator_list.lineno if node.decorator_list else node.lineno
            code_block = "\n".join(lines[start - 1 : node.end_lineno])
            sym_type = "class" if isinstance(node, ast.ClassDef) else "function"
            doc = ast.get_docstring(node)
            intent = doc.strip().splitlines() if doc and doc.strip() else f"Python {sym_type} `{node.name}`"

            header = f"# Symbol: {node.name} ({sym_type}) | File: {record.filename}\n# Intent: {intent}\n"
            enriched = f"{header}\n{imports_block}\n\n{code_block}" if imports_block else f"{header}\n{code_block}"
            edges = self.extract_edges(node, imported_symbols)

            chunks.extend(
                self.build_documents(record, enriched, node.name, sym_type, limit, edges=edges)
            )

        return chunks or self.build_documents(record, record.content, "module_block", "module", limit)

---

from functools import lru_cache
from typing import Any
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import EmbeddingClientOptions, merge_options, settings


@lru_cache(maxsize=8)
def build_cached_embedding_client(
    provider: str,
    model_name: str,
    device: str,
    base_url: str,
    api_key: str,
    normalize_embeddings: bool,
    check_ctx_length: bool,
) -> Embeddings:
    if provider == "api":
        return OpenAIEmbeddings(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            check_embedding_ctx_length=check_ctx_length,
        )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def get_embedding_client(
    options: EmbeddingClientOptions | dict[str, Any] | None = None,
) -> Embeddings:
    opts = merge_options(EmbeddingClientOptions(), options)
    provider = opts.provider or settings.embedding.provider

    if provider == "api":
        api_cfg = merge_options(settings.embedding.api, opts)
        return build_cached_embedding_client(
            provider="api",
            model_name=api_cfg.model_name,
            device="",
            base_url=api_cfg.base_url,
            api_key=api_cfg.api_key,
            normalize_embeddings=True,
            check_ctx_length=api_cfg.check_ctx_length,
        )

    local_cfg = merge_options(settings.embedding.local, opts)
    return build_cached_embedding_client(
        provider="local",
        model_name=local_cfg.model_name,
        device=local_cfg.device,
        base_url="",
        api_key="",
        normalize_embeddings=local_cfg.normalize_embeddings,
        check_ctx_length=False,
    )

---

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from sqlalchemy import text
from config import SearchOptions, merge_options, settings
from store.chroma_store import ChromaStoreManager
from store.graph_store import CodeGraphStore
from store.session import get_session


@dataclass
class RetrievalHit:
    document: Document
    rrf_score: float
    vector_distance: float | None = None
    bm25_rank: int | None = None
    via_graph_relation: str | None = None
    metadata: dict = field(default_factory=dict)


def build_fts_query(raw_query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", raw_query)
    if not tokens:
        return ""
    unique_tokens = list(dict.fromkeys(tokens))[:12]
    return " OR ".join(f'"{tok}"' for tok in unique_tokens)


class HybridRetriever:
    def __init__(
        self,
        chroma: ChromaStoreManager | None = None,
        graph: CodeGraphStore | None = None,
        db_path: str | Path | None = None,
    ):
        self.db_path = db_path
        self.chroma = chroma if chroma is not None else ChromaStoreManager(db_path=db_path)
        self.graph = graph if graph is not None else CodeGraphStore(db_path=db_path)

    def search_bm25(self, query: str, collection_name: str, limit: int) -> list[str]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        with get_session(self.db_path) as session:
            rows = session.execute(
                text(
                    """
                    SELECT chunk_id
                    FROM chunks_fts
                    WHERE chunks_fts MATCH :q AND collection_name = :col
                    ORDER BY bm25(chunks_fts)
                    LIMIT :lim
                    """
                ),
                {"q": fts_query, "col": collection_name, "lim": limit},
            ).all()
        return [r for r in rows]

    def search(
        self,
        query: str,
        options: SearchOptions | dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        opts = merge_options(SearchOptions(), options)
        target_col = opts.collection_name or settings.collections.code
        active_mode = opts.mode or settings.retrieval.mode
        top_k = opts.k if opts.k is not None else settings.retrieval.default_top_k
        rrf_k = settings.retrieval.rrf_k
        w_vec = settings.retrieval.vector_weight
        w_kw = settings.retrieval.keyword_weight
        use_graph = (
            opts.expand_ast_neighbors
            if opts.expand_ast_neighbors is not None
            else settings.retrieval.expand_ast_neighbors
        )

        candidate_k = top_k * 2
        doc_map: dict[str, Document] = {}
        vec_dist_map: dict[str, float] = {}
        rrf_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}

        if active_mode in ("hybrid", "vector"):
            vec_hits = self.chroma.search_vector(
                query=query,
                collection_name=target_col,
                k=candidate_k,
                max_distance=opts.max_distance,
                where=opts.where,
            )
            for rank, (doc, dist) in enumerate(vec_hits, start=1):
                cid = doc.metadata["chunk_id"]
                doc_map[cid] = doc
                vec_dist_map[cid] = dist
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_vec * (1.0 / (rrf_k + rank))

        if active_mode in ("hybrid", "keyword"):
            bm25_ids = self.search_bm25(query=query, collection_name=target_col, limit=candidate_k)
            missing_ids = [cid for cid in bm25_ids if cid not in doc_map]
            if missing_ids:
                for doc in self.chroma.get_by_ids(target_col, missing_ids):
                    doc_map[doc.metadata["chunk_id"]] = doc

            for rank, cid in enumerate(bm25_ids, start=1):
                if cid in doc_map:
                    bm25_ranks[cid] = rank
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_kw * (1.0 / (rrf_k + rank))

        ranked_ids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:top_k]
        hits: list[RetrievalHit] = [
            RetrievalHit(
                document=doc_map[cid],
                rrf_score=round(rrf_scores[cid], 5),
                vector_distance=round(vec_dist_map[cid], 4) if cid in vec_dist_map else None,
                bm25_rank=bm25_ranks.get(cid),
                metadata=doc_map[cid].metadata,
            )
            for cid in ranked_ids
        ]

        if use_graph and target_col == settings.collections.code and ranked_ids:
            neighbors = self.graph.get_1hop_neighbor_chunk_ids(
                ranked_ids, limit=settings.retrieval.max_neighbor_chunks
            )
            neighbor_ids = [n["chunk_id"] for n in neighbors]
            neighbor_docs = {
                d.metadata["chunk_id"]: d
                for d in self.chroma.get_by_ids(target_col, neighbor_ids)
            }
            for n in neighbors:
                ndoc = neighbor_docs.get(n["chunk_id"])
                if ndoc:
                    hits.append(
                        RetrievalHit(
                            document=ndoc,
                            rrf_score=0.0,
                            via_graph_relation=f"{n['relation']}:{n['symbol_name']}",
                            metadata=ndoc.metadata,
                        )
                    )

        return hits

---

import re
from datetime import datetime, timezone
from pathlib import Path
import yaml
from config import settings
from utils import resolve_path, sha256_hex


def slugify_text(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].strip("-")) or "untitled"


class MarkdownVaultStore:
    def __init__(self, vault_dir: str | Path | None = None):
        self.vault_dir = resolve_path(
            vault_dir if vault_dir is not None else settings.storage.markdown_vault_dir,
            is_dir=True,
        )

    def save_markdown(
        self,
        content: str,
        title: str,
        source_url: str,
        topic: str = "general",
        extra_meta: dict | None = None,
        vault_dir: str | Path | None = None,
    ) -> Path:
        target_dir = resolve_path(vault_dir, is_dir=True) if vault_dir is not None else self.vault_dir
        url_hash = sha256_hex(source_url, 10)
        filename = f"{slugify_text(topic)}_{slugify_text(title)}_{url_hash}.md"
        file_path = target_dir / filename

        frontmatter = {
            "title": title,
            "source_url": source_url,
            "topic": topic,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            **(extra_meta or {}),
        }
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        file_path.write_text(f"---\n{yaml_block}\n---\n\n{content.strip()}\n", encoding="utf-8")
        return file_path

---

from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from config import settings
from store.models import Base

ENGINE_CACHE: dict[str, Engine] = {}
SESSION_FACTORY_CACHE: dict[str, sessionmaker] = {}


def configure_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def get_engine(db_path: str | Path | None = None) -> Engine:
    resolved = str(
        Path(db_path).resolve()
        if db_path is not None
        else settings.storage.sqlite_db_path.resolve()
    )
    if resolved not in ENGINE_CACHE:
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{resolved}",
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )
        event.listen(engine, "connect", configure_sqlite_pragmas)
        ENGINE_CACHE[resolved] = engine
    return ENGINE_CACHE[resolved]


def get_session_factory(db_path: str | Path | None = None) -> sessionmaker:
    resolved = str(
        Path(db_path).resolve()
        if db_path is not None
        else settings.storage.sqlite_db_path.resolve()
    )
    if resolved not in SESSION_FACTORY_CACHE:
        engine = get_engine(resolved)
        SESSION_FACTORY_CACHE[resolved] = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
    return SESSION_FACTORY_CACHE[resolved]


def init_store(db_path: str | Path | None = None) -> None:
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    collection_name UNINDEXED,
                    source,
                    symbol_name,
                    content,
                    tokenize = 'porter unicode61 tokenchars "_"'
                );
                """
            )
        )


@contextmanager
def get_session(db_path: str | Path | None = None) -> Generator[Session, None, None]:
    init_store(db_path)
    factory = get_session_factory(db_path)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

---

from datetime import datetime, timezone
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class FileManifest(Base):
    __tablename__ = "file_manifest"

    file_path: Mapped[str] = mapped_column(String(1024), primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class CodeNode(Base):
    __tablename__ = "code_nodes"

    chunk_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    symbol_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    symbol_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)


class CodeEdge(Base):
    __tablename__ = "code_edges"
    __table_args__ = (
        UniqueConstraint("source_chunk_id", "target_symbol", "relation", name="uq_code_edge"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_chunk_id: Mapped[str] = mapped_column(
        String(256), ForeignKey("code_nodes.chunk_id", ondelete="CASCADE"), index=True
    )
    target_symbol: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String(32), nullable=False)


class PlanJob(Base):
    __tablename__ = "plan_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    codebase_context: Mapped[str] = mapped_column(Text, default="")
    skills_needed: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(128), default="initialized")
    final_plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="AgentMessage.id"
    )


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("plan_jobs.job_id", ondelete="CASCADE"), index=True
    )
    sender: Mapped[str] = mapped_column(String(32), nullable=False)
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped["PlanJob"] = relationship(back_populates="messages")


class CrawledUrl(Base):
    __tablename__ = "crawled_urls"

    url: Mapped[str] = mapped_column(String(2048), primary_key=True)
    topic: Mapped[str] = mapped_column(String(256), default="general", index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    markdown_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="indexed")
    crawled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

---

"""
Configuration & Override Models
===============================
Purpose:
    Loads `default_config.yaml` with `${ENV_VAR}` expansion into typed Pydantic
    settings, and defines compact override models (`IngestOptions`, `SearchOptions`,
    `EmbeddingClientOptions`) so methods accept a single options dict or model.
"""

import os
import re
from pathlib import Path
from typing import Any, Literal, TypeVar
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"
TModel = TypeVar("TModel", bound=BaseModel)


def merge_options(defaults: TModel, overrides: BaseModel | dict[str, Any] | None) -> TModel:
    """Return a copy of `defaults` updated with non-None fields from `overrides`."""
    if overrides is None:
        return defaults
    raw = (
        overrides.model_dump(exclude_none=True)
        if isinstance(overrides, BaseModel)
        else {k: v for k, v in overrides.items() if v is not None}
    )
    return defaults.model_copy(update=raw) if raw else defaults


def expand_env_vars(value: Any) -> Any:
    """Recursively resolve `${VAR_NAME}` placeholders in loaded YAML values."""
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


class StorageConfig(BaseModel):
    base_dir: Path = Path("./.rag_memory_data")
    chroma_dir: Path = Path("./.rag_memory_data/chroma_db")
    sqlite_db_path: Path = Path("./.rag_memory_data/state.sqlite3")
    markdown_vault_dir: Path = Path("./.rag_memory_data/vault/md")

    def ensure_dirs(self) -> None:
        for d in (self.base_dir, self.chroma_dir, self.markdown_vault_dir, self.sqlite_db_path.parent):
            d.mkdir(parents=True, exist_ok=True)


class CollectionsConfig(BaseModel):
    code: str = "reusable_code"
    docs: str = "research_docs"


class LocalEmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True


class ApiEmbeddingConfig(BaseModel):
    base_url: str = "https://openai.com"
    api_key: str = ""
    model_name: str = "text-embedding-3-small"
    check_ctx_length: bool = False


class EmbeddingConfig(BaseModel):
    provider: Literal["local", "api"] = "local"
    batch_size: int = 32
    local: LocalEmbeddingConfig = LocalEmbeddingConfig()
    api: ApiEmbeddingConfig = ApiEmbeddingConfig()


class CodexLlmConfig(BaseModel):
    cli_path: str = "codex"
    model_name: str = ""
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    timeout_seconds: int = 120


class OllamaLlmConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5-coder:3b"


class ApiLlmConfig(BaseModel):
    base_url: str = "https://openrouter.ai"
    api_key: str = ""
    model_name: str = "openrouter/free"


class LlmConfig(BaseModel):
    provider: Literal["codex", "ollama", "api"] = "codex"
    temperature: float = 0.0
    codex: CodexLlmConfig = CodexLlmConfig()
    ollama: OllamaLlmConfig = OllamaLlmConfig()
    api: ApiLlmConfig = ApiLlmConfig()


class ChunkingConfig(BaseModel):
    md_chunk_size: int = 900
    md_chunk_overlap: int = 100
    code_max_chars: int = 2400


class RetrievalConfig(BaseModel):
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    rrf_k: int = 60
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    expand_ast_neighbors: bool = True
    max_neighbor_chunks: int = 2
    code_max_distance: float = 0.38
    docs_max_distance: float = 0.50
    default_top_k: int = 4


class CrawlerConfig(BaseModel):
    max_concurrency: int = 4
    pruning_threshold: float = 0.45
    cache_ttl_hours: int = 168


class EmbeddingClientOptions(BaseModel):
    provider: Literal["local", "api"] | None = None
    model_name: str | None = None
    device: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    normalize_embeddings: bool | None = None


class IngestOptions(BaseModel):
    reset: bool = False
    code_collection: str | None = None
    docs_collection: str | None = None
    md_chunk_size: int | None = None
    md_chunk_overlap: int | None = None
    code_max_chars: int | None = None


class SearchOptions(BaseModel):
    collection_name: str | None = None
    mode: Literal["hybrid", "vector", "keyword"] | None = None
    k: int | None = None
    max_distance: float | None = None
    expand_ast_neighbors: bool | None = None
    where: dict[str, Any] | None = None


class AppConfig(BaseModel):
    storage: StorageConfig = StorageConfig()
    collections: CollectionsConfig = CollectionsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LlmConfig = LlmConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    crawler: CrawlerConfig = CrawlerConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    target = Path(config_path or os.getenv("RAG_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    raw_data = {}
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        raw_data = expand_env_vars(loaded)
    cfg = AppConfig(**raw_data)
    cfg.storage.ensure_dirs()
    return cfg


settings = load_config()

---

The overall goal of rag-memory-system is to act as a read-only, persistent research, architectural planning, and reusable code memory fleet for CLI coding agents (such as Codex CLI and Claude Code)—preventing them from rewriting solved logic from scratch or relying on outdated web knowledge.
Instead of editing your codebase directly, it operates as a standalone MCP (Model Context Protocol) tool server that receives high-level planning tasks from your main coding agent, clarifies ambiguities, searches local and crawled knowledge via Hybrid + AST Graph retrieval, and returns a verified, runnable Implementation Blueprint.

1. Core Problems Solved


Problem in CLI Coding Agents	How rag-memory-system Solves It
Amnesia across sessions	Verified helper functions, patterns, and crawled documentation persist locally in ChromaDB + SQLite + a Markdown vault, accumulating institutional memory over time.
Wasted tokens & broken snippets	Instead of stuffing entire files into context or returning half-cut code blocks, Python AST chunking extracts complete functions/classes with imports attached, and a 1-hop AST dependency graph automatically pulls called helper functions.
Stale or hallucinated library APIs	When local memory misses, a parallel Crawl4AI subagent fleet scrapes current documentation, strips HTML noise, saves clean Markdown files, and indexes them before drafting the plan.
Unsafe direct file edits during research	Subagents run inside an isolated virtual filesystem sandbox (StateBackend + jailed /vault/) and read-only Codex CLI subprocesses—never touching your active Git repository.
Under-specified prompts	A LangGraph deepagents Coordinator detects missing repo constraints early and pauses via interrupt() to message your main CLI agent (needs_context) before researching or planning.


2. End-to-End Operational Lifecycle

Show Code
Download diagram SVG

3. Architectural Pillars & Scale Target

• Dual-Mode Deployment (1 Laptop to 1–100 Users): Runs with zero external database servers on an individual developer machine, and scales cleanly to 1–100 concurrent users over a LAN home server or AWS instance using SQLite WAL mode, embedded ChromaDB, and a shared Markdown vault.
• Polyglot Persistence (store/):
	• ChromaDB: Dual collections (reusable_code with strict 0.38 cutoff; research_docs with 0.50 cutoff).
	• SQLite (WAL + FTS5): ACID job queue, bidirectional Coordinator <-> Main Agent mailbox, URL TTL cache, BM25 keyword search, and AST dependency edges.
	• Markdown Vault: Human-readable .md files with YAML frontmatter so crawled data can be audited or re-indexed anytime without re-crawling.
• Self-Compounding Feedback Loop: Every completed task feeds tested, verified helper functions back into reusable_code via commit_verified_snippet(), making each future planning request faster, cheaper, and more reliable.