import hashlib
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from store.chroma_store import ChromaStoreManager
from store.graph_store import CodeGraphStore
from store.hybrid_store import HybridRetriever
from store.job_store import JobStore
from store.manifest_store import ManifestStore
from store.markdown_store import MarkdownVaultStore


class DeterministicFakeEmbeddings(Embeddings):
    """Fast, offline 16-dim bag-of-characters embedding for unit testing ChromaDB."""

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
def isolated_stores(tmp_path: Path):
    """Creates isolated SQLite, ChromaDB, and Markdown vault instances inside pytest's tmp_path."""
    db_path = tmp_path / "test_state.sqlite3"
    chroma_dir = tmp_path / "test_chroma"
    vault_dir = tmp_path / "test_vault"
    fake_embed = DeterministicFakeEmbeddings()

    manifest = ManifestStore(db_path=db_path)
    jobs = JobStore(db_path=db_path)
    vault = MarkdownVaultStore(vault_dir=vault_dir)
    graph = CodeGraphStore(db_path=db_path)
    chroma = ChromaStoreManager(
        persist_dir=chroma_dir,
        db_path=db_path,
        embeddings=fake_embed,
        batch_size=8,
    )
    retriever = HybridRetriever(chroma=chroma, graph=graph, db_path=db_path)

    return {
        "manifest": manifest,
        "jobs": jobs,
        "vault": vault,
        "graph": graph,
        "chroma": chroma,
        "retriever": retriever,
    }


def test_manifest_store_lifecycle(isolated_stores):
    manifest: ManifestStore = isolated_stores["manifest"]

    assert manifest.get_entry("/repo/utils.py") is None

    manifest.upsert_entry(
        file_path="/repo/utils.py",
        file_hash="hash_v1",
        collection_name="reusable_code",
        chunk_ids=["c1", "c2"],
    )
    entry = manifest.get_entry("/repo/utils.py")
    assert entry == ("hash_v1", "reusable_code", ["c1", "c2"])

    # Update entry with new hash and fewer chunks
    manifest.upsert_entry(
        file_path="/repo/utils.py",
        file_hash="hash_v2",
        collection_name="reusable_code",
        chunk_ids=["c3"],
    )
    assert manifest.get_entry("/repo/utils.py") == ("hash_v2", "reusable_code", ["c3"])

    # Delete entry and verify old chunk IDs are returned for cleanup
    deleted_ids = manifest.delete_entry("/repo/utils.py")
    assert deleted_ids == ["c3"]
    assert manifest.get_entry("/repo/utils.py") is None


def test_job_store_bidirectional_messaging_and_url_ttl(isolated_stores):
    jobs: JobStore = isolated_stores["jobs"]

    job_id = jobs.create_job(
        task="Add retry wrapper to HTTP client",
        codebase_context="Python 3.12, httpx installed",
        skills_needed=["python", "async-http"],
    )

    # Coordinator asks a clarifying question -> status becomes needs_context
    jobs.post_message(
        job_id=job_id,
        sender="coordinator",
        message_type="question",
        content="Should retries apply to HTTP 429 and 503 status codes?",
        new_status="needs_context",
    )
    snap = jobs.get_job_snapshot(job_id)
    assert snap["status"] == "needs_context"
    assert snap["pending_question"] == "Should retries apply to HTTP 429 and 503 status codes?"

    # Main agent replies -> status moves back to running and pending_question clears
    jobs.post_message(
        job_id=job_id,
        sender="main_agent",
        message_type="answer",
        content="Yes, retry 429 and 5xx with exponential backoff.",
        new_status="running",
    )
    snap = jobs.get_job_snapshot(job_id)
    assert snap["status"] == "running"
    assert snap["pending_question"] is None
    assert len(snap["messages"]) == 2

    # Mark job completed
    jobs.update_job_status(
        job_id=job_id,
        status="completed",
        stage="done",
        final_plan="1. Add retry_async helper...",
    )
    final_snap = jobs.get_job_snapshot(job_id)
    assert final_snap["status"] == "completed"
    assert "retry_async" in final_snap["final_plan"]

    # URL freshness cache check
    url = "https://crawl4ai.com"
    assert jobs.is_url_fresh(url, ttl_hours=24) is False
    jobs.record_crawled_url(
        url=url,
        topic="crawl4ai",
        content_hash="abc123",
        markdown_path="/vault/crawl4ai.md",
    )
    assert jobs.is_url_fresh(url, ttl_hours=24) is True
    assert jobs.is_url_fresh(url, ttl_hours=0) is False


def test_markdown_vault_store(isolated_stores):
    vault: MarkdownVaultStore = isolated_stores["vault"]

    saved_path = vault.save_markdown(
        content="# Pruning Filter\nUse `PruningContentFilter` to strip navbars.",
        title="Crawl4AI Pruning Guide",
        source_url="https://crawl4ai.com",
        topic="web-crawling",
        extra_meta={"skills": "crawl4ai,rag"},
    )
    assert saved_path.exists()
    raw = saved_path.read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert "source_url: https://crawl4ai.com" in raw
    assert "# Pruning Filter" in raw


def test_hybrid_search_and_1hop_ast_graph_expansion(isolated_stores):
    chroma: ChromaStoreManager = isolated_stores["chroma"]
    graph: CodeGraphStore = isolated_stores["graph"]
    retriever: HybridRetriever = isolated_stores["retriever"]

    # Seed two functions where `fetch_json` calls `retry_async`
    helper_doc = Document(
        page_content="async def retry_async(fn, retries=3):\n    # exponential backoff loop\n    pass",
        metadata={
            "chunk_id": "retry_utils.py::retry_async::0",
            "source": "retry_utils.py",
            "symbol_name": "retry_async",
            "type": "python_code",
        },
    )
    caller_doc = Document(
        page_content="async def fetch_json(url: str) -> dict:\n    return await retry_async(lambda: get(url))",
        metadata={
            "chunk_id": "retry_utils.py::fetch_json::0",
            "source": "retry_utils.py",
            "symbol_name": "fetch_json",
            "type": "python_code",
        },
    )
    unrelated_doc = Document(
        page_content="def parse_csv_rows(path: str) -> list[dict]:\n    pass",
        metadata={
            "chunk_id": "csv_utils.py::parse_csv_rows::0",
            "source": "csv_utils.py",
            "symbol_name": "parse_csv_rows",
            "type": "python_code",
        },
    )

    chroma.upsert_chunks("reusable_code", [helper_doc, caller_doc, unrelated_doc])

    # Register AST nodes & `fetch_json --calls--> retry_async` edge
    graph.upsert_node_with_edges(
        chunk_id="retry_utils.py::retry_async::0",
        symbol_name="retry_async",
        symbol_type="function",
        file_path="/repo/retry_utils.py",
        source="retry_utils.py",
        collection_name="reusable_code",
        edges=[],
    )
    graph.upsert_node_with_edges(
        chunk_id="retry_utils.py::fetch_json::0",
        symbol_name="fetch_json",
        symbol_type="function",
        file_path="/repo/retry_utils.py",
        source="retry_utils.py",
        collection_name="reusable_code",
        edges=[("retry_async", "calls")],
    )

    # Query specifically for `fetch_json`
    hits = retriever.search(
        query="fetch_json",
        collection_name="reusable_code",
        mode="hybrid",
        k=1,
        max_distance=1.5,
        expand_ast_neighbors=True,
    )

    matched_ids = [h.document.metadata["chunk_id"] for h in hits]
    assert "retry_utils.py::fetch_json::0" in matched_ids
    # Verify 1-hop AST dependency `retry_async` was automatically attached
    assert "retry_utils.py::retry_async::0" in matched_ids
    graph_hit = next(h for h in hits if h.document.metadata["chunk_id"] == "retry_utils.py::retry_async::0")
    assert graph_hit.via_graph_relation == "calls:retry_async"


def test_keyword_filters_and_collection_scoped_delete(isolated_stores):
    chroma = isolated_stores["chroma"]
    retriever = isolated_stores["retriever"]
    docs = [
        Document(page_content="shared identifier", metadata={"chunk_id": "one", "source": "keep"}),
        Document(page_content="shared identifier", metadata={"chunk_id": "two", "source": "exclude"}),
    ]
    chroma.upsert_chunks("first_collection", docs)
    chroma.upsert_chunks("second_collection", docs)
    chroma.delete_ids("second_collection", ["one"])

    hits = retriever.search("shared", {
        "collection_name": "first_collection",
        "mode": "keyword",
        "where": {"source": "keep"},
    })
    assert [hit.metadata["chunk_id"] for hit in hits] == ["one"]
    hits = retriever.search("shared", "second_collection", mode="keyword")
    assert [hit.metadata["chunk_id"] for hit in hits] == ["two"]
