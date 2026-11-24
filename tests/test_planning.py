import asyncio
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from config import SearchOptions, settings
from crawler.fleet import CrawlFleet
from embedding.pipeline import EmbeddingPipeline
from planning.blueprint import BlueprintPlanner
from planning.coordinator import PlanningCoordinator
from store.chroma_store import ChromaStoreManager
from store.hybrid_store import HybridRetriever
from store.job_store import JobStore
from store.markdown_store import MarkdownVaultStore


class PlanningTestModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def planning_store(tmp_path):
    db = tmp_path / "state.sqlite3"
    chroma = ChromaStoreManager(
        persist_dir=tmp_path / "chroma",
        db_path=db,
        embeddings=DeterministicFakeEmbedding(size=16),
    )
    return JobStore(db), EmbeddingPipeline(chroma=chroma), HybridRetriever(chroma=chroma)


def test_planning_resumes_with_same_mailbox_and_evidence(tmp_path, planning_store):
    jobs, pipeline, retrieval = planning_store
    source = tmp_path / "utils.py"
    source.write_text("def retry_delay(attempt):\n    return 2 ** attempt\n")
    pipeline.ingest(source)
    assert retrieval.search("retry_delay", options=SearchOptions(mode="keyword"))
    plan = {
        "summary": "Reuse the exponential retry helper.",
        "steps": [{"action": "Reuse retry_delay in the transport module.", "evidence_ids": ["E1"]}],
        "validation": ["Run python -m pytest."],
        "open_questions": [],
    }
    response = AIMessage(content="", tool_calls=[{"name": "Blueprint", "args": plan, "id": "plan"}])
    planner = BlueprintPlanner(model=PlanningTestModel(responses=[response]))
    job_id = PlanningCoordinator(jobs, retrieval, planner=planner).submit("retry_delay")
    paused = PlanningCoordinator(jobs, retrieval, planner=planner).advance(job_id)
    assert paused["status"] == "needs_context"
    assert len(paused["messages"]) == 1

    resumed = PlanningCoordinator(jobs, retrieval, planner=planner).advance(job_id, "Python 3.12")
    assert resumed["status"] == "completed"
    assert resumed["blueprint"]["steps"] == plan["steps"]
    assert resumed["blueprint"]["sources"][0]["location"] == str(source.resolve())
    assert [message["message_type"] for message in resumed["messages"]] == ["question", "answer"]
    assert PlanningCoordinator(jobs, retrieval, planner=planner).advance(job_id) == resumed


def test_vault_read_rejects_traversal_and_external_symlinks(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "guide.md").write_text("guide")
    private = tmp_path / "private.md"
    private.write_text("outside")
    (vault / "link.md").symlink_to(private)
    planner = BlueprintPlanner(vault_dir=vault)
    assert planner.read_vault_document("/vault/guide.md") == "guide"
    for path in ("/vault/../private.md", "/vault/link.md", str(private)):
        with pytest.raises(ValueError, match="vault|Vault"):
            planner.read_vault_document(path)


def test_crawl_cache_requires_indexing(tmp_path, planning_store, monkeypatch):
    jobs, pipeline, retrieval = planning_store
    requested = "https://docs.python.org/3/library/example.html"

    class TestCrawler:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def arun(self, url, config):
            assert url.startswith("raw:<h1>Sample</h1>")
            markdown = "# Sample\n\nThe searchable sentinel word is documented here."
            return SimpleNamespace(success=True, markdown=SimpleNamespace(fit_markdown=markdown))

    async def validate_url(url):
        assert url == requested

    monkeypatch.setattr(CrawlFleet, "validate_url", staticmethod(validate_url))
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<h1>Sample</h1>", headers={"content-type": "text/html"})
    )
    fleet = CrawlFleet(
        MarkdownVaultStore(tmp_path / "vault"), jobs, pipeline, TestCrawler,
        http_transport=transport,
    )

    saved = asyncio.run(fleet.crawl_urls([requested], {"auto_embed": False}))
    assert saved.outcomes[0].status == "saved"
    assert not jobs.is_url_fresh(requested)

    indexed = asyncio.run(fleet.crawl_urls([requested]))
    assert indexed.outcomes[0].status == "indexed"
    assert indexed.chunks_indexed == 1
    assert jobs.is_url_fresh(requested)
    hits = retrieval.search("sentinel", settings.collections.docs, mode="keyword")
    assert hits[0].metadata["source_url"] == requested
    assert asyncio.run(fleet.crawl_urls([requested])).outcomes[0].status == "skipped_fresh"
