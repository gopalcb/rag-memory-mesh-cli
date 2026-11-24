import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from embedding.pipeline import EmbeddingPipeline
from planning.blueprint import BlueprintPlanner, Evidence
from planning.coordinator import PlanningCoordinator
from store.chroma_store import ChromaStoreManager
from store.hybrid_store import HybridRetriever
from utils import sha256_hex


class EvaluatedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def retrieval_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "retry.py").write_text(
        'def backoff_delay(attempt):\n    """Compute an exponential backoff schedule."""\n'
        '    return 2 ** attempt\n'
    )
    (corpus / "markdown.md").write_text(
        "# Markdown output\n\nPreserve breadcrumb headings when rendering a Markdown section."
    )
    (corpus / "unrelated.py").write_text("def trim_csv(value):\n    return value.strip()\n")
    chroma = ChromaStoreManager(
        persist_dir=tmp_path / "chroma", db_path=tmp_path / "state.sqlite3",
        embeddings=DeterministicFakeEmbedding(size=16),
    )
    EmbeddingPipeline(chroma=chroma).ingest(corpus)
    return HybridRetriever(chroma=chroma)


@pytest.mark.parametrize(("query", "collection", "source"), [
    ("backoff_delay exponential schedule", "reusable_code", "retry.py"),
    ("Markdown breadcrumb headings", "research_docs", "markdown.md"),
    ("trim_csv", "reusable_code", "unrelated.py"),
])
def test_expected_source_at_top_rank(retrieval_corpus, query, collection, source):
    hits = retrieval_corpus.search(query, collection, mode="hybrid", k=1, expand_ast_neighbors=False)
    assert hits and hits[0].metadata["source"] == source


def test_unanswerable_exact_query_has_no_lexical_evidence(retrieval_corpus):
    hits = retrieval_corpus.search(
        "unobserved_protocol_7305", "research_docs", mode="keyword", expand_ast_neighbors=False,
    )
    assert hits == []


@pytest.mark.parametrize(("ids", "expected"), [(["E1"], "cited_sources"), ([], "proposal")])
def test_blueprint_preserves_citation_identity(tmp_path, ids, expected):
    content = "Render Markdown section breadcrumbs."
    evidence = Evidence(id="E1", collection="research_docs", source="markdown.md",
                        location="/fixtures/markdown.md", content=content)
    payload = {
        "summary": "Render a breadcrumb.",
        "steps": [{"action": "Use section metadata.", "evidence_ids": ids}],
        "validation": ["Check breadcrumb appears at the top."],
        "open_questions": [],
    }
    model = EvaluatedModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "Blueprint", "args": payload, "id": "blueprint"}]),
    ])
    from store.job_store import JobStore
    coordinator = PlanningCoordinator(jobs=JobStore(tmp_path / "state.sqlite3"),
                                      planner=BlueprintPlanner(model=model))
    result = coordinator.draft({
        "task": "Render breadcrumbs.", "context": "Markdown.", "evidence": [evidence.model_dump()],
        "crawl_outcomes": [],
    })["artifact"]
    assert result["grounding"] == expected
    assert result["sources"][0]["excerpt_sha256"] == sha256_hex(content)
    assert result["steps"][0]["evidence_ids"] == ids


def test_blueprint_rejects_invented_citation():
    payload = {"summary": "Plan", "steps": [{"action": "Change source", "evidence_ids": ["E9"]}],
               "validation": ["Run tests"], "open_questions": []}
    model = EvaluatedModel(responses=[AIMessage(content=json.dumps(payload))])
    with pytest.raises(ValueError, match="unknown evidence IDs: E9"):
        BlueprintPlanner(model=model).draft("Plan the change.", "Python.", [])
