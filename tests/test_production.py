import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import load_config, settings
from embedding.pipeline import EmbeddingPipeline
from model_policy import require_external_permission
from planning.blueprint import Evidence
from planning.coordinator import PlanningCoordinator
from rag_cli import exit_status
from store.audit import audit_index
from store.chroma_store import ChromaStoreManager
from store.hybrid_store import HybridRetriever
from store.job_store import JobStore


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def managed_pipeline(tmp_path):
    chroma = ChromaStoreManager(
        persist_dir=tmp_path / "chroma", db_path=tmp_path / "state.sqlite3",
        embeddings=DeterministicFakeEmbedding(size=16), batch_size=1, model_identity="first-model",
    )
    return EmbeddingPipeline(chroma=chroma)


def write_python(path, *names):
    path.write_text(
        "\n\n".join(f"def {name}():\n    return 1" for name in names) + "\n",
        encoding="utf-8",
    )


def test_failed_encode_preserves_published_version(tmp_path, managed_pipeline, monkeypatch):
    pipeline = managed_pipeline
    path = tmp_path / "source.py"
    write_python(path, "published")
    pipeline.ingest(path)
    write_python(path, "first_batch", "second_batch")
    original = pipeline.chroma.embeddings.embed_documents

    def fail_second(texts):
        if "second_batch" in texts[0]:
            raise RuntimeError("embedding unavailable")
        return original(texts)

    with monkeypatch.context() as patch:
        patch.setattr(type(pipeline.chroma.embeddings), "embed_documents", lambda _, texts: fail_second(texts))
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            pipeline.ingest(path)
    assert pipeline.index_state.pending() == []
    assert audit_index(pipeline.chroma)["status"] == "ok"
    assert HybridRetriever(chroma=pipeline.chroma).search("published", mode="keyword")


@pytest.mark.parametrize("failure_point", ["vectors", "fts", "graph", "manifest", "reset"])
def test_recovery_replays_without_embedding_again(tmp_path, managed_pipeline, monkeypatch, failure_point):
    pipeline = managed_pipeline
    path = tmp_path / "source.py"
    write_python(path, "old_symbol")
    pipeline.ingest(path)
    write_python(path, "new_symbol", "another_symbol")

    owners = {
        "fts": (pipeline.chroma, "upsert_chunks"),
        "graph": (pipeline.graph, "upsert_node_with_edges"),
        "manifest": (pipeline.manifest, "upsert_entry"),
        "reset": (pipeline.chroma, "reset_collections"),
    }
    if failure_point == "vectors":
        collection = pipeline.chroma.raw_collection(settings.collections.code)
        owner, method = type(collection), "upsert"
    else:
        owner, method = owners[failure_point]
    original = getattr(owner, method)

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("stopped after write")

    with monkeypatch.context() as patch:
        patch.setattr(owner, method, interrupted)
        with pytest.raises(RuntimeError, match="stopped after write"):
            pipeline.ingest(path, {"reset": failure_point == "reset"})
    assert audit_index(pipeline.chroma)["issues"]["pending_operations"]

    def forbidden(*_, **__):
        pytest.fail("Journal replay must not call the embedding provider")

    monkeypatch.setattr(pipeline.chroma, "encode_chunks", forbidden)
    assert pipeline.recover() == 1
    assert pipeline.recover() == 0
    assert audit_index(pipeline.chroma)["status"] == "ok"
    retriever = HybridRetriever(chroma=pipeline.chroma)
    assert retriever.search("new_symbol", mode="keyword")
    assert not retriever.search("old_symbol", mode="keyword")


def test_reconcile_scope_and_invalid_reset(tmp_path, managed_pipeline):
    reviewed = tmp_path / "repo"
    reviewed.mkdir()
    outside = tmp_path / "repo-other"
    outside.mkdir()
    deleted = reviewed / "deleted.py"
    emptied = reviewed / "empty.py"
    other = outside / "keep.py"
    for path in (deleted, emptied, other):
        write_python(path, path.stem)
        managed_pipeline.ingest(path)
    deleted.unlink()
    emptied.write_text("")
    unsupported = reviewed / "data.txt"
    unsupported.write_text("fixture")
    for target in (unsupported, reviewed):
        with pytest.raises(ValueError, match="Reset requires"):
            managed_pipeline.ingest(target, {"reset": True})
        assert audit_index(managed_pipeline.chroma)["counts"]["dense"] == 3
    assert managed_pipeline.ingest(reviewed, {"reconcile": True}).removed_files == 2
    assert audit_index(managed_pipeline.chroma)["status"] == "ok"
    assert managed_pipeline.manifest.get_entry(str(other.resolve())) is not None


def test_fingerprint_and_same_dimension_model_switch(tmp_path, managed_pipeline):
    path = tmp_path / "document.md"
    path.write_text("# Context\n" + "More context to split into embeddings. " * 40)
    managed_pipeline.ingest(path)
    assert managed_pipeline.ingest(path).skipped_unchanged == 1
    assert managed_pipeline.ingest(path, {"md_chunk_size": 300}).indexed_files == 1

    original = managed_pipeline.chroma
    changed = ChromaStoreManager(
        persist_dir=original.persist_dir, db_path=original.db_path,
        embeddings=DeterministicFakeEmbedding(size=16), model_identity="second-model",
    )
    with pytest.raises(ValueError, match="model or dimensions changed"):
        EmbeddingPipeline(chroma=changed).ingest(path)
    with pytest.raises(ValueError, match="model or dimensions changed"):
        HybridRetriever(chroma=changed).search("context", settings.collections.docs, mode="vector")
    assert EmbeddingPipeline(chroma=changed).ingest(path, {"reset": True}).indexed_files == 1
    assert audit_index(changed)["status"] == "ok"


def test_explicit_config_and_private_profile(tmp_path):
    missing = tmp_path / "absent.yaml"
    with pytest.raises(ValueError, match="Configuration file"):
        load_config(missing)
    assert not (tmp_path / ".rag_memory_data").exists()
    config = tmp_path / "private.yaml"
    config.write_text("security:\n  private_profile: true\n")
    storage = load_config(config).storage
    assert storage.base_dir.stat().st_mode & 0o777 == 0o700
    storage.base_dir.chmod(0o755)
    with pytest.raises(ValueError, match="permissions 0700"):
        load_config(config)


def test_external_provider_denied_without_permission(monkeypatch):
    monkeypatch.setattr(settings.security, "allow_external_models", False)
    require_external_permission("local model", "http://127.0.0.1:11434")
    with pytest.raises(ValueError, match="allow_external_models"):
        require_external_permission("Codex")
    with pytest.raises(ValueError, match="allow_external_models"):
        require_external_permission("API embeddings", "https://example.com/v1")
    monkeypatch.setattr(settings.security, "allow_external_models", True)
    with pytest.raises(ValueError, match="HTTPS"):
        require_external_permission("API embeddings", "http://example.com/v1")


def test_planner_refreshes_explicit_stale_evidence(tmp_path):
    url = "https://docs.python.org/stale"

    class OfflineCrawler:
        async def crawl_urls(self, sources):
            assert sources == [url]
            from crawler.models import PageOutcome
            return SimpleNamespace(outcomes=[PageOutcome(url=url, topic="docs", status="failed")])

    coordinator = PlanningCoordinator(jobs=JobStore(tmp_path / "state.sqlite3"), crawler=OfflineCrawler())
    coordinator.collect_evidence = lambda _task, _sources: [
        Evidence(id="E1", collection="research_docs", source="stale.md", location=url, content="old")
    ]
    result = coordinator.research({"task": "current information", "urls": [url]})
    assert result["evidence"] == []
    assert result["crawl_outcomes"][0]["status"] == "failed"


def test_crawl_exit_status_reports_partial_and_total_failure():
    success, failed = {"status": "indexed"}, {"status": "failed"}
    assert exit_status("crawl", {"outcomes": [success]}) == 0
    assert exit_status("crawl", {"outcomes": [failed]}) == 1
    assert exit_status("crawl", {"outcomes": [failed, success]}) == 2


def command(config, *args, ok=True):
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT,
        env=dict(os.environ, RAG_CONFIG_PATH=str(config)), capture_output=True, text=True, timeout=35,
    )
    if ok:
        assert result.returncode == 0, result.stderr or result.stdout
    return result


def test_interprocess_lock_rejects_second_writer(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("runtime:\n  lock_timeout_seconds: 0.2\n")
    hold = "from store.runtime import index_lock\nwith index_lock():\n print('ready', flush=True)\n input()"
    with subprocess.Popen(
        [sys.executable, "-c", hold], cwd=ROOT, env=dict(os.environ, RAG_CONFIG_PATH=str(config)),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as first:
        try:
            assert first.stdout.readline().strip() == "ready"
            blocked = command(config, "rag_cli.py", "recover", ok=False)
            assert blocked.returncode == 1 and "Index is busy" in blocked.stderr
        finally:
            first.stdin.write("\n")
            first.stdin.flush()
    assert json.loads(command(config, "rag_cli.py", "recover").stdout)["operations_replayed"] == 0


def test_cli_backup_restore_drill(tmp_path):
    active = tmp_path / "active.yaml"
    restored = tmp_path / "restored.yaml"
    active.write_text("storage:\n  base_dir: active\n  chroma_dir: active/chroma\n"
                      "  sqlite_db_path: active/state.sqlite3\n  markdown_vault_dir: active/vault/md\n")
    restored.write_text(active.read_text().replace("active", "restored"))
    seed = r"""
from langchain_core.embeddings import DeterministicFakeEmbedding
from config import settings
from embedding.pipeline import EmbeddingPipeline
from store.chroma_store import ChromaStoreManager
from store.markdown_store import MarkdownVaultStore
from planning.coordinator import PlanningCoordinator

path = MarkdownVaultStore().save_markdown("# Recovery\n\nRecovery sentinel reference.", "Recovery",
                                         "https://docs.python.org/recovery")
EmbeddingPipeline(chroma=ChromaStoreManager(embeddings=DeterministicFakeEmbedding(size=16))).ingest(path)
code = settings.storage.base_dir / "vault" / "verified_code"
code.mkdir()
(code / "saved.py").write_text("x = 1\n")
coordinator = PlanningCoordinator()
job = coordinator.submit("Plan the restored task.")
assert coordinator.advance(job)["status"] == "needs_context"
print(job)
"""
    job_id = command(active, "-c", seed).stdout.strip()
    backup = tmp_path / "snapshot"
    assert json.loads(command(active, "rag_cli.py", "backup", str(backup)).stdout)["status"] == "ok"
    assert command(active, "rag_cli.py", "restore", str(backup), ok=False).returncode == 1
    assert json.loads(command(restored, "rag_cli.py", "restore", str(backup)).stdout)["status"] == "ok"
    assert json.loads(command(restored, "rag_cli.py", "doctor").stdout)["counts"]["dense"] == 1
    hits = json.loads(command(
        restored, "rag_cli.py", "search", "sentinel", "--mode", "keyword", "--collection", "research_docs",
    ).stdout)
    assert hits[0]["metadata"]["source_url"] == "https://docs.python.org/recovery"
    assert json.loads(command(restored, "rag_cli.py", "run", job_id).stdout)["status"] == "needs_context"
    assert (tmp_path / "restored" / "vault" / "verified_code" / "saved.py").is_file()
    saved_doc = next((backup / "data" / "markdown").glob("*.md"))
    saved_doc.write_text("tampered")
    damaged = command(restored, "rag_cli.py", "restore", str(backup), ok=False)
    assert damaged.returncode == 1 and "checksums" in damaged.stderr


def test_stdio_mcp_respects_private_profile(tmp_path):
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config = tmp_path / "mcp.yaml"
    config.write_text("security:\n  private_profile: true\n")

    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable, args=[str(ROOT / "mcp_server.py")], cwd=str(tmp_path),
            env={"RAG_CONFIG_PATH": str(config)},
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                names = {tool.name for tool in (await client.list_tools()).tools}
                assert {"get_plan", "search_memory", "record_task_changes"} <= names
                miss = await client.call_tool("search_memory", {
                    "query": "Reuse previous retry delay handling.",
                    "keywords": ["retry_delay", "exponential backoff"],
                })
                assert not miss.isError
                assert miss.structuredContent == {"status": "no_results", "results": []}
                queued = await client.call_tool("submit_plan", {"task": "Plan a private deployment."})
                assert not queued.isError
                job_id = queued.structuredContent["job_id"]
                paused = await client.call_tool("run_plan", {"job_id": job_id})
                assert not paused.isError and paused.structuredContent["status"] == "needs_context"
                found = await client.call_tool("get_plan", {"job_id": job_id})
                assert found.structuredContent["pending_question"]
                other_profile = tmp_path / "other"
                other_profile.mkdir()
                other_config = other_profile / "mcp.yaml"
                other_config.write_text(config.read_text())
                isolated = await asyncio.to_thread(command, other_config, "rag_cli.py", "status", job_id, ok=False)
                assert isolated.returncode == 1 and "Unknown planning job" in isolated.stderr

    asyncio.run(asyncio.wait_for(exercise(), timeout=45))


def test_process_exit_mid_batch_recovers_in_next_cli(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("embedding:\n  batch_size: 1\n")
    source = tmp_path / "samples.py"
    write_python(source, "old_revision")
    bootstrap = """
from langchain_core.embeddings import DeterministicFakeEmbedding
from embedding.pipeline import EmbeddingPipeline
from store.chroma_store import ChromaStoreManager

pipeline = EmbeddingPipeline(chroma=ChromaStoreManager(embeddings=DeterministicFakeEmbedding(size=16)))
"""
    command(config, "-c", bootstrap + f"\npipeline.ingest({str(source)!r})\n")
    write_python(source, "new_revision", "another_revision")
    crash = bootstrap + f"""
import os
collection_type = type(pipeline.chroma.raw_collection("reusable_code"))
original = collection_type.upsert
def crash_after_upsert(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(7)
collection_type.upsert = crash_after_upsert
pipeline.ingest({str(source)!r})
"""
    assert command(config, "-c", crash, ok=False).returncode == 7
    pending = command(config, "rag_cli.py", "doctor", ok=False)
    assert json.loads(pending.stdout)["issues"]["pending_operations"]
    assert json.loads(command(config, "rag_cli.py", "recover").stdout)["operations_replayed"] == 1
    assert json.loads(command(config, "rag_cli.py", "doctor").stdout)["counts"]["dense"] == 2
    assert json.loads(command(config, "rag_cli.py", "search", "old_revision", "--mode", "keyword").stdout) == []
