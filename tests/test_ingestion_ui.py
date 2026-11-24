import re
from importlib import import_module
from pathlib import Path

import pytest

create_app = pytest.importorskip("ingestion_ui").create_app
scan_watches = import_module("ingestion_ui.watcher").scan_watches
IngestionWorker = import_module("ingestion_ui.worker").IngestionWorker


def console(tmp_path: Path):
    app = create_app(tmp_path / "jobs.sqlite3")
    client = app.test_client()
    page = client.get("/")
    token = re.search(rb'name="ingestion-token" content="([^"]+)"', page.data)[1].decode()
    return app, client, {"X-Ingestion-Token": token}


def test_job_validation_failure_and_retry(tmp_path):
    app, client, headers = console(tmp_path)
    queue = app.extensions["ingestion_queue"]
    assert client.get("/health").json == {"service": "rag-memory-ingestion", "status": "ok"}
    assert client.post("/api/jobs", json={"path": "."}).status_code == 403
    assert client.post("/api/jobs", json={"path": "."}, headers=headers).status_code == 400
    source = tmp_path / "notes.md"
    source.write_text("notes")
    job = client.post("/api/jobs", json={"path": str(source)}, headers=headers)
    assert job.status_code == 202
    job_id = job.json["id"]

    def fail(_path):
        raise RuntimeError("Embedding failed")

    assert IngestionWorker(queue, fail).run_once()
    assert queue.job(job_id)["error"] == "Embedding failed"
    retry = client.post(f"/api/jobs/{job_id}/retry", json={}, headers=headers)
    assert retry.status_code == 202
    assert IngestionWorker(queue, lambda _: {"indexed_files": 1}).run_once()
    rows = client.get("/api/dashboard").json["jobs"]
    assert [row["status"] for row in rows] == ["completed", "failed"]
    assert rows[0]["result"]["indexed_files"] == 1


def test_folder_watch_coalesces_and_resumes_after_restart(tmp_path):
    app, client, headers = console(tmp_path)
    queue = app.extensions["ingestion_queue"]
    folder = tmp_path / "source"
    folder.mkdir()
    source = folder / "helper.py"
    source.write_text("one")

    response = client.post("/api/watchers", json={"path": str(folder)}, headers=headers)
    assert response.status_code == 201
    watch_id = response.json["id"]
    assert client.post("/api/watchers", json={"path": str(folder)}, headers=headers).status_code == 409

    scan_watches(queue)
    source.write_text("changed before the worker began")
    scan_watches(queue)
    assert len(queue.dashboard()["jobs"]) == 1
    assert IngestionWorker(queue, lambda _: {"indexed_files": 1}).run_once()

    assert client.patch(f"/api/watchers/{watch_id}", json={"enabled": False}, headers=headers).status_code == 200
    source.write_text("changed while paused")
    scan_watches(queue)
    assert len(queue.dashboard()["jobs"]) == 1

    app, client, headers = console(tmp_path)
    queue = app.extensions["ingestion_queue"]
    assert not queue.dashboard()["watchers"][0]["enabled"]
    assert client.patch(f"/api/watchers/{watch_id}", json={"enabled": True}, headers=headers).status_code == 200
    scan_watches(queue)
    scan_watches(queue)
    assert len(queue.dashboard()["jobs"]) == 2

    source.unlink()
    folder.rmdir()
    scan_watches(queue)
    assert queue.dashboard()["watchers"][0]["error"]


def test_inflight_job_can_restart(tmp_path):
    app, client, headers = console(tmp_path)
    queue = app.extensions["ingestion_queue"]
    job_id = client.post("/api/jobs", json={"path": str(tmp_path)}, headers=headers).json["id"]
    assert queue.claim()["id"] == job_id
    assert queue.job(job_id)["status"] == "running"
    queue.recover_running()
    assert queue.claim()["id"] == job_id


def test_open_console_can_refresh_mutation_token_after_restart(tmp_path):
    _, _, old_headers = console(tmp_path)
    app = create_app(tmp_path / "jobs.sqlite3")
    client = app.test_client()
    assert client.post("/api/jobs", json={"path": str(tmp_path)}, headers=old_headers).status_code == 403

    dashboard = client.get("/api/dashboard")
    assert dashboard.headers["Cache-Control"] == "no-store"
    refreshed = {"X-Ingestion-Token": dashboard.json["mutation_token"]}
    assert client.post("/api/jobs", json={"path": str(tmp_path)}, headers=refreshed).status_code == 202
