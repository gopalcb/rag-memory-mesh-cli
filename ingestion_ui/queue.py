import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionQueue:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS watchers (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    signature TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    watcher_id TEXT REFERENCES watchers(id),
                    status TEXT NOT NULL DEFAULT 'queued',
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT
                );
                """
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def insert_job(self, db: sqlite3.Connection, path: str, watcher_id: str | None = None) -> str:
        job_id = uuid4().hex
        db.execute(
            "INSERT INTO jobs (id, path, watcher_id, created_at) VALUES (?, ?, ?, ?)",
            (job_id, path, watcher_id, timestamp()),
        )
        return job_id

    def enqueue(self, path: str) -> str:
        with self.connect() as db:
            return self.insert_job(db, path)

    def job(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self.job_data(row) if row else None

    @staticmethod
    def job_data(row: sqlite3.Row) -> dict:
        job = dict(row)
        if job["result"]:
            job["result"] = json.loads(job["result"])
        return job

    def retry(self, job_id: str) -> str:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute(
                "SELECT path, watcher_id, status FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            if job is None:
                raise LookupError("This run does not exist.")
            if job["status"] != "failed":
                raise ValueError("Only failed runs can be retried.")
            queued = db.execute(
                """SELECT id FROM jobs
                   WHERE path = ? AND watcher_id IS ? AND status IN ('queued', 'running')
                   ORDER BY created_at DESC LIMIT 1""",
                (job["path"], job["watcher_id"]),
            ).fetchone()
            return queued["id"] if queued else self.insert_job(db, job["path"], job["watcher_id"])

    def recover_running(self) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'")

    def claim(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            job = self.job_data(row)
            job.update(status="running", started_at=timestamp())
            db.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (job["started_at"], job["id"]),
            )
            return job

    def finish(self, job_id: str, *, result: dict | None = None, error: str | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, ended_at = ? WHERE id = ?",
                ("failed" if error else "completed", json.dumps(result) if result is not None else None,
                 error, timestamp(), job_id),
            )

    def add_watch(self, path: str) -> str:
        with self.connect() as db:
            watcher_id = uuid4().hex
            db.execute(
                "INSERT INTO watchers (id, path, created_at) VALUES (?, ?, ?)",
                (watcher_id, path, timestamp()),
            )
            return watcher_id

    def set_watch(self, watcher_id: str, enabled: bool) -> bool:
        with self.connect() as db:
            return bool(db.execute(
                "UPDATE watchers SET enabled = ? WHERE id = ?", (enabled, watcher_id),
            ).rowcount)

    def active_watches(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT id, path, signature FROM watchers WHERE enabled = 1")
            return [dict(row) for row in rows]

    def note_watch_error(self, watcher_id: str, error: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE watchers SET error = ? WHERE id = ?", (error, watcher_id))

    def observe(self, watcher_id: str, signature: str, has_files: bool) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            watch = db.execute("SELECT * FROM watchers WHERE id = ?", (watcher_id,)).fetchone()
            if watch is None or not watch["enabled"]:
                return
            if signature != watch["signature"]:
                queued = db.execute(
                    "SELECT id FROM jobs WHERE watcher_id = ? AND status = 'queued'", (watcher_id,),
                ).fetchone()
                if queued is None and (has_files or watch["signature"] is None):
                    self.insert_job(db, watch["path"], watcher_id)
            db.execute(
                "UPDATE watchers SET signature = ?, error = NULL WHERE id = ?",
                (signature, watcher_id),
            )

    def dashboard(self) -> dict:
        with self.connect() as db:
            jobs = db.execute("SELECT * FROM jobs ORDER BY rowid DESC LIMIT 30").fetchall()
            watches = db.execute(
                """
                SELECT w.id, w.path, w.enabled, w.error, w.created_at, j.status AS last_status,
                       j.created_at AS last_run
                FROM watchers w
                LEFT JOIN jobs j ON j.id = (
                    SELECT id FROM jobs WHERE watcher_id = w.id ORDER BY rowid DESC LIMIT 1
                )
                ORDER BY w.rowid DESC
                """
            ).fetchall()
            return {
                "jobs": [self.job_data(row) for row in jobs],
                "watchers": [dict(row) for row in watches],
            }
