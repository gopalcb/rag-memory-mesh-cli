import json
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable
from threading import Event, Thread

from ingestion_ui.queue import IngestionQueue
from ingestion_ui.watcher import scan_watches


class EmbeddingRunner:
    def __call__(self, path: str) -> dict:
        cli = Path(__file__).resolve().parents[1] / "rag_cli.py"
        result = subprocess.run(
            [sys.executable, str(cli), "ingest", path], capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-1000:] or "Ingestion worker failed")
        return json.loads(result.stdout)


class IngestionWorker:
    def __init__(
        self, queue: IngestionQueue, ingest: Callable[[str], dict] | None = None,
        *, watch_interval: float = 2.0,
    ) -> None:
        self.queue = queue
        self.ingest = ingest or EmbeddingRunner()
        self.watch_interval = watch_interval
        self.stopped = Event()
        self.threads = [
            Thread(target=self.consume, name="ingestion-job-worker", daemon=True),
            Thread(target=self.watch, name="ingestion-folder-watcher", daemon=True),
        ]

    def run_once(self) -> bool:
        job = self.queue.claim()
        if job is None:
            return False
        try:
            result = self.ingest(job["path"])
        except Exception as error:
            self.queue.finish(job["id"], error=str(error) or type(error).__name__)
        else:
            self.queue.finish(job["id"], result=result)
        return True

    def consume(self) -> None:
        while not self.stopped.is_set():
            if not self.run_once():
                self.stopped.wait(0.3)

    def watch(self) -> None:
        while not self.stopped.is_set():
            scan_watches(self.queue)
            self.stopped.wait(self.watch_interval)

    def start(self) -> None:
        self.queue.recover_running()
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stopped.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
