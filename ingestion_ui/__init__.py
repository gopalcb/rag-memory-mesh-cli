import secrets
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from config import settings
from embedding.loaders import DEFAULT_EXTENSIONS
from ingestion_ui.queue import IngestionQueue


HEALTH = {"service": "rag-memory-ingestion", "status": "ok"}


def local_path(value: object, *, folder: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter an absolute path.")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("Enter a full local path.")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Check that the path exists and is readable.") from None
    if folder and not path.is_dir():
        raise ValueError("Choose a folder to watch.")
    if not path.is_dir() and (
        not path.is_file() or path.suffix.lower() not in DEFAULT_EXTENSIONS or path.name == "__init__.py"
    ):
        raise ValueError("Choose a folder or a Python or Markdown source file.")
    return str(path)


def create_app(queue_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"], MAX_CONTENT_LENGTH=16384)
    queue = IngestionQueue(queue_path or settings.storage.base_dir / "ingestion-ui.sqlite3")
    app.extensions["ingestion_queue"] = queue
    mutation_token = secrets.token_urlsafe(32)

    @app.before_request
    def check_mutation():
        if request.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
            received = request.headers.get("X-Ingestion-Token", "")
            if not secrets.compare_digest(received.encode(), mutation_token.encode()):
                return jsonify(error="Reload the console and try again."), 403
            if not request.is_json or not isinstance(request.get_json(silent=True), dict):
                return jsonify(error="Send a JSON request."), 400
        return None

    @app.after_request
    def prevent_dashboard_caching(response):
        if request.path == "/" or request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", mutation_token=mutation_token)

    @app.get("/health")
    def health():
        return jsonify(HEALTH)

    @app.get("/api/dashboard")
    def dashboard():
        return jsonify({**queue.dashboard(), "mutation_token": mutation_token})

    @app.post("/api/jobs")
    def create_job():
        try:
            path = local_path(request.json.get("path"))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        return jsonify(id=queue.enqueue(path)), 202

    @app.post("/api/jobs/<job_id>/retry")
    def retry_job(job_id: str):
        try:
            job_id = queue.retry(job_id)
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except ValueError as error:
            return jsonify(error=str(error)), 409
        return jsonify(id=job_id), 202

    @app.post("/api/watchers")
    def add_watch():
        try:
            path = local_path(request.json.get("path"), folder=True)
            watcher_id = queue.add_watch(path)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except sqlite3.IntegrityError:
            return jsonify(error="This folder already has a watch. Use its controls below."), 409
        return jsonify(id=watcher_id), 201

    @app.patch("/api/watchers/<watcher_id>")
    def set_watch(watcher_id: str):
        enabled = request.json.get("enabled")
        if not isinstance(enabled, bool):
            return jsonify(error="Choose whether to pause or resume the folder watch."), 400
        if not queue.set_watch(watcher_id, enabled):
            return jsonify(error="This folder watch does not exist."), 404
        return jsonify(id=watcher_id, enabled=enabled)

    return app
