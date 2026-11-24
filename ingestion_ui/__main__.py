import argparse
import fcntl

from config import settings
from ingestion_ui import create_app
from ingestion_ui.worker import IngestionWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local RAG memory ingestion console.")
    parser.add_argument("--port", type=int, default=27743)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    app = create_app()
    worker = IngestionWorker(app.extensions["ingestion_queue"])
    with (settings.storage.base_dir / "ingestion-ui.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "The ingestion console is already using this storage.\n")
        worker.start()
        try:
            app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
        finally:
            worker.stop()


if __name__ == "__main__":
    main()
