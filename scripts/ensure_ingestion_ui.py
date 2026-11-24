import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE = "rag-memory-ingestion"
DEFAULT_PORT = 27743
OPENER = build_opener(ProxyHandler({}))


def healthy(port: int) -> bool:
    try:
        with OPENER.open(f"http://127.0.0.1:{port}/health", timeout=0.35) as response:
            result = json.loads(response.read(4096))
    except (OSError, URLError, ValueError):
        return False
    return isinstance(result, dict) and result.get("service") == SERVICE and result.get("status") == "ok"


def port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.35):
            return True
    except ConnectionRefusedError:
        return False


def ensure_ui(port: int, runtime: Path, config: Path | None = None) -> dict:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (runtime / f"ingestion-ui-{port}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        url = f"http://127.0.0.1:{port}"
        if healthy(port):
            return {"status": "already_running", "url": url}
        if port_in_use(port):
            raise RuntimeError(f"Port {port} is already occupied by a service without the RAG ingestion identity")
        if not (PROJECT_ROOT / "ingestion_ui" / "__main__.py").is_file():
            return {"status": "unavailable", "reason": "The optional ingestion UI is absent from this checkout."}
        log_path = runtime / f"ingestion-ui-{port}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        env.pop("RAG_CONFIG_PATH", None)
        if config is not None:
            env["RAG_CONFIG_PATH"] = str(config)
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "ingestion_ui", "--port", str(port)],
                cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if healthy(port):
                return {"status": "started", "url": url, "pid": process.pid}
            if process.poll() is not None:
                break
            time.sleep(0.15)
        if process.poll() is None:
            process.terminate()
        raise RuntimeError(f"Ingestion UI did not become healthy. See {log_path}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure the local RAG ingestion UI is running")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / ".rag_memory_data" / "runtime")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hook", action="store_true")
    args = parser.parse_args()
    config = args.config.expanduser().resolve() if args.config else None
    if config is not None and not config.is_file():
        parser.error(f"Storage configuration is not a file: {config}")
    try:
        result = ensure_ui(args.port, args.runtime, config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"continue": True} if args.hook else result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
