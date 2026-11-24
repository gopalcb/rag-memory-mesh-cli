"""Exercise the installed CLI and real model on a temporary store."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://docs.python.org/3/library/textwrap.html"


def run(config: Path, *args: str, expected_status: int = 0) -> dict | list:
    result = subprocess.run(
        [sys.executable, str(ROOT / "rag_cli.py"), *args],
        cwd=ROOT, env=dict(os.environ, RAG_CONFIG_PATH=str(config)), text=True,
        capture_output=True, timeout=360,
    )
    if result.returncode != expected_status:
        raise RuntimeError(result.stdout or result.stderr)
    return json.loads(result.stdout) if result.stdout else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", nargs="?", const="ollama", choices=("ollama", "codex"))
    parser.add_argument("--crawl", action="store_true")
    selected = parser.parse_args()
    with TemporaryDirectory(prefix="rag-memory-smoke-") as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "restored"
        first.mkdir()
        second.mkdir()
        config = first / "config.yaml"
        config.write_text(
            f"llm:\n  provider: {selected.planner or 'ollama'}\n"
            f"security:\n  allow_external_models: {str(selected.planner == 'codex').lower()}\n"
            "crawler:\n  allowed_domains: [docs.python.org]\n",
            encoding="utf-8",
        )
        restore_config = second / "config.yaml"
        restore_config.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        source = root / "retry.py"
        source.write_text(
            'def retry_delay(attempt: int) -> int:\n'
            '    """Calculate exponential backoff delay for transient requests."""\n'
            '    return 2 ** attempt\n',
            encoding="utf-8",
        )
        assert run(config, "ingest", str(source))["indexed_files"] == 1
        assert run(config, "ingest", str(source))["skipped_unchanged"] == 1
        query = "exponential backoff retry delay"
        assert run(config, "search", query, "--mode", "hybrid")[0]["metadata"]["symbol_name"] == "retry_delay"
        rejected = root / "notes.txt"
        rejected.write_text("Unsupported sample.", encoding="utf-8")
        run(config, "ingest", str(rejected), "--reset", expected_status=1)
        assert run(config, "doctor")["counts"]["dense"] == 1

        if selected.crawl:
            crawl = run(config, "crawl", SOURCE_URL)
            assert crawl["outcomes"][0]["status"] == "indexed"
            assert run(config, "crawl", SOURCE_URL)["outcomes"][0]["status"] == "skipped_fresh"

        job = run(config, "submit", "Plan reuse of retry_delay for exponential backoff.")
        job_id = job["job_id"]
        assert run(config, "run", job_id)["status"] == "needs_context"
        snapshot = root / "snapshot"
        assert run(config, "backup", str(snapshot))["status"] == "ok"
        assert run(restore_config, "restore", str(snapshot))["status"] == "ok"
        assert run(restore_config, "doctor")["status"] == "ok"
        assert run(restore_config, "search", query, "--mode", "hybrid")[0]["metadata"]["symbol_name"] == "retry_delay"
        assert run(restore_config, "run", job_id)["status"] == "needs_context"

        if selected.planner:
            result = run(restore_config, "answer", job_id, "Python standard library, target transport.py, use pytest.")
            assert result["status"] == "completed"
            assert result["blueprint"]["sources"]
    print("Real-model ingestion, search, reset safety, checkpoint and backup/restore smoke passed.")
    if selected.crawl:
        print("Live documentation crawl and fresh cache hit passed.")
    if selected.planner:
        print(f"Resumed planning with {selected.planner} passed.")


if __name__ == "__main__":
    main()
