import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger("rag_memory")


class CliLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {"level": record.levelname.lower(), "event": record.getMessage()}
        for field in ("command", "elapsed_ms", "exit_code", "operation_id", "error_type"):
            if hasattr(record, field):
                event[field] = getattr(record, field)
        return json.dumps(event)


def run(args: argparse.Namespace):
    from config import settings

    match args.command:
        case "ingest":
            from embedding.pipeline import EmbeddingPipeline
            return EmbeddingPipeline().ingest(args.path, {
                "reset": args.reset, "reconcile": args.reconcile,
            }).model_dump()
        case "search":
            from store.hybrid_store import HybridRetriever
            collection = args.collection or settings.collections.code
            hits = HybridRetriever().search(args.query, collection, args.mode, args.k)
            return [asdict(hit) | {"document": hit.document.model_dump()} for hit in hits]
        case "search-memory":
            from planning.memory_coordinator import search_memory
            from planning.memory_inputs import MemoryLookup

            lookup = MemoryLookup(query=args.query, keywords=args.keywords)
            return search_memory(lookup)
        case "record-task-changes":
            from planning.memory_coordinator import record_task_changes
            from planning.memory_inputs import TaskChanges

            changes = TaskChanges(
                project=args.project, task=args.task, implementation_summary=args.implementation_summary,
                verification=args.verification, files=args.files,
            )
            return record_task_changes(changes)
        case "crawl":
            from crawler.fleet import CrawlFleet
            return asyncio.run(CrawlFleet().crawl_urls(
                args.urls, {"force_refresh": args.force_refresh},
            )).model_dump()
        case "commit-verified-snippet":
            from planning.snippets import commit_verified_snippet
            code = args.code if hasattr(args, "code") else Path(args.file).read_text(encoding="utf-8")
            return commit_verified_snippet(args.name, code, args.verification)
        case "doctor":
            from store.audit import audit_index
            return audit_index()
        case "recover":
            from embedding.pipeline import EmbeddingPipeline
            return {"status": "ok", "operations_replayed": EmbeddingPipeline().recover()}
        case "backup":
            from store.backup import export_backup
            from store.index_state import IndexState
            if IndexState().pending():
                raise RuntimeError("Run recover before backup to replay pending index writes")
            return export_backup(args.path)
        case "restore":
            from store.backup import import_backup
            return import_backup(args.path)

    from planning.coordinator import PlanningCoordinator
    coordinator = PlanningCoordinator()
    match args.command:
        case "submit":
            job_id = coordinator.submit(args.task, args.context, args.source)
            return coordinator.snapshot(job_id)
        case "run":
            return coordinator.advance(args.job_id)
        case "status":
            return coordinator.snapshot(args.job_id)
        case "answer":
            return coordinator.advance(args.job_id, args.answer)


def execute(args: argparse.Namespace):
    from store.runtime import index_lock

    with index_lock():
        if args.command in {
            "search", "search-memory", "crawl", "run", "answer", "commit-verified-snippet",
        }:
            from embedding.pipeline import EmbeddingPipeline
            EmbeddingPipeline().recover()
        return run(args)


def exit_status(command: str, result) -> int:
    if command == "crawl":
        outcomes = result["outcomes"]
        failed = sum(outcome["status"] == "failed" for outcome in outcomes)
        return (1 if failed == len(outcomes) else 2) if failed else 0
    return int(isinstance(result, dict) and result.get("status") == "failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG memory research and planning service")
    parser.add_argument("--log-json", action="store_true", help="Write structured operation events to stderr")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("path")
    reset = ingest.add_mutually_exclusive_group()
    reset.add_argument("--reset", action="store_true")
    reset.add_argument("--reconcile", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--collection")
    search.add_argument("--mode", choices=("hybrid", "keyword", "vector"))
    search.add_argument("-k", type=int)
    crawl = commands.add_parser("crawl")
    crawl.add_argument("urls", nargs="+")
    crawl.add_argument("--force-refresh", action="store_true")
    submit = commands.add_parser("submit")
    submit.add_argument("task")
    submit.add_argument("--context", default="")
    submit.add_argument("--source", action="append", default=[])
    for command in ("run", "status", "answer"):
        cmd = commands.add_parser(command)
        cmd.add_argument("job_id")
        if command == "answer":
            cmd.add_argument("answer")
    snippet = commands.add_parser("commit-verified-snippet")
    snippet.add_argument("name")
    snippet.add_argument("file")
    snippet.add_argument("--verification", required=True)
    commands.add_parser("doctor")
    commands.add_parser("recover")
    for command in ("backup", "restore"):
        commands.add_parser(command).add_argument("path")
    commands.add_parser("_mcp")

    args = parser.parse_args()
    structured_logs = args.log_json
    if structured_logs:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(CliLogFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    started = time.monotonic()
    try:
        if args.command == "_mcp":
            payload = sys.stdin.buffer.read(16_000_001)
            if len(payload) > 16_000_000:
                raise ValueError("MCP request exceeds its size limit")
            fields = json.loads(payload)
            if fields["command"] not in {
                "submit", "run", "status", "answer", "commit-verified-snippet",
                "search-memory", "record-task-changes",
            }:
                raise ValueError("Unsupported MCP command")
            args = argparse.Namespace(**fields)
        result = execute(args)
    except Exception as exc:
        expected = isinstance(exc, (ValueError, OSError, RuntimeError, SyntaxError))
        message = str(exc) if expected else f"Operation failed ({type(exc).__name__})"
        if structured_logs:
            print(json.dumps({"status": "failed", "error": message}))
            log.error("command_failed", extra={
                "command": args.command, "error_type": type(exc).__name__, "exit_code": 1,
                "elapsed_ms": round(1000 * (time.monotonic() - started)),
            })
        else:
            print(message, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, default=str))
    status = exit_status(args.command, result)
    log.info("command_finished", extra={
        "command": args.command, "elapsed_ms": round(1000 * (time.monotonic() - started)),
        "exit_code": status,
    })
    return status


if __name__ == "__main__":
    sys.exit(main())
