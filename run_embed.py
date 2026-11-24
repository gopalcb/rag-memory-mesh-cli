"""
Standalone Embedding CLI Entrypoint
===================================
Purpose:
    Provides a command-line interface to run `EmbeddingPipeline.ingest()` on any
    local file or directory using `IngestOptions` overrides.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run incremental AST + Markdown + Graph embedding pipeline.")
    parser.add_argument("target", help="Folder to scan recursively, or a single .py, .md, or .markdown file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reset", action="store_true", help="Rebuild the selected collections")
    mode.add_argument("--reconcile", action="store_true", help="Remove missing and empty files in this directory")
    parser.add_argument("--md-chunk-size", type=int, default=None, help="Override Markdown chunk size")
    parser.add_argument("--md-chunk-overlap", type=int, default=None, help="Override Markdown chunk overlap")
    parser.add_argument("--code-max-chars", type=int, default=None, help="Override Python AST max chars")
    args = parser.parse_args()

    try:
        from config import IngestOptions, settings
        from embedding.pipeline import EmbeddingPipeline
        from store.runtime import index_lock

        opts = IngestOptions(
            reset=args.reset,
            reconcile=args.reconcile,
            md_chunk_size=args.md_chunk_size,
            md_chunk_overlap=args.md_chunk_overlap,
            code_max_chars=args.code_max_chars,
        )
        with index_lock():
            stats = EmbeddingPipeline().ingest(target_path=args.target, options=opts)
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print("\n--- Ingestion Summary ---")
    print(f"ChromaDB location  : {settings.storage.chroma_dir.resolve()}")
    print(f"SQLite location    : {settings.storage.sqlite_db_path.resolve()}")
    print(f"Scanned files      : {stats.scanned_files}")
    print(f"Newly indexed      : {stats.indexed_files}")
    print(f"Removed files      : {stats.removed_files}")
    print(f"Skipped unchanged  : {stats.skipped_unchanged}")
    print(f"Chunks upserted    : {stats.total_chunks_upserted}")
    print(f"AST edges upserted : {stats.total_graph_edges_upserted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
