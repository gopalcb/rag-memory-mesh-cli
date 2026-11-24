"""
Embedding Subsystem Public Interface
====================================
Purpose:
    Exposes the class-based loaders, language-aware chunkers, cached embedding
    client factory, and incremental ingestion pipeline for `rag-memory-system`.

High-Level Call Flow:
    run_embed.py / CrawlFleet
          │
          ▼
    EmbeddingPipeline.ingest(target_path, options)
          ├──► FileLoader.scan(target_path)        -> list[RawFileRecord]
          ├──► PythonAstChunker.chunk(record)      -> list[Document] + AST graph edges
          ├──► MarkdownChunker.chunk(record)       -> list[Document] + section breadcrumbs
          └──► ChromaStoreManager / CodeGraphStore / ManifestStore
"""

from embedding.code_chunker import PythonAstChunker
from embedding.loaders import FileLoader, RawFileRecord
from embedding.markdown_chunker import MarkdownChunker
from embedding.model import get_embedding_client
from embedding.pipeline import EmbeddingPipeline, IngestionStats

__all__ = [
    "EmbeddingPipeline",
    "FileLoader",
    "IngestionStats",
    "MarkdownChunker",
    "PythonAstChunker",
    "RawFileRecord",
    "get_embedding_client",
]
