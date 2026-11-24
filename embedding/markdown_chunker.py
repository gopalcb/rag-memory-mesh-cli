"""
Structure-Aware Markdown Chunker
================================
Purpose:
    Splits Markdown guides and crawled web pages in two stages: first along
    heading boundaries (`#`, `##`, `###`) to attach section breadcrumbs, and
    second with a Markdown-aware splitter that keeps fenced code blocks intact.

Call Hierarchy:
    EmbeddingPipeline.ingest()
          │
          ▼
    MarkdownChunker.chunk(record, chunk_size, chunk_overlap)
          ├── Stage 1: MarkdownHeaderTextSplitter (extracts h1 > h2 > h3 breadcrumbs)
          └── Stage 2: RecursiveCharacterTextSplitter(MARKDOWN)
                └── Prepends section breadcrumb + merges YAML frontmatter metadata
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from config import settings
from embedding.loaders import RawFileRecord
from utils import sha256_hex


class MarkdownChunker:
    """Two-stage Markdown splitter preserving heading context and fenced code blocks."""

    HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3")]

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """
        Initialize the Markdown chunker with defaults from `default_config.yaml`.

        Args:
            chunk_size: Target maximum characters per Markdown chunk.
            chunk_overlap: Character overlap between consecutive chunks within a section.
        """
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunking.md_chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunking.md_chunk_overlap
        )

    def chunk(
        self,
        record: RawFileRecord,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        """
        Split a Markdown `RawFileRecord` into contextualized LangChain `Document` chunks.

        Args:
            record: Loaded Markdown `RawFileRecord` with parsed frontmatter.
            chunk_size: Optional per-call override for chunk size.
            chunk_overlap: Optional per-call override for chunk overlap.

        Returns:
            List of `Document` chunks enriched with section breadcrumbs and frontmatter.
        """
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT,
            strip_headers=False,
        ).split_text(record.content)

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size if chunk_size is not None else self.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else self.chunk_overlap,
        )

        chunks: list[Document] = []
        for sec_idx, sec in enumerate(sections):
            breadcrumb = " > ".join(
                sec.metadata[k] for k in ("h1", "h2", "h3") if k in sec.metadata
            )
            prefix = f"[Document: {record.filename} | Section: {breadcrumb}]\n" if breadcrumb else ""

            for sub_idx, part in enumerate(splitter.split_text(sec.page_content)):
                full_text = f"{prefix}{part}".strip()
                identity = sha256_hex(f"{record.resolved_path}\n{full_text}", 24)
                chunks.append(
                    Document(
                        page_content=full_text,
                        metadata={
                            **record.frontmatter,
                            "chunk_id": f"{record.filename}::s{sec_idx}_c{sub_idx}::{identity}",
                            "source": record.filename,
                            "file_path": record.resolved_path,
                            "type": "markdown_doc",
                            "language": "markdown",
                            "section": breadcrumb or "root",
                        },
                    )
                )
        return chunks
