from datetime import datetime, timezone
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class FileManifest(Base):
    """Tracks ingested .py and .md files to skip unchanged files and purge stale chunks."""

    __tablename__ = "file_manifest"

    file_path: Mapped[str] = mapped_column(String(1024), primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class CodeNode(Base):
    """AST symbol node representing a function, class, or module block stored in ChromaDB."""

    __tablename__ = "code_nodes"

    chunk_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    symbol_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    symbol_type: Mapped[str] = mapped_column(String(64), nullable=False)  # function | class | module
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)


class CodeEdge(Base):
    """Directed dependency edge between a chunk and a target symbol it calls, imports, or inherits."""

    __tablename__ = "code_edges"
    __table_args__ = (
        UniqueConstraint("source_chunk_id", "target_symbol", "relation", name="uq_code_edge"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_chunk_id: Mapped[str] = mapped_column(
        String(256), ForeignKey("code_nodes.chunk_id", ondelete="CASCADE"), index=True
    )
    target_symbol: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String(32), nullable=False)  # calls | imports | inherits


class PlanJob(Base):
    """Represents a plan-completion request submitted by the main CLI agent (Codex / Claude Code)."""

    __tablename__ = "plan_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    codebase_context: Mapped[str] = mapped_column(Text, default="")
    skills_needed: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(
        String(32), default="queued", index=True
    )  # queued | running | needs_context | completed | failed
    stage: Mapped[str] = mapped_column(String(128), default="initialized")
    final_plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="AgentMessage.id"
    )


class AgentMessage(Base):
    """Bidirectional message log between Coordinator agent and Main CLI agent."""

    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("plan_jobs.job_id", ondelete="CASCADE"), index=True
    )
    sender: Mapped[str] = mapped_column(String(32), nullable=False)  # "coordinator" | "main_agent"
    message_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "question" | "answer" | "status_note"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    job: Mapped["PlanJob"] = relationship(back_populates="messages")


class CrawledUrl(Base):
    """Tracks crawled URLs, content hashes, saved Markdown paths, and TTL expiration."""

    __tablename__ = "crawled_urls"

    url: Mapped[str] = mapped_column(String(2048), primary_key=True)
    topic: Mapped[str] = mapped_column(String(256), default="general", index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    markdown_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="indexed")  # indexed | failed
    crawled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PlanResearchSources(Base):
    __tablename__ = "plan_research_sources"

    job_id: Mapped[str] = mapped_column(
        ForeignKey("plan_jobs.job_id", ondelete="CASCADE"), primary_key=True
    )
    urls: Mapped[list[str]] = mapped_column(JSON, nullable=False)


class IndexJournalEntry(Base):
    __tablename__ = "index_journal"

    id: Mapped[int] = mapped_column(primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class IndexFingerprint(Base):
    __tablename__ = "index_fingerprints"

    file_path: Mapped[str] = mapped_column(
        ForeignKey("file_manifest.file_path", ondelete="CASCADE"), primary_key=True
    )
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)


class IndexCollectionContract(Base):
    __tablename__ = "index_collection_contracts"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
