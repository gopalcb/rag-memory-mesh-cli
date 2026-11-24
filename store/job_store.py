from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from config import settings
from store.model import AgentMessage, CrawledUrl, FileManifest, PlanJob, PlanResearchSources
from store.session import get_session
from utils import sha256_hex


class JobStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path

    def create_job(
        self, task: str, codebase_context: str = "", skills_needed: list[str] | None = None,
        *, research_urls: list[str] | None = None,
    ) -> str:
        job_id = uuid4().hex
        with get_session(self.db_path) as session:
            session.add(PlanJob(
                job_id=job_id,
                task=task,
                codebase_context=codebase_context,
                skills_needed=skills_needed or [],
            ))
            if research_urls:
                session.flush()
                session.add(PlanResearchSources(job_id=job_id, urls=research_urls))
        return job_id

    def post_message(
        self, job_id: str, sender: str, message_type: str, content: str,
        new_status: str | None = None,
    ) -> None:
        with get_session(self.db_path) as session:
            job = session.get_one(PlanJob, job_id)
            job.messages.append(AgentMessage(sender=sender, message_type=message_type, content=content))
            if new_status is not None:
                job.status = new_status
            job.updated_at = datetime.now(timezone.utc)

    def get_job_snapshot(self, job_id: str) -> dict | None:
        with get_session(self.db_path) as session:
            job = session.get(PlanJob, job_id)
            if job is None:
                return None
            messages = [
                {
                    "id": message.id,
                    "sender": message.sender,
                    "message_type": message.message_type,
                    "content": message.content,
                    "created_at": message.created_at,
                }
                for message in job.messages
            ]
            question = None
            if job.status == "needs_context":
                question = next(
                    (message["content"] for message in reversed(messages)
                     if message["message_type"] == "question"),
                    None,
                )
            research = session.get(PlanResearchSources, job_id)
            return {
                "research_urls": research.urls if research is not None else [],
                "job_id": job.job_id,
                "task": job.task,
                "codebase_context": job.codebase_context,
                "skills_needed": job.skills_needed,
                "status": job.status,
                "stage": job.stage,
                "final_plan": job.final_plan,
                "error_message": job.error_message,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "messages": messages,
                "pending_question": question,
            }

    def update_job_status(
        self, job_id: str, status: str, stage: str | None = None,
        final_plan: str | None = None, error_message: str | None = None,
    ) -> None:
        with get_session(self.db_path) as session:
            job = session.get_one(PlanJob, job_id)
            job.status = status
            for field, value in (("stage", stage), ("final_plan", final_plan), ("error_message", error_message)):
                if value is not None:
                    setattr(job, field, value)
            job.updated_at = datetime.now(timezone.utc)

    def is_url_fresh(self, url: str, ttl_hours: float | None = None) -> bool:
        ttl = ttl_hours if ttl_hours is not None else settings.crawler.cache_ttl_hours
        if ttl <= 0:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl)
        with get_session(self.db_path) as session:
            return session.scalar(select(CrawledUrl.url).where(
                CrawledUrl.url == url, CrawledUrl.status == "indexed", CrawledUrl.crawled_at > cutoff,
            )) is not None

    def indexed_url_path(self, url: str, ttl_hours: float | None = None) -> Path | None:
        with get_session(self.db_path) as session:
            row = session.get(CrawledUrl, url)
            if row is None:
                return None
            path = Path(row.markdown_path)
            if ttl_hours is not None:
                if ttl_hours <= 0 or row.status != "indexed":
                    return None
                cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
                if row.crawled_at.replace(tzinfo=timezone.utc) <= cutoff:
                    return None
                manifest = session.get(FileManifest, str(path.resolve()))
                if manifest is None or not manifest.chunk_ids or not path.is_file():
                    return None
                try:
                    with path.open(encoding="utf-8") as source:
                        content = source.read(settings.runtime.max_source_bytes + 1)
                except (OSError, UnicodeError):
                    return None
                if sha256_hex(content) != row.content_hash:
                    return None
            return path

    def record_crawled_url(
        self, url: str, topic: str, content_hash: str, markdown_path: str | Path,
        status: str = "indexed",
    ) -> None:
        with get_session(self.db_path) as session:
            session.merge(CrawledUrl(
                url=url,
                topic=topic,
                content_hash=content_hash,
                markdown_path=str(markdown_path),
                status=status,
                crawled_at=datetime.now(timezone.utc),
            ))
