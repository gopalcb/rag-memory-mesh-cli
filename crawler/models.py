from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class CrawlTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    topic: str = Field(default="general", min_length=1, max_length=128)
    tags: list[str] = Field(default_factory=list)

    @field_validator("topic", mode="before")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def url_str(self) -> str:
        return str(self.url)


class CrawlOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int | None = Field(default=None, ge=1, le=16)
    pruning_threshold: float | None = Field(default=None, ge=0, le=1)
    cache_ttl_hours: int | None = Field(default=None, ge=0)
    page_timeout_ms: int = Field(default=20000, ge=100, le=120000)
    force_refresh: bool = False
    auto_embed: bool = True


class PageOutcome(BaseModel):
    url: str
    topic: str
    status: Literal["indexed", "saved", "skipped_fresh", "failed"]
    markdown_path: str | None = None
    title: str | None = None
    char_count: int = 0
    error: str | None = None


class BatchCrawlReport(BaseModel):
    outcomes: list[PageOutcome] = Field(default_factory=list)
    chunks_indexed: int = 0
