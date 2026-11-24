"""
Configuration & Override Models
===============================
Purpose:
    Loads `default_config.yaml` with `${ENV_VAR}` expansion into typed Pydantic
    settings, and defines compact override models (`IngestOptions`, `SearchOptions`,
    `EmbeddingClientOptions`) so methods accept a single options dict or model.
"""

import os
import re
import stat
from pathlib import Path
from typing import Any, Literal, TypeVar
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"
TModel = TypeVar("TModel", bound=BaseModel)


def merge_options(defaults: TModel, overrides: BaseModel | dict[str, Any] | None) -> TModel:
    """Return a copy of `defaults` updated with non-None fields from `overrides`."""
    if overrides is None:
        return defaults
    raw = (
        overrides.model_dump(exclude_none=True, include=set(type(defaults).model_fields))
        if isinstance(overrides, BaseModel)
        else {k: v for k, v in overrides.items() if v is not None}
    )
    return type(defaults).model_validate({**defaults.model_dump(), **raw}) if raw else defaults


def expand_env_vars(value: Any) -> Any:
    """Recursively resolve `${VAR_NAME}` placeholders in loaded YAML values."""
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


class StorageConfig(BaseModel):
    base_dir: Path = Path("./.rag_memory_data")
    chroma_dir: Path = Path("./.rag_memory_data/chroma_db")
    sqlite_db_path: Path = Path("./.rag_memory_data/state.sqlite3")
    markdown_vault_dir: Path = Path("./.rag_memory_data/vault/md")

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for d in (self.base_dir, self.chroma_dir, self.markdown_vault_dir, self.sqlite_db_path.parent):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)


class CollectionsConfig(BaseModel):
    code: str = "reusable_code"
    docs: str = "research_docs"


class LocalEmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    revision: str | None = None
    device: str = "cpu"
    normalize_embeddings: bool = True

    @property
    def effective_revision(self) -> str | None:
        if self.revision is not None:
            return self.revision
        if self.model_name in {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}:
            return "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        return None


class ApiEmbeddingConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "text-embedding-3-small"
    check_ctx_length: bool = False


class EmbeddingConfig(BaseModel):
    provider: Literal["local", "api"] = "local"
    batch_size: int = Field(default=32, gt=0)
    local: LocalEmbeddingConfig = LocalEmbeddingConfig()
    api: ApiEmbeddingConfig = ApiEmbeddingConfig()


class CodexLlmConfig(BaseModel):
    cli_path: str = "codex"
    model_name: str = ""
    sandbox_mode: Literal["read-only"] = "read-only"
    timeout_seconds: int = Field(default=120, gt=0)


class OllamaLlmConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model_name: str = "qwen2.5-coder:3b"
    num_ctx: int = Field(default=8192, ge=4096, le=131072)


class ApiLlmConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model_name: str = "openrouter/free"


class LlmConfig(BaseModel):
    provider: Literal["codex", "ollama", "api"] = "ollama"
    temperature: float = 0.0
    codex: CodexLlmConfig = CodexLlmConfig()
    ollama: OllamaLlmConfig = OllamaLlmConfig()
    api: ApiLlmConfig = ApiLlmConfig()


class ChunkingConfig(BaseModel):
    md_chunk_size: int = Field(default=900, gt=0)
    md_chunk_overlap: int = Field(default=100, ge=0)
    code_max_chars: int = Field(default=2400, gt=0)


class RetrievalConfig(BaseModel):
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    rrf_k: int = Field(default=60, ge=0)
    vector_weight: float = Field(default=0.6, ge=0)
    keyword_weight: float = Field(default=0.4, ge=0)
    expand_ast_neighbors: bool = True
    max_neighbor_chunks: int = Field(default=2, ge=0)
    code_max_distance: float = 0.38
    docs_max_distance: float = 0.50
    default_top_k: int = Field(default=4, gt=0)


class CrawlerConfig(BaseModel):
    max_concurrency: int = Field(default=4, gt=0)
    pruning_threshold: float = Field(default=0.45, ge=0, le=1)
    cache_ttl_hours: int = Field(default=168, ge=0)
    allowed_domains: list[str] = []


class EmbeddingClientOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "api"] | None = None
    model_name: str | None = None
    revision: str | None = None
    device: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    normalize_embeddings: bool | None = None
    check_ctx_length: bool | None = None


class IngestOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reset: bool = False
    reconcile: bool = False
    code_collection: str | None = None
    docs_collection: str | None = None
    md_chunk_size: int | None = Field(default=None, gt=0)
    md_chunk_overlap: int | None = Field(default=None, ge=0)
    code_max_chars: int | None = Field(default=None, gt=0)


class SearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str | None = None
    mode: Literal["hybrid", "vector", "keyword"] | None = None
    k: int | None = None
    max_distance: float | None = None
    expand_ast_neighbors: bool | None = None
    keywords: list[str] | None = None
    where: dict[str, Any] | None = None


class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_external_models: bool = False
    private_profile: bool = False


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lock_timeout_seconds: float = Field(default=30, gt=0, le=600)
    max_source_bytes: int = Field(default=2_000_000, gt=0)
    max_scan_bytes: int = Field(default=64_000_000, gt=0)
    max_scan_files: int = Field(default=5000, gt=0)
    max_reset_journal_bytes: int = Field(default=128_000_000, gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    storage: StorageConfig = StorageConfig()
    collections: CollectionsConfig = CollectionsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LlmConfig = LlmConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    crawler: CrawlerConfig = CrawlerConfig()
    security: SecurityConfig = SecurityConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    target = Path(config_path or os.getenv("RAG_CONFIG_PATH") or DEFAULT_CONFIG_PATH).expanduser()
    raw_data = {}
    if not target.is_file() and (config_path is not None or os.getenv("RAG_CONFIG_PATH")):
        raise ValueError(f"Configuration file does not exist or is not a regular file: {target}")
    if target.is_file():
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid configuration YAML: {target}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"Configuration must contain a YAML mapping: {target}")
        raw_data = expand_env_vars(loaded or {})
    cfg = AppConfig(**raw_data)
    config_dir = target.expanduser().resolve().parent
    for name in type(cfg.storage).model_fields:
        value = getattr(cfg.storage, name).expanduser()
        setattr(cfg.storage, name, (config_dir / value).resolve())
    if cfg.security.private_profile:
        base = cfg.storage.base_dir
        for name in ("chroma_dir", "sqlite_db_path", "markdown_vault_dir"):
            if not getattr(cfg.storage, name).is_relative_to(base):
                raise ValueError(f"Private profiles require storage.{name} inside storage.base_dir")
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = base.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Private profile storage must be owned by the current user with permissions 0700")
    cfg.storage.ensure_dirs()
    return cfg


settings = load_config()
