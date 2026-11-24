"""
Filesystem Scanner & Frontmatter Loader
=======================================
Purpose:
    Recursively scans files or directories, skips ignored build/VCS folders,
    parses optional YAML frontmatter from Markdown documents, and emits
    validated, immutable `RawFileRecord` Pydantic models with SHA-256 hashes.

Call Hierarchy:
    EmbeddingPipeline.ingest(target_path)
          │
          ▼
    FileLoader.scan(target_path)
          ├── [file]      ──► FileLoader.load_file(target_path)
          └── [directory] ──► rglob("*") ──► FileLoader.load_file(item)
                                                  └── [if .md] ──► FileLoader.parse_frontmatter()
"""

from os import walk
from pathlib import Path, PurePath
from stat import S_ISREG
import yaml
from pydantic import BaseModel, ConfigDict, Field
from config import settings
from utils import resolve_path, sha256_hex

DEFAULT_EXTENSIONS = {".py", ".md", ".markdown"}
DEFAULT_IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".rag_memory_data"}


class RawFileRecord(BaseModel):
    """Immutable, validated representation of a loaded source file ready for chunking."""

    model_config = ConfigDict(frozen=True)

    path: Path
    resolved_path: str
    filename: str
    extension: str
    content: str = Field(min_length=1)
    file_hash: str = Field(min_length=64, max_length=64)
    frontmatter: dict[str, str] = Field(default_factory=dict)


class FileLoader:
    """Scans files and directories into validated `RawFileRecord` instances."""

    def __init__(
        self,
        supported_extensions: set[str] | None = None,
        ignored_dirs: set[str] | None = None,
    ):
        """
        Initialize the loader with allowed file suffixes and excluded directory names.

        Args:
            supported_extensions: Lowercase file suffixes to include (defaults to `.py`, `.md`, `.markdown`).
            ignored_dirs: Directory names skipped during recursive folder scans.
        """
        self.supported_extensions = supported_extensions if supported_extensions is not None else DEFAULT_EXTENSIONS
        self.ignored_dirs = ignored_dirs if ignored_dirs is not None else DEFAULT_IGNORED_DIRS

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """
        Extract leading YAML frontmatter (`--- ... ---`) from a Markdown string.

        Args:
            text: Full UTF-8 Markdown content.

        Returns:
            Tuple of `(frontmatter_dict, markdown_body_without_frontmatter)`.
        """
        if not text.startswith("---\n"):
            return {}, text
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            return {}, text
        try:
            meta = yaml.safe_load(parts[0][4:]) or {}
            if isinstance(meta, dict):
                return {str(k): str(v) for k, v in meta.items()}, parts[1].lstrip("\n")
        except yaml.YAMLError:
            pass
        return {}, text

    def supports_file(self, path: PurePath) -> bool:
        """Check suffix and filename eligibility without opening the file."""
        return path.suffix.lower() in self.supported_extensions and path.name != "__init__.py"

    def load_file(self, file_path: str | Path) -> RawFileRecord | None:
        """
        Read a single file, compute its SHA-256 hash, and extract frontmatter if Markdown.

        Args:
            file_path: Path to a candidate `.py` or `.md` file.

        Returns:
            A validated `RawFileRecord`, or `None` if unsupported, empty, or `__init__.py`.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if not self.supports_file(path):
            return None
        try:
            regular = S_ISREG(path.stat().st_mode)
        except FileNotFoundError:
            return None
        if not regular:
            return None

        limit = settings.runtime.max_source_bytes
        with path.open("rb") as source:
            raw = source.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f"Source exceeds {limit} bytes: {path}")
        raw_text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if not raw_text.strip():
            return None

        frontmatter, body = (
            self.parse_frontmatter(raw_text) if ext in {".md", ".markdown"} else ({}, raw_text)
        )
        if not body.strip():
            return None

        return RawFileRecord(
            path=path,
            resolved_path=str(resolve_path(path)),
            filename=path.name,
            extension=ext,
            content=body,
            file_hash=sha256_hex(raw_text),
            frontmatter=frontmatter,
        )

    def scan(self, target_path: str | Path) -> list[RawFileRecord]:
        """
        Recursively discover and load all supported files under `target_path`.

        Args:
            target_path: Path to a single file or a root directory.

        Returns:
            Sorted list of `RawFileRecord` models ready for chunking.

        Raises:
            FileNotFoundError: If `target_path` does not exist on disk.
        """
        target = Path(target_path)
        if not target.exists():
            raise FileNotFoundError(f"Target path does not exist: {target.resolve()}")

        if target.is_file():
            rec = self.load_file(target)
            return [rec] if rec else []

        def fail_walk(error: OSError) -> None:
            raise error

        target = target.resolve()
        records = []
        size = 0
        for directory, subdirs, files in walk(target, onerror=fail_walk):
            subdirs[:] = sorted(
                name for name in subdirs
                if name not in self.ignored_dirs and not (Path(directory) / name).is_symlink()
            )
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                record = self.load_file(path)
                if record is None:
                    continue
                records.append(record)
                size += path.stat().st_size
                if len(records) > settings.runtime.max_scan_files or size > settings.runtime.max_scan_bytes:
                    raise ValueError("Scan exceeds configured file or byte limits. Ingest smaller directories.")
        return records
