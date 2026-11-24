import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from config import settings
from utils import sha256_hex


def slugify_text(text: str, max_len: int = 80) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len].strip("-") or "document"


class MarkdownVaultStore:
    def __init__(self, vault_dir: str | Path | None = None):
        self.vault_dir = Path(vault_dir) if vault_dir is not None else settings.storage.markdown_vault_dir

    def save_markdown(
        self, content: str, title: str, source_url: str, topic: str = "general",
        extra_meta: dict[str, Any] | None = None,
        *, existing_path: Path | None = None,
    ) -> Path:
        slug = slugify_text(title)
        path = self.vault_dir / f"{slug}-{sha256_hex(source_url, 16)}.md"
        if existing_path is not None:
            prior = existing_path.resolve()
            if prior.parent == self.vault_dir.resolve() and prior.suffix.lower() == ".md":
                path = prior
        metadata = {
            **(extra_meta or {}), "title": title, "source_url": source_url, "topic": topic,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        }
        frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.vault_dir, delete=False) as draft:
            draft_path = Path(draft.name)
            try:
                draft.write(f"---\n{frontmatter}---\n\n{content.rstrip()}\n")
                draft.close()
                draft_path.replace(path)
            finally:
                draft_path.unlink(missing_ok=True)
        return path
