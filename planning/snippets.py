import ast
from pathlib import Path
from tempfile import NamedTemporaryFile

from config import settings
from embedding.pipeline import EmbeddingPipeline
from utils import sha256_hex


def commit_verified_snippet(name: str, code: str, verification: str) -> dict:
    """Index an explicitly verified Python snippet in the service-owned code vault."""
    if not name.isidentifier() or len(name) > 70:
        raise ValueError("Snippet name must be a Python identifier of at most 70 characters")
    if not code.strip() or not verification.strip():
        raise ValueError("Provide code and the observed verification result")
    if len(code.encode("utf-8")) > settings.runtime.max_source_bytes or len(verification) > 4000:
        raise ValueError("Snippet or verification exceeds the configured size limit")
    ast.parse(code)
    vault = settings.storage.base_dir.resolve() / "vault" / "verified_code"
    vault.mkdir(parents=True, exist_ok=True)
    suffix = sha256_hex(code, 16)
    path = vault / f"{name}-{suffix}.py"
    note = "\n".join(f"# {line}" for line in verification.strip().splitlines())
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=vault, delete=False) as draft:
        temp_path = Path(draft.name)
        try:
            draft.write(f"{note}\n\n{code.rstrip()}\n")
            draft.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    stats = EmbeddingPipeline().ingest(path)
    return {"path": str(path), "verification": verification.strip(), "ingestion": stats.model_dump()}
