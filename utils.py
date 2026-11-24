from hashlib import sha256
from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def sha256_hex(value: str, length: int = 64) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:length]
