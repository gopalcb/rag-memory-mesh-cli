import hashlib
import os
from pathlib import Path

from embedding.loaders import DEFAULT_EXTENSIONS, DEFAULT_IGNORED_DIRS
from ingestion_ui.queue import IngestionQueue


def raise_walk_error(error: OSError) -> None:
    raise error


def folder_signature(root: Path) -> tuple[str, bool]:
    """Fingerprint eligible files so unchanged folders do not create jobs."""
    digest = hashlib.sha256()
    has_files = False
    for parent, dirs, files in os.walk(root, onerror=raise_walk_error):
        dirs[:] = sorted(name for name in dirs if name not in DEFAULT_IGNORED_DIRS)
        for name in sorted(files):
            file = Path(parent) / name
            if file.suffix.lower() not in DEFAULT_EXTENSIONS or name == "__init__.py":
                continue
            try:
                if not file.is_file():
                    continue
                stat = file.stat()
            except FileNotFoundError:
                continue
            has_files = True
            relative = file.relative_to(root)
            digest.update(os.fsencode(relative))
            digest.update(f"\0{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest(), has_files


def scan_watches(queue: IngestionQueue) -> None:
    for watch in queue.active_watches():
        try:
            signature, has_files = folder_signature(Path(watch["path"]))
        except OSError as error:
            queue.note_watch_error(watch["id"], error.strerror or str(error))
        else:
            queue.observe(watch["id"], signature, has_files)
