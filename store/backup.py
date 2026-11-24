import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from config import StorageConfig, settings


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def marker(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Unexpected file type in snapshot: {path.relative_to(root)}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = checksum(path)
    return files


def sqlite_snapshot(source: Path, target: Path) -> None:
    original = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    replica = sqlite3.connect(target)
    try:
        original.backup(replica)
        replica.execute("PRAGMA journal_mode=DELETE")
        if replica.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError(f"SQLite integrity check failed for {source.name}")
    finally:
        replica.close()
        original.close()


def components(storage: StorageConfig) -> dict[str, Path]:
    return {
        "chroma": storage.chroma_dir,
        "state.sqlite3": storage.sqlite_db_path,
        "checkpoints.sqlite3": storage.sqlite_db_path.with_suffix(".checkpoints.sqlite3"),
        "markdown": storage.markdown_vault_dir,
        "verified_code": storage.base_dir / "vault" / "verified_code",
        "ingestion-ui.sqlite3": storage.base_dir / "ingestion-ui.sqlite3",
    }


def copy_components(source: dict[str, Path], target: dict[str, Path]) -> None:
    for name, original in source.items():
        if name not in target or not original.exists():
            continue
        if original.is_symlink():
            raise ValueError(f"Unexpected symlink in {name}")
        destination = target[name]
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if original.is_dir():
            shutil.copytree(original, destination, symlinks=True, dirs_exist_ok=True)
        elif original.is_file():
            sqlite_snapshot(original, destination)
        else:
            raise ValueError(f"Unexpected file type in {name}")


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_snapshot(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        parent = Path(directory)
        for name in files:
            with (parent / name).open("rb") as file:
                os.fsync(file.fileno())
        fsync_directory(parent)


def export_backup(path: str | Path) -> dict:
    target = Path(path).expanduser().resolve()
    active = components(settings.storage)
    if target.is_relative_to(settings.storage.base_dir) or any(
        target.is_relative_to(path) for path in active.values()
    ):
        raise ValueError("Backup destination must be outside active storage")
    if target.exists():
        raise FileExistsError("Backup destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rag-backup-", dir=target.parent) as workspace:
        snapshot = Path(workspace) / "snapshot"
        snapshot.mkdir(mode=0o700)
        root = snapshot / "data"
        root.mkdir(mode=0o700)
        copy_components(active, {name: root / name for name in active})
        files = marker(root)
        (snapshot / "manifest.json").write_text(
            json.dumps({"version": 1, "files": files}, indent=2) + "\n", encoding="utf-8",
        )
        sync_snapshot(snapshot)
        # Rename the complete snapshot. An existing nonempty destination is protected by rename.
        if target.exists():
            raise FileExistsError("Backup destination already exists")
        os.rename(snapshot, target)
        fsync_directory(target.parent)
    return {"status": "ok", "snapshot": str(target), "files": len(files)}


def import_backup(path: str | Path) -> dict:
    snapshot = Path(path).expanduser().resolve()
    manifest_path = snapshot / "manifest.json"
    root = snapshot / "data"
    if manifest_path.is_symlink() or root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid snapshot layout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = marker(root)
    if manifest.get("version") != 1 or files != manifest.get("files"):
        raise ValueError("Snapshot format or checksums do not match")

    storage = settings.storage
    base = storage.base_dir
    live = components(storage)
    if not all(path.is_relative_to(base) and path != base for path in live.values()):
        raise ValueError("Restore requires all configured components under storage.base_dir")
    existing = list(base.rglob("*"))
    if any(path.is_symlink() or not path.is_dir() for path in existing):
        raise FileExistsError("Restore requires an empty storage directory")
    if snapshot.is_relative_to(base):
        raise ValueError("Snapshot must be outside the restore destination")

    source = {name: root / name for name in live}
    with TemporaryDirectory(prefix=".rag-restore-", dir=base.parent) as workspace:
        destination = Path(workspace) / "store"
        destination.mkdir(mode=0o700)
        targets = {name: destination / path.relative_to(base) for name, path in live.items()}
        copy_components(source, targets)
        for name in ("chroma", "markdown"):
            targets[name].mkdir(parents=True, exist_ok=True, mode=0o700)
        if marker(root) != files:
            raise ValueError("Snapshot changed during restore")
        sync_snapshot(destination)
        # The live destination is empty. Never remove or replace existing data files.
        for directory in sorted(existing, key=lambda path: len(path.parts), reverse=True):
            directory.rmdir()
        os.replace(destination, base)
        fsync_directory(base.parent)
    return {"status": "ok", "restored": str(base), "files": len(files)}
