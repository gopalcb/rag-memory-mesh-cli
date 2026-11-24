from pathlib import Path
from sqlalchemy import delete, select
from store.model import FileManifest
from store.session import get_session, init_store


class ManifestStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path
        init_store(self.db_path)

    def get_entry(
        self,
        file_path: str,
        db_path: str | Path | None = None,
    ) -> tuple[str, str, list[str]] | None:
        """Returns (file_hash, collection_name, chunk_ids) if tracked, else None."""
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            entry = session.execute(
                select(FileManifest).where(FileManifest.file_path == file_path)
            ).scalar_one_or_none()
            if not entry:
                return None
            return entry.file_hash, entry.collection_name, list(entry.chunk_ids)

    def upsert_entry(
        self,
        file_path: str,
        file_hash: str,
        collection_name: str,
        chunk_ids: list[str],
        db_path: str | Path | None = None,
    ) -> None:
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            entry = session.get(FileManifest, file_path)
            if entry:
                entry.file_hash = file_hash
                entry.collection_name = collection_name
                entry.chunk_ids = chunk_ids
            else:
                session.add(
                    FileManifest(
                        file_path=file_path,
                        file_hash=file_hash,
                        collection_name=collection_name,
                        chunk_ids=chunk_ids,
                    )
                )

    def delete_entry(self, file_path: str, db_path: str | Path | None = None) -> list[str]:
        """Removes a tracked file and returns its old chunk_ids for Chroma & Graph cleanup."""
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            entry = session.get(FileManifest, file_path)
            if not entry:
                return []
            old_ids = list(entry.chunk_ids)
            session.delete(entry)
            return old_ids

    def clear_all(self, db_path: str | Path | None = None) -> None:
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            session.execute(delete(FileManifest))

    def clear_collections(self, collections: list[str]) -> None:
        with get_session(self.db_path) as session:
            session.execute(delete(FileManifest).where(FileManifest.collection_name.in_(collections)))
