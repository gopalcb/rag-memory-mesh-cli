from pathlib import Path

from sqlalchemy import delete, select

from store.model import FileManifest, IndexCollectionContract, IndexFingerprint, IndexJournalEntry
from store.session import get_session


class IndexState:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path

    def stage(self, payload: dict) -> int:
        with get_session(self.db_path) as session:
            row = IndexJournalEntry(payload=payload)
            session.add(row)
            session.flush()
            return row.id

    def pending(self) -> list[tuple[int, dict]]:
        with get_session(self.db_path) as session:
            rows = session.scalars(select(IndexJournalEntry).order_by(IndexJournalEntry.id))
            return [(row.id, row.payload) for row in rows]

    def finish(self, op_id: int) -> None:
        with get_session(self.db_path) as session:
            session.execute(delete(IndexJournalEntry).where(IndexJournalEntry.id == op_id))

    def fingerprint(self, path: str) -> str | None:
        with get_session(self.db_path) as session:
            row = session.get(IndexFingerprint, path)
            return row.fingerprint if row is not None else None

    def set_fingerprint(self, path: str, fingerprint: str) -> None:
        with get_session(self.db_path) as session:
            session.merge(IndexFingerprint(file_path=path, fingerprint=fingerprint))

    def paths_under(self, root: Path, collections: set[str]) -> set[str]:
        with get_session(self.db_path) as session:
            paths = session.scalars(
                select(FileManifest.file_path).where(FileManifest.collection_name.in_(collections))
            )
            return {path for path in paths if Path(path).is_relative_to(root)}

    def contract(self, name: str) -> tuple[str, int] | None:
        with get_session(self.db_path) as session:
            row = session.get(IndexCollectionContract, name)
            return (row.model_id, row.dimension) if row is not None else None

    def set_contract(self, name: str, model_id: str, dimension: int) -> None:
        with get_session(self.db_path) as session:
            session.merge(IndexCollectionContract(name=name, model_id=model_id, dimension=dimension))

    def clear_contracts(self, names: list[str]) -> None:
        with get_session(self.db_path) as session:
            session.execute(delete(IndexCollectionContract).where(IndexCollectionContract.name.in_(names)))
