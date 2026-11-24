from pathlib import Path
from sqlalchemy import delete, select
from store.model import CodeEdge, CodeNode
from store.session import get_session, init_store


class CodeGraphStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path
        init_store(self.db_path)

    def upsert_node_with_edges(
        self,
        chunk_id: str,
        symbol_name: str,
        symbol_type: str,
        file_path: str,
        source: str,
        collection_name: str,
        edges: list[tuple[str, str]] | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        """Registers a code symbol node and its (target_symbol, relation) edges."""
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            existing = session.get(CodeNode, chunk_id)
            if existing:
                existing.symbol_name = symbol_name
                existing.symbol_type = symbol_type
                existing.file_path = file_path
                existing.source = source
                existing.collection_name = collection_name
                session.execute(delete(CodeEdge).where(CodeEdge.source_chunk_id == chunk_id))
            else:
                session.add(
                    CodeNode(
                        chunk_id=chunk_id,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        file_path=file_path,
                        source=source,
                        collection_name=collection_name,
                    )
                )
                session.flush()

            seen: set[tuple[str, str]] = set()
            for target_symbol, relation in edges or []:
                if not target_symbol or (target_symbol, relation) in seen:
                    continue
                seen.add((target_symbol, relation))
                session.add(
                    CodeEdge(
                        source_chunk_id=chunk_id,
                        target_symbol=target_symbol,
                        relation=relation,
                    )
                )

    def get_1hop_neighbor_chunk_ids(
        self,
        chunk_ids: list[str],
        limit: int = 3,
        db_path: str | Path | None = None,
        *,
        collection_name: str | None = None,
    ) -> list[dict]:
        """Finds chunk_ids defining symbols called/imported/inherited by `chunk_ids`."""
        if not chunk_ids or limit <= 0:
            return []
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            edge_rows = session.execute(
                select(CodeEdge.target_symbol, CodeEdge.relation).where(
                    CodeEdge.source_chunk_id.in_(chunk_ids)
                )
            ).all()
            if not edge_rows:
                return []

            symbol_to_rel = {sym: rel for sym, rel in edge_rows}
            neighbor_nodes = session.execute(
                select(CodeNode)
                .where(
                    CodeNode.symbol_name.in_(list(symbol_to_rel.keys())),
                    CodeNode.chunk_id.notin_(chunk_ids),
                )
                .where(CodeNode.collection_name == collection_name if collection_name is not None else True)
                .order_by(CodeNode.symbol_name, CodeNode.chunk_id)
                .limit(limit)
            ).scalars().all()

            return [
                {
                    "chunk_id": node.chunk_id,
                    "symbol_name": node.symbol_name,
                    "relation": symbol_to_rel.get(node.symbol_name, "depends_on"),
                    "collection_name": node.collection_name,
                    "source": node.source,
                }
                for node in neighbor_nodes
            ]

    def delete_by_chunk_ids(
        self, chunk_ids: list[str], db_path: str | Path | None = None
    ) -> None:
        if not chunk_ids:
            return
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            session.execute(delete(CodeNode).where(CodeNode.chunk_id.in_(chunk_ids)))

    def clear_all(self, db_path: str | Path | None = None) -> None:
        target_db = db_path if db_path is not None else self.db_path
        with get_session(target_db) as session:
            session.execute(delete(CodeEdge))
            session.execute(delete(CodeNode))

    def clear_collections(self, collections: list[str]) -> None:
        with get_session(self.db_path) as session:
            session.execute(delete(CodeNode).where(CodeNode.collection_name.in_(collections)))
