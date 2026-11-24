from collections import Counter
from pathlib import Path

from sqlalchemy import select, text

from store.chroma_store import ChromaStoreManager
from store.index_state import IndexState
from store.model import CodeNode, FileManifest, IndexCollectionContract
from store.session import get_session


def audit_index(chroma: ChromaStoreManager | None = None) -> dict:
    chroma = chroma or ChromaStoreManager()
    expected = set()
    expected_nodes = set()
    with get_session(chroma.db_path) as session:
        for manifest in session.scalars(select(FileManifest)):
            keys = {(manifest.collection_name, chunk_id) for chunk_id in manifest.chunk_ids}
            expected.update(keys)
            if Path(manifest.file_path).suffix.lower() == ".py":
                expected_nodes.update(keys)
        lexical_counts = Counter(tuple(row) for row in session.execute(
            text("SELECT collection_name, chunk_id FROM chunks_fts")
        ))
        graph = set(map(tuple, session.execute(select(CodeNode.collection_name, CodeNode.chunk_id))))
        contracts = {
            row.name: row.dimension for row in session.scalars(select(IndexCollectionContract))
        }
        foreign_key_errors = len(session.execute(text("PRAGMA foreign_key_check")).all())
        integrity_checks = list(session.scalars(text("PRAGMA quick_check")))

    dense = set()
    unknown_model = []
    dimension_mismatch = []
    for collection in chroma.client.list_collections():
        name = collection.name
        count = collection.count()
        if count and name not in contracts:
            unknown_model.append(name)
        if count and name in contracts:
            sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
            if sample is None or len(sample[0]) != contracts[name]:
                dimension_mismatch.append(name)
        for offset in range(0, count, 1000):
            dense.update((name, chunk_id) for chunk_id in collection.get(
                offset=offset, limit=1000, include=[]
            )["ids"])

    lexical = set(lexical_counts)
    pending = IndexState(chroma.db_path).pending()
    issues = {
        "missing_dense": sorted(expected - dense),
        "orphan_dense": sorted(dense - expected),
        "missing_lexical": sorted(expected - lexical),
        "orphan_lexical": sorted(lexical - expected),
        "duplicate_lexical": sorted(key for key, count in lexical_counts.items() if count > 1),
        "missing_graph_nodes": sorted(expected_nodes - graph),
        "orphan_graph_nodes": sorted(graph - expected_nodes),
        "unknown_embedding_contract": sorted(unknown_model),
        "embedding_dimension_mismatch": sorted(dimension_mismatch),
        "pending_operations": [op_id for op_id, _ in pending],
        "foreign_key_errors": foreign_key_errors,
        "sqlite_integrity_errors": [] if integrity_checks == ["ok"] else integrity_checks,
    }
    return {
        "status": "failed" if any(issues.values()) else "ok",
        "counts": {"manifest": len(expected), "dense": len(dense), "lexical": len(lexical), "graph": len(graph)},
        "issues": issues,
    }
