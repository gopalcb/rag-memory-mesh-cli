import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from langchain_core.documents import Document
from sqlalchemy import text
from config import SearchOptions, settings
from store.chroma_store import ChromaStoreManager
from store.graph_store import CodeGraphStore
from store.session import get_session


@dataclass
class RetrievalHit:
    document: Document
    rrf_score: float
    vector_distance: float | None = None
    bm25_rank: int | None = None
    via_graph_relation: str | None = None  # e.g., "calls:retry_async"
    metadata: dict = field(default_factory=dict)


def build_fts_query(raw_query: str) -> str:
    """Converts natural language / identifiers into a safe OR-joined FTS5 token query."""
    tokens = re.findall(r"\w+", raw_query)
    if not tokens:
        return ""
    unique_tokens = list(dict.fromkeys(tokens))[:12]
    return " OR ".join(f'"{tok}"' for tok in unique_tokens)


class HybridRetriever:
    def __init__(
        self,
        chroma: ChromaStoreManager | None = None,
        graph: CodeGraphStore | None = None,
        db_path: str | Path | None = None,
    ):
        self.db_path = db_path if db_path is not None else (chroma.db_path if chroma is not None else None)
        self.chroma = chroma if chroma is not None else ChromaStoreManager(db_path=self.db_path)
        self.graph = graph if graph is not None else CodeGraphStore(db_path=self.db_path)

    def search_bm25(
        self,
        query: str,
        collection_name: str,
        limit: int,
        keywords: list[str] | None = None,
    ) -> list[str]:
        if keywords is None:
            fts_query = build_fts_query(query)
        else:
            phrases = dict.fromkeys(" ".join(re.findall(r"\w+", keyword)) for keyword in keywords)
            fts_query = " OR ".join(f'"{phrase}"' for phrase in phrases if phrase)
        if not fts_query:
            return []
        with get_session(self.db_path) as session:
            return list(session.scalars(
                text(
                    """
                    SELECT chunk_id
                    FROM chunks_fts
                    WHERE chunks_fts MATCH :q AND collection_name = :col
                    ORDER BY bm25(chunks_fts), chunk_id
                    LIMIT :lim
                    """
                ),
                {"q": fts_query, "col": collection_name, "lim": limit},
            ))

    def search(
        self,
        query: str,
        collection_name: str | SearchOptions | dict | None = None,
        mode: Literal["hybrid", "vector", "keyword"] | None = None,
        k: int | None = None,
        max_distance: float | None = None,
        expand_ast_neighbors: bool | None = None,
        where: dict | None = None,
        *,
        options: SearchOptions | dict | None = None,
    ) -> list[RetrievalHit]:
        """Executes Hybrid (Vector + BM25 RRF) search and expands 1-hop AST dependencies."""
        if options is not None:
            if isinstance(collection_name, (dict, SearchOptions)):
                raise TypeError("Specify search options once")
            opts = SearchOptions.model_validate(options)
        else:
            opts = collection_name if isinstance(collection_name, (dict, SearchOptions)) else {
                "collection_name": collection_name,
            }
            opts = SearchOptions.model_validate(opts)
        explicit = SearchOptions(
            collection_name=collection_name if isinstance(collection_name, str) else None,
            mode=mode, k=k, max_distance=max_distance,
            expand_ast_neighbors=expand_ast_neighbors, where=where,
        )
        opts = opts.model_copy(update=explicit.model_dump(exclude_none=True))
        config = settings.retrieval
        target_col = opts.collection_name or settings.collections.code
        active_mode = opts.mode or config.mode
        top_k = opts.k if opts.k is not None else config.default_top_k
        use_graph = opts.expand_ast_neighbors if opts.expand_ast_neighbors is not None else config.expand_ast_neighbors
        if not query.strip() or top_k <= 0:
            return []

        candidate_k = top_k * 2
        doc_map: dict[str, Document] = {}
        vec_dist_map: dict[str, float] = {}
        rrf_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}

        # 1. Dense Vector Search (ChromaDB)
        if active_mode in ("hybrid", "vector"):
            vec_hits = self.chroma.search_vector(
                query=query,
                collection_name=target_col,
                k=candidate_k,
                max_distance=opts.max_distance,
                where=opts.where,
            )
            for rank, (doc, dist) in enumerate(vec_hits, start=1):
                cid = doc.metadata["chunk_id"]
                doc_map[cid] = doc
                vec_dist_map[cid] = dist
                rrf_scores[cid] = config.vector_weight / (config.rrf_k + rank)

        # 2. Lexical BM25 Search (SQLite FTS5)
        if active_mode in ("hybrid", "keyword"):
            bm25_ids = self.search_bm25(
                query=query, collection_name=target_col, limit=candidate_k, keywords=opts.keywords,
            )
            missing_ids = [cid for cid in bm25_ids if cid not in doc_map]
            if missing_ids:
                for doc in self.chroma.get_by_ids(target_col, missing_ids, where=opts.where):
                    doc_map[doc.metadata["chunk_id"]] = doc

            for rank, cid in enumerate(bm25_ids, start=1):
                if cid in doc_map:
                    bm25_ranks[cid] = rank
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + config.keyword_weight / (config.rrf_k + rank)

        # 3. Rank by Fused RRF Score
        ranked_ids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:top_k]
        hits: list[RetrievalHit] = [
            RetrievalHit(
                document=doc_map[cid],
                rrf_score=round(rrf_scores[cid], 5),
                vector_distance=round(vec_dist_map[cid], 4) if cid in vec_dist_map else None,
                bm25_rank=bm25_ranks.get(cid),
                metadata=doc_map[cid].metadata,
            )
            for cid in ranked_ids
        ]

        # 4. 1-Hop AST Graph Expansion (collections with code graph nodes)
        if use_graph and ranked_ids:
            neighbors = self.graph.get_1hop_neighbor_chunk_ids(
                ranked_ids, limit=config.max_neighbor_chunks, collection_name=target_col,
            )
            neighbor_ids = [n["chunk_id"] for n in neighbors]
            neighbor_docs = {
                d.metadata["chunk_id"]: d
                for d in self.chroma.get_by_ids(target_col, neighbor_ids, where=opts.where)
            }
            for n in neighbors:
                ndoc = neighbor_docs.get(n["chunk_id"])
                if ndoc:
                    hits.append(
                        RetrievalHit(
                            document=ndoc,
                            rrf_score=0.0,
                            via_graph_relation=f"{n['relation']}:{n['symbol_name']}",
                            metadata=ndoc.metadata,
                        )
                    )

        return hits
