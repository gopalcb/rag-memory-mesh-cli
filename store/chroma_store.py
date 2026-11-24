from itertools import batched
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text
from config import settings
from store.index_state import IndexState
from store.session import get_session
from utils import sha256_hex


class ChromaStoreManager:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        embeddings: Embeddings | None = None,
        batch_size: int | None = None,
        model_identity: str | None = None,
    ):
        self.persist_dir = (
            str(Path(persist_dir).resolve())
            if persist_dir is not None
            else str(settings.storage.chroma_dir.resolve())
        )
        self.db_path = db_path
        self._embeddings = embeddings
        self.managed_model = embeddings is None or model_identity is not None
        self.model_identity = sha256_hex(model_identity or self.describe_model(embeddings))
        self.index_state = IndexState(db_path)
        self.batch_size = batch_size if batch_size is not None else settings.embedding.batch_size
        self._client = None

    @staticmethod
    def describe_model(embeddings: Embeddings | None) -> str:
        if embeddings is not None:
            return type(embeddings).__module__ + "." + type(embeddings).__qualname__
        config = settings.embedding
        if config.provider == "api":
            base = urlsplit(config.api.base_url)
            return f"api:{base.scheme}:{base.hostname}:{base.port}:{base.path}:{config.api.model_name}"
        return f"local:{config.local.model_name}:{config.local.effective_revision}:{config.local.normalize_embeddings}"

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from embedding.model import get_embedding_client
            self._embeddings = get_embedding_client()
        return self._embeddings

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def raw_collection(self, name: str):
        return self.client.get_or_create_collection(
            name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        )

    def existing_collection(self, name: str):
        try:
            return self.client.get_collection(name, embedding_function=None)
        except NotFoundError:
            return None

    def check_contract(self, name: str, model_id: str, dimension: int | None = None) -> None:
        contract = self.index_state.contract(name)
        if contract:
            if contract[0] != model_id or (dimension is not None and contract[1] != dimension):
                raise ValueError(f"Embedding model or dimensions changed for {name}. Rebuild with ingest --reset.")
        elif self.managed_model:
            collection = self.existing_collection(name)
            if collection is not None and collection.count():
                raise ValueError(f"{name} has a legacy index with no model identity. Rebuild with ingest --reset.")

    def encode_chunks(self, chunks: list[Document], batch_size: int | None = None) -> list[list[float]]:
        step = self.batch_size if batch_size is None else batch_size
        if step < 1:
            raise ValueError("Embedding batch size must be positive")
        vectors: list[list[float]] = []
        dimensions = None
        for group in batched(chunks, step):
            batch = self.embeddings.embed_documents([doc.page_content for doc in group])
            if len(batch) != len(group):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            for vector in batch:
                values = [float(value) for value in vector]
                if dimensions is None:
                    dimensions = len(values)
                if not values or len(values) != dimensions or not all(map(isfinite, values)):
                    raise ValueError("Embedding provider returned invalid vectors")
                vectors.append(values)
        return vectors

    def get_collection(self, collection_name: str) -> Chroma:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            client=self.client,
            embedding_function=self._embeddings,
            collection_configuration={"hnsw": {"space": "cosine"}},
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Document],
        batch_size: int | None = None,
        *, vectors: list[list[float]] | None = None,
    ) -> list[str]:
        """Upserts chunks into both ChromaDB (dense) and SQLite FTS5 (BM25 lexical)."""
        if not chunks:
            return []
        step = batch_size if batch_size is not None else self.batch_size
        if step < 1:
            raise ValueError("Embedding batch size must be positive")
        vectors = self.encode_chunks(chunks, step) if vectors is None else vectors
        if len(vectors) != len(chunks):
            raise ValueError("Missing prepared embeddings")
        store = self.raw_collection(collection_name)
        step = min(step, self.client.get_max_batch_size())
        ids = [doc.metadata["chunk_id"] for doc in chunks]

        for offset in range(0, len(chunks), step):
            group = chunks[offset : offset + step]
            store.upsert(
                ids=ids[offset : offset + step],
                documents=[doc.page_content for doc in group],
                metadatas=[doc.metadata for doc in group],
                embeddings=vectors[offset : offset + step],
            )

        with get_session(self.db_path) as session:
            for doc in chunks:
                cid = doc.metadata["chunk_id"]
                session.execute(
                    text("DELETE FROM chunks_fts WHERE collection_name = :col AND chunk_id = :cid"),
                    {"cid": cid, "col": collection_name},
                )
                session.execute(
                    text(
                        """
                        INSERT INTO chunks_fts (chunk_id, collection_name, source, symbol_name, content)
                        VALUES (:cid, :col, :src, :sym, :content)
                        """
                    ),
                    {
                        "cid": cid,
                        "col": collection_name,
                        "src": doc.metadata.get("source", ""),
                        "sym": doc.metadata.get("symbol_name", ""),
                        "content": doc.page_content,
                    },
                )
        return ids

    def get_by_ids(
        self, collection_name: str, ids: list[str], where: dict | None = None,
    ) -> list[Document]:
        """Fetches full Document objects by chunk_id (used for BM25 hits & AST graph neighbors)."""
        if not ids:
            return []
        store = self.get_collection(collection_name)
        raw = store.get(ids=ids, where=where, include=["documents", "metadatas"])
        docs: list[Document] = []
        for content, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def delete_ids(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        collection = self.existing_collection(collection_name)
        if collection is not None:
            for group in batched(ids, self.client.get_max_batch_size()):
                collection.delete(ids=list(group))
        with get_session(self.db_path) as session:
            for cid in ids:
                session.execute(
                    text("DELETE FROM chunks_fts WHERE collection_name = :col AND chunk_id = :cid"),
                    {"cid": cid, "col": collection_name},
                )

    def reset_collections(self, collections: list[str] | None = None) -> None:
        targets = (
            collections
            if collections is not None
            else [settings.collections.code, settings.collections.docs]
        )
        for name in targets:
            if self.existing_collection(name) is not None:
                self.client.delete_collection(name)
            with get_session(self.db_path) as session:
                session.execute(
                    text("DELETE FROM chunks_fts WHERE collection_name = :col"),
                    {"col": name},
                )

    def search_vector(
        self,
        query: str,
        collection_name: str | None = None,
        k: int | None = None,
        max_distance: float | None = None,
        where: dict | None = None,
    ) -> list[tuple[Document, float]]:
        target_col = (
            collection_name if collection_name is not None else settings.collections.code
        )
        top_k = k if k is not None else settings.retrieval.default_top_k
        default_cutoff = (
            settings.retrieval.code_max_distance
            if target_col == settings.collections.code
            else settings.retrieval.docs_max_distance
        )
        cutoff = max_distance if max_distance is not None else default_cutoff

        if top_k <= 0:
            return []
        collection = self.existing_collection(target_col)
        if collection is None or not collection.count():
            return []
        self.check_contract(target_col, self.model_identity)
        query_vector = self.embeddings.embed_query(query)
        self.check_contract(target_col, self.model_identity, len(query_vector))
        store = self.get_collection(target_col)
        results = store.similarity_search_by_vector_with_relevance_scores(query_vector, k=top_k, filter=where)
        return [(doc, score) for doc, score in results if score <= cutoff]
