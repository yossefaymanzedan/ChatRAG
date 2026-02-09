from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create()

    def _get_or_create(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create()

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)

    def query(self, query_embedding: list[float], n_results: int) -> dict[str, Any]:
        if n_results <= 0:
            return {}
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances", "metadatas", "embeddings", "documents"],
        )
