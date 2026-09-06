"""ChromaDB 向量存储实现"""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.core.types import ChunkRecord, RetrievalResult


class ChromaStore(BaseVectorStore):
    def __init__(self, persist_directory: str, collection_name: str = "default", **kwargs):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, records: List[ChunkRecord]) -> List[str]:
        ids = [r.id for r in records]
        texts = [r.text for r in records]
        vectors = [r.dense_vector for r in records]
        metadatas = [r.metadata for r in records]
        self.collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        return ids

    def query(self, query_vector: List[float], top_k: int = 10) -> List[RetrievalResult]:
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        if not results["ids"]:
            return []
        items = []
        for i, cid in enumerate(results["ids"][0]):
            items.append(RetrievalResult(
                chunk_id=cid,
                score=results["distances"][0][i] if results["distances"] else 0,
                text=results["documents"][0][i] if results["documents"] else "",
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))
        return items