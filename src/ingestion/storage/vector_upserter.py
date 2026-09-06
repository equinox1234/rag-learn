"""向量上载器"""
from typing import List
from src.libs.vector_store.base_vector_store import BaseVectorStore


class VectorUpserter:
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def upsert(self, records: List) -> List[str]:
        return self.vector_store.upsert(records)