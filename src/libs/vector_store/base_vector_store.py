"""VectorStore 抽象接口：所有向量存储实现必须实现 upsert 和 query"""
from abc import ABC, abstractmethod
from typing import List
from src.core.types import ChunkRecord, RetrievalResult


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, records: List[ChunkRecord]) -> List[str]:
        """存入向量记录，返回 ID 列表"""
        ...

    @abstractmethod
    def query(self, query_vector: List[float], top_k: int = 10) -> List[RetrievalResult]:
        """用向量检索，返回最相似的 top_k 条"""
        ...