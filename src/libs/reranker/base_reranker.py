"""Reranker 抽象接口：所有重排实现必须实现 rerank 方法"""
from abc import ABC, abstractmethod
from typing import List
from src.core.types import RetrievalResult


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        """对检索结果按 query 相关性重新排序，返回 top_k"""
        ...