"""Embedding 抽象接口：所有 Embedding 实现必须实现 embed 方法"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """把文本列表转成向量列表"""
        ...