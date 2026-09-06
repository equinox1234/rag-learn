"""LLM 抽象接口：所有 LLM 实现必须实现 ask 方法"""
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def ask(self, prompt: str, temperature: float = 0.0) -> str:
        """调用 LLM 生成回答"""
        ...