"""Ollama 本地 Embedding 实现"""
from typing import List, Optional
from openai import OpenAI
from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "nomic-embed-text", **kwargs):
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]