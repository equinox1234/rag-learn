"""OpenAI / DashScope 兼容的 Embedding 实现"""
from typing import List, Optional
from openai import OpenAI
from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002",
                 base_url: Optional[str] = None, **kwargs):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]