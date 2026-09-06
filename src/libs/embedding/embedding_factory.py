"""Embedding Factory：根据配置创建对应的 Embedding 实例"""
from src.libs.embedding.base_embedding import BaseEmbedding


def create_embedding(cfg: dict) -> BaseEmbedding:
    """根据配置创建 Embedding 实例

    cfg 来自 settings.yaml 的 embedding 段：
      embedding:
        provider: "openai" | "dashscope" | "ollama"
        ...
    """
    provider = cfg.get("provider", "openai")

    if provider in ("openai", "dashscope"):
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        return OpenAIEmbedding(
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url"),
            model=cfg.get("model", "text-embedding-ada-002"),
        )

    if provider == "ollama":
        from src.libs.embedding.ollama_embedding import OllamaEmbedding
        return OllamaEmbedding(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            model=cfg.get("model", "nomic-embed-text"),
        )

    raise ValueError(f"不支持的 Embedding provider: {provider}")