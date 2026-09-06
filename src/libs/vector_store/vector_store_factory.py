"""VectorStore Factory：根据配置创建对应的向量存储实例"""
from src.libs.vector_store.base_vector_store import BaseVectorStore


def create_vector_store(cfg: dict) -> BaseVectorStore:
    """根据配置创建 VectorStore 实例

    cfg 来自 settings.yaml 的 vector_store 段：
      vector_store:
        provider: "chroma"
        ...
    """
    provider = cfg.get("provider", "chroma")

    if provider == "chroma":
        from src.libs.vector_store.chroma_store import ChromaStore
        return ChromaStore(
            persist_directory=cfg.get("persist_directory", "./data/db/chroma"),
            collection_name=cfg.get("collection_name", "default"),
        )

    raise ValueError(f"不支持的 VectorStore provider: {provider}")