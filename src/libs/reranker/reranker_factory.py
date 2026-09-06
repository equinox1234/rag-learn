"""Reranker Factory：根据配置创建对应的 Reranker 实例"""
from src.libs.reranker.base_reranker import BaseReranker


def create_reranker(cfg: dict, llm=None) -> BaseReranker:
    """根据配置创建 Reranker

    cfg 来自 settings.yaml 的 rerank 段：
      rerank:
        enabled: true
        provider: "cross_encoder" | "llm"
        model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """
    if not cfg.get("enabled", False):
        return None

    provider = cfg.get("provider", "cross_encoder")

    if provider == "cross_encoder":
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker(
            model_name=cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        )

    if provider == "llm":
        if llm is None:
            raise ValueError("LLM Reranker 需要传入 llm 实例")
        from src.libs.reranker.llm_reranker import LLMReranker
        return LLMReranker(llm=llm)

    raise ValueError(f"不支持的 Reranker provider: {provider}")