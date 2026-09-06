"""LLM Factory：根据配置创建对应的 LLM 实例"""
from src.libs.llm.base_llm import BaseLLM


def create_llm(cfg: dict) -> BaseLLM:
    """根据配置创建 LLM 实例

    cfg 来自 settings.yaml 的 llm 段：
      llm:
        provider: "openai" | "dashscope" | "ollama"
        ...
    """
    provider = cfg.get("provider", "openai")

    if provider in ("openai", "dashscope"):
        from src.libs.llm.openai_llm import OpenAILLM
        return OpenAILLM(
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url"),
            model=cfg.get("model", "gpt-4o"),
        )

    if provider == "ollama":
        from src.libs.llm.ollama_llm import OllamaLLM
        return OllamaLLM(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            model=cfg.get("model", "llama3"),
        )

    raise ValueError(f"不支持的 LLM provider: {provider}")