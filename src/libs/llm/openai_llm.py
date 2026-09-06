"""OpenAI / DashScope 兼容的 LLM 实现"""
from typing import Optional
from openai import OpenAI
from src.libs.llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: str = "gpt-4o", **kwargs):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def ask(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""