"""Ollama 本地 LLM 实现"""
from typing import Optional
from openai import OpenAI
from src.libs.llm.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3", **kwargs):
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self.model = model

    def ask(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""