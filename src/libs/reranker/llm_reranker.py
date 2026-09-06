"""LLM Reranker：用 LLM 对检索结果重新排序

原理：让 LLM 判断每个 chunk 与 query 的相关性，按相关度排序。
适用场景：没有 Cross-Encoder 模型时，复用已有的 LLM。
"""
from typing import List
from src.libs.reranker.base_reranker import BaseReranker
from src.core.types import RetrievalResult


class LLMReranker(BaseReranker):
    def __init__(self, llm, **kwargs):
        self.llm = llm

    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        if not results:
            return []

        # 对每个 chunk 让 LLM 打分
        for r in results:
            prompt = f"""请判断以下文档片段与问题的相关程度，只输出一个 0-100 的分数。

问题：{query}

文档片段：{r.text[:300]}

相关度分数（0=完全不相关，100=高度相关）："""
            try:
                score_str = self.llm.ask(prompt, temperature=0.0).strip()
                r.score = float(score_str.split()[0])
            except (ValueError, IndexError):
                r.score = 0

        # 按新分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]