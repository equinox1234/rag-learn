"""Cross-Encoder Reranker：用专用模型对 (query, chunk) 逐对打分

原理：
  普通的向量检索（Bi-Encoder）把 query 和 chunk 分别编码成向量，再算距离。
  这种方式快但精度有限，因为 query 和 chunk 是独立编码的。

  Cross-Encoder 把 query 和 chunk 拼在一起输入模型，让模型同时看到两者，
  直接输出相关性分数。精度更高，但速度慢（每对都要算一次）。

  所以 Rerank 只对 RRF 融合后的 Top-N（N=20~30）重新打分，不做全量。
"""
from typing import List
from src.libs.reranker.base_reranker import BaseReranker
from src.core.types import RetrievalResult


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 **kwargs):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        if not results:
            return []

        # 构造 (query, chunk_text) 对
        pairs = [(query, r.text) for r in results]

        # 模型打分
        scores = self.model.predict(pairs)

        # 按新分数排序
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        for r, s in scored:
            r.score = float(s)

        return [r for r, _ in scored[:top_k]]