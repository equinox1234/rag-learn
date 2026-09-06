"""RRF 融合：把向量检索和 BM25 检索的结果合并排序

RRF (Reciprocal Rank Fusion) 公式：
  score(chunk) = 1/(k + rank_dense) + 1/(k + rank_sparse)

不依赖原始分数，只依赖排名，所以两路结果分数差异大也不影响。
"""
from typing import List, Dict
from src.core.types import RetrievalResult


class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k  # 平滑常数，默认 60

    def fuse(self, dense_results: List[RetrievalResult],
             sparse_results: List[RetrievalResult],
             top_k: int = 10) -> List[RetrievalResult]:
        """融合两路结果，返回排序后的列表"""

        # 给每个 chunk 算 RRF 分数
        scores: Dict[str, float] = {}

        # 稠密检索：排名越靠前，分数越高
        for rank, r in enumerate(dense_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (self.k + rank + 1)

        # 稀疏检索：排名越靠前，分数越高
        for rank, r in enumerate(sparse_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (self.k + rank + 1)

        # 按 RRF 分数排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 把 chunk_id 映射回完整的 RetrievalResult（保留分数高的那个）
        result_map = {r.chunk_id: r for r in dense_results + sparse_results}

        final = []
        seen = set()
        for chunk_id, score in ranked:
            if chunk_id not in seen and chunk_id in result_map:
                r = result_map[chunk_id]
                r.score = score  # 用 RRF 分数替换原始分数
                final.append(r)
                seen.add(chunk_id)

        return final[:top_k]