"""BM25 索引器：摄入时建关键词索引，查询时按关键词匹配"""
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import jieba


class BM25Indexer:
    """轻量 BM25 索引，基于 rank_bm25"""

    def __init__(self, index_dir: str = "./data/db/bm25"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []       # chunk_id 列表
        self.doc_texts: List[str] = []      # 原始文本

    def _tokenize(self, text: str) -> List[str]:
        """分词：中文用 jieba，英文保留原样"""
        # jieba 对英文会保留原词，对中文会切词
        tokens = jieba.lcut(text)
        # 去掉空白字符
        return [t.strip() for t in tokens if t.strip()]

    def build_index(self, chunk_ids: List[str], texts: List[str]):
        """从 chunk 列表构建 BM25 索引"""
        self.doc_ids = chunk_ids
        self.doc_texts = texts
        tokenized = [self._tokenize(t) for t in texts]
        self.index = BM25Okapi(tokenized)
        print(f"  [BM25] 索引构建完成: {len(chunk_ids)} 个文档")

    def save(self, name: str = "default"):
        """保存索引到磁盘"""
        path = self.index_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "doc_texts": self.doc_texts,
            }, f)
        print(f"  [BM25] 索引已保存: {path}")

    def load(self, name: str = "default"):
        """从磁盘加载索引"""
        path = self.index_dir / f"{name}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        self.build_index(self.doc_ids, self.doc_texts)
        return True

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """搜索：返回 [(chunk_id, score, text), ...]"""
        if self.index is None:
            return []
        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)
        # 按分数排序
        ranked = sorted(
            [(self.doc_ids[i], scores[i], self.doc_texts[i]) for i in range(len(self.doc_ids))],
            key=lambda x: x[1],
            reverse=True,
        )
        # 过滤掉分数为 0 的
        return [(cid, s, t) for cid, s, t in ranked if s > 0][:top_k]