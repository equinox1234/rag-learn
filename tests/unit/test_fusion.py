"""测试 RRF 融合算法"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.types import RetrievalResult
from src.core.query_engine.fusion import RRFFusion


def make_result(chunk_id: str, score: float, text: str = "") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=score, text=text)


class TestRRFFusion:
    def setup_method(self):
        self.fusion = RRFFusion(k=60)

    def test_fusion_returns_correct_count(self):
        """测试融合后返回 top_k 条结果"""
        dense = [make_result(f"d_{i}", 0.9 - i * 0.1) for i in range(5)]
        sparse = [make_result(f"s_{i}", 0.8 - i * 0.1) for i in range(5)]
        result = self.fusion.fuse(dense, sparse, top_k=3)
        assert len(result) == 3

    def test_fusion_dedup(self):
        """测试同一 chunk 在两路中都出现时，只出现一次"""
        dense = [make_result("a", 0.9), make_result("b", 0.8)]
        sparse = [make_result("a", 0.7), make_result("c", 0.6)]
        result = self.fusion.fuse(dense, sparse, top_k=3)
        ids = [r.chunk_id for r in result]
        assert ids.count("a") == 1

    def test_fusion_ordering(self):
        """测试两路都出现的结果排在最前"""
        dense = [make_result("a", 0.9), make_result("b", 0.8)]
        sparse = [make_result("c", 0.7), make_result("a", 0.6)]
        result = self.fusion.fuse(dense, sparse, top_k=3)
        # "a" 在两路中都出现，RRF 分数最高，应该排第一
        assert result[0].chunk_id == "a"

    def test_fusion_empty(self):
        """测试空结果"""
        assert self.fusion.fuse([], [], top_k=5) == []

    def test_fusion_one_side_empty(self):
        """测试只有一路有结果"""
        dense = [make_result("a", 0.9), make_result("b", 0.8)]
        result = self.fusion.fuse(dense, [], top_k=5)
        assert len(result) == 2

    def test_rrf_score_range(self):
        """测试 RRF 分数在合理范围内"""
        dense = [make_result(f"d_{i}", 1.0) for i in range(10)]
        sparse = [make_result(f"s_{i}", 1.0) for i in range(10)]
        result = self.fusion.fuse(dense, sparse, top_k=10)
        for r in result:
            assert 0 < r.score < 1, f"RRF 分数 {r.score} 不在 (0,1) 范围内"