"""测试配置加载"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.settings import Settings, load_settings


class TestSettings:
    def test_load_defaults(self):
        """测试配置加载"""
        s = load_settings()
        assert s.llm.get("provider") is not None

    def test_settings_get(self):
        """测试获取配置"""
        s = load_settings()
        assert s.llm.get("provider") in ("openai", "dashscope", "ollama")
        assert s.embedding.get("provider") is not None
        assert s.vector_store.get("provider") == "chroma"

    def test_ingestion_defaults(self):
        """测试摄入配置默认值"""
        s = load_settings()
        ingest = s.ingestion or {}
        assert ingest.get("chunk_size", 500) > 0
        assert ingest.get("chunk_overlap", 0) >= 0