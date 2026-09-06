"""配置加载"""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    def __init__(self, data: Dict[str, Any]):
        self.llm = _get(data, "llm", {})
        self.embedding = _get(data, "embedding", {})
        self.vector_store = _get(data, "vector_store", {})
        self.retrieval = _get(data, "retrieval", {})
        self.rerank = _get(data, "rerank", {})
        self.ingestion = _get(data, "ingestion", {})


def _get(d, key, default):
    return d.get(key, default) if d else default


def load_settings(path: Optional[str] = None) -> Settings:
    if path is None:
        path = REPO_ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(data)