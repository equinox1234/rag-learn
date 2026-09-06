"""核心数据模型：RAG 全流程的通用契约"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """原始文档（Loader 的输出）"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """文本块（Splitter 的输出）"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    """带向量的完整记录（存储到向量数据库）"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: Optional[List[float]] = None) -> "ChunkRecord":
        return cls(id=chunk.id, text=chunk.text, metadata=chunk.metadata.copy(), dense_vector=vector)


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)