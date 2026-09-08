"""测试核心数据类型"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.types import Document, Chunk, ChunkRecord, RetrievalResult


class TestDocument:
    def test_create_document(self):
        """测试创建 Document"""
        doc = Document(id="test", text="hello world",
                       metadata={"source_path": "test.pdf"})
        assert doc.id == "test"
        assert doc.text == "hello world"
        assert doc.metadata["source_path"] == "test.pdf"

    def test_document_to_dict(self):
        """测试 Document 转字典"""
        doc = Document(id="test", text="hello",
                       metadata={"source_path": "test.pdf"})
        d = doc.to_dict()
        assert d["id"] == "test"
        assert d["text"] == "hello"


class TestChunk:
    def test_create_chunk(self):
        """测试创建 Chunk"""
        chunk = Chunk(id="chunk_1", text="some text",
                      metadata={"source_path": "test.pdf", "chunk_index": 0},
                      source_ref="doc_1")
        assert chunk.id == "chunk_1"
        assert chunk.source_ref == "doc_1"

    def test_chunk_to_dict(self):
        """测试 Chunk 转字典"""
        chunk = Chunk(id="chunk_1", text="text",
                      metadata={"source_path": "test.pdf"})
        d = chunk.to_dict()
        assert d["id"] == "chunk_1"


class TestChunkRecord:
    def test_from_chunk(self):
        """测试从 Chunk 创建 ChunkRecord"""
        chunk = Chunk(id="chunk_1", text="text",
                      metadata={"source_path": "test.pdf"})
        record = ChunkRecord.from_chunk(chunk, vector=[0.1, 0.2, 0.3])
        assert record.id == "chunk_1"
        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.metadata["source_path"] == "test.pdf"

    def test_from_chunk_no_vector(self):
        """测试不传向量"""
        chunk = Chunk(id="chunk_1", text="text",
                      metadata={"source_path": "test.pdf"})
        record = ChunkRecord.from_chunk(chunk)
        assert record.dense_vector is None


class TestRetrievalResult:
    def test_create_result(self):
        """测试创建检索结果"""
        r = RetrievalResult(chunk_id="c1", score=0.95, text="content",
                            metadata={"source": "test.pdf"})
        assert r.chunk_id == "c1"
        assert r.score == 0.95
        assert r.text == "content"