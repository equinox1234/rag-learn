"""测试文档切块"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker


class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    def test_split_empty_document(self):
        """测试空文档"""
        doc = Document(id="test", text="", metadata={"source_path": "test.pdf"})
        chunks = self.chunker.split_document(doc)
        assert len(chunks) == 0

    def test_split_short_document(self):
        """测试短文档（小于 chunk_size）"""
        doc = Document(id="test", text="Hello world",
                       metadata={"source_path": "test.pdf"})
        chunks = self.chunker.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_split_long_document(self):
        """测试长文档（大于 chunk_size）"""
        text = "word " * 200
        doc = Document(id="test", text=text, metadata={"source_path": "test.pdf"})
        chunks = self.chunker.split_document(doc)
        assert len(chunks) > 1

    def test_chunk_has_correct_id(self):
        """测试 chunk ID 格式"""
        doc = Document(id="doc123", text="Hello world",
                       metadata={"source_path": "test.pdf"})
        chunks = self.chunker.split_document(doc)
        assert chunks[0].id == "doc123_chunk_0000"

    def test_chunk_has_source_ref(self):
        """测试 chunk 保留文档引用"""
        doc = Document(id="doc123", text="Hello world",
                       metadata={"source_path": "test.pdf"})
        chunks = self.chunker.split_document(doc)
        assert chunks[0].source_ref == "doc123"

    def test_chunk_retains_metadata(self):
        """测试 chunk 继承 document 的 metadata"""
        doc = Document(id="test", text="Hello world",
                       metadata={"source_path": "test.pdf", "doc_type": "pdf"})
        chunks = self.chunker.split_document(doc)
        assert chunks[0].metadata["source_path"] == "test.pdf"
        assert chunks[0].metadata["doc_type"] == "pdf"