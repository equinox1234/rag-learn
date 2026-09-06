"""文档分块器"""
from typing import List
from src.core.types import Document, Chunk


class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> List[Chunk]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n## ", "\n### ", "\n", ". ", " ", ""],
            )
            texts = splitter.split_text(document.text)
        except ImportError:
            # fallback: 按字符数切
            texts = [document.text[i:i+self.chunk_size]
                     for i in range(0, len(document.text), self.chunk_size - self.chunk_overlap)]

        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=f"{document.id}_chunk_{i:04d}",
                text=text,
                metadata={**document.metadata, "chunk_index": i},
                source_ref=document.id,
            )
            chunks.append(chunk)
        return chunks