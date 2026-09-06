"""稠密编码器：把 Chunk 变成向量"""
from typing import List
from src.core.types import Chunk, ChunkRecord


class DenseEncoder:
    def __init__(self, embedding_client, batch_size: int = 20):
        self.embedding_client = embedding_client
        self.batch_size = batch_size

    def encode(self, chunks: List[Chunk]) -> List[ChunkRecord]:
        texts = [c.text for c in chunks]
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            vectors.extend(self.embedding_client.embed(batch))
        return [ChunkRecord.from_chunk(c, v) for c, v in zip(chunks, vectors)]