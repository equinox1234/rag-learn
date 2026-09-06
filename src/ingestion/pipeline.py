"""Pipeline：PDF → Chunk → 向量 → 存储 + BM25 索引 + Trace

现在所有组件通过 Factory 创建，不硬编码。
"""
import time
from pathlib import Path
from typing import Optional
from src.core.types import Document, Chunk
from src.core.settings import Settings, load_settings
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_writer import write_trace
from src.libs.embedding.embedding_factory import create_embedding
from src.libs.vector_store.vector_store_factory import create_vector_store
from src.libs.vector_store.bm25_indexer import BM25Indexer
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.storage.vector_upserter import VectorUpserter


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        ingest_cfg = settings.ingestion or {}

        self.loader = PdfLoader(extract_images=True)
        self.chunker = DocumentChunker(
            chunk_size=ingest_cfg.get("chunk_size", 500),
            chunk_overlap=ingest_cfg.get("chunk_overlap", 50),
        )

        # 通过 Factory 创建 Embedding 和 VectorStore
        emb_cfg = settings.embedding or {}
        embedding = create_embedding(emb_cfg)
        self.dense_encoder = DenseEncoder(embedding, batch_size=ingest_cfg.get("batch_size", 20))

        vs_cfg = settings.vector_store or {}
        collection = vs_cfg.get("collection_name", "default")
        self.vector_store = create_vector_store(vs_cfg)
        self.upserter = VectorUpserter(self.vector_store)

        self.bm25_indexer = BM25Indexer(index_dir="./data/db/bm25")
        self._collection = collection

    def run(self, file_path: str) -> dict:
        trace = TraceContext(trace_type="ingestion")
        path = Path(file_path)

        _t0 = time.monotonic()
        print(f"[Pipeline] 1/5 加载: {path.name}")
        doc = self.loader.load(str(path))
        trace.record_stage("load", {"file": path.name, "chars": len(doc.text)},
                           elapsed_ms=(time.monotonic() - _t0) * 1000)

        _t0 = time.monotonic()
        print(f"[Pipeline] 2/5 分块 ({len(doc.text)} 字符)")
        chunks = self.chunker.split_document(doc)
        print(f"          → {len(chunks)} 个块")
        trace.record_stage("split", {"chunk_count": len(chunks)},
                           elapsed_ms=(time.monotonic() - _t0) * 1000)

        _t0 = time.monotonic()
        print(f"[Pipeline] 3/5 编码向量")
        records = self.dense_encoder.encode(chunks)
        trace.record_stage("embed", {"vector_count": len(records)},
                           elapsed_ms=(time.monotonic() - _t0) * 1000)

        _t0 = time.monotonic()
        print(f"[Pipeline] 4/5 存储到 Chroma")
        ids = self.upserter.upsert(records)
        trace.record_stage("store", {"vector_ids": len(ids)},
                           elapsed_ms=(time.monotonic() - _t0) * 1000)

        _t0 = time.monotonic()
        print(f"[Pipeline] 5/5 构建 BM25 关键词索引")
        self.bm25_indexer.build_index(chunk_ids=[r.id for r in records], texts=[r.text for r in records])
        self.bm25_indexer.save(self._collection)
        trace.record_stage("bm25", {"doc_count": len(records)},
                           elapsed_ms=(time.monotonic() - _t0) * 1000)

        trace.finish()
        write_trace(trace.to_dict())
        print(f"[Pipeline] ✅ 完成! 存储了 {len(ids)} 个向量 + BM25 索引")
        print(f"[Trace]   🆔 {trace.trace_id} | 总耗时 {trace.to_dict()['total_ms']}ms")
        return {"doc_id": doc.id, "chunk_count": len(chunks), "vector_ids": ids}


def run_pipeline(file_path: str, settings_path: Optional[str] = None):
    settings = load_settings(settings_path)
    pipeline = IngestionPipeline(settings)
    return pipeline.run(file_path)