"""查询脚本：混合检索（向量 + BM25）→ RRF 融合 → LLM 生成 + Trace

所有组件通过 Factory 创建，修改 settings.yaml 即可切换后端。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import load_settings
from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_writer import write_trace
from src.core.query_engine.fusion import RRFFusion
from src.libs.embedding.embedding_factory import create_embedding
from src.libs.vector_store.vector_store_factory import create_vector_store
from src.libs.vector_store.bm25_indexer import BM25Indexer
from src.libs.llm.llm_factory import create_llm
from src.libs.reranker.reranker_factory import create_reranker


def dense_search(settings, query: str, top_k: int):
    """向量检索：用 Factory 创建 Embedding 和 VectorStore"""
    emb_cfg = settings.embedding or {}
    embedding = create_embedding(emb_cfg)
    query_vec = embedding.embed([query])[0]

    vs_cfg = settings.vector_store or {}
    store = create_vector_store(vs_cfg)
    return store.query(query_vec, top_k=top_k)


def sparse_search(settings, query: str, top_k: int):
    """BM25 关键词检索"""
    vs_cfg = settings.vector_store or {}
    collection = vs_cfg.get("collection_name", "default")
    indexer = BM25Indexer(index_dir="./data/db/bm25")
    if not indexer.load(collection):
        return []
    results = indexer.search(query, top_k=top_k)
    return [RetrievalResult(chunk_id=cid, score=score, text=text) for cid, score, text in results]


def build_context(results) -> str:
    sections = []
    for i, r in enumerate(results, 1):
        sections.append(f"[片段 {i}]\n{r.text}")
    return "\n\n".join(sections)


def ask_llm(settings, query: str, context: str) -> str:
    """用 Factory 创建 LLM"""
    llm_cfg = settings.llm or {}
    llm = create_llm(llm_cfg)
    prompt = f"""你是一个技术文档助手。请根据以下文档内容回答问题。

文档内容：
{context}

问题：{query}

请用中文回答，并引用文档中的具体内容作为依据。如果文档内容不足以回答问题，请如实说明。"""
    return llm.ask(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="混合检索 + Trace")
    parser.add_argument("--query", required=True, help="你的问题")
    parser.add_argument("--top-k", type=int, default=5, help="最终返回多少结果")
    args = parser.parse_args()

    trace = TraceContext(trace_type="query")
    settings = load_settings()

    print(f"🔍 混合检索: {args.query}\n")
    provider = settings.embedding.get("provider", "?")
    print(f"   Embedding: {provider}")
    provider = settings.llm.get("provider", "?")
    print(f"   LLM: {provider}\n")

    _t0 = time.monotonic()
    print("❶ 向量检索中...")
    dense_results = dense_search(settings, args.query, args.top_k * 2)
    trace.record_stage("dense_search", {"result_count": len(dense_results)},
                       elapsed_ms=(time.monotonic() - _t0) * 1000)
    print(f"   找到 {len(dense_results)} 个结果")

    _t0 = time.monotonic()
    print("❷ 关键词检索中...")
    sparse_results = sparse_search(settings, args.query, args.top_k * 2)
    trace.record_stage("sparse_search", {"result_count": len(sparse_results)},
                       elapsed_ms=(time.monotonic() - _t0) * 1000)
    print(f"   找到 {len(sparse_results)} 个结果")

    _t0 = time.monotonic()
    print("❸ RRF 融合排序...")
    fusion = RRFFusion(k=60)
    final_results = fusion.fuse(dense_results, sparse_results, top_k=args.top_k)
    trace.record_stage("fusion", {"final_count": len(final_results)},
                       elapsed_ms=(time.monotonic() - _t0) * 1000)
    print(f"   最终 {len(final_results)} 个结果\n")

    for i, r in enumerate(final_results, 1):
        print(f"  [{i}] [得分:{r.score:.4f}] {r.text[:80]}...")

    # 3.5 Rerank 精排
    rerank_cfg = settings.rerank or {}
    llm = None
    if rerank_cfg.get("enabled") and rerank_cfg.get("provider") == "llm":
        llm = create_llm(settings.llm or {})
    reranker = create_reranker(rerank_cfg, llm=llm)

    if reranker:
        _t0 = time.monotonic()
        print(f"\n❸½ Rerank 精排中...")
        final_results = reranker.rerank(args.query, final_results, top_k=args.top_k)
        trace.record_stage("rerank", {"final_count": len(final_results)},
                           elapsed_ms=(time.monotonic() - _t0) * 1000)
        print(f"   重排后 {len(final_results)} 个结果\n")
        for i, r in enumerate(final_results, 1):
            print(f"  [{i}] [得分:{r.score:.4f}] {r.text[:80]}...")

    _t0 = time.monotonic()
    print(f"\n❹ LLM 生成回答...")
    context = build_context(final_results)
    answer = ask_llm(settings, args.query, context)
    trace.record_stage("llm_generate", {"context_chars": len(context), "answer_chars": len(answer)},
                       elapsed_ms=(time.monotonic() - _t0) * 1000)

    trace.finish()
    write_trace(trace.to_dict())

    print(f"\n{'='*50}")
    print(f"回答:\n{answer}")
    print(f"\n[Trace] 🆔 {trace.trace_id} | 总耗时 {trace.to_dict()['total_ms']}ms")
    print(f"{'='*50}")