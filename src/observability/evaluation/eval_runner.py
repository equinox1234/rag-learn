"""评估流程编排：读取测试集 → 逐条检索 → 生成回答 → 计算指标 → 产出报告

支持两种模式：
  quick（默认）：关键词匹配 Hit Rate，无 LLM 额外开销
  full（--full）：LLM 判 Hit + Faithfulness + Relevancy
"""
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from src.core.settings import load_settings, Settings
from src.core.types import RetrievalResult
from src.libs.embedding.embedding_factory import create_embedding
from src.libs.vector_store.vector_store_factory import create_vector_store
from src.libs.vector_store.bm25_indexer import BM25Indexer
from src.core.query_engine.fusion import RRFFusion
from src.libs.llm.llm_factory import create_llm
from src.libs.reranker.reranker_factory import create_reranker


@dataclass
class EvalResult:
    """一条测试结果"""
    query: str
    retrieved_texts: List[str] = field(default_factory=list)
    answer: str = ""
    hit: bool = False
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0


@dataclass
class EvalReport:
    """评估报告"""
    total: int = 0
    hit_rate: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    results: List[EvalResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    mode: str = "quick"


def hybrid_search(settings, query: str, top_k: int = 5):
    """混合检索"""
    emb_cfg = settings.embedding or {}
    embedding = create_embedding(emb_cfg)
    query_vec = embedding.embed([query])[0]
    vs_cfg = settings.vector_store or {}
    store = create_vector_store(vs_cfg)
    dense_results = store.query(query_vec, top_k=top_k * 2)

    collection = vs_cfg.get("collection_name", "default")
    indexer = BM25Indexer(index_dir="./data/db/bm25")
    sparse_results = []
    if indexer.load(collection):
        raw = indexer.search(query, top_k=top_k * 2)
        sparse_results = [RetrievalResult(chunk_id=cid, score=score, text=text) for cid, score, text in raw]

    fusion = RRFFusion(k=60)
    results = fusion.fuse(dense_results, sparse_results, top_k=top_k)

    rerank_cfg = settings.rerank or {}
    if rerank_cfg.get("enabled"):
        llm = None
        if rerank_cfg.get("provider") == "llm":
            llm = create_llm(settings.llm or {})
        reranker = create_reranker(rerank_cfg, llm=llm)
        if reranker:
            results = reranker.rerank(query, results, top_k=top_k)

    return results


def ask_llm(settings, query: str, context: str) -> str:
    """LLM 生成回答"""
    llm = create_llm(settings.llm or {})
    prompt = f"""你是一个技术文档助手。请根据以下文档内容回答问题。

文档内容：
{context}

问题：{query}

请用中文回答，并引用文档中的具体内容作为依据。如果文档内容不足以回答问题，请如实说明。"""
    return llm.ask(prompt)


# ── Quick 模式（无 LLM 开销） ─────────────────

def compute_hit_quick(answer: str, relevant_text: str) -> bool:
    """关键词匹配 Hit Rate

    提取关键术语（数字、大写专有名词），检查是否出现在回答中。
    速度快，但无法处理否定（"不是80x80"也会匹配）。
    """
    terms = re.findall(r'\d+[\dx]*[a-zA-Z]*', relevant_text)
    terms += re.findall(r'[A-Z]{2,}', relevant_text)
    terms = list(set(terms))
    if not terms:
        return relevant_text.lower().replace(" ", "") in answer.lower().replace(" ", "")
    for t in terms:
        if t.lower() in answer.lower():
            return True
    return False


# ── Full 模式（LLM 判分） ────────────────────

def compute_hit_llm(settings, answer: str, relevant_text: str) -> bool:
    """LLM 判 Hit Rate（理解否定、同义改写、中英文）"""
    llm = create_llm(settings.llm or {})
    prompt = f"""判断以下回答是否包含关键信息。

关键信息：{relevant_text}
回答：{answer}

回答中是否准确包含了关键信息？（注意：如果回答否定了关键信息，回答"不包含"）
只输出"包含"或"不包含"："""
    try:
        result = llm.ask(prompt, temperature=0.0).strip()
        return "包含" in result
    except Exception:
        return False


def judge_faithfulness(settings, query: str, answer: str, context: str) -> float:
    """LLM 判 Faithfulness"""
    llm = create_llm(settings.llm or {})
    prompt = f"""判断以下回答是否严格基于提供的文档内容。

文档内容：{context}
回答：{answer}

只输出一个 0-100 的分数（0=完全脱离文档，100=完全基于文档）："""
    try:
        score_str = llm.ask(prompt, temperature=0.0).strip()
        return float(score_str.split()[0]) / 100.0
    except (ValueError, IndexError):
        return 0.5


def judge_relevancy(settings, query: str, answer: str) -> float:
    """LLM 判 Relevancy"""
    llm = create_llm(settings.llm or {})
    prompt = f"""判断以下回答是否与问题相关。

问题：{query}
回答：{answer}

只输出一个 0-100 的分数（0=完全不相关，100=高度相关）："""
    try:
        score_str = llm.ask(prompt, temperature=0.0).strip()
        return float(score_str.split()[0]) / 100.0
    except (ValueError, IndexError):
        return 0.5


def run_evaluation(settings: Settings,
                   test_set_path: str = "tests/fixtures/golden_test_set.json",
                   mode: str = "quick") -> EvalReport:
    """运行评估

    Args:
        settings: 系统配置
        test_set_path: 测试集路径
        mode: "quick"（关键词匹配，免费） 或 "full"（LLM 判分，费钱）
    """
    _t0 = time.monotonic()

    with open(test_set_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    for case in test_cases:
        query = case["query"]
        relevant_text = case.get("relevant_text", "")
        print(f"  [{len(results)+1}/{len(test_cases)}] {query}")

        retrieved = hybrid_search(settings, query, top_k=5)
        retrieved_texts = [r.text for r in retrieved]

        context = "\n\n".join([f"[{i+1}] {r.text}" for i, r in enumerate(retrieved)])
        answer = ask_llm(settings, query, context)

        if mode == "full":
            hit = compute_hit_llm(settings, answer, relevant_text) if relevant_text else False
            faithfulness = judge_faithfulness(settings, query, answer, context)
            relevancy = judge_relevancy(settings, query, answer)
        else:
            # quick 模式：只算关键词匹配的 Hit Rate，不调 LLM
            hit = compute_hit_quick(answer, relevant_text) if relevant_text else False
            faithfulness = 0.0
            relevancy = 0.0

        results.append(EvalResult(
            query=query, retrieved_texts=retrieved_texts,
            answer=answer, hit=hit,
            faithfulness=faithfulness, answer_relevancy=relevancy,
        ))

    n = len(results)
    return EvalReport(
        total=n, mode=mode,
        hit_rate=sum(1 for r in results if r.hit) / n,
        avg_faithfulness=sum(r.faithfulness for r in results) / n,
        avg_answer_relevancy=sum(r.answer_relevancy for r in results) / n,
        results=results,
        elapsed_ms=(time.monotonic() - _t0) * 1000,
    )