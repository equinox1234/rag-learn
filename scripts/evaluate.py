"""评估脚本：运行测试集，计算 RAG 质量指标

用法：
    python scripts/evaluate.py                          # quick 模式（免费）
    python scripts/evaluate.py --full                   # full 模式（LLM 判分）
    python scripts/evaluate.py --test-set xxx.json      # 自定义测试集
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import load_settings
from src.observability.evaluation.eval_runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="RAG 质量评估")
    parser.add_argument("--test-set", default="tests/fixtures/golden_test_set.json")
    parser.add_argument("--output", default="logs/eval_report.json")
    parser.add_argument("--full", action="store_true",
                        help="完整模式：LLM 判 Hit + Faithfulness + Relevancy（费钱）")
    args = parser.parse_args()

    mode = "full" if args.full else "quick"
    settings = load_settings()

    print(f"📊 RAG 评估开始")
    print(f"   模式: {'full（LLM 判分）' if mode == 'full' else 'quick（关键词匹配）'}")
    print(f"   测试集: {args.test_set}")
    print(f"   Embedding: {settings.embedding.get('provider', '?')}")
    print(f"   LLM: {settings.llm.get('provider', '?')}")
    print(f"   Rerank: {settings.rerank.get('enabled', False)}\n")

    report = run_evaluation(settings, args.test_set, mode=mode)

    print(f"\n{'='*50}")
    print(f"📊 评估报告（{mode} 模式）")
    print(f"{'='*50}")
    print(f"   测试用例:      {report.total}")
    print(f"   总耗时:        {report.elapsed_ms/1000:.1f}s")
    print(f"   Hit Rate:      {report.hit_rate:.1%}")
    if mode == "full":
        print(f"   Faithfulness:  {report.avg_faithfulness:.1%}")
        print(f"   Answer Relevancy: {report.avg_answer_relevancy:.1%}")
    print()

    for r in report.results:
        suffix = ""
        if mode == "full":
            suffix = f" Faith={r.faithfulness:.0%} Rel={r.answer_relevancy:.0%}"
        print(f"  [{'✅' if r.hit else '❌'}] {r.query}{suffix}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump = {
        "mode": report.mode, "total": report.total,
        "hit_rate": report.hit_rate,
        "avg_faithfulness": report.avg_faithfulness,
        "avg_answer_relevancy": report.avg_answer_relevancy,
        "elapsed_ms": report.elapsed_ms,
        "results": [{"query": r.query, "hit": r.hit,
                      "faithfulness": r.faithfulness, "answer_relevancy": r.answer_relevancy,
                      "answer": r.answer[:200]} for r in report.results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()