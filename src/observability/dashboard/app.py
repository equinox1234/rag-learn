"""Trace Dashboard — 查看索引、查询和评估的追踪记录

启动：streamlit run src/observability/dashboard/app.py
"""
import json
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.core.trace.trace_writer import read_traces

TYPE_LABELS = {"ingestion": "索引", "query": "查询"}

st.set_page_config(page_title="RAG Dashboard", layout="wide")
st.title("RAG Dashboard")

tab1, tab2, tab3 = st.tabs(["Trace 记录", "评估报告", "系统配置"])

# ── Tab 1: Trace ──
with tab1:
    traces = read_traces()
    if not traces:
        st.info("还没有 Trace 记录，先运行 scripts/ingest.py 或 scripts/query.py")
    else:
        col1, _ = st.columns([1, 3])
        with col1:
            filter_type = st.selectbox("类型", ["全部", "ingestion", "query"],
                                       format_func=lambda x: {"全部": "全部", "ingestion": "索引", "query": "查询"}.get(x, x))
        filtered = [t for t in traces if filter_type == "全部" or t["trace_type"] == filter_type]
        st.write(f"共 {len(filtered)} 条记录")

        for t in reversed(filtered):
            type_label = TYPE_LABELS.get(t["trace_type"], t["trace_type"])
            with st.expander(f"[{type_label}] {t['trace_id']} — 总耗时 {t.get('total_ms', 0):.0f}ms"):
                stages = t.get("stages", [])
                if stages:
                    chart_data = {"阶段": [s["stage"] for s in stages],
                                  "耗时(ms)": [s.get("elapsed_ms", 0) for s in stages]}
                    st.bar_chart(chart_data, x="阶段", y="耗时(ms)")
                    rows = []
                    for s in stages:
                        data = s.get("data", {})
                        detail = " | ".join(f"{k}={v}" for k, v in data.items()) if data else "-"
                        rows.append({"阶段": s["stage"], "耗时(ms)": f"{s.get('elapsed_ms', 0):.0f}", "详情": detail})
                    st.table(rows)

# ── Tab 2: 评估报告 ──
with tab2:
    report_path = Path(__file__).resolve().parents[3] / "logs" / "eval_report.json"
    if not report_path.exists():
        st.info("还没有评估报告，先运行 scripts/evaluate.py")
    else:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("测试用例", report["total"])
        col2.metric("Hit Rate", f"{report['hit_rate']:.0%}")
        col3.metric("Faithfulness", f"{report['avg_faithfulness']:.0%}")
        col4.metric("Answer Relevancy", f"{report['avg_answer_relevancy']:.0%}")

        for r in report["results"]:
            with st.expander(f"[{'✅' if r['hit'] else '❌'}] {r['query']}"):
                st.write(f"**Faithfulness:** {r['faithfulness']:.0%} | **Relevancy:** {r['answer_relevancy']:.0%}")
                st.write(f"**回答:** {r['answer'][:300]}...")

# ── Tab 3: 系统配置 ──
with tab3:
    from src.core.settings import load_settings
    s = load_settings()
    st.write("### LLM")
    st.json({"provider": s.llm.get("provider"), "model": s.llm.get("model")})
    st.write("### Embedding")
    st.json({"provider": s.embedding.get("provider"), "model": s.embedding.get("model")})
    st.write("### Rerank")
    st.json({"enabled": s.rerank.get("enabled"), "provider": s.rerank.get("provider")})
    st.write("### 摄入")
    st.json({"chunk_size": s.ingestion.get("chunk_size"), "chunk_overlap": s.ingestion.get("chunk_overlap")})