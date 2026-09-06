"""MCP Server — 把 RAG 检索包装成 MCP 工具，供 Claude Desktop 调用

启动：python src/mcp_server/server.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

from src.core.settings import load_settings
from src.core.types import RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_writer import write_trace
from src.core.query_engine.fusion import RRFFusion
from src.libs.embedding.embedding_factory import create_embedding
from src.libs.vector_store.vector_store_factory import create_vector_store
from src.libs.vector_store.bm25_indexer import BM25Indexer


# ── 工具背后的实际功能 ─────────────────────────────

def hybrid_search(settings, query: str, top_k: int = 5):
    """混合检索：向量 + BM25 → RRF 融合"""
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
        sparse_results = [
            RetrievalResult(chunk_id=cid, score=score, text=text)
            for cid, score, text in raw
        ]

    fusion = RRFFusion(k=60)
    return fusion.fuse(dense_results, sparse_results, top_k=top_k)


def fmt_results(results) -> str:
    if not results:
        return "未找到相关结果。"
    lines = [f"找到 {len(results)} 条相关结果：\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] 相关度: {r.score:.2%}")
        lines.append(f"    {r.text[:200]}...\n")
    return "\n".join(lines)


def list_collections():
    """列出知识库中所有集合"""
    vs_cfg = load_settings().vector_store or {}
    store = ChromaStore(
        persist_directory=vs_cfg.get("persist_directory", "./data/db/chroma"),
        collection_name=vs_cfg.get("collection_name", "default"),
    )
    cols = store.client.list_collections()
    if not cols:
        return "暂无集合。"
    return "\n".join([f"- {c.name}" for c in cols])


# ── MCP 协议层：注册 + 路由 ────────────────────────

server = Server("rag-knowledge-hub")


@server.list_tools()
async def list_tools():
    """【注册】告诉 MCP 客户端：我有以下工具"""
    from mcp.types import Tool
    return [
        Tool(
            name="query_knowledge_hub",
            description="从知识库检索文档，返回最相关的片段。支持语义搜索和关键词搜索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索问题"},
                    "top_k": {"type": "integer", "description": "返回多少结果", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_collections",
            description="列出知识库中所有文档集合。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """【路由】根据工具名，分发到对应的功能函数"""
    settings = load_settings()

    if name == "query_knowledge_hub":
        trace = TraceContext(trace_type="query")

        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        results = hybrid_search(settings, query, top_k)
        text = fmt_results(results)

        trace.finish()
        write_trace(trace.to_dict())

        return [TextContent(type="text", text=text)]

    elif name == "list_collections":
        text = list_collections()
        return [TextContent(type="text", text=text)]

    else:
        raise ValueError(f"未知工具: {name}")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())