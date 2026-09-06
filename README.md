# RAG Learn — 从零搭建的 RAG 系统

一个模块化的 RAG（检索增强生成）系统，支持混合检索、Rerank 重排、可观测 Trace 和 MCP 协议。

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 配置 API Key
#    编辑 config/settings.yaml，填入你的 API Key
#    支持 OpenAI / DashScope / Ollama

# 3. 摄入文档
python scripts/ingest.py --file "你的文件.pdf"

# 4. 查询
python scripts/query.py --query "你的问题"

# 5. 启动 Dashboard
streamlit run src/observability/dashboard/app.py
```

## 系统架构

```
摄入 (ingest.py)
  PDF → 文本清洗 → 递归切块 → 向量编码 → ChromaDB + BM25 索引 → Trace

查询 (query.py)
  问题 → 向量检索 + BM25 → RRF 融合 → Rerank 精排 → LLM 生成 → Trace

评估 (evaluate.py)
  测试集 → 逐条检索 + 生成 → 计算指标（Hit Rate / Faithfulness / Relevancy）

MCP Server (main.py)
  Claude Desktop 可直接调用知识库 → 配置后即可使用
```

## 功能特性

| 功能 | 说明 |
|------|------|
| **混合检索** | 向量语义搜索 + BM25 关键词搜索，RRF 融合 |
| **Rerank 重排** | 支持 Cross-Encoder / LLM 重排，提高排序质量 |
| **可插拔架构** | Factory 模式，切换 Provider 只需改配置 |
| **Trace 追踪** | 每个环节的耗时和数据都记录，可可视化 |
| **评估体系** | Quick 模式（免费） + Full 模式（LLM 判分） |
| **MCP 协议** | 标准 MCP Server，Claude Desktop 可直接调用 |
| **Dashboard** | Streamlit 可视化面板 |

## 配置切换

编辑 `config/settings.yaml`，一行代码不用改：

```yaml
# 切换 LLM
llm:
  provider: "dashscope"   # 改成 "ollama" 即可切换本地模型

# 切换 Embedding
embedding:
  provider: "dashscope"   # 改成 "ollama" 即可切换

# 开关 Rerank
rerank:
  enabled: true           # 改成 false 关闭
```

## 项目结构

```
RAG_LEARN/
├── main.py                     # MCP Server 入口
├── config/settings.yaml        # 配置
├── scripts/
│   ├── ingest.py               # 摄入文档
│   ├── query.py                # 混合检索 + LLM 生成
│   ├── evaluate.py             # 评估
│   └── start_dashboard.py      # Dashboard
├── src/
│   ├── core/                   # 核心 + Trace
│   ├── ingestion/              # 摄入管道
│   ├── libs/                   # 可插拔组件
│   │   ├── llm/                # LLM (OpenAI / Ollama)
│   │   ├── embedding/          # Embedding (OpenAI / Ollama)
│   │   ├── loader/             # PDF 加载 + 清洗
│   │   ├── vector_store/       # ChromaDB + BM25
│   │   └── reranker/           # Cross-Encoder / LLM Rerank
│   ├── mcp_server/             # MCP 协议
│   └── observability/          # Trace + Dashboard + 评估
├── tests/fixtures/             # 测试集
└── docs/SUMMARY.md             # 学习笔记
```

## MCP 配置（Claude Desktop）

```json
{
  "mcpServers": {
    "rag-knowledge-hub": {
      "command": "python",
      "args": ["D:\\path\\to\\RAG_LEARN\\main.py"]
    }
  }
}
```

## 面试考点

每个模块对应一个面试高频题，详见 `docs/SUMMARY.md`：

- 数据契约 vs 硬编码
- 混合检索：为什么 RRF 比加权平均好？
- Rerank：为什么需要两段式排序？
- Trace：可观测性三 pillar
- MCP：和传统 API 有什么区别？
- Factory 模式：设计模式在 AI 工程中的应用