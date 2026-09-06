# RAG 项目总结文档

> 从零搭建一个完整的 RAG（检索增强生成）系统，理解每个模块的原理和代码。

---

## 目录

1. [项目全景](#1-项目全景)
2. [数据流纵览](#2-数据流纵览)
3. [模块详解](#3-模块详解)
   - 3.1 数据契约 — `types.py`
   - 3.2 PDF 加载 — `pdf_loader.py`
   - 3.3 文本清洗 — `text_cleaner.py`
   - 3.4 递归切块 — `document_chunker.py`
   - 3.5 向量编码 — `dense_encoder.py`
   - 3.6 向量存储 — `chroma_store.py`
   - 3.7 BM25 索引 — `bm25_indexer.py`
   - 3.8 RRF 融合 — `fusion.py`
   - 3.9 LLM 生成 — `llm_client.py`
   - 3.10 管道编排 — `pipeline.py`
   - 3.11 Trace 追踪 — `trace_context.py`
   - 3.12 MCP Server — `server.py`
4. [面试考点](#4-面试考点)
5. [快速命令](#5-快速命令)

---

## 1. 项目全景

```
┌────────────────────────────────────────────────────────────┐
│                    RAG 系统架构图                            │
│                                                            │
│  摄取管道 (ingest.py)             查询管道 (query.py)        │
│  ┌──────────────────┐            ┌──────────────────┐      │
│  │ PDF → 文本       │            │ 问题 → 向量检索   │      │
│  │ 文本 → 清洗      │            │ 问题 → BM25 检索  │      │
│  │ 清洗 → 切块      │            │ 两路 → RRF 融合   │      │
│  │ 切块 → 向量      │            │ 融合 → LLM 生成   │      │
│  │ 向量 + BM25 → 存  │            │ 全流程 → Trace   │      │
│  └──────┬───────────┘            └──────┬───────────┘      │
│         │                               │                  │
│         ▼                               ▼                  │
│  ┌────────────────────────────────────────────┐            │
│  │          Trace Dashboard (可视化)            │           │
│  └────────────────────────────────────────────┘            │
│         │                                                   │
│         ▼                                                   │
│  ┌────────────────────────────────────────────┐            │
│  │     MCP Server → Claude Desktop 调用       │            │
│  └────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────┘
```

### 核心思想

RAG = Retrieval (检索) + Augmented (增强) + Generation (生成)

不是让 LLM 凭空回答，而是**先从知识库里找到相关材料，再让 LLM 基于材料回答**。

---

## 2. 数据流纵览

### 2.1 数据在流程中的三种形态

| 形态 | 类名 | 产生于 | 说明 |
|------|------|--------|------|
| 原始文档 | `Document` | Loader | 整篇 PDF 的文字内容 |
| 文本块 | `Chunk` | Chunker | 切碎后的小块 |
| 带向量记录 | `ChunkRecord` | Encoder | 小块 + 向量 |

### 2.2 完整流程

```
摄入 (ingest.py):
  PDF 文件 → Loader → Document → 文本清洗 → Chunker → Chunk[]
  → Encoder → ChunkRecord[] → ChromaDB (向量) + BM25 (关键词索引)

查询 (query.py):
  问题 → 向量检索 (ChromaDB) → 结果 A
  问题 → 关键词检索 (BM25) → 结果 B
  结果 A + 结果 B → RRF 融合 → 排序后的结果 → LLM 生成回答
```

---

## 3. 模块详解

### 3.1 数据契约 — `src/core/types.py`

**是什么：** 定义 RAG 流程中所有模块之间传递的数据格式。

**为什么需要：** 没有固定格式，每个模块各写各的，上下游无法衔接。

**三种核心类型：**

```python
@dataclass
class Document:             # 最原始的文档
    id: str                 # 唯一标识
    text: str               # 全文内容
    metadata: dict          # 附加信息（来源路径、文档类型等）

@dataclass
class Chunk:                # 切碎后的文本块
    id: str                 # 块ID
    text: str               # 块内容
    metadata: dict          # 继承自 Document + 块索引
    source_ref: str | None  # 指向原始 Document

@dataclass
class ChunkRecord:          # 带向量的完整记录
    id: str
    text: str
    metadata: dict
    dense_vector: list | None  # 新增：向量

@dataclass
class RetrievalResult:      # 检索结果
    chunk_id: str
    score: float
    text: str
    metadata: dict
```

**设计原则：** 数据契约 = 各模块之间约好的数据格式。像快递包装——不管里面是什么，包装规格统一，运输环节就不用关心内容。

---

### 3.2 PDF 加载 — `src/libs/loader/pdf_loader.py`

**是什么：** 把 PDF 文件解析成纯文本。

**原理：** 用 MarkItDown 库（微软开源的文档转 Markdown 工具）解析 PDF，提取文字内容。

**代码核心：**

```python
class PdfLoader:
    def load(self, file_path: str) -> Document:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(path))
        text = result.text_content or ""
        return Document(id=doc_id, text=text, metadata={...})
```

**注意：** PDF 提取的文字换行是乱的，需要下一步清洗。

---

### 3.3 文本清洗 — `src/libs/loader/text_cleaner.py`

**是什么：** 修复 PDF 提取导致的换行混乱。

**问题：** PDF 的换行 `\n` 分两种：

| 类型 | 含义 | 例子 |
|------|------|------|
| 假换行 | 页面排不下，单纯折行 | `V.\nVillamayor` → 应合并为 `V. Villamayor` |
| 真换行 | 段落结束 | `Spain\nAbstract` → 应保留，Abstract 是标题 |

**解决方案：** 规则判断当前行是否"没说完"。

```python
def _should_merge(self, current_line, next_line):
    # 下一行是标题 → 不合并
    if next_line 是 Abstract/Introduction/Fig. 等:
        return False
    # 当前行以连字符结尾 → 合并（detec-\ntion）
    if current_line 以 - 结尾:
        return True
    # 当前行以缩写点结尾 → 合并（V.\nVillamayor）
    if current_line 匹配 r'[A-Z]\.\s*$':
        return True
    # 当前行以小写字母结尾 → 合并（大概率单词被截断）
    if current_line 匹配 r'[a-z]\s*$':
        return True
    # 当前行以句号结尾 → 不合并
    return False
```

**处理流程：**

```
原始: "V.\nVillamayor"     → "V. Villamayor"  ✅
原始: "detec-\ntion"       → "detection"      ✅（连字符直接拼接）
原始: "80x80\nFPA"         → "80x80 FPA"     ✅
原始: "Spain\nAbstract"    → "Spain\nAbstract" ✅（保留段落）
```

---

### 3.4 递归切块 — `src/ingestion/chunking/document_chunker.py`

**是什么：** 把一篇长文档切成若干小块。

**为什么切块：** LLM 有上下文窗口限制，而且一篇文档中只有部分内容与问题相关。

**用什么算法：** 递归字符切分（RecursiveCharacterTextSplitter）。

**核心逻辑：**

```python
# 按优先级尝试不同的分隔符
separators = ["\n## ", "\n### ", "\n", ". ", " ", ""]
# 先按标题切 → 太大了按段落切 → 还太大按句子切 → 最后按字符切
```

**chunk_size=500 不是"每块 500 字"，而是"最大 500 字，能少则少"。**

```
切 "## Abstract\n300字...\n## Introduction\n800字...\n## Experiment\n400字..."

第1轮：按 ## 标题切 → ["Abstract 300字", "Introduction 800字", "Experiment 400字"]
第2轮：检查大小
  → "Abstract 300字" ✅ 通过
  → "Introduction 800字" ❌ 太大，进入下一轮
  → "Experiment 400字" ✅ 通过
第3轮：按 \n 段落切 "Introduction 800字"
  → ["第一段250字", "第二段300字", "第三段250字"]
  → 全部 ✅ 通过
```

**经验值：**
- 英文论文：500-1000 字符
- 中文文档：500 字
- 代码文档：200-500 字符

---

### 3.5 向量编码 — `src/ingestion/embedding/dense_encoder.py`

**是什么：** 把文本变成一堆数字（向量）。

**为什么：** 计算机不理解"意思"，但能计算数字之间的距离。意思相近的文本，向量距离近。

**原理：**

```python
"红外传感器" → [0.1, 0.5, -0.2, ...]  ← 1536 个浮点数
"热成像仪"   → [0.15, 0.48, -0.18, ...]  ← 离得近（余弦相似度高）
"苹果手机"   → [-0.8, 0.1, 0.9, ...]  ← 离得远
```

**代码核心：**

```python
class DenseEncoder:
    def encode(self, chunks: List[Chunk]) -> List[ChunkRecord]:
        texts = [c.text for c in chunks]
        vectors = self.embedding_client.embed(texts)  # 调 API 转向量
        return [ChunkRecord.from_chunk(c, v) for c, v in zip(chunks, vectors)]
```

---

### 3.6 向量存储 — `src/libs/vector_store/chroma_store.py`

**是什么：** 把向量存到 ChromaDB，查询时按距离搜索。

**为什么用 ChromaDB：** 本地嵌入式数据库，零配置，支持向量索引和相似度搜索。

**两个核心操作：**

```python
# 存
def upsert(self, records: List[ChunkRecord]):
    self.collection.upsert(ids=ids, documents=texts, embeddings=vectors)

# 查（找最接近的向量）
def query(self, query_vector, top_k=10):
    results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
```

---

### 3.7 BM25 索引 — `src/libs/vector_store/bm25_indexer.py`

**是什么：** 建关键词索引，支持精确匹配搜索。

**为什么需要它：** 向量搜索擅长语义匹配，但不擅长精确关键词（如 `Fig. 6`、`200mm`）。BM25 补这个短板。

**原理：** BM25 是一种排名函数，基于词频（TF）和逆文档频率（IDF）计算相关性。

```
搜索 "VPD PbSe"：
  → 包含 "VPD" 和 "PbSe" 的文档得分高
  → 出现次数越多得分越高
  → 但在很多文档中都出现的词（如 "the"）权重会降低
```

**代码核心：**

```python
class BM25Indexer:
    def build_index(self, chunk_ids, texts):
        tokenized = [jieba.lcut(t) for t in texts]  # 分词
        self.index = BM25Okapi(tokenized)            # 建索引

    def search(self, query, top_k=10):
        tokens = jieba.lcut(query)                   # 查询分词
        scores = self.index.get_scores(tokens)        # 算分
        # 按分数排序返回
```

---

### 3.8 RRF 融合 — `src/core/query_engine/fusion.py`

**是什么：** 把向量检索和 BM25 检索的结果合并排序。

**为什么不用加权平均：** 两路检索的分数尺度不同（Chroma 返回距离，BM25 返回词频分），直接加权没有意义。

**RRF 公式：**

```
score(chunk) = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

k 是平滑常数（默认 60），rank 是排名（从 1 开始）。

**例子：** 一个 chunk 在向量里排第 1，BM25 里排第 3：

```
score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
```

**代码核心：**

```python
class RRFFusion:
    def fuse(self, dense_results, sparse_results, top_k=10):
        scores = {}
        # 稠密检索：排名越靠前，分数越高
        for rank, r in enumerate(dense_results):
            scores[r.chunk_id] += 1 / (self.k + rank + 1)
        # 稀疏检索：排名越靠前，分数越高
        for rank, r in enumerate(sparse_results):
            scores[r.chunk_id] += 1 / (self.k + rank + 1)
        # 按 RRF 分数排序，返回 top_k
```

---

### 3.9 LLM 生成 — `src/libs/llm/llm_client.py`

**是什么：** 把检索到的原文片段喂给 LLM，让它生成完整的回答。

**原理：** 把检索结果拼成上下文，构造 prompt 发给 LLM。

```python
prompt = f"""根据以下文档内容回答问题。

文档内容：
{context}

问题：{query}

请用中文回答，并引用文档中的具体内容作为依据。"""
```

**这就是 RAG 的 G（Generation）：** 不是只返回原文，而是让 LLM 理解后重新组织语言。

---

### 3.10 管道编排 — `src/ingestion/pipeline.py`

**是什么：** 把 1~5 步串起来，形成一个完整的"一键摄入"流程。

**为什么叫 Pipeline（管道）：** 像工厂流水线，每个工位做一件事，做完传给下一个。

```python
def run(self, file_path):
    doc = self.loader.load(file_path)        # 工位1：加载
    chunks = self.chunker.split_document(doc) # 工位2：切块
    records = self.dense_encoder.encode(chunks) # 工位3：转向量
    ids = self.upserter.upsert(records)       # 工位4：存向量
    self.bm25_indexer.build_index(...)        # 工位5：建 BM25 索引
    return ids
```

**好处：** 可以单独换掉任何一个工位，其他代码不用动。

---

### 3.11 Trace 追踪 — `src/core/trace/trace_context.py`

**是什么：** 记录每个环节的耗时和数据，让系统从"黑盒"变"白盒"。

**为什么需要：** 没有 Trace，出了问题只能猜。有了 Trace，能精确知道瓶颈在哪。

**用法：**

```python
trace = TraceContext(trace_type="query")

_t0 = time.monotonic()
# 执行某个环节...
trace.record_stage("检索", {"result_count": 5},
                   elapsed_ms=(time.monotonic() - _t0) * 1000)

trace.finish()
write_trace(trace.to_dict())  # 写入 logs/traces.jsonl
```

**输出示例：**

```json
{
  "trace_id": "9fa3f10b",
  "trace_type": "ingestion",
  "total_ms": 5438.0,
  "stages": [
    {"stage": "load", "elapsed_ms": 2703},
    {"stage": "split", "elapsed_ms": 297},
    {"stage": "embed", "elapsed_ms": 1953},
    {"stage": "store", "elapsed_ms": 141},
    {"stage": "bm25", "elapsed_ms": 344}
  ]
}
```

**Dashboard 可视化：** `streamlit run src/observability/dashboard/app.py` 查看。

---

### 3.12 MCP Server — `src/mcp_server/server.py`

**是什么：** 把你的 RAG 检索包装成 MCP 工具，让 Claude Desktop 能直接调用。

**MCP 协议本质：** 客户端和服务器之间传 JSON。

```
客户端 → 服务器: {"method": "tools/list", "params": {}}
服务器 → 客户端: {"result": {"tools": [{"name": "query_knowledge_hub", ...}]}}

客户端 → 服务器: {"method": "tools/call", "params": {"name": "query_knowledge_hub", "arguments": {...}}}
服务器 → 客户端: {"result": {"content": [{"type": "text", "text": "..."}]}}
```

**MCP SDK 帮你做了什么：** 解析 JSON-RPC 消息、路由到正确的函数、包装返回结果。你只需要写：

```python
@server.list_tools()      # 注册工具清单
async def list_tools():
    return [Tool(name="query_knowledge_hub", ...)]

@server.call_tool()        # 处理工具调用
async def call_tool(name, arguments):
    # 根据 name 路由到不同的功能函数
    if name == "query_knowledge_hub":
        return hybrid_search(arguments["query"])
```

**MCP 角色分工：**

| 角色 | 是什么 | 你需不需要写 |
|------|--------|-------------|
| Host | Claude Desktop | 不需要 |
| Client | Claude 内置的 MCP 客户端 | 不需要 |
| Server | 你的 server.py | 需要，按固定格式写 |

---

## 4. 面试考点

### 4.1 基础概念

| 问题 | 答案 |
|------|------|
| 什么是 RAG？ | 检索增强生成 = 先检索相关知识，再让 LLM 基于知识回答 |
| 为什么需要 RAG？ | 解决 LLM 知识过时、幻觉、无法访问私有数据的问题 |
| 什么是 Embedding？ | 把文本变成向量的技术，意思相近的文本向量距离近 |
| 什么是 Chunk？ | 文档切碎后的小块，控制输入给 LLM 的上下文大小 |
| 什么是数据契约？ | 模块之间约定好的数据格式，保证上下游衔接 |

### 4.2 混合检索

| 问题 | 答案 |
|------|------|
| 为什么用混合检索？ | 向量检索擅长语义，BM25 擅长精确匹配，互补 |
| 什么是 RRF？ | 基于排名的融合算法，不依赖原始分数 |
| 为什么 RRF 比加权平均好？ | 两路检索分数尺度不同，无法直接加权 |
| k 值怎么选？ | 默认 60，越大排名差异的影响越小 |

### 4.3 工程实践

| 问题 | 答案 |
|------|------|
| PDF 提取文字有什么坑？ | 假换行（行尾折行）和真换行（段落结束）混在一起 |
| chunk_size 怎么选？ | 500-1000 字符，取决于文档类型和 LLM 上下文窗口 |
| 什么是 Pipeline 模式？ | 把流程拆成独立步骤，每个步骤只做一件事 |
| 什么是可观测性？ | 记录每个环节的耗时和数据，让系统可排查 |
| MCP 是什么？ | 让 AI 客户端（Claude）调用外部工具的标准化协议 |

### 4.4 设计思路

| 问题 | 答案 |
|------|------|
| 为什么用 Chroma 不是 Pinecone？ | 本地开发用 Chroma（零配置、嵌入式），生产用 Pinecone（分布式、托管） |
| BM25 和向量搜索各有什么优缺点？ | BM25 精确但无法处理同义词，向量搜索语义强但可能漏精确匹配 |
| RAG 的瓶颈通常在哪？ | 检索质量（召回率） > LLM 生成质量 |
| 怎么评估 RAG 好不好？ | 用 Ragas 指标：Faithfulness（忠实度）、Answer Relevancy（回答相关度）、Context Precision（上下文精确度） |

---

## 5. 快速命令

```bash
# 摄入文档
python scripts/ingest.py --file "你的文件.pdf"

# 查询（混合检索 + LLM 生成）
python scripts/query.py --query "你的问题"

# 启动 Trace Dashboard
streamlit run src/observability/dashboard/app.py

# 启动 MCP Server（供 Claude Desktop 调用）
python main.py
```

### Claude Desktop 配置

文件位置：`%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-knowledge-hub": {
      "command": "python",
      "args": ["D:\\1AI_study\\AI_program\\RAG_LEARN\\main.py"]
    }
  }
}
```

---

## 项目文件结构

```
RAG_LEARN/
├── main.py                          # MCP Server 入口
├── config/settings.yaml             # 配置文件
├── scripts/
│   ├── ingest.py                    # 摄入文档
│   ├── query.py                     # 查询（混合检索 + LLM 生成）
│   └── start_dashboard.py           # 启动 Dashboard
├── src/
│   ├── core/
│   │   ├── types.py                 # 数据契约
│   │   ├── settings.py              # 配置加载
│   │   ├── query_engine/
│   │   │   └── fusion.py            # RRF 融合
│   │   └── trace/
│   │       ├── trace_context.py     # 追踪上下文
│   │       └── trace_writer.py      # 追踪持久化
│   ├── ingestion/
│   │   ├── pipeline.py              # 管道编排
│   │   ├── chunking/
│   │   │   └── document_chunker.py  # 递归切块
│   │   ├── embedding/
│   │   │   └── dense_encoder.py     # 向量编码
│   │   └── storage/
│   │       └── vector_upserter.py   # 向量存储
│   ├── libs/
│   │   ├── loader/
│   │   │   ├── pdf_loader.py        # PDF 加载
│   │   │   └── text_cleaner.py      # 文本清洗
│   │   ├── embedding/
│   │   │   └── openai_embedding.py   # Embedding API
│   │   ├── llm/
│   │   │   └── llm_client.py        # LLM 客户端
│   │   └── vector_store/
│   │       ├── chroma_store.py      # ChromaDB 封装
│   │       └── bm25_indexer.py      # BM25 索引
│   ├── mcp_server/
│   │   └── server.py                # MCP Server
│   └── observability/
│       └── dashboard/
│           └── app.py               # Trace 可视化
└── docs/
    └── SUMMARY.md                   # 本文件
```