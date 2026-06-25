# Tiny RAG

一个极简的 RAG（检索增强生成）系统，用于学习目的。

## 技术栈

| 组件           | 选型                  | 说明                          |
|----------------|-----------------------|-------------------------------|
| Web 框架       | Flask                 | 轻量、同步、简单              |
| LLM           | GLM 4.7               | 通过 OpenAI 兼容接口调用      |
| Embedding     | Qwen text-embedding-v2 | 通过 OpenAI 兼容接口调用      |
| 向量数据库     | ChromaDB              | 本地轻量，无需额外服务        |
| 检索方式       | 稠密检索（向量相似度） | 后续可扩展混合检索 + Rerank   |
| 配置管理       | pydantic-settings     | 支持 .env 文件                |
| 文本分块       | 固定 token 数 + 重叠   | 支持可扩展的分词策略          |

## 快速开始

### 前置条件

- Python 3.12+
- 智谱 GLM API Key（用于 LLM）
- 阿里云 DashScope API Key（用于 Embedding）

### 安装

```bash
pip install -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件：

```env
LLM_API_KEY=your_zhipu_api_key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4-plus

DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v2
```

### 运行

```bash
python -m src.tiny_rag.app
```

访问 http://localhost:5000

## API 文档

### `POST /upload`
上传文档。

**Request (multipart/form-data)：**
| 参数 | 类型 | 说明 |
|------|------|------|
| file | File | .txt 文件 |

**Response (200)：**
```json
{
  "id": "doc_xxx",
  "filename": "example.txt",
  "chunks": 12
}
```

### `POST /ask`
提问。

**Request (application/json)：**
```json
{
  "question": "文档里提到了什么？"
}
```

**Response (200)：**
```json
{
  "answer": "根据文档...",
  "sources": ["doc_xxx", "doc_yyy"]
}
```

### `GET /documents`
查看已上传文档列表。

**Response (200)：**
```json
{
  "documents": [
    {
      "id": "doc_xxx",
      "filename": "example.txt",
      "chunks": 12,
      "created_at": "2026-05-27T12:00:00"
    }
  ]
}
```

## 项目结构

```
tiny-rag/
├── src/tiny_rag/          # 包根目录
│   ├── __init__.py
│   ├── app.py             # Flask 入口 + 路由
│   ├── config.py          # pydantic-settings 配置
│   ├── ingestion/         # 文档加载、分块
│   │   ├── __init__.py
│   │   ├── loader.py      # 文档加载器
│   │   ├── chunker.py     # 文本分块（可扩展分词策略）
│   │   └── tokenizer.py   # tokenizer 接口与实现
│   ├── embedding/         # 嵌入模型接口
│   │   ├── __init__.py
│   │   └── client.py      # Qwen Embedding 客户端
│   ├── storage/           # 向量存储
│   │   ├── __init__.py
│   │   └── vector_store.py# ChromaDB 封装
│   ├── generation/        # LLM 查询 / 生成
│   │   ├── __init__.py
│   │   └── llm.py         # GLM 客户端
│   └── templates/         # Web 界面模板
│       └── index.html
├── tests/
├── data/                  # 测试文档
├── .env                   # 配置文件（不提交）
├── requirements.txt
└── README.md
```

## 开发

```bash
# 运行测试
pytest

# 带覆盖率
pytest --cov=src.tiny_rag

# 格式化代码
black src/ tests/
```

## 路线图

- [ ] 基础 RAG 流程（上传 → 分块 → 向量化 → 检索 → 生成）
- [ ] 简洁的 Web 界面
- [ ] 扩展文档格式（Markdown, PDF）
- [ ] 扩展分词策略（jieba 等）
- [ ] 混合检索（向量 + BM25）
- [ ] Rerank 重排序
