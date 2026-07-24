# 法律多智能体平台（Legal Multi-Agent）

面向不懂法的普通市民的法律多智能体系统：从「描述遭遇」到「拿到可执行的维权行动方案」。

采用「总助手 / 专项助手」（Supervisor / Worker）架构，Supervisor 识别用户意图并路由到对应专项 Agent。

## 架构概览

| Agent | 面向对象 | 职责 | 状态 |
|---|---|---|---|
| **公民法律指引**（`guide_agent`） | 普通市民 | 描述具体纠纷 → 法律依据 + 证据清单 + 维权路径比较 + 可操作渠道 | 🔄 调试中 |
| **法律知识问答**（`legal_qa_agent`） | 通用 | 法条含义、制度性法律知识问答 | ✅ 已完成 |
| **专业法律助手**（`professional_agent`） | 法律从业者 | 裁决预测、案件分析、文书摘要 | ⏳ 占位 |
| **法考助手**（`exam_agent`） | 法学学生 | 真题练习、知识点讲解 | ⏳ 占位 |

- **维权侧**：`guide_agent` 用 LangGraph 9 节点状态机（`check_urgency` 高危熔断 → 问题标准化 → 追问 → 并行 RAG 检索 → 分级输出）。
- **咨询侧**：`legal_qa_agent` 用 LangChain RAG 工具箱（法条 HyDE 检索 + 类案 + 知识图谱 + NL2SQL）。

### 核心安全 / 质量能力

- **高危多轮熔断**：每轮强制过 `check_urgency`，识别人身安全威胁立即终止流程并推送 110 / 12348 求助渠道。
- **法条幻觉自省**：`statute_rag` 生成回答后用检索到的法条原文校验，无依据的陈述追加免责提示。
- **置信度分级输出**：五维加权打分 → HIGH / MEDIUM / LOW 三档，决定维权方案的确定程度。

## 技术栈

- **Web / API**：FastAPI + Uvicorn（SSE 流式）、Gradio 调试台
- **Agent 框架**：LangChain 1.2、LangGraph 1.1（Redis Checkpointer）、LangChain-DeepSeek
- **大模型**：DeepSeek（`deepseek-chat`）、DashScope（`text-embedding-v3`，1024 维）
- **存储**：PostgreSQL 17 · Neo4j 5.20 · Milvus 2.6 · Redis Stack 7.4 · MinIO
- **迁移 / 评估 / 解析**：Alembic · TruLens · MinerU · python-docx · BeautifulSoup

## 快速开始

```sh
# 1. 环境
conda create -n legal python=3.13 && conda activate legal
pip install -r requirements.txt

# 2. 配置：复制 .env.example 为 .env，填写 DeepSeek/DashScope Key 及各数据库连接
cp .env.example .env

# 3. 启动依赖（Postgres/Neo4j/Milvus/Redis/MinIO）
docker compose up -d
alembic upgrade head

# 4. 数据入库
python scripts/init_legal_postgres.py
python scripts/init_legal_neo4j.py
python scripts/init_milvus_indexes.py

# 5. 启动服务
uvicorn src.main:app --port 8080 --reload

# 6. 对话测试台（另开终端）
python scripts/gradio_chat_demo.py   # http://localhost:7860
```

## 测试

```sh
pytest test/ -v
```

## 更多

完整的架构说明、目录结构、数据现状与开发进度见 [项目说明.md](项目说明.md)。
