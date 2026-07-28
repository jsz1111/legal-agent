# 法律多智能体平台（Legal Multi-Agent Platform）

[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-orange)](https://langchain-ai.github.io/langgraph/)

面向普通市民的法律 AI 智能体系统。用户用大白话描述纠纷，系统自动分析案情、检索法条和类案、评估维权胜算，最终输出可执行的行动方案。

> **核心理念**：从"描述遭遇"到"拿到可执行的维权行动方案"
>
> **目标用户**：不懂法的普通市民（ToC），架构预留 ToB 扩展接口

---

## 一、项目概述

采用「总助手 / 专项助手」（Supervisor / Worker）多智能体架构，Supervisor 识别用户意图并路由到对应的专项 Agent：

| Agent | 面向对象 | 职责 | 状态 |
|-------|----------|------|------|
| **公民法律指引**（`guide_agent`） | 普通市民 | 描述具体纠纷 → 法律依据 + 证据清单 + 维权路径比较 + 可操作渠道 | ✅ 运行中 |
| **法律知识问答**（`legal_qa_agent`） | 通用 | 法律概念、法条含义、制度性知识问答 | ✅ 运行中 |
| **专业法律助手**（`professional_agent`） | 法律从业者 | 裁决预测、案件分析、文书摘要 | 🚧 占位 |
| **法考助手**（`exam_agent`） | 法学学生 | 法考真题练习、知识点讲解 | 🚧 占位 |

### 典型场景

> "公司已经3个月没给我发工资了，我该怎么办？"

输出结构化的 **五段式行动方案**：

1. **理解情况** — 一句共情
2. **法律依据** — 《劳动合同法》第85条等，锚定真实条文
3. **类似案例** — 类案参考 + 裁判要旨
4. **维权路径比较** — 调解（免费）/ 仲裁（免费）/ 起诉（收费）三项对比
5. **行动清单** — 证据清单 + 具体步骤 + 渠道联系方式

### 权威参考文书双交付

维权方案结束后，用户回复“生成文书”，系统会同时提供两份材料：

- **智能填写参考稿 DOCX**：根据本次会话中已确认的事实、证据和检索到的法律依据生成，可继续编辑；缺失信息保留为明确占位符。
- **官方空白模板 PDF**：从发布机关原始 PDF 中按原页拆分，不修改模板内容，并展示发布机关、文号、适用版本、官方发布页和文件哈希。

首批接入 8 类最高人民法院示范文本，覆盖离婚、买卖合同、房屋买卖、民间借贷、房屋租赁、劳动争议、机动车交通事故责任和著作权纠纷。模板依据为最高人民法院、司法部、中华全国律师协会发布的《部分案件起诉状答辩状示范文本》（法〔2025〕82号）。系统生成稿始终标注“非发布机关出具”，避免与官方空白模板混淆。

---

## 二、技术栈

| 类别 | 技术 |
|------|------|
| **API 层** | FastAPI 0.135、Uvicorn 0.42、SSE 流式、Gradio 6.12（调试台） |
| **Agent 框架** | LangChain 1.2、LangGraph 1.1（StateGraph、Redis Checkpointer）、LangChain-DeepSeek |
| **大模型** | DeepSeek（deepseek-v4-flash、deepseek-chat）、DashScope Qwen-VL（可选多模态） |
| **Embedding** | DashScope qwen3.7-text-embedding（1024维），可选：Ollama bge-large / 火山引擎 |
| **精排模型** | DashScope qwen3-rerank |
| **数据库** | PostgreSQL 17 · Milvus 2.6 · Neo4j 5.20 · Redis Stack 7.4 · MinIO |
| **中文 NLP** | jieba（BM25 分词）、pypinyin |
| **文档解析** | MinerU（PDF/图片/DOCX/PPTX/XLSX → Markdown）、python-docx、BeautifulSoup |
| **评估工具** | TruLens、pytest |
| **基础设施** | Docker Compose、Alembic |

---

## 三、系统架构

### 1. 双层路由设计

```
用户输入
    |
    v
[FastAPI /api/v1/chat]
    |
    +-- Redis 有 guide_active? ----> [指引状态机 LangGraph]（续轮绕过 Supervisor）
    |
    +-- 无活跃指引 ----------------> [Supervisor Agent]
                                         |
                           +-------------+-------------+
                           |             |             |
                     [指引 Worker]  [法律问答]   [专业/法考]
```

**第一层 — Supervisor 路由**（`src/agents/supervisor_agent.py`）：
- 基于 DeepSeek 的意图识别与路由
- Worker 以 Agent-as-Tool 方式暴露（`call_guide_agent`、`call_legal_qa_agent` 等）
- 附带记忆工具（`save_memory` / `search_memory`）
- Redis Checkpointer 维持短期会话记忆
- **透传规则**：Worker 的原始回复通过 Redis 键直传路由层，禁止 Supervisor 改写

**第二层 — Worker 执行**：
- guide_agent 激活后，API 绕过 Supervisor 直驱 LangGraph 状态机
- Redis 持久化 GuideState，保证多轮追问连贯
- 指导结束前，所有后续请求直接进入状态机，不经过 Supervisor

### 2. 公民法律指引状态机（13 节点）

核心技术引擎 —— 一个带循环的 LangGraph `StateGraph`：

```
dispatcher → load_context → check_urgency（每轮必过，安全检测）
                |
          extract_issues（三层法律术语标准化）
           ↙        ↘
     clarify       score（置信度打分，零 I/O 零 LLM）
     (最多2轮)       |
                   retrieve（并行检索：法条 + 案例 + 知识图谱）
                      |
                   conclude（五段式行动方案 + 胜算评估）
                      |
            route_after_conclude（按置信度分流）
        ↙              |               ↘
  ask_facts        save_record      ask_evidence
  (LOW 档)         (HIGH 档)        (MEDIUM 档)
      |                                |
  parse_details ←──────────────────────┘
      |
  extract_issues（重新标准化 → 重新打分 → 重新检索）
      |
  ...（循环直到收敛或达到轮次上限）
      |
  save_record → END
      |
  generate_doc → END（用户回复"生成文书"时触发）
```

### 3. 关键设计决策

| 决策 | 理由 |
|------|------|
| **两个独立的 issue 池**（标准术语 vs 口语词） | 防止口语词污染 BM25 精确匹配通道，BM25 对中文口语零召回 |
| **置信度打分放在检索之前** | 零 I/O、不依赖外部服务，确保所有档位都有法律依据可输出 |
| **始终先检索再输出** | LOW 档也给初步法律依据，避免"问了5个问题然后什么都没给"的糟糕体验 |
| **分级追问** | HIGH 直接结束 → MEDIUM 追问证据 → LOW 追问事实，符合法律分析顺序 |
| **双路法条检索（领域过滤 + 全库）** | 领域分类错误不会导致零召回 |
| **HyDE 仅用于 HIGH 档** | 避免低质量查询被放大为幻觉文档 |
| **每轮执行安全检测** | 用户可能在多轮对话中途追加高危信息（家暴、拘押等） |
| **透传规则** | Supervisor 不干预专项助手的回复，通过 Redis 键保证原始输出直达用户 |

### 4. 五数据库架构

| 存储 | 用途 | 端口 | 技术 |
|------|------|------|------|
| **PostgreSQL 17** | 结构化法律数据：75 部法律、7,105 条法条、231 个案例、103 个渠道 | 5433 | Async SQLAlchemy |
| **Milvus 2.6** | 向量嵌入：法条索引、案例索引、法律术语索引（13,268）、长期记忆 | 19531 | IVF_FLAT + BM25 稀疏 |
| **Neo4j 5.20** | 法律知识图谱：法律-领域-渠道-LegalConcept 关系网络 | 7688 | Cypher |
| **Redis Stack 7.4** | 短期记忆、LangGraph Checkpointer、Guide 状态持久化 | 6380 | RedisSaver |
| **MinIO** | 对象存储：知识文档、用户上传图片（可选） | 9010 | S3 兼容 |

### 5. 混合检索 RAG 流水线

```
用户查询（中文口语）
    |
    v
[三层问题标准化]
    L1: LLM 提取法律问题 + 粗标准化
    L2: Neo4j LegalConcept 精确匹配（逐字相等）
    L3: Milvus legal_term_index 语义兜底（阈值 0.75，同领域优先）
    |
    v
[并行检索]
    ├── 法条 RAG：Dense (1024d) + Sparse (jieba BM25) + RRF 融合 + 精排
    │     └─ 双路并行：领域过滤 + 全库检索 + PostgreSQL LIKE 兜底
    ├── 案例 RAG：Dense + BM25 + RRF + 精排
    │     └─ 质量过滤 + 未命中时裁判文书网搜索提示
    └── 图谱 RAG：参数化 Cypher 查询领域/法律/渠道
    |
    v
[自省审查]（仅 HIGH 档 — 法条适用性、时效、管辖权检查）
    |
    v
[答案生成]
    └─ 幻觉校验（法条引用 grounding check）
```

---

## 四、项目结构

```
src/
├── main.py                          # FastAPI 应用工厂 + 健康检查
├── agents/
│   ├── supervisor_agent.py          # Supervisor 意图路由 Agent
│   ├── tools/
│   │   ├── worker_tools.py          # Agent-as-Tool 定义（5 个 Worker）
│   │   ├── store_tools.py           # 长期记忆工具
│   │   └── multimodal_tools.py      # 多模态图片分析（可选）
│   ├── workers/
│   │   ├── guide_agent.py           # 公民法律指引 Worker
│   │   ├── legal_qa_agent.py        # 法律知识问答 Worker
│   │   └── ...（专业 / 法考 / 运营）
│   ├── legal_guide/                 # ★ 核心法律指引状态机
│   │   ├── graph.py                 # 13 节点 LangGraph 定义
│   │   ├── state.py                 # GuideState / GuidePhase 数据结构
│   │   ├── prompts.py               # 所有 LLM 提示词常量
│   │   ├── issue_normalizer.py      # 三层法律术语标准化
│   │   ├── confidence.py            # 三维度置信度评分
│   │   ├── convergence.py           # 收敛判断逻辑
│   │   ├── channel_catalog.py       # 渠道地区标准化、分层排序和全国兜底
│   │   ├── neo4j_queries.py         # 图谱查询函数
│   │   ├── db_queries.py            # PostgreSQL 上下文、渠道 Repository、记录保存
│   │   ├── formatters.py            # 输出格式化
│   │   ├── document_templates.py    # 权威模板注册、选择与来源元数据
│   │   └── doc_generator.py         # 智能填写参考稿与 DOCX 输出
│   └── legal_knowledge/             # RAG 工具箱
│       ├── statute_rag.py           # 法条混合检索
│       ├── case_rag.py              # 类案检索
│       ├── graph_rag.py             # 知识图谱 NL2Cypher
│       ├── nl2sql.py                # 通用结构化数据 NL2SQL（渠道改用确定性 Repository）
│       └── runtime.py               # 共享依赖注入
├── api/routers/
│   ├── chat.py                      # 聊天 API + 双路由 + 图片上传
│   └── legal.py                     # 法律模块健康检查
├── core/                            # 配置、ORM 基类、异常、日志
├── infra/                           # 数据库客户端（5 种存储）
├── rag/                             # 通用 RAG 引擎（分块、生成、评估）
└── modules/                         # SQLAlchemy ORM 模型

scripts/                             # 数据入库、索引、评估
├── init_legal_postgres.py           # 导入法律数据到 PostgreSQL
├── supplement_channels.py           # 建立渠道详情字段并导入全国/北京/上海试点数据
├── init_legal_neo4j.py              # 构建 Neo4j 知识图谱
├── init_milvus_indexes.py           # 构建 Milvus 集合 + 嵌入
├── build_legal_concepts.py          # 从法条构建法律概念图谱
└── gradio_chat_demo.py              # Gradio 调试测试台

resources/legal_document_templates/  # 权威空白 PDF、版本清单与官方源文件
```

---

## 五、数据规模

### PostgreSQL

| 表 | 数量 | 来源 |
|----|------|------|
| `laws`（法律） | 75 部 | 53 部法律 + 14 行政法规 + 5 司法解释 + 3 其他 |
| `articles`（法条） | 7,105 条 | python-docx 解析，最长 968 字 |
| `legal_cases`（案例） | 231 个 | CAIL2019 刑事(200) + 劳动/消费 HTML(13) + 预付卡(12) |
| `channels`（渠道） | 103 个 | 联系方式；其中10条全国/北京/上海试点记录含适用事项、材料、来源和核验日期 |

### Milvus

| 集合 | 数量 | 嵌入策略 |
|------|------|----------|
| `statute_index`（法条索引） | 7,077 | Dense (1024维) + Sparse (jieba BM25) |
| `case_index`（案例索引） | 24 | Dense (1024维) + Sparse (BM25) |
| `legal_term_index`（术语索引） | 13,268 | Dense (1024维) |
| `agent_long_term_memory`（长期记忆） | 9 | Dense (1024维) |

### Neo4j

| 节点 | 数量 | 关系 | 数量 |
|------|------|------|------|
| Law | 75 | HAS_ARTICLE | 7,105 |
| Article | 7,105 | APPLIES_TO | 77 |
| LegalCase | 1,680 | HANDLES | 103 |
| Channel | 103 | BELONGS_TO | 12,706 |
| Domain | 23 | CITES | 0 |
| LegalConcept | 13,268 | 总计 | 19,991 |

覆盖 **13 个法律领域**：劳动、消费、合同、房产、交通、家庭、侵权、刑事、行政、环境、知识产权、公司、税务。

---

## 六、安全与质量特性

- **多轮高危熔断**：`check_urgency` 每轮执行，检测人身安全威胁立即终止并推送 110/12348
- **法条幻觉校验**：statute_rag 输出用检索到的法条原文做 grounding check，无依据陈述追加免责提示
- **置信度分级输出**：HIGH/MEDIUM/LOW 三档控制结论的确定性程度
- **不利事实跟踪**：`adverse_facts` 跨轮积累用户自身不利法律事实，纳入胜算评估
- **NL2SQL 护栏**：只允许 SELECT，拦截 DDL/DML，强制 LIMIT 20，10 秒超时
- **时效提醒**：识别到时间信息时 RAG 检索对应时效法条，输出中加警示
- **NL2Cypher 保护**：仅提示词约束，建议生产环境增加只读语法校验
- **文书来源可追溯**：官方空白模板记录发布机关、文号、原始页面和 SHA256；智能填写稿使用随机下载令牌并随会话 TTL 过期

---

## 七、快速开始

### 前提条件

```bash
# 启动基础设施（PostgreSQL, Redis, Milvus, Neo4j, MinIO）
docker compose up -d
```

### 安装配置

```bash
# 1. 创建环境
conda create -n legal python=3.13
conda activate legal
pip install -r requirements.txt

# 2. 配置
# 复制 .env.example 为 .env，填写 DeepSeek/DashScope API Key
# 注意：DB_HOST 必须设为 127.0.0.1（不能是 localhost，asyncpg 会优先走 IPv6）
cp .env.example .env

# 3. 数据库迁移
alembic upgrade head

# 4. 数据入库
python scripts/init_legal_postgres.py
python scripts/supplement_channels.py
python scripts/init_legal_neo4j.py
python scripts/init_milvus_indexes.py

# 5. 启动 API 服务
uvicorn src.main:app --port 8001 --reload

# 6. 启动 Gradio 调试台（另开终端）
python scripts/gradio_chat_demo.py   # 访问 http://localhost:7862
```

---

## 八、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 非流式对话（双路由：指引/监督） |
| POST | `/api/v1/chat/stream` | SSE 流式对话 |
| POST | `/api/v1/chat/upload-image` | 多模态图片证据分析（可选） |
| GET | `/api/v1/chat/document-templates` | 可用的权威空白模板及来源元数据 |
| GET | `/api/v1/chat/document-templates/{template_id}/official` | 下载官方空白模板 PDF |
| GET | `/api/v1/chat/documents/{document_id}` | 下载短期有效的智能填写参考稿 DOCX |
| GET | `/health` | 应用存活检查 |
| GET | `/health/deps` | 5 种数据库连接状态 |
| GET | `/api/v1/legal/health` | 法律模块状态 |

### 调试信息

法律指引流程额外返回 `debug` 字段：

```json
{
  "reply": "...",
  "session_id": "...",
  "debug": {
    "domain": "劳动",
    "confidence_tier": "HIGH",
    "statute_hits": "《劳动合同法》第85条、第87条...",
    "case_hits": "...",
    "graph_laws": ["劳动合同法", "劳动争议调解仲裁法"],
    "graph_channels": ["劳动监察投诉", "劳动仲裁申请"]
  }
}
```

---

## 九、开发进度

| 阶段 | 状态 |
|------|------|
| 基础设施（FastAPI、数据库、Alembic、健康检查） | ✅ 完成 |
| 数据入库（PostgreSQL / Neo4j / Milvus） | ✅ 完成 |
| RAG 工具箱 + 法律问答 Agent | ✅ 完成 |
| 指引状态机（13 节点 LangGraph） | ✅ 完成，持续迭代 |
| 三层标准化 + BM25 中文分词 | ✅ 完成 |
| 系统集成（双路由、调试 API、胜算评估） | ✅ 完成 |
| 测试与验证 | 🔄 进行中 |
| 多模态图片分析（可选，独立分支） | ✅ 完成 |

### 近期优先项

- [ ] 数据一致性：对齐 PostgreSQL / Milvus / Neo4j 数据
- [ ] 案例库扩充（劳动、消费、民事领域）
- [ ] Legal QA Agent 数据库会话注入修复
- [ ] 接口鉴权、速率限制、文件上传安全
- [ ] CI/CD 流水线 + 自动化测试

---

## 十、核心技术挑战与解决方案

1. **中文口语到法律术语的映射**：三层标准化流水线（LLM → Neo4j 精确匹配 → Milvus 语义兜底），解决"老板炒了我"到"违法解除劳动合同赔偿"的语义鸿沟

2. **中文法律文本混合检索**：Dense 嵌入 + jieba 分词 BM25 稀疏 + RRF 融合 + 精排，双路并行（领域过滤 + 全库检索）防止领域分类错误导致零召回

3. **置信度驱动的对话深度**：检索前的零 I/O 打分决定追问策略（LOW 问事实 / MEDIUM 问证据 / HIGH 直接输出），所有档位都保证输出法律依据

4. **多轮状态持久化**：Redis 存储 LangGraph 状态 + `guide_active` 标志实现 Supervisor 旁路，保证多轮法律访谈的连贯性

5. **法律 AI 安全机制**：每轮安全检测、法条幻觉校验、不利事实跟踪、NL2SQL 只读防护、NL2Cypher 约束

---

## 十一、测试

```bash
# 单元测试
pytest test/ -v

# 场景测试
python _test_state_trace.py        # 状态追踪验证
python _test_comprehensive.py       # 综合场景测试
python _test_e2e_flow.py            # 端到端流程测试

# 回归测试
python _test_regression_p0.py       # P0 回归套件
```
