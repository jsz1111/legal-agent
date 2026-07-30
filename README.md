# 法律多智能体平台

面向普通市民与企业用户的法律咨询、维权行动指引和证据辅助系统。用户可以用自然语言描述纠纷，系统会区分“需要围绕具体事件推进的维权问题”和“需要查询法律知识的问答问题”，分别交由维权 Agent 与法律问答 Agent 处理，并通过检索、结构化状态、文书和多模态能力提供后续支持。

项目的核心目标是将法律信息检索、事实梳理、证据整理、渠道推荐和参考文书生成组织为可复用的工程流程，帮助用户形成清晰的下一步行动。

## 核心能力

| 能力 | 面向场景 | 系统输出 |
| --- | --- | --- |
| 维权 Agent | 欠薪、消费争议、租房、合同、交通与人身损害等具体事件 | 案情梳理、关键追问、法律依据、证据建议、维权渠道、行动清单与参考文书 |
| 法律问答 Agent | 法条含义、制度流程、类案、投诉渠道、专业文档和法律统计 | 基于检索结果的通俗回答、来源化法条/案例信息、渠道信息或统计分析结果 |
| 多模态证据理解 | 合同、订单、聊天记录、支付凭证、通知文书、现场照片等图片 | 可见文字与关键事实提取、可能证明事项、待核验信息、图片指纹 |
| 法律统计分析 | 中国法律年鉴中的数量、趋势、比例、年度对比 | 自然语言统计回答、结构化数据及推荐图表配置 |
| 参考文书 | 投诉、仲裁、起诉等需要形成材料的场景 | 可编辑 DOCX 参考稿、缺失字段占位及关联的官方空白模板 |

## 系统架构

```text
Gradio Web UI / API Client
            |
            v
      FastAPI Chat API
            |
            +-- 已进行中的维权会话 --> GuideGraph（维权 Agent）
            |
            `-- 新会话 --> Supervisor（意图识别与路由）
                              |
                              +-- 具体纠纷 --> 维权 Agent
                              +-- 法律知识 --> 法律问答 Agent
                              `-- 法律统计 --> 法律问答 Agent 的 NL2SQL 工具

检索与存储层：PostgreSQL | Milvus | Neo4j | Redis | MinIO
```

`Supervisor` 是统一入口。它根据用户意图把具体维权事件路由至 `Guide Worker`，把法条、概念、程序、渠道或统计类问题路由至 `Legal QA Worker`。一旦进入维权流程，后续消息会直接回到同一份 `GuideState`，保持事实、证据、追问和方案的连续性。

## 维权 Agent

### 服务内容

维权 Agent 面向“我遇到了什么事、下一步怎么处理”这类具体问题。它支持劳动与社会保障、消费与市场、合同与房产、交通与人身损害等场景，并将用户的口语化叙述转化为可执行的维权方案。

每个会话以 `GuideState` 保存案情状态，包括法律领域、已确认问题、原子事实、证据状况、时间和地域信息、待追问事项、检索结果与最终方案。事实保存用户原话、来源轮次、确定性及修订状态，使同一事实能够在跨轮对话中合并、补充或更正。

### 九节点工作流

维权 Agent 使用 LangGraph 实现固定的九节点业务图：

```text
prepare_turn -> check_urgency -> extract_issues
                                      |        |
                                      |        +-> clarify -> END
                                      v
                              assess_retrieve
                                  |        |
                                  |        +-> ask_followup -> END
                                  v
                              conclude -> save_record -> END

用户回答追问后：prepare_turn -> check_urgency -> parse_details
                                                   |          |
                                                   +----------+-> extract_issues / assess_retrieve
```

| 节点 | 作用 |
| --- | --- |
| `prepare_turn` | 载入上下文和历史记忆，推进轮次，识别用户是否希望直接收束为方案或生成文书。 |
| `check_urgency` | 每轮检查人身安全、紧急风险和时效风险，优先输出安全行动提示。 |
| `extract_issues` | 提取法律问题、归类领域，并将叙述拆分为带来源的原子事实。 |
| `clarify` | 当案情尚无法归类时，以低负担问题完成基础澄清。 |
| `assess_retrieve` | 评估案情完整度，执行法条、类案、图谱和渠道检索，规划下一步。 |
| `ask_followup` | 每轮只提出一个对责任、请求、时效、管辖、程序或证据有实质影响的问题。 |
| `parse_details` | 解析用户补充，更新事实、证据、冲突与缺失状态。 |
| `conclude` | 汇总检索结果，生成维权路径、风险提示、证据建议和行动清单。 |
| `save_record` | 保存本次咨询的结构化结果，便于后续会话衔接。 |

### 动态追问与方案生成

追问并非固定问卷。系统从领域追问目录中选择候选维度，结合已知事实、已确认或缺失的证据、已提问的决策键及本轮检索到的法律依据，选择信息增益更高的问题。用户可随时输入“现在生成方案”收束流程。

在输出方案时，维权 Agent 会提供：

- 经过整理的案件事实和主要争点；
- 已检索到的法律依据与类案参考；
- 证据清单、证明方向及补充建议；
- 可行的投诉、调解、仲裁、诉讼或法律援助路径；
- 按优先级组织的行动清单和材料准备建议。

法律依据采用检索接地机制：方案中使用的法律名称、条号和原文均来自本轮真实召回的法条上下文。证据评估则分开记录证据的持有状态、真实性、关联性、可采性与证明边界，避免把材料描述直接等同于最终认定。

### 参考文书与官方模板

当用户在方案后请求生成文书时，文书服务从已确认事实和证据中生成可编辑 DOCX 参考稿。系统将未提供或需核验的内容保留为明确占位符，并对生成稿执行事实审校和规则式兜底。

项目同时维护官方空白模板目录。对匹配场景，接口会返回系统生成的“智能填写参考稿”以及官方原始空白 PDF 的下载入口，并标明发布机关、文号、适用类型和来源链接，便于用户按对应程序准备材料。

## 法律问答 Agent

法律问答 Agent 处理不需要建立个案状态的知识型问题，例如法律概念、法条规定、程序区别、维权部门、类案情况、专业文档内容及法律统计。它通过 LangChain Agent 调度工具，并要求回答以工具返回结果为基础，采用普通用户可理解的表达。

### 六类检索工具

| 工具 | 数据与实现 | 适用问题 |
| --- | --- | --- |
| `search_statute` | 查询改写 + Milvus 法条语义检索 + PostgreSQL 元数据兜底与重排 | 某项规定、权利义务、法律后果 |
| `search_similar_cases` | Milvus 类案召回与 PostgreSQL 案例元数据补全 | 类似纠纷如何处理、裁判要点参考 |
| `search_legal_graph` | Neo4j 法律、领域、概念和渠道关系查询 | 某领域适用法律、法律关系和主管渠道 |
| `search_channels` | PostgreSQL 中的权威渠道目录和地域筛选 | 投诉电话、网址、法律援助和办理方向 |
| `search_legal_docs` | Milvus 文书知识库检索 | 合同条款、裁判文书或专业材料内容 |
| `search_legal_statistics` | 中国法律年鉴专用 NL2SQL 与 ChatBI 流程 | 案件数量、趋势、比例、年度或指标对比 |

查询进入工具前可进行法律术语改写，以提升口语问题在法律语料中的召回效果。检索调用同时记录用户、查询、工具、索引、耗时和结果摘要，可用于审计和运营分析。

### 法律统计问答

统计问题采用独立于业务库的法律统计数据库。NL2SQL 流程将自然语言映射为受约束的查询，返回回答、结构化统计数据、SQL 元信息和推荐的图表类型。前端使用 Plotly 展示折线图、柱状图或表格。

对于连续追问，系统会保存上一次统计 SQL 的上下文；当用户提出“再加一个指标”“同图对比”等要求时，系统将已有指标和新增指标一并重新查询，以输出完整可比的数据集。

## 多模态证据理解

多模态能力面向维权流程中的图片材料，可通过 `POST /api/v1/chat/upload-image` 上传。功能由 `ENABLE_MULTIMODAL` 和视觉模型密钥控制，使用 DashScope 多模态模型进行图文理解。

处理流程如下：

1. 读取上传内容并校验真实图像格式、文件大小、像素数和完整性；支持 JPG、PNG、WEBP、GIF、BMP。
2. 计算原图 SHA-256，生成会话关联的临时文件标识。
3. 按当前法律领域、已识别争点、已确认或缺失的证据和最近追问构造上下文提示。
4. 以“证据类型、清晰度、可见原文、关键事实、可能证明事项、局限与待核验”为固定结构返回识别结果。
5. 可将结果自动注入当前对话，作为维权 Agent 后续事实与证据整理的输入。

视觉提示明确要求只提取客观可见信息，隐藏身份证号、银行卡号、手机号和住址等敏感信息的非必要部分；图像识别结果用于证据梳理和待核验提示，并保留原图指纹以支持材料对应。

## 技术实现

| 层级 | 技术与职责 |
| --- | --- |
| 服务接口 | FastAPI、Pydantic、Uvicorn；提供非流式对话、SSE 流式对话、文书、模板与图片上传接口。 |
| 对话与编排 | LangChain、LangGraph；Supervisor 负责路由，GuideGraph 负责多轮维权状态机。 |
| 模型 | DeepSeek 兼容聊天模型用于 Agent、检索回答与文书生成；DashScope 向量模型和可选视觉模型用于语义检索与图片分析。 |
| 关系数据 | PostgreSQL + SQLAlchemy + Alembic，保存法律元数据、渠道、咨询记录和统计数据。 |
| 向量检索 | Milvus，承载法条、案例、文书以及长期记忆的向量索引，结合 BM25、RRF 和重排序能力。 |
| 图谱 | Neo4j，维护法律、领域、概念和维权渠道间的关联关系。 |
| 会话与缓存 | Redis Stack + LangGraph Redis Checkpointer，保存活跃 GuideState、短期会话、下载文件和统计上下文。 |
| 对象与文件 | MinIO 存放知识文档等对象；`python-docx` 生成可编辑 DOCX。 |
| 交互与展示 | Gradio 演示界面，展示对话、图片证据、统计图表、检索详情和文书下载入口。 |
| 工程化 | Docker Compose 编排基础设施，pytest 覆盖路由、状态机、追问、检索接地、文书、统计与多模态流程。 |

## 目录说明

```text
src/
  agents/
    supervisor_agent.py          # 意图路由、短期/长期记忆协调
    workers/guide_agent.py       # 维权 Agent Worker
    workers/legal_qa_agent.py    # 法律问答 Agent Worker
    legal_guide/                 # 维权状态、图、追问、证据、方案和文书
    legal_knowledge/             # 法条、类案、图谱、文书和统计检索工具
    tools/multimodal_tools.py    # 图片校验、上下文提示和视觉分析
  api/routers/chat.py            # 对话、SSE、文书、模板与图片接口
  infra/                         # PostgreSQL、Redis、Milvus、Neo4j、MinIO 客户端
scripts/                         # 数据初始化、索引、导入、评估和 Gradio 演示
resources/legal_document_templates/  # 官方模板元数据与空白文件
database/                        # 案例库、法律统计数据库结构
test/                            # 主要业务能力的自动化测试
```

## API 概览

基础前缀为 `/api/v1/chat`。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | 非流式对话；自动路由到维权或法律问答流程。 |
| `POST` | `/api/v1/chat/stream` | SSE 流式对话。 |
| `POST` | `/api/v1/chat/upload-image` | 上传、分析并可选地注入图片证据。 |
| `GET` | `/api/v1/chat/document-templates` | 获取官方空白模板目录。 |
| `GET` | `/api/v1/chat/document-templates/{id}/official` | 下载官方空白模板 PDF。 |
| `GET` | `/api/v1/chat/documents/{id}` | 下载会话内生成的 DOCX 参考稿。 |
| `GET` | `/health` | 服务存活检查。 |
| `GET` | `/health/deps` | PostgreSQL、Redis、MinIO、Milvus、Neo4j 依赖检查。 |

对话请求示例：

```json
{
  "user_id": "demo-user",
  "session_id": "demo-session",
  "message": "公司三个月没有发工资，我有劳动合同、工资流水和考勤记录。"
}
```

## 快速启动

### 1. 安装 Python 依赖

```powershell
conda create -n legal python=3.13
conda activate legal
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中配置聊天模型、嵌入模型及服务连接信息。要启用图片证据分析，设置：

```ini
ENABLE_MULTIMODAL=true
VL_API_KEY=your-key
VL_MODEL=qwen3.7-plus
```

### 2. 启动基础设施与初始化数据

```powershell
docker compose up -d
alembic upgrade head
```

首次部署时，可按数据准备情况运行以下脚本：

```powershell
python scripts/init_legal_postgres.py
python scripts/init_legal_neo4j.py
python scripts/init_milvus_indexes.py
```

`docker-compose.yml` 默认暴露 PostgreSQL `5433`、Redis `6380`、Milvus `19531`、Neo4j Bolt `7688`、MinIO API `9010`。

### 3. 启动服务与演示界面

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8085
$env:LEGAL_AGENT_API_BASE='http://127.0.0.1:8085'
$env:LEGAL_AGENT_GRADIO_PORT='7864'
python scripts/gradio_chat_demo.py
```

也可运行 `start_dev.bat`。启动后访问：

- Gradio 界面：<http://127.0.0.1:7864/>
- OpenAPI 文档：<http://127.0.0.1:8085/docs>
- 服务健康检查：<http://127.0.0.1:8085/health>

## 测试

```powershell
python -m pytest test -q
```

测试覆盖维权状态图、动态追问、上下文记忆、法条接地、类案检索、渠道推荐、文书生成、统计 NL2SQL/ChatBI、Supervisor 路由及多模态图片处理等关键流程。

## 可扩展方向

项目已经按模块化依赖注入和工具注册方式预留扩展空间：

- 增加法律领域、地区化渠道、法条、类案和专业文书数据，只需扩充数据源、索引与领域目录；
- 增加新的专项 Worker，由 Supervisor 的工具注册与路由策略接入统一对话入口；
- 接入新的视觉模型、OCR 服务或文件解析器，复用现有的图片验证、上下文提示和证据注入接口；
- 扩充官方模板目录及文书类型，沿用模板元数据、来源标注和 DOCX 生成链路；
- 扩充法律统计主题与数据集，复用受约束 NL2SQL、统计上下文和 Plotly 展示协议；
- 面向不同客户端接入 Web、移动端或企业内部系统，复用 FastAPI/SSE 和结构化响应协议。

## 项目边界

本项目提供法律信息检索、维权行动辅助与材料整理能力。具体案件的法律判断、代理服务及最终处理结果仍应由有权机关或具备执业资格的专业人士依法作出。
