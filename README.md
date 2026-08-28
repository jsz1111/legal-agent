# 法护通 AI：智能法律咨询与维权系统

> 开源的 ToC 法律咨询与维权辅助项目，由本人独立设计和开发。当前聚焦两个核心模块：**智能法律维权多智能体系统**与 **RAG 多维法律知识库系统**。

## 项目介绍

法护通 AI 面向普通用户的法律咨询场景：当用户只知道“公司拖欠工资”“商家不退款”或“合同出了问题”时，往往难以自行判断争点、准备材料或选择维权渠道。项目的目标不是把大模型包装成“在线律师”，而是将案情梳理、法律信息检索、证据准备与下一步行动组织为一套可追踪的工程流程。

系统首先区分“需要围绕具体事件推进的维权问题”和“需要查询法律知识的问答问题”。前者进入带状态的案件引导工作流，系统通过多轮追问补全事实和证据，并以明确的收敛条件生成行动建议；后者通过法规、案例、图谱、渠道和统计数据等多通道检索，返回附带来源和适用边界的通俗回答。

当前版本的核心价值在于：用确定性路由和 LangGraph 状态机控制复杂多轮任务，用记忆、法律术语标准化和检索接地减少上下文断裂与无依据生成，再以收敛和复核机制约束最终输出。

**使用边界：** 项目提供法律信息检索、维权行动辅助与材料整理能力，不替代律师意见、法律代理或司法裁判。

## 核心设计

| 核心模块 | 解决的问题 | 代表性实现 |
| --- | --- | --- |
| 智能法律维权多智能体系统 | 让复杂法律咨询在多轮对话中保持连续、可控并能收束为行动方案 | Supervisor Tool Calling、FastAPI + Redis 确定性分流、LangGraph 九节点状态机、Redis + Milvus 双层记忆、决策充分度收敛、Critique-Revise 复核 |
| RAG 多维法律知识库系统 | 为上层 Agent 提供可组合、可追溯的法律知识检索与结构化查询能力 | Doc RAG / Graph RAG / SQL RAG、法律术语三层标准化、HyDE / Reranker / RRF、NL2Cypher 重试、受约束 NL2SQL、TruLens 评估 |

## 当前能力边界

| 已接入能力 | 用途 |
| --- | --- |
| 维权 Agent 与法律问答 Agent | 分别处理具体案件引导与法律知识型问答 |
| 多模态证据理解 | 对图片材料提取可见文字和待核验事实，辅助证据整理 |
| 法律统计与参考文书 | 提供统计查询、结构化数据展示及 DOCX 参考稿生成能力 |

项目的核心目标是将法律信息检索、事实梳理、证据整理、渠道推荐和参考文书生成组织为可复用的工程流程；多模态、统计和文书能力作为这两个核心模块的支撑能力接入。

## 系统架构

```text
Gradio Web UI / API Client
            |
            v
      FastAPI Chat API
            |
            +-- mode=case / 已进行案件 --> GuideGraph（维权 Agent）
            |
            +-- mode=qa / 已进行问答 ----> 法律问答 Agent
            |
            `-- mode=auto 的未分类首轮 --> Supervisor（意图识别与路由）
                              |
                              +-- 具体纠纷 --> 维权 Agent
                              +-- 法律知识 --> 法律问答 Agent
                              `-- 法律统计 --> 法律问答 Agent 的 NL2SQL 工具

检索与存储层：PostgreSQL | Milvus | Neo4j | Redis | MinIO
```

前端首屏会同时介绍“法律问答”和“案件维权”，并提供示例。用户明确选择后，请求携带 `mode=qa|case` 并直接进入对应 Worker；用户直接描述而未选择时，才由 `Supervisor` 在首轮识别。类型确定后写入服务端会话模式，同一 `session_id` 不允许在问答历史与案件事实库之间静默切换。一旦进入维权流程，后续消息会直接回到同一份 `GuideState`，保持事实、证据、追问和方案的连续性。

网页端支持跨会话并发：一个案件或问答在后台处理时，可以切换或新建其他会话继续发送。进度和结果按 `session_id` 写回原记录；同一会话仍串行执行，避免结构化案件状态发生覆盖。

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
| `ask_followup` | 每轮生成 2–5 个高价值动态问题的混合表单；确实只剩一个关键缺口时允许只问一个。 |
| `parse_details` | 解析用户补充，更新事实、证据、冲突与缺失状态。 |
| `conclude` | 汇总检索结果，生成维权路径、风险提示、证据建议和行动清单。 |
| `save_record` | 保存本次咨询的结构化结果，便于后续会话衔接。 |

### 动态追问与方案生成

追问并非固定问卷。首轮及后续回答先归入原子事实细节库，系统再针对未解决的责任、请求、时效、管辖和程序问题进行轻量法条/知识图谱检索（此阶段不检索类案），实时生成 2–5 个高信息增益问题。问题可混合短文本、长文本、单选和多选；用户也可直接在聊天框回答。已确认、已否认、明确不知道或已经回答的决策点不会重复追问，预设目录只作为模型或检索失败时的安全兜底。

证据需求随事实变化增量维护，使用稳定 ID 和版本号记录证明目标、触发事实、检索依据、替代材料和当前覆盖状态。事实没有新的高价值缺口时，系统集中展示证据准备清单，用户可统一上传多份 PDF、DOCX、TXT 或图片，也可把文件直接绑定到某个清单项。上传后完成单份材料与整体证明目标覆盖评估；此后才执行包含类案在内的完整检索并生成方案。用户可在任何阶段输入“现在生成方案”，按当前事实与证据状态生成条件式建议。

在输出方案时，维权 Agent 会提供：

- 经过整理的案件事实和主要争点；
- 已检索到的法律依据与类案参考；
- 证据清单、证明方向及补充建议；
- 可行的投诉、调解、仲裁、诉讼或法律援助路径；
- 按优先级组织的行动清单和材料准备建议。

法律依据采用检索接地机制：方案中使用的法律名称、条号和原文均来自本轮真实召回的法条上下文。证据评估分开记录材料的持有/上传状态、可能用途、完整性、主体与时间可见性、冲突和证明边界，不直接认定真实性、合法性、可采性或最终证明力；评估结果会进入最终方案的“证据作用与缺口”部分。

网页端通过 SSE 展示可公开核验的处理进度（风险检查、事实整理、法条检索、证据评估和方案生成），不暴露模型内部思维链。追问与证据面板支持收起、放大、缩小和拖动；提交后旧面板自动隐藏。用户可在生成方案后继续补交或重新提交证据，系统按证据评估版本重新评估并更新方案。

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
| 模型 | 通过 OpenAI 兼容接口接入聊天模型；DashScope 向量模型和可选视觉模型用于语义检索与图片分析。 |
| 关系数据 | PostgreSQL + SQLAlchemy + Alembic，保存法律元数据、渠道、咨询记录和统计数据。 |
| 向量检索 | Milvus，承载法条、案例、文书以及长期记忆的向量索引，结合 BM25、RRF 和重排序能力。 |
| 图谱 | Neo4j，维护法律、领域、概念和维权渠道间的关联关系。 |
| 会话与缓存 | Redis Stack + LangGraph Redis Checkpointer，保存活跃 GuideState、短期会话、下载文件和统计上下文。 |
| 对象与文件 | MinIO 存放知识文档等对象；`python-docx` 生成可编辑 DOCX。 |
| 交互与展示 | Gradio 演示界面，展示对话、图片证据、统计图表、检索详情和文书下载入口。 |
| 工程化 | Docker Compose 编排基础设施；仓库包含覆盖路由、状态机、追问、检索接地、文书、统计与多模态流程的自动化测试用例。 |

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
data/sources/                    # 法律、案例、办事指南和统计原始来源的统一目录
test/                            # 全部自动化测试、验收执行器和 materials 测试物料
docs/                            # 保留的历史优化记录
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

## 自动化测试

```powershell
python -m pytest test -q
```

仓库包含维权状态图、动态追问、上下文记忆、法条接地、类案检索、渠道推荐、文书生成、统计 NL2SQL/ChatBI、Supervisor 路由及多模态图片处理等关键流程的测试用例。

## 可扩展方向

项目已经按模块化依赖注入和工具注册方式预留扩展空间：

- 增加法律领域、地区化渠道、法条、类案和专业文书数据，只需扩充数据源、索引与领域目录；
- 增加新的专项 Worker，由 Supervisor 的工具注册与路由策略接入统一对话入口；
- 接入新的视觉模型、OCR 服务或文件解析器，复用现有的图片验证、上下文提示和证据注入接口；
- 扩充官方模板目录及文书类型，沿用模板元数据、来源标注和 DOCX 生成链路；
- 扩充法律统计主题与数据集，复用受约束 NL2SQL、统计上下文和 Plotly 展示协议；
- 面向不同客户端接入 Web、移动端或企业内部系统，复用 FastAPI/SSE 和结构化响应协议。
