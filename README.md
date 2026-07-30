# 法律多智能体平台

面向普通用户的法律咨询与行动指引系统。用户可以用自然语言描述纠纷，系统会结合对话上下文识别法律问题、检索法条和类案、按信息增益追问关键事实或证据，并生成可执行的维权方案及参考文书。

当前项目重点不是“让模型一次性回答所有问题”，而是构建一条可审计、可收敛的法律工作流：

- Supervisor 识别咨询意图，将具体纠纷和通用法律问答分流到不同 Worker。
- 九节点 LangGraph 管理案情理解、安全检测、动态追问、检索、结论与状态保存。
- PostgreSQL、Milvus、Neo4j 和 Redis 分别承担结构化数据、向量检索、知识图谱和会话状态。
- 事实、证据和法律结论分层存储，所有进入方案或文书的事实都保留用户原文来源。
- 法律依据只能引用本轮知识库实际召回的条文。
- 用户可随时回复“现在生成方案”停止追问，也可在方案后回复“生成文书”。

完整设计和数据流说明见 [项目说明.md](项目说明.md)。

## 核心能力

### 上下文法律指引

- 承接用户已经说明的主体、行为、金额、时间、地点、损失、诉求和材料。
- 不重复询问已覆盖的信息，不把题库当成固定问卷。
- 每轮最多询问一个会实质影响责任、请求、时效、管辖、程序、证据或安全措施的问题。
- 达到软轮次上限后，只保留高信息增益问题；没有必要继续问时自动生成方案。
- 对人身安全场景维护独立状态：`danger / safe / unknown / not_applicable`。

### 法律知识检索

- 法条：领域检索与全库检索双路召回，融合 PostgreSQL 兜底并统一精排。
- 类案：Milvus 混合检索、PostgreSQL 元数据补全和质量护栏。
- 图谱：Neo4j 提供法律、领域、概念和渠道关系。
- 通用问答：由法律知识 Worker 独立处理，不进入纠纷追问状态机。
- 法律统计：支持 NL2SQL、表格和 Plotly 图表返回。

### 事实、证据与记忆

- `case_facts` 保存稳定语义键、用户原文、轮次、确定性和修订状态。
- 同一事实跨轮复述会合并；更正、否定和冲突不会被静默覆盖。
- `evidence_assessments` 区分存在性、真实性、相关性、可采性和证明边界。
- Redis 保存短期会话和 GuideState；长期记忆由 Supervisor 的向量 Store 管理。
- 历史案情不会默认混入当前案件，只有用户明确回忆历史时才进入推理。

### 方案与文书

- 输出法律依据、维权路径、风险评估和行动清单。
- 结论阶段不再返回批量“待补充信息表单”。信息不足只作为风险说明。
- 生成文书时返回可编辑 DOCX；存在匹配资源时同时返回官方空白模板 PDF。
- 官方模板和系统生成稿在前端明确区分来源、适用阶段与文件类型。

## 九节点工作流

```text
prepare_turn
    -> check_urgency
    -> extract_issues
       -> clarify -> END
       -> assess_retrieve
          -> ask_followup -> END
          -> conclude -> save_record -> END

下一轮存在 pending question 时：
prepare_turn -> check_urgency -> parse_details -> assess_retrieve / extract_issues
```

九个业务节点：

1. `prepare_turn`：恢复上下文、推进轮次、识别收敛指令。
2. `check_urgency`：判断当前危险、时效风险和安全状态。
3. `extract_issues`：提取法律问题与带来源的原子事实。
4. `clarify`：无法识别基本纠纷时进行一次低负担澄清。
5. `assess_retrieve`：评分、法条/类案/图谱检索和下一问题规划。
6. `ask_followup`：展示一个上下文化问题及其权威依据。
7. `parse_details`：解析回答，更新事实、证据、冲突和缺失状态。
8. `conclude`：生成并审校行动方案，只保留真实召回法条。
9. `save_record`：保存咨询结果。

文书生成是 Guide 结束后的独立服务能力，不伪装成第十个状态机节点。

## 技术栈

| 层级 | 主要技术 |
|---|---|
| API / UI | FastAPI、Uvicorn、Gradio、SSE、Plotly |
| Agent | LangChain、LangGraph、Supervisor / Worker |
| 模型 | DashScope 或 DeepSeek 兼容聊天模型，配置以 `.env` 为准 |
| 检索 | PostgreSQL、Milvus Dense + BM25、RRF、Reranker、Neo4j |
| 状态与记忆 | Redis Checkpointer、Milvus Store |
| 文书 | python-docx、官方 PDF 模板注册表 |
| 工程 | Docker Compose、Alembic、pytest |

## 快速启动

### 1. 安装依赖

```powershell
conda create -n legal python=3.13
conda activate legal
pip install -r requirements.txt
```

### 2. 配置环境

```powershell
Copy-Item .env.example .env
```

至少配置聊天模型、Embedding 和数据库连接。不要提交 `.env` 或真实密钥。

本项目 Docker Compose 默认使用以下宿主机端口：

| 服务 | 端口 |
|---|---:|
| PostgreSQL | 5433 |
| Redis | 6380 |
| Milvus | 19531 |
| Neo4j Bolt | 7688 |
| MinIO API | 9010 |

### 3. 启动基础设施

```powershell
docker compose up -d
alembic upgrade head
```

首次初始化法律数据时，根据本地数据情况执行 `scripts/init_legal_postgres.py`、`scripts/init_legal_neo4j.py` 和 `scripts/init_milvus_indexes.py`。

### 4. 启动前后端

```powershell
D:\develop\Miniconda\envs\legal\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8085
$env:LEGAL_AGENT_API_BASE='http://127.0.0.1:8085'
$env:LEGAL_AGENT_GRADIO_PORT='7864'
D:\develop\Miniconda\envs\legal\python.exe scripts/gradio_chat_demo.py
```

也可直接运行 `start_dev.bat`。

- 前端：http://127.0.0.1:7864/
- 后端健康检查：http://127.0.0.1:8085/health
- 依赖健康检查：http://127.0.0.1:8085/health/deps

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/chat` | 非流式对话 |
| POST | `/api/v1/chat/stream` | SSE 对话 |
| GET | `/api/v1/chat/document-templates` | 官方模板列表 |
| GET | `/api/v1/chat/document-templates/{id}/official` | 下载官方模板 PDF |
| GET | `/api/v1/chat/documents/{id}` | 下载智能填写 DOCX |
| POST | `/api/v1/chat/upload-image` | 可选图片分析入口，默认关闭 |
| GET | `/health` | 后端存活检查 |
| GET | `/health/deps` | 外部依赖检查 |

对话请求：

```json
{
  "user_id": "demo-user",
  "session_id": "demo-session",
  "message": "公司三个月没有发工资，我有劳动合同和工资流水"
}
```

## 测试

纯文本主套件：

```powershell
python -m pytest test --ignore=test/test_multimodal_pipeline.py -k "not multimodal" -q
```

最近一次验收结果：

```text
209 passed
5 subtests passed
1 multimodal test deselected
```

九节点数量有独立回归测试；上下文追问、安全门、事实来源、证据边界、记忆、知识库召回、文书下载和 Supervisor 路由均有测试覆盖。

## 当前边界

- 这是法律信息与行动指引工具，不替代律师、公安机关、法院或行政机关的个案判断。
- 开发界面尚未提供生产级鉴权、限流和租户隔离。
- 检索和模型调用较重，真实追问常见延迟约 18-30 秒，完整方案可能接近 48 秒。
- 多模态接口仍为可选实验能力，不属于当前文本主流程验收范围。
- 数据量以实际数据库为准，不在文档中维护容易过期的静态条数。

## 安全提示

- 不要提交 `.env`、API Key、用户证据、上传文件、数据库卷或运行日志。
- 方案中的事实必须能追溯到用户原话；未核验材料不得表述为已经生效的证据。
- 法律名称、条号和原文必须来自本轮召回，不能由模型自行补写。
