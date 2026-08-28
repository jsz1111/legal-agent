# 法条检索 RAG 量化测评说明书（TruLens）

> 版本：v1.0（2026-08-02）  
> 被测项目：`D:\learn\legal-agent`（只读引用，不修改任何文件）  
> 测评工程：`D:\learn\legal-tools\trulens-statute-eval\`（完全隔离）  
> 测评对象：已入库法条检索链路（`search_statutes_raw` + Milvus `statute_index`）

## 1. 测评目标

对当前已入库的法条检索链路做量化评测，回答三个问题：

1. **检索准不准**：给定一个问题，正确的法条（金标）能否被召回、排在第几位；
2. **回答贴不贴**：基于召回法条生成的回答是否有据、是否回答了问题；
3. **哪个环节值得改**：通过消融实验对比 rerank、HyDE、RRF 混合检索各自的贡献。

评测结论用于指导后续检索策略调整，不改变生产检索逻辑。

## 2. 被测对象与数据现状

### 2.1 检索链路

被测函数：`src/agents/legal_knowledge/statute_rag.py:search_statutes_raw`

```text
问题
 -> 向量化（volcengine/doubao-embedding-vision-251215）
 -> Dense 向量检索 +（可选）BM25 稀疏检索 RRF 融合
 ->（可选）HyDE 查询改写
 ->（可选）rerank 精排（rerank_top_k=8）
 -> 返回 [{law_id, article_no, domain, text, score}]
```

### 2.2 当前数据

| 数据 | 数量 | 位置 |
| --- | ---: | --- |
| `laws` | 75 | PostgreSQL |
| `articles` | 7,105 | PostgreSQL |
| `statute_index` | 7,077 | Milvus |

以上数据为**被测资源**，全程只读。

## 3. 评测集设计

### 3.1 金标来源

金标直接取自已入库的 `articles` 表，不重新标注、不入向量库：

- 从 `articles` 按领域分层抽样；
- 每条记录以条文原文为基础整理成一个口语化/场景化问题；
- 金标 = 该条文本身（`law_id` + `article_no`）；
- 评测集保存为独立 JSON 文件，落在本工程 `data/` 下，与项目隔离。

### 3.2 评测集结构

```json
{
  question: 公司拖欠我两个月工资，我可以要求什么赔偿？,
  law_title: 中华人民共和国劳动合同法,
  article_no: 第八十五条,
  expected_law_id: 42,
  expected_article_no: 85,
  domain: labor_social_security,
  reference: 用人单位未足额支付劳动报酬的，由劳动行政部门责令限期支付……
}
```

### 3.3 规模与抽样

- **首轮小样本：10 条**，覆盖 5 个领域（劳动、消费、合同/房产、交通、婚姻家庭 等，每领域约 2 条），用于验证评测链路；
- 后续扩展：按领域扩充到 50–100 条，保持领域均衡；
- 每条问题不包含条文原文（避免“答案就在问题里”），只保留口语化表述。

## 4. 指标设计

### 4.1 确定性检索指标（不依赖 LLM，主指标）

| 指标 | 说明 |
| --- | --- |
| `Hit@8` | 金标条文是否出现在 top-8 |
| `MRR` | 金标条文排名倒数的均值 |
| `NDCG@8` | 按排序质量加权（相关分：金标=1，同法他条=0.5，其他=0） |
| 法律级命中率 | 不要求条号一致，只看是否正确法律 |
| 延迟 | 每次检索耗时 P50 / P95 |

以上按整体和分领域分别统计。

### 4.2 TruLens RAG Triad（LLM 判官）

| 指标 | 方向 | 说明 |
| --- | --- | --- |
| 答案相关性 | 问题 → 回答 | 回答是否针对问题 |
| 上下文相关性 | 问题 → 召回上下文 | 召回的条文是否与问题相关 |
| 有据性 | 召回上下文 → 回答 | 回答是否基于召回条文、不编造 |

判官模型：DeepSeek（通过阿里 DashScope OpenAI 兼容接口），与生成模型分离配置。

## 5. 消融实验设计

同一评测集、同一 `app_name`，不同检索配置注册为不同 `app_version`，在 TruLens 中直接对比：

| version | 配置 | 验证点 |
| --- | --- | --- |
| baseline | 纯 Dense，无 RRF / HyDE / rerank | 下限基准 |
| v1_rerank | Dense + rerank | 精排单独贡献 |
| v2_hyde | Dense + HyDE | 查询改写贡献 |
| v3_rrf | Dense + BM25 RRF | 混合检索贡献 |
| v4_production | RRF + rerank（当前默认） | 现状基线 |
| v5_all | RRF + HyDE + rerank | 上限 |

首轮小样本先跑 `baseline` 与 `v4_production` 两组，验证消融链路；全量时跑全部六组。

## 6. TruLens 接入方式

参考 tiangong-agent 的 TruLens 2.x 结构：

1. **会话**：`TruSession(database_url=sqlite:///.../trulens_eval.db)`，评估记录落在本工程目录，不写入项目业务库；
2. **判官**：`LiteLLM(model_engine=openai/deepseek-..., completion_kwargs={api_key: ..., api_base: ...})`，复用项目 `.env` 中的模型配置（只读读取）；
3. **指标**：`Metric` + `Selector` 实现 RAG Triad（`relevance_with_cot_reasons` / `context_relevance_with_cot_reasons` / `groundedness_measure_with_cot_reasons`）；
4. **追踪**：用 `@instrument(span_type=SpanType.RETRIEVAL / GENERATION)` 包装 `retrieve → generate → query` 三段，不改项目源码，只在本工程内做适配转发；
5. **运行**：`TruApp(app_name=statute_rag, app_version=..., feedbacks=..., session=...)`，逐条 `with tru_app as recording: await rag.query(q)`，结束后 `force_flush → stop_evaluator → compute_feedbacks → get_leaderboard`；
6. **确定性指标**：从每条 record 的 `meta`（金标 + 召回列表）用纯 Python 计算 `Hit@8 / MRR / NDCG`，与 RAG Triad 一起写入报告。

## 7. 执行环境与隔离保证

| 项 | 做法 |
| --- | --- |
| 工程目录 | `D:\learn\legal-tools\trulens-statute-eval\`（独立） |
| Python 环境 | 独立 venv（`--system-site-packages` 继承 `legal` 环境依赖），TruLens 只装入该 venv |
| 项目源码 | 只读 import，不修改、不新增任何文件 |
| 数据库 | PostgreSQL / Milvus 只读查询，不写库、不建表、不删数据、不重建索引 |
| 依赖 | 不改 `requirements.txt`，TruLens 依赖只在独立 venv 中安装 |
| 输出 | 评测集、TruLens 记录、报告全部落在本工程目录 |

## 8. 执行流程

```text
1) 只读检查（数据在、连接通）        ← 已完成
2) 构建 10 条评测集（金标=已入库条文）
3) 建独立 venv，安装 TruLens 依赖
4) 运行消融评测（baseline + v4_production）
5) 汇总确定性指标 + RAG Triad
6) 生成《法条检索测评报告.md》
```

## 9. 报告输出

报告包含：

- 评测配置与数据规模；
- 各 version 的确定性指标（整体 + 分领域）；
- RAG Triad 三项得分；
- 逐条样例（问题、金标、top-8 命中情况、判官理由）；
- 结论与建议（是否调整 rerank / HyDE / RRF 参数）。

## 10. 后续扩展（不在本轮范围）

- 追问依据检索（`followup_basis`）：法条 + 图谱的轻量召回；
- 类案检索（`case_rag`）：`case_index` + PG 元数据；
- 渠道检索（`channels`）：覆盖率和字段正确性；
- 法律统计 NL2SQL：SQL 正确性评测。
