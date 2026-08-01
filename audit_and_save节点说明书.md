# `audit_and_save` 审校发布与保存节点说明书

> 文档状态：正式节点说明，已接入 GuideGraph、FastAPI DebugInfo、Worker/Gradio 和电脑网页端
> 编写日期：2026-08-01
> 所属工作流：维权助手 GuideGraph
> 节点序号：节点八
> 关联文档：`维权工作流优化说明书.md`、`generate_solution节点说明书.md`

## 1. 节点定位

`audit_and_save` 是八节点工作流的最后一个核心处理节点。它消费节点七生成的候选方案，但不重新收集事实、不重新规划证据，也不直接相信候选 Markdown。

```text
generate_solution
        |
        v
audit_and_save
        |
        +-- 上游版本过期 --> 对应事实、证据规划或证据评估节点
        |
        +-- 审校通过 --> 发布正式方案版本 --> END
```

一句话定义：

> 节点八回答“这份方案是否仍绑定当前案件版本、是否遵守事实和法律边界，以及通过审校的内容如何可靠发布和长期保存”。

## 2. 职责边界

### 2.1 应当负责

- 校验案件标识、案件代次和草稿指纹；
- 校验事实快照、法律模型、证据清单和证据评估版本；
- 审查未知、冲突和失效事实是否被写成确定事实；
- 审查法律和渠道依据是否来自节点五、六的可核对结果；
- 审查已提交、未提交、明确没有、可调取和冲突材料是否混淆；
- 审查五维判断、行动、路径和任务是否自洽；
- 删除胜诉概率、结果保证、犯罪定性和确定证据效力表述；
- 用确定性模板重建正式 Markdown；
- 分配或复用正式 `plan_version`；
- 保存完整版本包和审校历史；
- 向网页端与 Gradio 返回同一份最终 Markdown。

### 2.2 不应负责

- 不从对话重新提取事实；
- 不修改事实黑板；
- 不静默解决事实冲突；
- 不重新确定法律关系、请求权或举证责任；
- 不新增未经检索的法条、期限、费用、机构或地址；
- 不重新理解上传文件；
- 不把审校模型当作新的方案生成器；
- 不物理删除旧方案、旧任务或旧证据记录。

## 3. 输入契约

节点八必须读取：

```text
case_id
case_generation

solution_draft
solution_draft_status
solution_draft_fingerprint
plan_version_candidate

fact_snapshot_version
fact_snapshot_hash
legal_model_version
evidence_plan_version
evidence_review_version

fact_blackboard
fact_snapshot_draft
legal_model
formal_evidence_requirements
evidence_review_report
plan_basis_refs
evidence_basis_refs
retrieved_law_refs
relevant_channels

plan_version
solution_versions
case_tasks
case_progress
```

只有 `solution_draft_status` 为 `awaiting_audit` 或兼容恢复状态，且草稿指纹有效时才能进入正式审校。

## 4. 第一层：发布前版本门禁

以下检查属于不可自动修正的上游问题：

| 检查 | 不一致时处理 |
|---|---|
| `case_id` / `case_generation` | 停止发布，返回案件恢复流程 |
| `fact_snapshot_version` / hash | 返回 `decide_facts` |
| `legal_model_version` | 返回 `plan_evidence` |
| `evidence_plan_version` | 返回 `plan_evidence` |
| `evidence_review_version` | 返回 `assess_evidence` |
| 新事实候选待确认 | 返回 `update_facts` |
| 材料核验未完成 | 返回 `assess_evidence` |
| 草稿指纹不匹配 | 返回 `generate_solution` |

发生这些问题时：

```text
solution_audit_status = blocked
solution_draft_status = audit_blocked
pending_solution_audit = false
```

旧草稿不得覆盖当前案件，也不得分配正式版本号。

## 5. 第二层：六类正式审校

### 5.1 事实边界

- 用户可见“已确认事实”只从当前事实快照重建；
- `unknown`、`conflicted`、`denied`、`superseded` 不进入确定事实；
- 草稿中无法映射到当前事实编号的内容被移除；
- 节点八只修正方案展示，不回写事实黑板。

### 5.2 法律和来源边界

- 方案依据必须能映射到节点五、六的检索结果；
- `pending`、`unreviewed`、`needs_pinpoint`、`rejected`、失效或废止依据不能作为确定依据；
- 无精确依据时保留“需通过官方渠道核对”的限制；
- 具体期限、费用、管辖和办理地址不得由模型补造。

### 5.3 证据边界

- 证据摘要从当前 `evidence_review_report` 重建；
- “未提交”不改写成“没有”；
- “用户称持有”不改写成“已经核验”；
- “已提交”不等于真实性、合法性、可采性或最终证明力已确定；
- 冲突材料保留冲突状态，不由系统选择某一份为真。

### 5.4 推理一致性

节点八核对五个维度：

1. 权利基础；
2. 事实清晰度；
3. 证据覆盖；
4. 程序可行性；
5. 履行风险。

最终等级只能是：

- 较有利；
- 条件性有利；
- 不确定；
- 风险较高。

行动依赖的事实编号和证据需求编号必须仍然存在。缺少标题或结构损坏的行动项不得发布。

### 5.5 表达边界

系统自动修正：

- 胜诉率、成功率或其他具体结果概率；
- “保证胜诉”“一定成功”“包赢”等结果承诺；
- “现有证据已经足够”“法院必然采纳”等越权表述；
- 将民事违约直接认定为诈骗等犯罪的表述。

证据评估和维权可能性始终是条件式、版本化判断。

### 5.6 Markdown 格式

最终 Markdown 由审校后的结构化方案重新渲染，至少包含：

```text
## 核心判断
## 已确认事实
## 法律依据与适用条件
## 证据检验结果
## 证据缺口与替代材料
## 有利、不利和不确定因素
## 当前维权可能性
## 推荐行动方案
## 替代与升级路径
## 下一步任务清单
## 参考文书
## 版本变化与限制
## 后续更新
```

最终文本不得出现 Redis 键、内部追踪 ID、候选版本 ID、异常堆栈或重复栏目。

## 6. 可修正问题与阻断问题

| 类型 | 示例 | 处理 |
|---|---|---|
| 可修正 | 未知事实被展示为确认事实 | 只重建事实展示 |
| 可修正 | 引用了未定位来源 | 移除该引用并记录依据缺口 |
| 可修正 | 证据状态措辞混淆 | 按当前评估重建证据摘要 |
| 可修正 | 概率、保证或越权措辞 | 保守改写 |
| 可修正 | Markdown 缺标题、重复或含调试标识 | 重新渲染 |
| 阻断 | 事实快照已变化 | 返回节点四 |
| 阻断 | 法律模型或证据清单过期 | 返回节点五 |
| 阻断 | 证据评估过期或核验未完成 | 返回节点六 |
| 阻断 | 草稿被修改且指纹不一致 | 返回节点七 |

节点八不使用大模型自由重写事实。结构化、可定位的问题由确定性审校器修正；上游问题由图路由回流。

## 7. 正式版本

首次发布：

```text
previous_plan_version = 0
plan_version = 1
```

后续事实、证据或程序进展导致实质变化：

```text
previous_plan_version = N
plan_version = N + 1
```

相同 `plan_version_candidate` 和 `solution_draft_fingerprint` 重试时复用原版本，不重复递增。

正式版本记录至少包含：

- 案件标识和案件代次；
- 当前版本、上一版本和发布时间；
- 来源版本；
- 事实快照；
- 法律模型；
- 证明目标、证据需求和交付入口；
- 证据评估；
- 结构化方案和最终 Markdown；
- 任务、案件进展和参考文书建议；
- 变化摘要和审校报告。

旧版本只追加，不覆盖。

## 8. 持久化

### 8.1 长期案件状态

`GuideState.solution_versions` 保存完整版本历史。项目的 `GUIDE_SESSION_TTL=0`，案件状态默认不设置过期时间，直到用户手动删除会话。

### 8.2 PostgreSQL 咨询索引

数字型数据库用户同步写入 `consultations`：

- `legal_advice` 保存当前最终 Markdown；
- `action_plan` 保存当前版本和完整版本历史 JSON；
- 同一用户、同一会话更新同一咨询索引记录。

公共非数字用户不强行写入外键表，使用长期案件状态保存。数据库索引临时不可用时，节点八记录降级状态，但不丢弃已经审校的案件版本。

### 8.3 删除

节点八不会自动删除案件或历史版本。用户手动删除会话时，由统一删除接口清理案件状态、附件关联、证据评估、方案版本和生成文书。

## 9. 输出状态

审校通过：

```text
phase = END
workflow_stage = plan_issued
solution_draft_status = published
pending_solution_audit = false
solution_audit_status = passed | passed_with_corrections
plan_version = N
```

主要输出字段：

```text
solution_audit_id
solution_reviewed_at
solution_audit_report
solution_audit_history

published_solution
published_solution_markdown
published_solution_fingerprint

plan_version
previous_plan_version
plan_published_at
solution_versions
solution_persistence_status
```

## 10. 前端与 Gradio

电脑网页端优先展示 `published_solution`：

- 显示正式版本号，不展示内部候选版本 ID；
- 显示审校后的定性等级和五维判断；
- 显示行动、路径、任务和版本变化；
- 审校未完成时明确显示“审校中”；
- 正式发布后显示“第 N 版已完成审校并长期保存”。

Gradio 不复制审校逻辑。它继续通过同一后端状态机获得节点八的最终 Markdown，因此网页端和 Gradio 的等级、路径、任务和版本一致。

## 11. 代码映射

| 文件或函数 | 职责 |
|---|---|
| `src/agents/legal_guide/audit_and_save.py::validate_audit_inputs` | 上游版本和草稿指纹门禁 |
| `src/agents/legal_guide/audit_and_save.py::audit_solution_draft` | 六类确定性审校和局部修正 |
| `src/agents/legal_guide/audit_and_save.py::build_published_version` | 正式版本、幂等和版本包 |
| `src/agents/legal_guide/audit_and_save.py::run_audit_and_save` | 节点八入口 |
| `src/agents/legal_guide/db_queries.py::save_solution_version` | PostgreSQL 咨询索引同步 |
| `src/agents/legal_guide/graph.py::node_audit_and_save` | GuideGraph 节点入口 |
| `src/agents/legal_guide/graph.py::route_after_audit_and_save` | 阻断问题上游回流 |
| `src/api/routers/chat.py::DebugInfo` | 网页端正式版本调试契约 |
| `src/agents/workers/guide_agent.py` | Worker/Gradio 正式版本透传 |

## 12. 必测场景

1. 正常草稿通过审校并发布第 1 版；
2. 事实快照变化后旧草稿被阻止；
3. 法律模型或证据清单过期后返回节点五；
4. 证据评估过期后返回节点六；
5. 草稿指纹被修改后返回节点七；
6. 未知事实不会进入已确认事实；
7. 未定位法律依据不会进入正式方案；
8. 未提交材料不会写成明确没有；
9. 胜诉概率和结果保证被移除；
10. 民事违约不会被直接定性为犯罪；
11. 正式 Markdown 栏目完整且无调试信息；
12. 同一候选版本重试不增加版本号；
13. 新事实或新证据形成第 N+1 版；
14. 旧版本和旧任务状态仍可读取；
15. 公共非数字用户仍能长期保存案件版本；
16. 网页端和 Gradio 返回同一最终 Markdown。

## 13. 验收标准

满足以下条件才视为节点八完成：

1. 正式路由为 `generate_solution -> audit_and_save -> END`；
2. 旧 `conclude/save_record` 未被删除，历史会话仍可运行；
3. 上游版本过期时绝不发布旧草稿；
4. 事实、法律、证据、推理、表达和 Markdown 六类审校均有记录；
5. 可修正问题只修改方案展示，不污染事实黑板；
6. 不输出胜诉百分比、结果保证或确定证据效力；
7. 每个正式方案具有可追溯版本和来源版本；
8. 相同候选版本重试幂等；
9. 旧版本、任务和进展不会被覆盖；
10. 案件默认长期保存，直到用户手动删除；
11. 网页端和 Gradio 使用同一正式方案；
12. 前端只显示正式版本号，不暴露内部候选 ID。
