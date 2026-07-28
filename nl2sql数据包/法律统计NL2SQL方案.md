# 法律统计 NL2SQL 方案

## 整体架构

```
legal_qa_agent
├── search_statutes              法条检索（已有，不动）
├── search_cases                 类案检索（导入中）
├── search_channels              渠道查询（已有，不动）
├── search_legal_graph           图谱查询（已有，不动）
└── search_legal_statistics      统计 NL2SQL（新增）
```

路由逻辑：

| 问题 | 调用能力 |
|---|---|
| "劳动合同解除有哪些规定？" | 法条 RAG |
| "类似案件法院一般怎么判？" | 案例 RAG |
| "2020年劳动争议案件多少？" | 统计 NL2SQL |
| "2020年法院受理案件总量变化趋势？" | 统计 NL2SQL |
| "劳动争议案件变化及相关规定？" | 统计 + 法条并行 |

---

## 一、数据源

**数据包**：`nl2sql数据包/`

| 项目 | 值 |
|---|---|
| 来源 | 中国法律年鉴 2020-2021 |
| 工作簿数 | 193 个 XLS |
| 数据集 | 193 个 |
| 记录行 | 3,044 条 |
| 数值事实 | 11,612 条 |
| 单元格 | 23,573 个 |
| 年份范围 | 2015-2020（统计年份） |
| 年鉴版本 | 2019、2020、2021 |

主题覆盖：法院统计、刑事、民事、行政、检察、交通事故、公安、社会救助、仲裁、条约等。

**已校验通过**（`validation_report.json`），数据完整。

---

## 二、数据库设计

### 隔离策略

- 独立数据库 `legal_statistics_db`，schema `legal_statistics`
- 与现有 `legal_db`（法条、案例、用户）完全隔离
- 统计数据不能被当作法律依据
- 法条不能参与数值计算

### 表结构

**datasets** — 数据集目录（193 条）：

| 字段 | 说明 |
|---|---|
| `dataset_id` | 主键 |
| `yearbook_year` | 年鉴版本年份 |
| `statistical_year_start/end` | 统计实际年份 |
| `title` | 表名，如"2020年全国法院受理民事一审案件情况统计表" |
| `institution` | 统计机构，如"最高人民法院" |
| `topic` | 主题分类 |
| `unit` | 单位 |
| `source_file` | 来源文件 |

**records** — 原表行级 JSON 记录（3,044 条）：

| 字段 | 说明 |
|---|---|
| `record_id` | 主键 |
| `dataset_id` | 关联 datasets |
| `source_row` | 原始行号 |
| `row_label` | 行标签，如"民事一审收案" |
| `row_data` | JSONB 整行数据 |
| `search_text` | 全文检索用文本 |

**facts** — 数值长表（11,612 条）：

| 字段 | 说明 |
|---|---|
| `fact_id` | 主键 |
| `dataset_id` | 关联 datasets |
| `record_id` | 关联 records |
| `statistical_year_start/end` | 统计年份 |
| `institution` | 机构 |
| `topic` | 主题 |
| `dimension_label` | 维度标签 |
| `metric` | 指标名 |
| `numeric_value` | 数值 |
| `text_value` | 文本值 |
| `value_kind` | number / percentage |
| `unit` | 单位 |
| `quality_flag` | 质量标记 |

**cells** — 原始单元格审计（23,573 条）：

| 字段 | 说明 |
|---|---|
| `dataset_id` | 关联 datasets |
| `source_row/column` | 行列号 |
| `cell_text` | 原始文本 |
| `resolved_text` | 解析后文本 |
| `cell_type` | 类型 |

**category_mappings** — 跨年口径映射：

| 字段 | 说明 |
|---|---|
| `source_label` | 原始名称 |
| `canonical_label` | 标准名称 |
| `valid_from/to_year` | 适用年份 |
| `mapping_scope` | 作用域 |

---

## 三、NL2SQL 管线

### 工作流

```
自然语言问题："2020年劳动争议案件多少？"
  │
  1. 提取查询要素：年份、机构、类别、指标
  2. 查询 datasets 目录，找到相关数据集
  3. LLM 生成 SQL
  4. SQL 安全检查
  5. 执行查询（legal_statistics_db）
  6. 校验年份、单位和统计口径
  7. 返回：数据 + 来源文件 + 必要提示
```

### 安全限制

| 规则 | 说明 |
|---|---|
| 只允许 SELECT | 禁止 DDL、DML、多语句、任意函数 |
| 只开放 `legal_statistics` schema | 不能访问其他数据库 |
| 行数上限 | LIMIT 100 |
| 超时 | 10 秒 |

### 复用现有代码

`nl2sql.py` 已有的安全机制直接复用：

```python
# 安全检查
def _validate_sql(sql):
    if not sql.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"
    for pattern in _FORBIDDEN:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作"

# 重试 + 超时
result = await asyncio.wait_for(db.execute(text(sql)), timeout=10)
```

### SQL 生成 prompt 要点

让 LLM 生成的 SQL 遵循以下结构：

```sql
-- 查询某年某类案件数量
SELECT f.metric, f.numeric_value, f.unit, d.title
FROM legal_statistics.facts f
JOIN legal_statistics.datasets d ON f.dataset_id = d.dataset_id
WHERE f.statistical_year_end = 2020
  AND f.metric LIKE '%劳动争议%'
ORDER BY f.numeric_value DESC
LIMIT 10;
```

---

## 四、接入 legal_qa_agent

### 新增工具

在 `src/agents/legal_knowledge/tools.py` 新增：

```python
@tool
async def search_legal_statistics(question: str) -> str:
    """从中国法律年鉴统计数据库中查询法律统计数据。
    适用：查询案件数量、趋势、比例等数值型法律统计信息。
    示例："2020年劳动争议案件有多少"、"民事案件占比多少"
    question: 用户的法律统计问题
    """
    # 连接 legal_statistics_db
    # NL2SQL 生成 + 执行
    # 返回带来源的答案
```

### 更新 system prompt

在 `legal_qa_agent.py` 的系统提示词中补充：

```
6. search_legal_statistics — 法律统计查询
   适用：查询案件数量、趋势变化、比例等数值问题
   示例："2020年全国法院受理多少案件"、"劳动争议案件增长趋势"
   注意：统计数据不能作为法律依据，仅作参考
```

### 数据隔离

```
数据库连接：
  legal_db（主库）→ search_statute / search_cases / search_channels
  legal_statistics_db（统计库）→ search_legal_statistics（新增）
```

---

## 五、实现步骤

| 步骤 | 内容 | 预估 |
|---|---|---|
| 1 | 建库：创建 PostgreSQL `legal_statistics_db` + 导入数据 | 10 分钟 |
| 2 | 建连接：在 `src/infra/` 新增统计库连接配置 | 15 分钟 |
| 3 | 写工具：实现 `search_legal_statistics`（复用 nl2sql.py） | 40 分钟 |
| 4 | 接入：注册到 `legal_qa_agent.py` | 10 分钟 |
| 5 | 测试：验证各类型统计查询的正确性 | 20 分钟 |

---

## 六、与类案检索的优先级

| 任务 | 状态 | 建议 |
|---|---|---|
| 类案检索导入 | 方案已定，待实现 | 先做这个（案例是核心能力） |
| 统计 NL2SQL | 数据包就绪 | 做完案例后做 |
