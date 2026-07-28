"""中国法律年鉴统计库专用 NL2SQL / ChatBI 管线。"""
from __future__ import annotations

import asyncio
import re
from decimal import Decimal
from typing import Any

import jieba
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from sqlalchemy import text

from src.infra.legal_statistics_database import LegalStatisticsSessionLocal


MAX_SQL_RETRIES = 2
MAX_RESULT_ROWS = 100
SQL_TIMEOUT_SECONDS = 10
ALLOWED_TABLES = {"datasets", "records", "facts", "category_mappings"}

_FORBIDDEN_SQL = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|MERGE|UPSERT|DROP|ALTER|CREATE|TRUNCATE|"
    r"GRANT|REVOKE|COPY|CALL|DO|SET|RESET|SHOW|VACUUM|ANALYZE|REINDEX|"
    r"CLUSTER|REFRESH|LOCK|LISTEN|NOTIFY|UNLISTEN|EXECUTE|PREPARE|"
    r"DEALLOCATE|INTO|UNION|INTERSECT|EXCEPT)\b",
    re.IGNORECASE,
)
_DANGEROUS_OBJECTS = re.compile(
    r"\b(?:pg_catalog|information_schema|public)\.|\bpg_[a-z0-9_]*\b|"
    r"\b(?:pg_sleep|current_setting|set_config|dblink|lo_import|lo_export)\s*\(",
    re.IGNORECASE,
)
_TABLE_REFERENCE = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][\w$]*(?:\.[a-zA-Z_][\w$]*)?)",
    re.IGNORECASE,
)
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d{2}")
_AGGREGATE_FUNCTION = re.compile(
    r"\b(?:SUM|AVG|COUNT|MIN|MAX|STRING_AGG|ARRAY_AGG|JSON_AGG)\s*\(",
    re.IGNORECASE,
)

_STOPWORDS = {
    "查询", "统计", "数据", "多少", "数量", "情况", "全国", "我国", "中国",
    "案件", "法院", "年", "年度", "分别", "请问", "一下", "有没有", "进行",
    "变化", "趋势", "对比", "比较", "占比", "比例", "总数", "总量",
}
_PHRASES = (
    "劳动争议", "人事争议", "民事一审", "民事二审", "刑事案件", "民事案件",
    "行政案件", "行政复议", "行政应诉", "交通事故", "婚姻家庭", "知识产权",
    "执行案件", "检察机关", "公安机关", "人民法院", "收案", "结案", "立案",
    "判决", "调解", "撤诉", "未结", "受理", "审结",
)
_SYNONYMS = {
    "受理": "收案",
    "审结": "结案",
    "收到": "收案",
    "办结": "结案",
}


STATISTICS_SQL_PROMPT = """你是中国法律年鉴统计数据库的 PostgreSQL 查询专家。

用户问题：{question}

数据库只包含统计资料，不包含法律条文、个案裁判或用户信息。

可查询 Schema：
1. legal_statistics.datasets
   dataset_id, yearbook_year, statistical_year_start, statistical_year_end,
   title, institution, topic, unit, year_quality, extraction_status
2. legal_statistics.facts
   fact_id, dataset_id, record_id, yearbook_year, statistical_year_start,
   statistical_year_end, institution, topic, dimension_label, metric,
   metric_path, numeric_value, text_value, value_kind, unit, quality_flag
3. legal_statistics.records
   record_id, dataset_id, source_row, row_label, row_data, search_text
4. legal_statistics.category_mappings
   source_label, canonical_label, valid_from_year, valid_to_year, mapping_scope

系统找到的候选统计字段如下。优先使用候选中的 dataset_id、dimension_label 和 metric，
不要凭空猜测数据表或字段：
{catalog_context}

生成规则：
1. 只输出一条 SELECT，必须使用 legal_statistics schema 全限定表名。
2. 只用显式 JOIN，禁止逗号连接、子查询、CTE、UNION 和多语句。
3. 默认查询 facts，并 JOIN datasets 返回 d.title AS dataset_title。
4. 用户说“2020年”时默认指 statistical_year_end=2020，不是 yearbook_year=2020。
5. 中文标签可能含空格，筛选时使用 REPLACE(field, ' ', '') ILIKE '%关键词%'。
6. “受理/收案”对应 metric 的“收案”；“审结/办结”对应“结案”。
7. 不同 metric、不同 unit、不同统计表的数据不得直接相加，除非用户明确要求且口径一致。
8. 若用户只问“案件多少”但口径不明确，同时返回收案、结案等相关指标，不擅自选一个。
9. SELECT 至少返回 statistical_year_end、dimension_label、metric、numeric_value、unit、
   dataset_title、institution、quality_flag 和 dataset_id，便于核验来源和口径。
10. LIMIT 默认 20，最大 100。
11. facts 每行已经是一个统计单元格。禁止使用 SUM、AVG、COUNT、MIN、MAX 等聚合函数，
    否则容易把“合计”行与分类明细重复计算；比例和增减由回答阶段根据原始行计算。
12. 数字只用 f.numeric_value IS NOT NULL 判断；禁止自行猜测或过滤 value_kind 枚举值。
13. 用户问总数时，优先筛选 REPLACE(f.dimension_label, ' ', '') IN ('合计', '总计')，
    或使用汇总表中与问题精确匹配的维度，绝不把分类行相加。
14. 跨年或趋势问题使用 statistical_year_end BETWEEN 起始年 AND 结束年，
    不得固定单个 dataset_id，因为不同年份通常来自不同统计表。

只返回 SQL，不要解释，不要使用 Markdown 代码块。"""


STATISTICS_FOLLOWUP_PROMPT = """这是法律统计 ChatBI 的多轮查询。

上一次已经通过安全校验并成功执行的 SQL：
{previous_sql}

请结合用户本轮问题修改查询。若用户说“再加、只看、改成、排除、下钻、对比”，
必须继承上次 SQL 中未被用户修改的年份、案件类别、机构和指标；“再加某指标”要保留原指标。
若本轮明显是全新的独立问题，则可以放弃上次条件重新生成。

无论如何，都必须重新遵守下方完整 schema 和安全规则，不能直接执行或机械拼接上次 SQL。

{base_prompt}"""


def _extract_sql(content: str) -> str:
    value = (content or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    return (fenced.group(1) if fenced else value).strip()


def validate_statistics_sql(sql: str) -> tuple[bool, str]:
    """对统计库 SQL 做保守的白名单校验并强制限制返回行数。"""
    stripped = (sql or "").strip()
    if not stripped:
        return False, "SQL 为空"
    if len(stripped) > 8000:
        return False, "SQL 过长"
    if "--" in stripped or "/*" in stripped or "*/" in stripped:
        return False, "不允许 SQL 注释"

    if stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if ";" in stripped:
        return False, "只允许单条 SQL"
    if not re.match(r"^SELECT\b", stripped, flags=re.IGNORECASE):
        return False, "只允许 SELECT 查询"
    if len(re.findall(r"\bSELECT\b", stripped, flags=re.IGNORECASE)) != 1:
        return False, "不允许子查询"
    if _FORBIDDEN_SQL.search(stripped):
        return False, "查询包含禁止的 SQL 操作"
    if _DANGEROUS_OBJECTS.search(stripped):
        return False, "查询试图访问系统对象或危险函数"
    if _AGGREGATE_FUNCTION.search(stripped):
        return False, "禁止聚合统计单元格，请直接返回原始指标行"
    if re.search(r"\bFOR\s+(?:UPDATE|SHARE)\b", stripped, flags=re.IGNORECASE):
        return False, "不允许行锁"
    if re.search(r"\bFROM\s*\(", stripped, flags=re.IGNORECASE):
        return False, "不允许子查询"

    table_refs = _TABLE_REFERENCE.findall(stripped)
    if not table_refs:
        return False, "查询必须访问法律统计表"
    for reference in table_refs:
        if "." not in reference:
            return False, f"表必须使用 legal_statistics schema 全限定名称：{reference}"
        schema, table_name = reference.lower().split(".", 1)
        if schema != "legal_statistics" or table_name not in ALLOWED_TABLES:
            return False, f"不允许访问表 {reference}"

    # 禁止 FROM 子句中的逗号表连接；生成提示要求一律使用显式 JOIN。
    for segment in re.findall(
        r"\bFROM\b(.*?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        if re.search(r",\s*(?:[a-zA-Z_]\w*\.)?[a-zA-Z_]\w*(?:\s|$)", segment):
            return False, "不允许逗号表连接"

    limits = _LIMIT.findall(stripped)
    if len(limits) > 1:
        return False, "只允许一个 LIMIT"
    if limits:
        current = int(limits[0])
        if current > MAX_RESULT_ROWS:
            stripped = _LIMIT.sub(f"LIMIT {MAX_RESULT_ROWS}", stripped, count=1)
    else:
        stripped += " LIMIT 20"
    return True, stripped


def extract_catalog_terms(question: str) -> tuple[list[str], list[int]]:
    compact = re.sub(r"\s+", "", question or "")
    years = sorted({int(value) for value in _YEAR.findall(compact)})
    if len(years) == 2 and years[1] - years[0] <= 30:
        range_pattern = rf"{years[0]}(?:到|至|[-—~～]){years[1]}"
        if re.search(range_pattern, compact):
            years = list(range(years[0], years[1] + 1))
    terms: list[str] = []

    for phrase in _PHRASES:
        if phrase in compact and phrase not in terms:
            terms.append(phrase)
    for token in jieba.lcut(compact):
        token = token.strip()
        if len(token) < 2 or token in _STOPWORDS or _YEAR.fullmatch(token):
            continue
        if token not in terms:
            terms.append(token)
    for token in list(terms):
        synonym = _SYNONYMS.get(token)
        if synonym and synonym not in terms:
            terms.append(synonym)
    return terms[:10], years


def _candidate_score(row: dict[str, Any], terms: list[str], years: list[int]) -> int:
    haystack = re.sub(
        r"\s+",
        "",
        " ".join(
            str(row.get(key) or "")
            for key in ("title", "institution", "topic", "dimension_label", "metric")
        ),
    )
    score = sum((len(term) + 1) * 3 for term in terms if term.replace(" ", "") in haystack)
    if row.get("statistical_year_end") in years:
        score += 8
    if row.get("metric") and any(
        synonym.replace(" ", "") in re.sub(r"\s+", "", str(row["metric"]))
        for synonym in ("收案", "结案", "立案", "判决", "调解")
        if synonym in terms
    ):
        score += 6
    return score


async def build_catalog_context(question: str) -> str:
    """先做确定性目录检索，为 LLM 提供真实 dataset/指标候选。"""
    terms, years = extract_catalog_terms(question)
    conditions: list[str] = ["f.numeric_value IS NOT NULL"]
    params: dict[str, Any] = {}

    if years:
        year_names = []
        for index, year in enumerate(years):
            name = f"year_{index}"
            params[name] = year
            year_names.append(f":{name}")
        conditions.append(f"f.statistical_year_end IN ({', '.join(year_names)})")

    if terms:
        term_conditions = []
        for index, term in enumerate(terms):
            name = f"term_{index}"
            params[name] = f"%{term.replace(' ', '')}%"
            term_conditions.append(
                "REPLACE(CONCAT_WS(' ', d.title, d.institution, d.topic, "
                "f.dimension_label, f.metric), ' ', '') ILIKE :" + name
            )
        conditions.append("(" + " OR ".join(term_conditions) + ")")

    statement = text(
        """
        SELECT DISTINCT
            f.dataset_id, d.title, d.institution, d.topic,
            f.statistical_year_start, f.statistical_year_end,
            f.dimension_label, f.metric, f.unit, f.quality_flag
        FROM legal_statistics.facts AS f
        JOIN legal_statistics.datasets AS d ON d.dataset_id = f.dataset_id
        WHERE """
        + " AND ".join(conditions)
        + " LIMIT 5000"
    )

    async with LegalStatisticsSessionLocal() as session:
        result = await session.execute(statement, params)
        candidates = [dict(row) for row in result.mappings().all()]

    candidates.sort(
        key=lambda row: (
            -_candidate_score(row, terms, years),
            row.get("statistical_year_end") or 0,
            str(row.get("title") or ""),
        )
    )
    selected = candidates[:30]
    if not selected:
        return "未找到明确候选；请在 facts 与 datasets 中保守查询，并返回多个可能口径。"

    lines = []
    for item in selected:
        lines.append(
            " | ".join(
                [
                    f"dataset_id={item['dataset_id']}",
                    f"统计年={item['statistical_year_start']}-{item['statistical_year_end']}",
                    f"表={item['title']}",
                    f"机构={item['institution']}",
                    f"维度={item['dimension_label'] or ''}",
                    f"指标={item['metric']}",
                    f"单位={item['unit'] or ''}",
                    f"质量={item['quality_flag']}",
                ]
            )
        )
    return "\n".join(lines)


async def _generate_sql(
    question: str,
    llm: BaseChatModel,
    catalog_context: str,
    error_hint: str = "",
    previous_sql: str = "",
) -> str:
    base_prompt = STATISTICS_SQL_PROMPT.format(
        question=question,
        catalog_context=catalog_context,
    )
    prompt = (
        STATISTICS_FOLLOWUP_PROMPT.format(
            previous_sql=previous_sql,
            base_prompt=base_prompt,
        )
        if previous_sql
        else base_prompt
    )
    if error_hint:
        prompt += f"\n\n上一次 SQL 未通过或执行失败：{error_hint}\n请修正后重新生成。"
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return _extract_sql(response.content)


async def _execute_readonly(sql: str) -> list[dict[str, Any]]:
    async with LegalStatisticsSessionLocal() as session:
        async with session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(text("SET LOCAL statement_timeout = '10s'"))
            result = await asyncio.wait_for(
                session.execute(text(sql)),
                timeout=SQL_TIMEOUT_SECONDS,
            )
            return [dict(row) for row in result.mappings().all()]


async def search_legal_statistics_raw(
    question: str,
    llm: BaseChatModel,
    previous_sql: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """目录召回 → SQL 生成 → 白名单校验 → 独立只读数据库执行。"""
    catalog_context = await build_catalog_context(question)
    error_hint = ""
    last_sql = ""

    for attempt in range(MAX_SQL_RETRIES + 1):
        raw_sql = await _generate_sql(
            question,
            llm,
            catalog_context,
            error_hint,
            previous_sql=previous_sql,
        )
        valid, checked = validate_statistics_sql(raw_sql)
        if not valid:
            error_hint = checked
            logger.warning("法律统计 SQL 校验失败 attempt={} reason={}", attempt + 1, checked)
            continue

        last_sql = checked
        try:
            rows = await _execute_readonly(checked)
            logger.info(
                "法律统计 NL2SQL 成功 attempt={} rows={} sql={}",
                attempt + 1,
                len(rows),
                checked,
            )
            if not rows and attempt < MAX_SQL_RETRIES:
                error_hint = (
                    "查询执行成功但返回 0 行。请删除候选目录中不存在的过滤条件，"
                    "不要过滤 value_kind，并对中文标签使用 REPLACE(..., ' ', '')。"
                )
                continue
            return rows, checked
        except asyncio.TimeoutError:
            error_hint = "查询超过 10 秒，请缩小范围并使用候选 dataset_id"
        except Exception as exc:
            error_hint = str(exc)[:500]
        logger.warning(
            "法律统计 SQL 执行失败 attempt={} reason={}", attempt + 1, error_hint
        )
    return [], last_sql


async def answer_legal_statistics_rows(
    question: str,
    rows: list[dict[str, Any]],
    _llm: BaseChatModel,
) -> str:
    """确定性格式化统计行，避免回答模型补造原因、指标或数字。"""
    if not rows:
        return (
            "当前法律年鉴统计库中未查询到匹配数据。请补充统计年份、案件类别和指标口径，"
            "例如“2020年全国法院劳动争议一审收案数量”。"
        )

    def compact(value: Any) -> str:
        return "".join(str(value or "").split())

    def display_number(value: Any) -> str:
        if isinstance(value, Decimal):
            if value == value.to_integral_value():
                return f"{int(value):,}"
            return f"{value:,.2f}".rstrip("0").rstrip(".")
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.2f}".rstrip("0").rstrip(".")
        return str(value)

    def ordered_values(field: str) -> list[Any]:
        seen: list[Any] = []
        for row in rows:
            value = row.get(field)
            if value is not None and value not in seen:
                seen.append(value)
        return seen

    years = sorted(int(value) for value in ordered_values("statistical_year_end"))
    dimensions = [compact(value) for value in ordered_values("dimension_label")]
    metrics = [compact(value) for value in ordered_values("metric")]
    units = [str(value or "") for value in ordered_values("unit")]
    one_unit = units[0] if len(units) == 1 else ""

    normalized_rows = [
        {
            **row,
            "dimension_label": compact(row.get("dimension_label")),
            "metric": compact(row.get("metric")),
        }
        for row in rows
    ]
    lines = ["根据《中国法律年鉴》统计资料，查询结果如下：", ""]

    if len(normalized_rows) == 1:
        row = normalized_rows[0]
        descriptor = "".join(
            part
            for part in (
                f"{row.get('statistical_year_end')}年" if row.get("statistical_year_end") else "",
                str(row.get("dimension_label") or ""),
                str(row.get("metric") or ""),
            )
        )
        lines.append(
            f"**{descriptor or '统计值'}：{display_number(row.get('numeric_value'))}"
            f"{row.get('unit') or ''}。**"
        )
    elif len(dimensions) == 1:
        unit_label = f"（{one_unit}）" if one_unit else ""
        headers = ["统计年份"] + [f"{metric}{unit_label}" for metric in metrics]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        value_map = {
            (int(row["statistical_year_end"]), row["metric"]): row.get("numeric_value")
            for row in normalized_rows
            if row.get("statistical_year_end") is not None
        }
        for year in years:
            values = [
                display_number(value_map[(year, metric)])
                if (year, metric) in value_map
                else "—"
                for metric in metrics
            ]
            lines.append("| " + " | ".join([str(year), *values]) + " |")
    elif len(years) == 1:
        unit_label = f"（{one_unit}）" if one_unit else ""
        headers = ["案件类别"] + [f"{metric}{unit_label}" for metric in metrics]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        value_map = {
            (row["dimension_label"], row["metric"]): row.get("numeric_value")
            for row in normalized_rows
        }
        for dimension in dimensions:
            values = [
                display_number(value_map[(dimension, metric)])
                if (dimension, metric) in value_map
                else "—"
                for metric in metrics
            ]
            lines.append("| " + " | ".join([dimension or "未分类", *values]) + " |")
    else:
        lines.extend(
            [
                "| 统计年份 | 案件类别 | 指标 | 数值 | 单位 |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for row in sorted(
            normalized_rows,
            key=lambda item: (
                item.get("statistical_year_end") or 0,
                str(item.get("dimension_label") or ""),
                str(item.get("metric") or ""),
            ),
        ):
            lines.append(
                f"| {row.get('statistical_year_end') or '—'} | "
                f"{row.get('dimension_label') or '未分类'} | {row.get('metric') or '未注明'} | "
                f"{display_number(row.get('numeric_value'))} | {row.get('unit') or ''} |"
            )

    trend_requested = any(
        word in question for word in ("趋势", "变化", "增减", "增长", "下降", "对比")
    )
    if trend_requested and len(years) > 1 and len(dimensions) == 1:
        lines.extend(["", "**数值变化：**"])
        for metric in metrics:
            series = sorted(
                (
                    (int(row["statistical_year_end"]), row.get("numeric_value"))
                    for row in normalized_rows
                    if row.get("metric") == metric
                    and row.get("statistical_year_end") is not None
                    and isinstance(row.get("numeric_value"), (int, float, Decimal))
                ),
                key=lambda item: item[0],
            )
            for (previous_year, previous), (current_year, current) in zip(series, series[1:]):
                change = current - previous
                direction = "增加" if change >= 0 else "减少"
                rate = (change / previous * 100) if previous else None
                rate_text = f"（{rate:+.2f}%）" if rate is not None else ""
                prefix = f"{metric}：" if len(metrics) > 1 else ""
                lines.append(
                    f"- {prefix}{previous_year}→{current_year} 年{direction} "
                    f"{display_number(abs(change))}{one_unit}{rate_text}。"
                )

    sources: list[tuple[str, str]] = []
    for row in normalized_rows:
        source = (str(row.get("dataset_title") or ""), str(row.get("institution") or ""))
        if source[0] and source not in sources:
            sources.append(source)
    if sources:
        lines.extend(["", "**来源口径：**"])
        lines.extend(
            f"- 《{title}》{f'（{institution}）' if institution else ''}"
            for title, institution in sources
        )

    if any("year_inferred" in str(row.get("quality_flag") or "") for row in rows):
        lines.extend(["", "其中部分统计年份根据年鉴版本推断，使用时请结合原表复核。"])

    disclaimer = "以上为《中国法律年鉴》统计资料，仅供数据分析，不构成法律依据或个案结论。"
    lines.extend(["", disclaimer])
    return "\n".join(lines)


async def search_legal_statistics(question: str, llm: BaseChatModel) -> str:
    rows, _sql = await search_legal_statistics_raw(question, llm)
    return await answer_legal_statistics_rows(question, rows, llm)
