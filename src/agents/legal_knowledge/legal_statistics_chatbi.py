"""法律年鉴统计查询的结构化 ChatBI 输出与确定性图表推荐。"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import json
import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from src.agents.legal_knowledge.legal_statistics_nl2sql import (
    answer_legal_statistics_rows,
    search_legal_statistics_raw,
)


ChartType = Literal["bar", "line", "pie", "scatter", "heatmap", "table"]

STATISTICS_FOLLOWUP_MARKERS = (
    "再加", "加上", "同时", "同一张图", "只看", "改成", "改为", "换成",
    "排除", "下钻", "继续", "对比", "上一年", "下一年",
)

CHART_RECOMMEND_PROMPT = """你是法律统计图表选择器。根据用户问题和查询结果特征，从
bar、line、pie、scatter、heatmap、table 中选择一种。

规则：时间趋势优先 line；同年类别/指标比较优先 bar；单年构成且单位一致可用 pie；
两个连续指标关系才用 scatter；多年多类别矩阵才用 heatmap；单值、单位混合或不适合绘图用 table。
程序规则引擎已经推荐：{fallback_type}
数据特征：{profile}
用户问题：{question}

只返回 JSON：{{"type":"line","reason":"一句话理由"}}。不得输出或改写任何统计数字。"""

SUMMARY_PROMPT = """你是法律年鉴统计摘要器。仅根据给定数据写 2 至 3 句中文关键发现。

严格要求：
1. 只能使用数据中已有的年份和数值，不得创造新数字或计算新指标。
2. 只描述高低、增减或并列关系，不得解释原因，不得出现“说明、表明、反映、可能、导致、由于、因为”。
3. 不提供法律建议，不把统计资料用于推断个案结果。

用户问题：{question}
数据：{rows_json}

只返回摘要正文。"""

PUBLIC_ROW_FIELDS = (
    "statistical_year_end",
    "dimension_label",
    "metric",
    "numeric_value",
    "unit",
    "dataset_title",
    "institution",
    "quality_flag",
    "dataset_id",
)


def is_statistics_followup(message: str, previous_sql: str) -> bool:
    if not previous_sql:
        return False
    if any(marker in message for marker in STATISTICS_FOLLOWUP_MARKERS):
        return True
    compact = "".join((message or "").split())
    short_statistical_reference = (
        len(compact) <= 40
        and (
            bool(re.search(r"(?:19|20)\d{2}", compact))
            or any(word in compact for word in ("收案", "结案", "案件", "指标", "图表", "柱状图", "折线图"))
        )
    )
    return short_statistical_reference


class StatisticsChartSeries(BaseModel):
    name: str
    data: list[Any] = Field(default_factory=list)


class StatisticsChart(BaseModel):
    type: ChartType
    title: str
    reason: str
    x_label: str = ""
    y_label: str = ""
    x_values: list[Any] = Field(default_factory=list)
    y_values: list[Any] = Field(default_factory=list)
    series: list[StatisticsChartSeries] = Field(default_factory=list)
    echarts_option: dict[str, Any] = Field(default_factory=dict)


class StatisticsSource(BaseModel):
    dataset_id: str
    title: str
    institution: str = ""
    years: list[int] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class LegalStatisticsChatBIResult(BaseModel):
    question: str
    answer: str
    summary: str
    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    chart: StatisticsChart
    sources: list[StatisticsSource] = Field(default_factory=list)


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def normalize_statistics_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只暴露绘图和来源核验所需字段，并转换 Decimal 等 JSON 不兼容类型。"""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = {
            field: _json_value(row.get(field))
            for field in PUBLIC_ROW_FIELDS
            if field in row
        }
        if item:
            normalized.append(item)
    return normalized


def _compact(value: Any) -> str:
    return "".join(str(value or "").split())


def _is_total_label(value: Any) -> bool:
    return _compact(value) in {"合计", "总计", "总数", "总量"}


def _series_name(row: dict[str, Any], dimensions: list[str], metrics: list[str]) -> str:
    dimension = str(row.get("dimension_label") or "")
    metric = str(row.get("metric") or "")
    if len(dimensions) > 1 and len(metrics) > 1:
        return f"{dimension} · {metric}"
    if len(metrics) > 1:
        return metric
    if len(dimensions) > 1:
        return dimension
    return metric or dimension or "数值"


def _make_axis_chart(
    chart_type: Literal["line", "bar"],
    title: str,
    reason: str,
    x_values: list[Any],
    series: list[StatisticsChartSeries],
    x_label: str,
    y_label: str,
) -> StatisticsChart:
    option = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 32},
        "grid": {"left": 56, "right": 24, "top": 76, "bottom": 52},
        "xAxis": {"type": "category", "name": x_label, "data": x_values},
        "yAxis": {"type": "value", "name": y_label},
        "series": [
            {
                "name": item.name,
                "type": chart_type,
                "data": item.data,
                "smooth": chart_type == "line",
            }
            for item in series
        ],
    }
    return StatisticsChart(
        type=chart_type,
        title=title,
        reason=reason,
        x_label=x_label,
        y_label=y_label,
        x_values=x_values,
        series=series,
        echarts_option=option,
    )


def _recommend_heatmap(
    question: str,
    rows: list[dict[str, Any]],
    years: list[int],
    dimensions: list[str],
    unit: str,
) -> StatisticsChart:
    x_values = years
    y_values = dimensions
    points: list[list[Any]] = []
    values: list[float] = []
    for row in rows:
        year = row.get("statistical_year_end")
        dimension = str(row.get("dimension_label") or "")
        value = row.get("numeric_value")
        if year not in x_values or dimension not in y_values or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
        points.append([x_values.index(year), y_values.index(dimension), value])

    series = [StatisticsChartSeries(name="统计值", data=points)]
    option = {
        "title": {"text": question[:60], "left": "center"},
        "tooltip": {"position": "top"},
        "grid": {"left": 120, "right": 32, "top": 64, "bottom": 48},
        "xAxis": {"type": "category", "name": "统计年份", "data": x_values},
        "yAxis": {"type": "category", "name": "类别", "data": y_values},
        "visualMap": {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
        },
        "series": [{"name": "统计值", "type": "heatmap", "data": points}],
    }
    return StatisticsChart(
        type="heatmap",
        title=question[:60],
        reason="结果同时包含多个年份和多个案件类别，热力图便于识别高低分布。",
        x_label="统计年份",
        y_label=f"类别（{unit}）" if unit else "类别",
        x_values=x_values,
        y_values=y_values,
        series=series,
        echarts_option=option,
    )


def _recommend_scatter(
    question: str,
    rows: list[dict[str, Any]],
    metrics: list[str],
    unit: str,
) -> StatisticsChart | None:
    if len(metrics) != 2:
        return None
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = row.get("numeric_value")
        if not isinstance(value, (int, float)):
            continue
        label = str(row.get("dimension_label") or row.get("statistical_year_end") or "")
        grouped[label][str(row.get("metric") or "")] = value

    points = [
        [values[metrics[0]], values[metrics[1]], label]
        for label, values in grouped.items()
        if all(metric in values for metric in metrics)
    ]
    if len(points) < 2:
        return None
    series = [StatisticsChartSeries(name=f"{metrics[0]} / {metrics[1]}", data=points)]
    option = {
        "title": {"text": question[:60], "left": "center"},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "name": metrics[0]},
        "yAxis": {"type": "value", "name": metrics[1]},
        "series": [{"name": series[0].name, "type": "scatter", "data": points}],
    }
    return StatisticsChart(
        type="scatter",
        title=question[:60],
        reason="问题明确要求比较两个数值指标的关系。",
        x_label=f"{metrics[0]}（{unit}）" if unit else metrics[0],
        y_label=f"{metrics[1]}（{unit}）" if unit else metrics[1],
        series=series,
        echarts_option=option,
    )


def recommend_statistics_chart(
    question: str,
    rows: list[dict[str, Any]],
) -> StatisticsChart:
    """根据实际结果形状选择图表，绝不让模型重写或补造绘图数值。"""
    usable = [
        row for row in rows if isinstance(row.get("numeric_value"), (int, float))
    ]
    if not usable:
        return StatisticsChart(
            type="table",
            title=question[:60],
            reason="没有可绘制的数值结果，使用表格保留原始口径。",
        )

    units = sorted({str(row.get("unit") or "") for row in usable})
    if len(units) != 1:
        return StatisticsChart(
            type="table",
            title=question[:60],
            reason="查询结果包含不同单位，不能放在同一坐标轴中比较。",
        )
    unit = units[0]
    years = sorted({int(row["statistical_year_end"]) for row in usable if row.get("statistical_year_end") is not None})
    dimensions = sorted({str(row.get("dimension_label") or "") for row in usable})
    metrics = sorted({str(row.get("metric") or "") for row in usable})

    # 详细分类与合计同时出现时，图表排除合计行，原始表格仍完整保留。
    # 否则柱状图/热力图会把整体与其组成部分并列，造成重复比较。
    if len(dimensions) > 1:
        detail_rows = [
            row for row in usable if not _is_total_label(row.get("dimension_label"))
        ]
        if detail_rows:
            usable = detail_rows
            years = sorted({int(row["statistical_year_end"]) for row in usable if row.get("statistical_year_end") is not None})
            dimensions = sorted({str(row.get("dimension_label") or "") for row in usable})
            metrics = sorted({str(row.get("metric") or "") for row in usable})

    if "散点" in question:
        scatter = _recommend_scatter(question, usable, metrics, unit)
        if scatter:
            return scatter

    if len(years) > 1 and len(dimensions) > 2 and ("热力" in question or len(usable) > 12):
        return _recommend_heatmap(question, usable, years, dimensions, unit)

    ratio_question = any(word in question for word in ("占比", "比例", "构成"))
    pie_rows = [row for row in usable if not _is_total_label(row.get("dimension_label"))]
    if ratio_question and len(years) <= 1 and 2 <= len(pie_rows) <= 12:
        labels = [
            str(row.get("dimension_label") or row.get("metric") or "未命名")
            for row in pie_rows
        ]
        values = [row["numeric_value"] for row in pie_rows]
        series = [StatisticsChartSeries(name=unit or "数值", data=values)]
        option = {
            "title": {"text": question[:60], "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "name": unit or "数值",
                    "type": "pie",
                    "radius": ["35%", "68%"],
                    "data": [
                        {"name": label, "value": value}
                        for label, value in zip(labels, values)
                    ],
                }
            ],
        }
        return StatisticsChart(
            type="pie",
            title=question[:60],
            reason="问题询问单一年度的类别构成，且各项单位一致。",
            y_label=unit,
            x_values=labels,
            series=series,
            echarts_option=option,
        )

    if len(years) > 1:
        x_values = years
        grouped: dict[str, dict[int, Any]] = defaultdict(dict)
        for row in usable:
            name = _series_name(row, dimensions, metrics)
            grouped[name][int(row["statistical_year_end"])] = row["numeric_value"]
        series = [
            StatisticsChartSeries(
                name=name,
                data=[year_values.get(year) for year in x_values],
            )
            for name, year_values in sorted(grouped.items())
        ]
        return _make_axis_chart(
            "line",
            question[:60],
            "结果包含多个统计年份，折线图适合展示变化趋势。",
            x_values,
            series,
            "统计年份",
            unit,
        )

    if len(usable) > 1:
        use_dimension = len(dimensions) > 1
        x_values = dimensions if use_dimension else metrics
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in usable:
            category = str(row.get("dimension_label") if use_dimension else row.get("metric"))
            name = str(row.get("metric") if use_dimension else row.get("dimension_label")) or "数值"
            grouped[name][category] = row["numeric_value"]
        series = [
            StatisticsChartSeries(
                name=name,
                data=[category_values.get(category) for category in x_values],
            )
            for name, category_values in sorted(grouped.items())
        ]
        return _make_axis_chart(
            "bar",
            question[:60],
            "结果是同一统计期内的类别或指标比较，柱状图便于横向比较。",
            x_values,
            series,
            "案件类别" if use_dimension else "统计指标",
            unit,
        )

    return StatisticsChart(
        type="table",
        title=question[:60],
        reason="查询只返回一个统计值，表格比图表更清晰。",
    )


def collect_statistics_sources(rows: list[dict[str, Any]]) -> list[StatisticsSource]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id:
            continue
        item = grouped.setdefault(
            dataset_id,
            {
                "dataset_id": dataset_id,
                "title": str(row.get("dataset_title") or ""),
                "institution": str(row.get("institution") or ""),
                "years": set(),
                "quality_flags": set(),
            },
        )
        if row.get("statistical_year_end") is not None:
            item["years"].add(int(row["statistical_year_end"]))
        if row.get("quality_flag"):
            item["quality_flags"].add(str(row["quality_flag"]))

    return [
        StatisticsSource(
            dataset_id=item["dataset_id"],
            title=item["title"],
            institution=item["institution"],
            years=sorted(item["years"]),
            quality_flags=sorted(item["quality_flags"]),
        )
        for item in grouped.values()
    ]


def _extract_json_object(content: str) -> dict[str, Any]:
    value = (content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


async def recommend_statistics_chart_with_llm(
    question: str,
    rows: list[dict[str, Any]],
    llm: BaseChatModel,
) -> StatisticsChart:
    """LLM 提议图表类型；数据形状规则负责最终裁决和降级。"""
    fallback = recommend_statistics_chart(question, rows)
    profile = {
        "row_count": len(rows),
        "years": sorted({row.get("statistical_year_end") for row in rows if row.get("statistical_year_end") is not None}),
        "dimension_count": len({str(row.get("dimension_label") or "") for row in rows}),
        "metrics": sorted({str(row.get("metric") or "") for row in rows}),
        "units": sorted({str(row.get("unit") or "") for row in rows}),
    }
    prompt = CHART_RECOMMEND_PROMPT.format(
        fallback_type=fallback.type,
        profile=json.dumps(profile, ensure_ascii=False),
        question=question,
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        proposal = _extract_json_object(str(response.content))
    except Exception:
        return fallback

    proposed_type = proposal.get("type")
    reason = str(proposal.get("reason") or "").strip()[:160]
    # 规则引擎已检查单位、行数和维度。LLM 只能确认其类型，或保守降级为表格。
    if proposed_type == fallback.type:
        if reason:
            fallback.reason = reason
        return fallback
    if proposed_type == "table":
        return StatisticsChart(
            type="table",
            title=question[:60],
            reason=reason or "图表提议未通过数据形状校验，使用表格展示。",
        )
    return fallback


def _fallback_summary(rows: list[dict[str, Any]]) -> str:
    usable = [
        row for row in rows if isinstance(row.get("numeric_value"), (int, float))
    ]
    if not usable:
        return "当前查询没有返回可用于分析的数值。"
    years = sorted({int(row["statistical_year_end"]) for row in usable if row.get("statistical_year_end") is not None})
    metrics = sorted({str(row.get("metric") or "") for row in usable})
    if len(years) > 1 and len(metrics) == 1:
        values_by_year = {
            int(row["statistical_year_end"]): row["numeric_value"]
            for row in usable
            if row.get("statistical_year_end") is not None
        }
        first_year, last_year = years[0], years[-1]
        first_value, last_value = values_by_year[first_year], values_by_year[last_year]
        direction = "增加" if last_value >= first_value else "减少"
        peak_year = max(values_by_year, key=values_by_year.get)
        return (
            f"查询覆盖 {first_year} 至 {last_year} 年，{metrics[0]}从 "
            f"{first_value:,} 变为 {last_value:,}，总体{direction}。"
            f"其中 {peak_year} 年数值最高，为 {values_by_year[peak_year]:,}。"
        )
    year_text = f"，覆盖 {years[0]} 至 {years[-1]} 年" if years else ""
    metric_text = "、".join(metrics) if metrics else "统计指标"
    return f"本次查询返回 {len(usable)} 个数值{year_text}。结果包含{metric_text}，具体口径见数据表。"


def _summary_is_grounded(summary: str, rows: list[dict[str, Any]]) -> bool:
    forbidden = (
        "原因", "说明", "表明", "反映", "可能", "导致", "由于", "因为",
        "疫情", "积压", "成效", "力度", "促进", "影响",
    )
    if not summary or any(word in summary for word in forbidden):
        return False
    allowed_numbers = set()
    for row in rows:
        year = row.get("statistical_year_end")
        value = row.get("numeric_value")
        if year is not None:
            allowed_numbers.add(str(int(year)))
        if isinstance(value, (int, float)):
            allowed_numbers.add(str(value).replace(",", ""))
            if float(value).is_integer():
                allowed_numbers.add(str(int(value)))
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", summary):
        normalized = token.replace(",", "")
        if normalized not in allowed_numbers:
            return False
    return True


async def summarize_statistics_with_llm(
    question: str,
    rows: list[dict[str, Any]],
    llm: BaseChatModel,
) -> str:
    fallback = _fallback_summary(rows)
    if not rows:
        return fallback
    prompt = SUMMARY_PROMPT.format(
        question=question,
        rows_json=json.dumps(rows[:30], ensure_ascii=False),
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        summary = str(response.content).strip()
    except Exception:
        return fallback
    return summary if _summary_is_grounded(summary, rows) else fallback


async def run_legal_statistics_chatbi(
    question: str,
    llm: BaseChatModel,
    previous_sql: str = "",
) -> LegalStatisticsChatBIResult:
    rows, sql = await search_legal_statistics_raw(
        question,
        llm,
        previous_sql=previous_sql,
    )
    answer = await answer_legal_statistics_rows(question, rows, llm)
    normalized = normalize_statistics_rows(rows)
    chart = await recommend_statistics_chart_with_llm(question, normalized, llm)
    summary = await summarize_statistics_with_llm(question, normalized, llm)
    columns = [field for field in PUBLIC_ROW_FIELDS if any(field in row for row in normalized)]
    return LegalStatisticsChatBIResult(
        question=question,
        answer=answer,
        summary=summary,
        sql=sql,
        columns=columns,
        rows=normalized,
        chart=chart,
        sources=collect_statistics_sources(normalized),
    )
