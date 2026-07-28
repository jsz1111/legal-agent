"""法律年鉴 ChatBI 结构化数据和图表推荐测试。"""
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from src.agents.legal_knowledge.legal_statistics_chatbi import (
    collect_statistics_sources,
    is_statistics_followup,
    normalize_statistics_rows,
    recommend_statistics_chart,
    recommend_statistics_chart_with_llm,
    summarize_statistics_with_llm,
)


def _row(year, dimension, metric, value, unit="件", dataset_id="d1"):
    return {
        "statistical_year_end": year,
        "dimension_label": dimension,
        "metric": metric,
        "numeric_value": value,
        "unit": unit,
        "dataset_title": f"表14 {year}年全国法院民事一审统计表",
        "institution": "人民法院",
        "quality_flag": "auto_extracted",
        "dataset_id": dataset_id,
    }


def test_multi_year_result_recommends_line_chart():
    rows = [
        _row(2018, "劳动争议、人事争议", "收案", 452289, dataset_id="d18"),
        _row(2019, "劳动争议、人事争议", "收案", 483767, dataset_id="d19"),
        _row(2020, "劳动争议、人事争议", "收案", 439678, dataset_id="d20"),
    ]
    chart = recommend_statistics_chart("2018到2020年劳动争议收案趋势", rows)
    assert chart.type == "line"
    assert chart.x_values == [2018, 2019, 2020]
    assert chart.series[0].data == [452289, 483767, 439678]


def test_same_year_metrics_recommend_grouped_bar_chart():
    rows = [
        _row(2020, "合计", "收案", 13136436),
        _row(2020, "合计", "结案", 13305873),
    ]
    chart = recommend_statistics_chart("2020年民事一审收案和结案对比", rows)
    assert chart.type == "bar"
    assert set(chart.x_values) == {"收案", "结案"}
    assert set(chart.series[0].data) == {13136436, 13305873}


def test_pie_chart_excludes_total_row():
    rows = [
        _row(2020, "合计", "收案", 100),
        _row(2020, "合同纠纷", "收案", 60),
        _row(2020, "劳动争议", "收案", 40),
    ]
    chart = recommend_statistics_chart("2020年两类案件收案占比", rows)
    assert chart.type == "pie"
    assert set(chart.x_values) == {"合同纠纷", "劳动争议"}
    assert set(chart.series[0].data) == {60, 40}


def test_mixed_units_fall_back_to_table():
    rows = [
        _row(2020, "劳动争议", "收案", 439678, unit="件"),
        _row(2020, "劳动争议", "占比", 3.35, unit="%"),
    ]
    chart = recommend_statistics_chart("2020年劳动争议情况", rows)
    assert chart.type == "table"
    assert "不同单位" in chart.reason


def test_explicit_heatmap_request_builds_matrix_points():
    rows = [
        _row(year, dimension, "收案", year + index)
        for year in (2018, 2019, 2020)
        for index, dimension in enumerate(("合同纠纷", "劳动争议", "侵权纠纷"))
    ]
    chart = recommend_statistics_chart("用热力图比较各类案件", rows)
    assert chart.type == "heatmap"
    assert chart.x_values == [2018, 2019, 2020]
    assert len(chart.y_values) == 3
    assert len(chart.series[0].data) == 9


def test_normalization_and_sources_are_json_safe():
    rows = [_row(2020, "劳动争议", "收案", Decimal("439678"))]
    normalized = normalize_statistics_rows(rows)
    sources = collect_statistics_sources(normalized)
    assert normalized[0]["numeric_value"] == 439678
    assert isinstance(normalized[0]["numeric_value"], int)
    assert sources[0].dataset_id == "d1"
    assert sources[0].years == [2020]


def test_incompatible_llm_chart_proposal_falls_back_to_rules():
    rows = [
        _row(2018, "劳动争议", "收案", 452289),
        _row(2019, "劳动争议", "收案", 483767),
        _row(2020, "劳动争议", "收案", 439678),
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"type":"pie","reason":"使用饼图"}')
    )
    chart = asyncio.run(
        recommend_statistics_chart_with_llm("劳动争议趋势", rows, llm)
    )
    assert chart.type == "line"


def test_llm_summary_with_causal_claim_falls_back():
    rows = [
        _row(2019, "劳动争议", "收案", 483767),
        _row(2020, "劳动争议", "收案", 439678),
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="2019年为483,767件，2020年为439,678件，下降说明疫情产生影响。"
        )
    )
    summary = asyncio.run(summarize_statistics_with_llm("变化趋势", rows, llm))
    assert "疫情" not in summary
    assert "2019" in summary
    assert "2020" in summary


def test_statistics_followup_detection_requires_existing_sql():
    previous_sql = "SELECT * FROM legal_statistics.facts LIMIT 20"
    assert is_statistics_followup("再加上结案数", previous_sql) is True
    assert is_statistics_followup("只看2020年", previous_sql) is True
    assert is_statistics_followup("2020年呢", previous_sql) is True
    assert is_statistics_followup("劳动合同解除有什么规定", previous_sql) is False
    assert is_statistics_followup("只看2020年", "") is False
