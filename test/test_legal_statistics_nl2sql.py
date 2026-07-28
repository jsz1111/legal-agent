"""法律年鉴统计库 NL2SQL 的白名单、真实查询与工具接入测试。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from src.agents.legal_knowledge.legal_statistics_nl2sql import (
    _generate_sql,
    answer_legal_statistics_rows,
    extract_catalog_terms,
    search_legal_statistics_raw,
    validate_statistics_sql,
)
from src.agents.legal_knowledge.tools import (
    LegalKnowledgeDeps,
    build_legal_knowledge_tools,
)


BASE_SQL = """
SELECT
    f.statistical_year_end,
    f.dimension_label,
    f.metric,
    f.numeric_value,
    f.unit,
    d.title AS dataset_title,
    d.institution,
    f.quality_flag,
    f.dataset_id
FROM legal_statistics.facts AS f
JOIN legal_statistics.datasets AS d ON d.dataset_id = f.dataset_id
WHERE f.statistical_year_end = 2020
  AND REPLACE(f.dimension_label, ' ', '') ILIKE '%劳动争议%'
  AND REPLACE(f.metric, ' ', '') ILIKE '%收案%'
""".strip()


def _llm_returning(content: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def test_valid_select_gets_default_limit():
    valid, checked = validate_statistics_sql(BASE_SQL)
    assert valid is True
    assert checked.endswith("LIMIT 20")


def test_oversized_limit_is_capped():
    valid, checked = validate_statistics_sql(f"{BASE_SQL} LIMIT 1000")
    assert valid is True
    assert checked.endswith("LIMIT 100")


def test_rejects_mutation_multistatement_comments_and_unsafe_objects():
    invalid_sql = [
        "INSERT INTO legal_statistics.datasets (dataset_id) VALUES ('x')",
        "UPDATE legal_statistics.datasets SET title = 'x'",
        "DELETE FROM legal_statistics.datasets",
        f"{BASE_SQL}; SELECT * FROM legal_statistics.datasets",
        f"{BASE_SQL} -- bypass",
        "SELECT * FROM public.datasets",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT pg_sleep(1) FROM legal_statistics.datasets",
        "SELECT SUM(numeric_value) FROM legal_statistics.facts",
        "SELECT * FROM legal_statistics.unknown_table",
        "SELECT * FROM datasets",
        (
            "SELECT * FROM legal_statistics.datasets d "
            "JOIN other_table x ON x.id = d.dataset_id"
        ),
        (
            "SELECT * FROM legal_statistics.datasets d WHERE EXISTS "
            "(SELECT 1 FROM legal_statistics.facts f WHERE f.dataset_id=d.dataset_id)"
        ),
    ]
    for sql in invalid_sql:
        valid, _reason = validate_statistics_sql(sql)
        assert valid is False, sql


def test_extract_catalog_terms_understands_year_and_intake_synonym():
    terms, years = extract_catalog_terms("2020年劳动争议受理多少")
    assert years == [2020]
    assert "劳动争议" in terms
    assert "受理" in terms
    assert "收案" in terms


def test_extract_catalog_terms_expands_explicit_year_range():
    _terms, years = extract_catalog_terms("2018到2020年劳动争议收案趋势")
    assert years == [2018, 2019, 2020]


def test_followup_prompt_contains_last_validated_sql():
    llm = _llm_returning(f"{BASE_SQL} LIMIT 20")
    previous_sql = "SELECT * FROM legal_statistics.facts LIMIT 20"
    generated = asyncio.run(
        _generate_sql(
            "再加上结案数",
            llm,
            "候选目录",
            previous_sql=previous_sql,
        )
    )
    prompt = llm.ainvoke.await_args.args[0][0].content
    assert generated.endswith("LIMIT 20")
    assert previous_sql in prompt
    assert "再加某指标" in prompt


def test_real_statistics_query_returns_expected_labor_case_count():
    rows, checked_sql = asyncio.run(
        search_legal_statistics_raw(
            "2020年全国法院劳动争议一审收案多少？",
            _llm_returning(f"{BASE_SQL} LIMIT 20"),
        )
    )
    assert checked_sql.endswith("LIMIT 20")
    assert rows
    assert any(
        row["statistical_year_end"] == 2020
        and "劳动争议" in row["dimension_label"].replace(" ", "")
        and "收案" in row["metric"].replace(" ", "")
        and int(row["numeric_value"]) == 439678
        and row["unit"] == "件"
        for row in rows
    )


def test_statistics_answer_is_deterministic_and_does_not_invent_causes():
    rows = [
        {
            "statistical_year_end": 2019,
            "dimension_label": "劳动争议、 人事争议",
            "metric": "收 案",
            "numeric_value": 483767,
            "unit": "件",
            "dataset_title": "表14 2019年全国法院审理民事一审案件情况统计表",
            "institution": "人民法院",
            "quality_flag": "auto_extracted",
        },
        {
            "statistical_year_end": 2020,
            "dimension_label": "劳动争议、 人事争议",
            "metric": "收 案",
            "numeric_value": 439678,
            "unit": "件",
            "dataset_title": "表14 2020年全国法院受理民事一审案件情况统计表",
            "institution": "人民法院",
            "quality_flag": "auto_extracted",
        },
    ]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=AssertionError("回答阶段不应调用 LLM"))
    answer = asyncio.run(
        answer_legal_statistics_rows("2019到2020年收案变化趋势", rows, llm)
    )
    assert "483,767" in answer
    assert "439,678" in answer
    assert "减少 44,089件（-9.11%）" in answer
    assert "疫情" not in answer
    llm.ainvoke.assert_not_awaited()


def test_legal_qa_toolbox_registers_statistics_tool():
    deps = LegalKnowledgeDeps(
        llm=MagicMock(),
        embedding_model=MagicMock(),
        milvus_client=MagicMock(),
        neo4j_driver=MagicMock(),
    )
    tool_names = {item.name for item in build_legal_knowledge_tools(deps)}
    assert "search_legal_statistics" in tool_names
    assert len(tool_names) == 6
