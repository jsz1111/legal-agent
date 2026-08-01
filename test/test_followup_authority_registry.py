import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide import graph as guide_graph
from src.agents.legal_guide.authority_registry import (
    DOMAIN_SOURCE_KEYS,
    build_authority_index_rows,
    build_citation_snapshots,
    build_source_snapshots,
    format_domain_authority_summary,
)
from src.agents.legal_guide.followup_catalog import (
    assess_fact_answer,
    fact_followups,
    load_followup_catalog,
)
from src.agents.legal_guide.graph import GuideDeps
from src.agents.legal_guide.state import GuideState


def _deps(parsed: dict) -> GuideDeps:
    deps = MagicMock(spec=GuideDeps)
    deps.llm = MagicMock()
    deps.llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps(parsed, ensure_ascii=False)))
    return deps


def _parsed(**updates) -> dict:
    payload = {
        "is_answer": True,
        "new_issues": [],
        "collected_facts": [],
        "evidence": [],
        "evidence_unavailable": [],
        "adverse_facts": [],
        "region": "",
        "time_info": "",
        "user_question": "",
    }
    payload.update(updates)
    return payload


def test_every_domain_has_sources_and_every_rule_has_a_citation():
    catalog = load_followup_catalog()
    sources = build_source_snapshots()
    citations = build_citation_snapshots(sources)
    citation_rule_ids = {citation.rule_id for citation in citations}
    expected_rule_ids = {
        rule.id
        for domain in catalog.domains.values()
        for rule in [*domain.facts, *domain.evidence]
    }

    assert set(DOMAIN_SOURCE_KEYS) == set(catalog.domains)
    assert expected_rule_ids == citation_rule_ids
    assert len(sources) == 18
    assert len(citations) == 115


def test_official_local_files_pass_integrity_check_without_claiming_legal_review():
    sources = build_source_snapshots()
    official = [source for source in sources if source.source_type != "system_rule"]

    assert all(source.sha256 for source in official)
    assert all(source.review_status == "integrity_verified_pending_legal_review" for source in official)
    assert all((Path(guide_graph.__file__).resolve().parents[3] / source.local_path).is_file() for source in official)


def test_mapping_status_distinguishes_file_level_and_system_guidance():
    citations = build_citation_snapshots()
    statuses = {citation.mapping_status for citation in citations}

    assert statuses == {"needs_pinpoint", "source_located", "system_only"}
    assert all(
        citation.mapping_status == "system_only"
        for citation in citations
        if citation.domain == "other"
    )
    assert "具体条款仍待精标" in format_domain_authority_summary("administrative_remedies")


def test_authority_index_rows_keep_traceability_fields():
    rows = build_authority_index_rows()
    assert len(rows) == 115
    assert len({row["id"] for row in rows}) == 115
    assert all(row["rule_id"] and row["source_key"] and row["source_url"] for row in rows if row["domain"] != "other")
    assert all("依据来源：" in row["text"] for row in rows)


def test_fact_answer_is_user_statement_not_verified_fact():
    rule = fact_followups("labor_social_security")[0]
    record = assess_fact_answer(rule, "大概去年十月份离职")

    assert record["status"] == "approximate"
    assert record["verification"] == "not_independently_verified"
    assert record["source"] == "user_statement"


def test_changed_fact_answer_is_marked_conflicted_until_user_explicitly_corrects_it():
    rule = fact_followups("labor_social_security")[0]
    previous = assess_fact_answer(rule, "去年十月离职")
    conflicted = assess_fact_answer(rule, "今年三月离职", previous)
    corrected = assess_fact_answer(rule, "更正一下，是今年三月离职", previous)

    assert conflicted["status"] == "conflicted"
    assert corrected["status"] == "corrected"


def test_user_claimed_evidence_is_not_marked_authentic_or_admissible():
    state = GuideState(
        legal_domain="labor_social_security",
        messages=[HumanMessage(content="有劳动合同，在我手里")],
        pending_ask_details=["劳动合同、工牌、考勤、工作群记录中，您现在有任何一种吗？"],
        pending_ask_type="evidence",
        pending_followup_ids=["labor_relationship_evidence"],
    )
    result = asyncio.run(guide_graph.node_parse_details(
        state,
        _deps(_parsed(evidence=["劳动合同"])),
    ))
    assessment = result["evidence_assessments"]["labor_relationship_evidence"]

    assert assessment["availability"] == "user_claimed_present"
    assert assessment["authenticity"] == "not_verified"
    assert assessment["legal_admissibility"] == "not_determined"


def test_explicit_evidence_absence_is_progress_not_low_information():
    state = GuideState(
        legal_domain="labor_social_security",
        messages=[HumanMessage(content="没有，找不到了")],
        pending_ask_details=["工资流水、工资条或单位确认欠薪的聊天记录，您有其中一种吗？"],
        pending_ask_type="evidence",
        pending_followup_ids=["labor_payment_evidence"],
        consecutive_low_info_answers=1,
    )
    result = asyncio.run(guide_graph.node_parse_details(
        state,
        _deps(_parsed(evidence_unavailable=["工资流水"])),
    ))

    assert result["consecutive_low_info_answers"] == 0
    assert result["force_conclude"] is False
    assert result["pending_followup_ids"] == []


def test_three_consecutive_ambiguous_answers_force_conclusion():
    state = GuideState(
        legal_domain="labor_social_security",
        messages=[HumanMessage(content="这个需要怎么确认？")],
        pending_ask_details=["是否签有书面劳动合同？"],
        pending_ask_type="facts",
        pending_followup_ids=["labor_employer_relation"],
        consecutive_low_info_answers=2,
    )
    result = asyncio.run(guide_graph.node_parse_details(
        state,
        _deps(_parsed(
            answers_asked_question=True,
            collected_facts=[],
        )),
    ))

    assert result["consecutive_low_info_answers"] == 3
    assert result["force_conclude"] is True
    assert result["pending_followup_ids"] == []


def test_every_followup_text_contains_at_most_one_question_mark():
    catalog = load_followup_catalog()
    questions = [
        rule.question
        for domain in catalog.domains.values()
        for rule in [*domain.facts, *domain.evidence]
    ]
    assert all(question.count("？") + question.count("?") <= 1 for question in questions)
