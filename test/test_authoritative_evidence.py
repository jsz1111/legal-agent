from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.legal_guide import graph as guide_graph
from src.agents.legal_guide.evidence_rules import (
    format_evidence_source,
    resolve_evidence_checklist,
    resolve_state_evidence_checklist,
)
from src.agents.legal_guide.graph import GuideDeps
from src.agents.legal_guide.state import GuideState


def test_labor_evidence_is_grounded_before_litigation_stage():
    checklist = resolve_evidence_checklist(
        "labor_social_security",
        "公司拖欠工资，尚未申请劳动仲裁",
    )

    assert checklist.is_officially_grounded
    assert checklist.source
    assert checklist.source["template_id"] == "spc_2025_labor_dispute_complaint"
    assert any("工资单" in item and "银行流水" in item for item in checklist.items)
    assert "法〔2025〕82号" in format_evidence_source(checklist)


def test_contract_evidence_uses_specific_case_type():
    lease = resolve_evidence_checklist(
        "contracts_property_housing",
        "房东拒绝退还租房押金",
    )
    lending = resolve_evidence_checklist(
        "contracts_property_housing",
        "朋友借钱后一直不还，有借条和转账",
    )

    assert lease.source and lease.source["case_type"] == "房屋租赁合同纠纷"
    assert any("押金" in item for item in lease.items)
    assert lending.source and lending.source["case_type"] == "民间借贷纠纷"
    assert any("款项实际交付" in item for item in lending.items)


def test_uncovered_domain_is_explicitly_system_guidance():
    checklist = resolve_evidence_checklist(
        "administrative_remedies",
        "对行政处罚决定不服",
    )

    assert not checklist.is_officially_grounded
    assert checklist.authority_level == "system_guidance"
    assert checklist.source is None
    assert "不是官方固定材料目录" in format_evidence_source(checklist)


def test_state_resolution_uses_accumulated_conversation():
    state = GuideState(
        legal_domain="contracts_property_housing",
        confirmed_issues=["返还租赁押金"],
        collected_facts=["租期届满后房东拒绝退押金"],
        messages=[HumanMessage(content="我租房交了5000元押金")],
    )

    checklist = resolve_state_evidence_checklist(state)

    assert checklist.source
    assert checklist.source["template_id"] == "spc_2025_house_lease_complaint"


def test_conclusion_prompt_and_reply_keep_official_evidence_source():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="最终维权方案，依据法〔2025〕82号。")
    )
    deps = MagicMock(spec=GuideDeps)
    deps.llm = llm
    state = GuideState(
        legal_domain="traffic_personal_injury",
        confirmed_issues=["机动车交通事故责任纠纷"],
        collected_facts=["交警已出具事故认定书"],
        evidence_confirmed=["事故认定书"],
        confidence_tier="MEDIUM",
    )

    updates = asyncio.run(guide_graph.node_conclude(state, deps))
    prompt = llm.ainvoke.await_args_list[0].args[0][0].content
    reply = updates["messages"][0].content

    assert "医疗费凭证、费用清单和病历资料" in prompt
    assert "证据清单性质与来源" in prompt
    assert "法〔2025〕82号" in prompt
    assert "法〔2025〕82号" in reply
    assert "官方发布页" in reply
    assert "https://www.court.gov.cn/zixun/xiangqing/468671.html" in reply


def test_conclusion_timeout_returns_a_complete_deterministic_plan():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=TimeoutError("slow model"))
    deps = MagicMock(spec=GuideDeps)
    deps.llm = llm
    state = GuideState(
        legal_domain="consumer_market",
        confirmed_issues=["消费纠纷"],
        collected_facts=["用户称购买的商品存在问题"],
        law_context_str=(
            "法条1【中华人民共和国消费者权益保护法 第五十五条】\n"
            "经营者提供商品或者服务有欺诈行为的，依法承担相应责任。"
        ),
        confidence_tier="LOW",
    )

    updates = asyncio.run(guide_graph.node_conclude(state, deps))
    reply = updates["messages"][0].content

    for section in (
        "理解您的情况",
        "法律依据",
        "维权路径比较",
        "维权胜算评估",
        "行动清单",
    ):
        assert section in reply
    assert "中华人民共和国消费者权益保护法" in reply
    assert "第五十五条" in reply
