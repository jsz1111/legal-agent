"""二次对抗审视：主审缺口扫描收敛后，从对方/办案机关视角补漏。

主审判定无更多高价值缺口时不直接收敛，先跑一次对方/办案机关视角的
二次审视（ADVERSARIAL_GAP_PROMPT），抓到主审漏掉的本案特有缺口；
二次审视也无题才以 fact_dimensions_converged 收敛。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from src.agents.legal_guide.followup_planner import plan_followup_batch
from src.agents.legal_guide.state import GuideState


def _satisfied_dims(*effects: str) -> dict:
    return {
        "dimensions": [
            {"effect": effect, "label": f"{effect}已满足", "satisfied": True}
            for effect in effects
        ]
    }


def _base_state(**kw: object) -> GuideState:
    defaults: dict = dict(
        legal_domain="criminal_public_security",
        decision_sufficiency=_satisfied_dims("limitation", "jurisdiction", "procedure"),
        followup_basis_refs=[{
            "source_type": "statute",
            "title": "中华人民共和国刑法",
            "article_no": "第二百三十四条",
            "text": "故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。",
        }],
        case_facts=[{
            "key": "event.assault",
            "category": "event",
            "statement": "昨晚十点在车站被打，对方至少两人，记了车牌号",
            "status": "asserted",
            "source_text": "昨晚十点在车站被打，对方至少两人，记了车牌号",
            "turn": 1,
        }],
        asked_details=["您被殴打后是否受伤，是否已去医院检查或做伤情鉴定"],
    )
    defaults.update(kw)
    return GuideState(**defaults)


def _sequential_llm(*payloads: dict) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps(p, ensure_ascii=False)) for p in payloads
    ])
    return llm


def _weapon_field() -> dict:
    return {
        "field_id": "weapon_tool_used",
        "question": "对方当时用什么方式、什么工具打的（徒手/棍棒/刀具/车辆等）？",
        "input_type": "long_text",
        "options": [],
        "placeholder": "请尽量回忆对方人数、使用的工具或方式",
        "answer_hint": "记不清时可填写“不清楚”",
        "decision_effects": ["responsibility"],
        "basis_indices": [0],
    }


def test_adversarial_scan_fires_when_main_converges():
    """主审判定无缺口后，二次审视仍能抓到场景特有缺口（用何种工具打）。"""
    llm = _sequential_llm(
        {"should_ask": False, "fields": []},  # 主审：无更多缺口
        {"should_ask": True, "fields": [_weapon_field()]},  # 二次审视：抓到工具缺口
    )
    plan = asyncio.run(plan_followup_batch(_base_state(), llm))

    assert plan["should_ask"] is True
    assert plan["planner_mode"] == "adversarial_retrieval_batch"
    assert plan["questions"][0]["field_id"] == "weapon_tool_used"
    assert plan["questions"][0]["basis_refs"][0]["title"] == "中华人民共和国刑法"


def test_adversarial_scan_converges_when_no_new_gap():
    """二次审视也判定无新缺口 → 最终收敛，不出现对抗批次。"""
    llm = _sequential_llm(
        {"should_ask": False, "fields": []},
        {"should_ask": False, "fields": []},
    )
    plan = asyncio.run(plan_followup_batch(_base_state(), llm))

    assert plan["should_ask"] is False
    assert plan["planner_mode"] == "fact_dimensions_converged"


def test_adversarial_scan_converges_when_llm_raises():
    """二次审视 LLM 异常不得让追问挂起，按主审结论收敛。"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps(
            {"should_ask": False, "fields": []}, ensure_ascii=False
        )),
        RuntimeError("adversarial llm down"),
    ])
    plan = asyncio.run(plan_followup_batch(_base_state(), llm))

    assert plan["should_ask"] is False
    assert plan["planner_mode"] == "fact_dimensions_converged"


def test_adversarial_scan_dedupes_already_asked_questions():
    """二次审视若只提出已问过的问题 → 无新增字段 → 收敛。"""
    state = _base_state(
        asked_details=[
            "您被殴打后是否受伤，是否已去医院检查或做伤情鉴定",
            "对方当时用什么方式、什么工具打的（徒手/棍棒/刀具/车辆等）？",
        ]
    )
    llm = _sequential_llm(
        {"should_ask": False, "fields": []},
        {"should_ask": True, "fields": [_weapon_field()]},  # 与已问问题重复
    )
    plan = asyncio.run(plan_followup_batch(state, llm))

    assert plan["should_ask"] is False
    assert plan["planner_mode"] == "fact_dimensions_converged"


def test_adversarial_scan_filters_out_of_range_basis_indices():
    """对抗批次字段带 basis_refs（供前端展示依据来源），越界引用被丢弃。"""
    field = _weapon_field()
    field["basis_indices"] = [0, 9]  # 9 越界
    llm = _sequential_llm(
        {"should_ask": False, "fields": []},
        {"should_ask": True, "fields": [field]},
    )
    plan = asyncio.run(plan_followup_batch(_base_state(), llm))

    assert plan["planner_mode"] == "adversarial_retrieval_batch"
    assert [r["title"] for r in plan["questions"][0]["basis_refs"]] == ["中华人民共和国刑法"]
