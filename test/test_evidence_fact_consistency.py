"""证据清单与细节库一致（合情理）+ 评估结合上下文 + 时效 + 程序序列 + 方案 Word。

覆盖“证据点相关性门控”这一本质机制在四类消费方（评估、追问、收敛、渲染）的
行为，以及维权路径序列、时效统一提示、方案 Word 交付物的回归保障。
"""
from __future__ import annotations

from src.agents.legal_guide.evidence_analysis import (
    coverage_for_rule,
    evidence_decay_banner,
    evaluate_state_evidence,
    merge_evidence_requirements,
)
from src.agents.legal_guide.decision_sufficiency import assess_decision_sufficiency
from src.agents.legal_guide.followup_catalog import evidence_followups
from src.agents.legal_guide.graph import _format_evidence_collection_reply
from src.agents.legal_guide.prompts import CONCLUDE_PROMPT
from src.agents.legal_guide.situation_review import (
    UserSituationVerdict,
    situation_guidance,
)
from src.agents.legal_guide.state import GuideState


def _brawl_state(*, with_contact: bool = False, **overrides) -> GuideState:
    facts = ["在小区门口被对方打伤"]
    case_facts = [
        {"key": "f1", "statement": "在小区门口被对方打伤", "status": "asserted", "turn": 1},
    ]
    if with_contact:
        facts.append("我们微信联系过")
        case_facts.append(
            {"key": "f2", "statement": "我们微信联系过", "status": "asserted", "turn": 1}
        )
    return GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["故意伤害"],
        collected_facts=facts,
        case_facts=case_facts,
        **overrides,
    )


def test_stranger_brawl_merges_only_relevant_evidence_points():
    state = _brawl_state()
    report = evaluate_state_evidence(state)
    requirements, _version = merge_evidence_requirements(state, report)

    labels = {item["label"] for item in requirements}
    assert "现场监控/行车记录仪录像" in labels          # 暴力事件总需影像线索
    assert "伤情照片" in labels
    assert "病历/诊断证明/医疗费票据" in labels
    assert "报警回执/受案回执/案件编号" in labels
    # 陌生互殴无先前接触/资金往来：聊天/通话/转账记录不参与清单。
    assert "与对方/在场人员的聊天记录" not in labels
    assert "通话记录" not in labels
    assert "转账/支付记录" not in labels


def test_chat_records_surface_when_detail_library_mentions_contact():
    state = _brawl_state(with_contact=True)
    report = evaluate_state_evidence(state)
    requirements, _version = merge_evidence_requirements(state, report)

    labels = {item["label"] for item in requirements}
    assert "与对方/在场人员的聊天记录" in labels
    chat_row = next(item for item in requirements if "聊天记录" in item["label"])
    assert chat_row["decay_risk"] is True


def test_negated_and_compound_phrases_do_not_light_contact_evidence():
    # “对方我不认识”含“认识”但被“不”否定；“证人联系方式”含“联系”但是名词，
    # 均不得点亮聊天/通话记录。这是相关性门控对否定与复合词的通用防误判。
    state = GuideState(
        legal_domain="criminal_public_security",
        confirmed_issues=["故意伤害"],
        collected_facts=[
            "在小区门口被对方打伤",
            "对方我不认识，应该是陌生人",
            "我没有证人联系方式",
        ],
        case_facts=[
            {"key": "f1", "statement": "在小区门口被对方打伤", "status": "asserted", "turn": 1},
            {"key": "f2", "statement": "对方我不认识，应该是陌生人", "status": "asserted", "turn": 1},
            {"key": "f3", "statement": "我没有证人联系方式", "status": "asserted", "turn": 1},
        ],
    )
    report = evaluate_state_evidence(state)
    requirements, _version = merge_evidence_requirements(state, report)

    labels = {item["label"] for item in requirements}
    assert "与对方/在场人员的聊天记录" not in labels
    assert "通话记录" not in labels
    assert "转账/支付记录" not in labels


def test_irrelevant_but_submitted_row_is_never_hidden():
    state1 = _brawl_state(with_contact=True, evidence_confirmed=["聊天记录"])
    report1 = evaluate_state_evidence(state1)
    requirements1, version1 = merge_evidence_requirements(state1, report1)
    chat_row = next(item for item in requirements1 if "聊天记录" in item["label"])
    assert chat_row["supporting_evidence_ids"]

    # 后续细节库不再命中“微信”，但用户已提交过聊天记录，行必须保留。
    state2 = _brawl_state(
        evidence_requirements=requirements1,
        evidence_requirement_version=version1,
        round=1,
    )
    report2 = evaluate_state_evidence(state2)
    requirements2, _version2 = merge_evidence_requirements(state2, report2)

    kept = [item for item in requirements2 if item["id"] == "proof_target:criminal_chat_records"]
    assert kept and kept[0]["active"] is True


def test_cctv_rule_is_retrieve_mode_with_retrieval_guidance():
    state = _brawl_state()
    report = evaluate_state_evidence(state)
    requirements, _version = merge_evidence_requirements(state, report)

    cctv = next(item for item in requirements if "监控" in item["label"])
    assert cctv["collect_mode"] == "retrieve"
    assert cctv["decay_risk"] is True
    assert "调取" in cctv["next_action"]


def test_unflagged_domain_shows_all_rules():
    state = GuideState(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        collected_facts=["公司拖欠工资"],
        case_facts=[
            {"key": "f1", "statement": "公司拖欠工资", "status": "asserted", "turn": 1}
        ],
    )
    report = evaluate_state_evidence(state)
    assert len(report.targets) == len(evidence_followups("labor_social_security"))


def test_merge_is_deterministic_and_version_stable():
    requirements1, version1 = merge_evidence_requirements(
        _brawl_state(),
        evaluate_state_evidence(_brawl_state()),
    )
    requirements2, version2 = merge_evidence_requirements(
        _brawl_state(),
        evaluate_state_evidence(_brawl_state()),
    )
    assert version1 == version2
    assert requirements1 == requirements2


def test_conclude_prompt_requires_procedural_sequence_for_criminal_paths():
    assert "按程序先后" in CONCLUDE_PROMPT
    assert "不得把报警、民事诉讼、调解、伤情鉴定并列" in CONCLUDE_PROMPT


def test_time_sensitive_hint_reaches_both_frameworks():
    normal = situation_guidance(
        UserSituationVerdict(time_sensitive=True)
    )
    assert "数天至数周" in normal
    assert "尽快调取/备份" in normal

    party = situation_guidance(
        UserSituationVerdict(
            own_risk_level="warning",
            own_risk_kinds=["criminal"],
            reasons=["还手致对方骨折"],
            time_sensitive=True,
        )
    )
    assert "数天至数周" in party
    assert "尽快调取/备份" in party

    quiet = situation_guidance(UserSituationVerdict())
    assert "数天至数周" not in quiet


def test_evidence_collection_reply_shows_retrieval_and_decay_banner():
    state = _brawl_state()
    report = evaluate_state_evidence(state)
    requirements, _version = merge_evidence_requirements(state, report)
    reply = _format_evidence_collection_reply(state, requirements)

    assert "需提供的调取线索" in reply
    assert "易消失证据提示" in reply
    assert "监控" in reply


def test_evidence_decay_banner_is_single_and_only_when_decay_risk():
    banner = evidence_decay_banner([
        {"id": "a", "label": "现场监控/行车记录仪录像", "decay_risk": True, "active": True},
        {"id": "b", "label": "与对方/在场人员的聊天记录", "decay_risk": True, "active": True},
        {"id": "c", "label": "伤情照片", "decay_risk": False, "active": True},
    ])
    assert "易消失证据提示" in banner
    assert "现场监控" in banner
    assert "聊天记录" in banner

    assert evidence_decay_banner([
        {"id": "c", "label": "伤情照片", "decay_risk": False, "active": True},
    ]) == ""


def test_missing_text_and_seed_question_are_single_shared_provider():
    state = _brawl_state(with_contact=True)
    report = evaluate_state_evidence(state)
    chat_coverage = coverage_for_rule(report, "criminal_chat_records")
    assert chat_coverage
    # 相关但尚未提供：unresolved 的缺失文案就是清单项本身。
    assert chat_coverage.missing_text == "与对方/在场人员的聊天记录"

    sufficiency = assess_decision_sufficiency(state)
    evidence_dim = next(
        dimension
        for dimension in sufficiency.dimensions
        if dimension.effect == "evidence_gap"
    )
    # 收敛方消费同一份 missing_text，不再各自硬编码文案。
    assert "与对方/在场人员的聊天记录" in evidence_dim.missing_information


def test_known_missing_missing_text_is_shared_between_evaluate_and_sufficiency():
    state = _brawl_state(with_contact=True, evidence_unavailable=["聊天记录"])
    report = evaluate_state_evidence(state)
    chat_coverage = coverage_for_rule(report, "criminal_chat_records")
    assert chat_coverage
    assert chat_coverage.status == "known_missing"
    assert chat_coverage.missing_text == "缺少与对方/在场人员的聊天记录"

    sufficiency = assess_decision_sufficiency(state)
    evidence_dim = next(
        dimension
        for dimension in sufficiency.dimensions
        if dimension.effect == "evidence_gap"
    )
    assert "缺少与对方/在场人员的聊天记录" in evidence_dim.missing_information
