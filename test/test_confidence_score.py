"""验证法律维权置信度打分与分档。"""
from src.agents.legal_guide.confidence import (
    score_confidence, tier_guidance, TIER_HIGH, TIER_MEDIUM,
)


def _score(**kw):
    base = dict(
        confirmed_issues=[], evidence_confirmed=[], candidate_laws=[],
        milvus_hit=False, case_hit=False, domain_locked=False, region_known=False,
    )
    base.update(kw)
    return score_confidence(**base)


def test_full_signals_high_tier():
    """信号齐全 → HIGH 档，分数接近上限。"""
    r = _score(
        confirmed_issues=["拖欠工资", "未签合同", "未缴社保"],
        evidence_confirmed=["劳动合同", "工资流水", "考勤记录"],
        candidate_laws=[{}, {}, {}],
        milvus_hit=True, case_hit=True, domain_locked=True, region_known=True,
    )
    assert r["tier"] == "HIGH"
    assert r["score"] >= TIER_HIGH
    assert r["score"] <= 1.0


def test_empty_signals_low_tier():
    """无任何信号 → LOW 档，分数为0。"""
    r = _score()
    assert r["tier"] == "LOW"
    assert r["score"] == 0.0


def test_partial_signals_medium_tier():
    """部分信号（命中法条+锁定领域，但无证据无类案）→ 落在 MEDIUM。"""
    r = _score(
        confirmed_issues=["消费欺诈"],
        candidate_laws=[{}, {}],
        milvus_hit=True, domain_locked=True,
    )
    # 0.25+0.10(基) +0.10+0.05(事实) = 0.50
    assert r["tier"] == "MEDIUM"
    assert TIER_MEDIUM <= r["score"] < TIER_HIGH


def test_weights_capped():
    """单维度得分不超过其权重上限（法条数很多也不溢出）。"""
    r = _score(
        candidate_laws=[{}] * 20,
        evidence_confirmed=["a", "b", "c", "d", "e", "f", "g"],
        milvus_hit=True,
    )
    assert r["breakdown"]["legal_basis"] <= 0.40
    assert r["breakdown"]["evidence"] <= 0.20


def test_score_monotonic_in_evidence():
    """证据越多分数不降低（单调性）。"""
    r1 = _score(evidence_confirmed=["a"])
    r2 = _score(evidence_confirmed=["a", "b", "c"])
    assert r2["score"] >= r1["score"]


def test_tier_guidance_distinct():
    """三档引导语各不相同且非空。"""
    g = {t: tier_guidance(t) for t in ("HIGH", "MEDIUM", "LOW")}
    assert len(set(g.values())) == 3
    assert all(g.values())
    assert "12348" in g["LOW"]


if __name__ == "__main__":
    test_full_signals_high_tier()
    test_empty_signals_low_tier()
    test_partial_signals_medium_tier()
    test_weights_capped()
    test_score_monotonic_in_evidence()
    test_tier_guidance_distinct()
    print("ALL PASS")
