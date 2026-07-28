"""验证法律维权置信度打分与分档（三维度前置版本）。"""
from src.agents.legal_guide.confidence import (
    score_confidence, tier_guidance, TIER_HIGH, TIER_MEDIUM,
)


def _score(**kw):
    base = dict(
        confirmed_issues=[], evidence_confirmed=[], evidence_total=4,
        domain_locked=False, region_known=False, time_known=False,
    )
    base.update(kw)
    return score_confidence(**base)


def test_full_signals_high_tier():
    """信号齐全 → HIGH 档。"""
    r = _score(
        confirmed_issues=["拖欠工资", "未签合同"],
        evidence_confirmed=["劳动合同", "工资流水", "考勤记录", "通讯记录"],
        evidence_total=4,
        domain_locked=True, region_known=True, time_known=True,
    )
    # 证据4/4=1.0*0.4=0.4; 事实0.1+0.1+0.1=0.3; 权责0.2+0.1=0.3; 总=1.0
    assert r["tier"] == "HIGH"
    assert r["score"] >= TIER_HIGH


def test_empty_signals_low_tier():
    """无任何信号 → LOW 档，分数为0。"""
    r = _score()
    assert r["tier"] == "LOW"
    assert r["score"] == 0.0


def test_partial_signals_medium_tier():
    """部分信号（有1个issue+领域锁定，但证据少）→ MEDIUM。"""
    r = _score(
        confirmed_issues=["消费欺诈"],
        evidence_confirmed=["订单截图"],
        evidence_total=4,
        domain_locked=True,
        region_known=True,
    )
    # 证据0.1 + 事实0.2 + 权责0.2 = 0.5，落在当前配置的 MEDIUM 区间。
    assert r["tier"] == "MEDIUM"
    assert TIER_MEDIUM <= r["score"] < TIER_HIGH


def test_weights_capped():
    """单维度得分不超过其权重上限。"""
    r = _score(
        confirmed_issues=["a", "b", "c", "d", "e"],  # 权责上限0.3
        evidence_confirmed=["1", "2", "3", "4", "5", "6"],
        evidence_total=4,  # 证据比例>1，但上限0.4
    )
    assert r["breakdown"]["evidence"] <= 0.40
    assert r["breakdown"]["rights_clarity"] <= 0.30


def test_score_monotonic_in_evidence():
    """证据越多分数不降低（单调性）。"""
    r1 = _score(evidence_confirmed=["a"], evidence_total=4)
    r2 = _score(evidence_confirmed=["a", "b", "c"], evidence_total=4)
    assert r2["score"] >= r1["score"]


def test_effective_evidence_weight_does_not_treat_claim_as_verified_copy():
    claimed = _score(
        evidence_confirmed=["劳动合同"],
        effective_evidence_count=0.45,
    )
    uploaded_copy = _score(
        evidence_confirmed=["劳动合同"],
        effective_evidence_count=0.70,
    )

    assert claimed["breakdown"]["evidence"] < uploaded_copy["breakdown"]["evidence"]
    assert uploaded_copy["breakdown"]["evidence"] < 0.40


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
