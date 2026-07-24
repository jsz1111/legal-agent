"""法律维权置信度打分：五维加权 → 0~1 分数 → HIGH/MEDIUM/LOW 三档。

用于维权助手（legal_guide）的分级收敛输出：分数越高，输出越明确、可直接执行；
分数低则转为谨慎建议并强烈引导咨询专业律师。规则式打分（确定性、零额外 LLM 调用、可单测）。
"""
from __future__ import annotations

# ── 各维度权重（合计 1.0）────────────────────────────────────────────────
W_LEGAL_BASIS = 0.40   # 法律依据充分度（最核心）
W_FACT_CLARITY = 0.25  # 事实清晰度
W_EVIDENCE = 0.20      # 证据完备度
W_CASE = 0.10          # 类案支撑
W_REGION = 0.05        # 地区明确度

# ── 分档阈值 ───────────────────────────────────────────────────────────
TIER_HIGH = 0.70
TIER_MEDIUM = 0.40


def score_confidence(
    confirmed_issues: list[str],
    evidence_confirmed: list[str],
    candidate_laws: list[dict],
    milvus_hit: bool,
    case_hit: bool,
    domain_locked: bool,
    region_known: bool,
) -> dict:
    """计算维权方案置信度。

    Returns:
        {
            "score": float,      # 0~1 总分
            "tier": str,         # HIGH / MEDIUM / LOW
            "breakdown": dict,   # 各维度得分（便于日志与调试）
        }
    """
    # 1. 法律依据：milvus 命中 + Neo4j 法条数量
    basis = 0.0
    if milvus_hit:
        basis += 0.25
    basis += min(len(candidate_laws) * 0.05, 0.15)
    basis = min(basis, W_LEGAL_BASIS)

    # 2. 事实清晰度：领域锁定 + 已标准化问题数
    fact = (0.10 if domain_locked else 0.0)
    fact += min(len(confirmed_issues) * 0.05, 0.15)
    fact = min(fact, W_FACT_CLARITY)

    # 3. 证据完备度
    evidence = min(len(evidence_confirmed) * 0.07, W_EVIDENCE)

    # 4. 类案支撑
    case = W_CASE if case_hit else 0.0

    # 5. 地区明确度
    region = W_REGION if region_known else 0.0

    score = round(basis + fact + evidence + case + region, 3)

    if score >= TIER_HIGH:
        tier = "HIGH"
    elif score >= TIER_MEDIUM:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return {
        "score": score,
        "tier": tier,
        "breakdown": {
            "legal_basis": round(basis, 3),
            "fact_clarity": round(fact, 3),
            "evidence": round(evidence, 3),
            "case_support": round(case, 3),
            "region": round(region, 3),
        },
    }


# ── 各档位对结论生成的引导语 ────────────────────────────────────────────
_TIER_GUIDANCE = {
    "HIGH": (
        "【置信度：高】法律依据与事实较为充分。请给出明确、可直接执行的完整维权方案，"
        "路径推荐要具体（首选哪条、为什么），语气笃定但仍提示个案差异。"
    ),
    "MEDIUM": (
        "【置信度：中】依据或事实存在一定缺口。可给出维权方案，但需在关键判断处标注"
        "“需进一步核实”，并说明还需补充哪些信息才能更准确。"
    ),
    "LOW": (
        "【置信度：低】法律依据或案情信息不足。请以谨慎的初步指引为主，避免给出确定性结论，"
        "明确提示信息有限，并强烈建议拨打 12348 或咨询专业律师后再行动。"
    ),
}


def tier_guidance(tier: str) -> str:
    """返回对应档位、注入 CONCLUDE_PROMPT 的引导语。"""
    return _TIER_GUIDANCE.get(tier, _TIER_GUIDANCE["MEDIUM"])
