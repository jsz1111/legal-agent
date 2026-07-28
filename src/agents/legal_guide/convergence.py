"""check_convergence：判断法律指引是否可以收敛输出结论。"""
from __future__ import annotations

from src.agents.legal_guide.evidence_rules import resolve_state_evidence_checklist


def check_convergence(
    laws: list[dict],
    domain: str,
    current_round: int,
    max_rounds: int = 4,
    milvus_hit: bool = False,
) -> tuple[bool, bool]:
    """
    判断是否可以输出行动方案。

    Returns:
        (should_conclude, force_conclude)
        should_conclude : True = 可以输出结论
        force_conclude  : True = 因达到轮次上限被迫结束（输出中加兜底说明）
    """
    # 达到轮次上限，强制结束
    if current_round >= max_rounds:
        return True, True

    # Milvus 检索到法条 → 有具体法律依据，可以给出行动方案（不强求 domain 已锁定）
    if milvus_hit:
        return True, False

    # domain 已锁定且 Neo4j 检索到足够法条
    if domain and len(laws) >= 3:
        return True, False

    # 有足够 Neo4j 法条（即使 domain 未锁定）
    if len(laws) >= 5:
        return True, False

    # 多轮后仍无任何相关法条 → 兜底收敛（降低到1轮，避免无限追问）
    if not laws and not milvus_hit and current_round >= 1:
        return True, True

    return False, False


def should_conclude(state, max_rounds: int = 12) -> tuple[bool, bool]:
    """新版收敛判断：基于 total_rounds + confidence_tier + evidence_ratio。

    Args:
        state: GuideState
        max_rounds: 最大总轮次（澄清+追问细节+追问证据）

    Returns:
        (should_stop, force_conclude)
        should_stop: 是否停止追问，进入 conclude
        force_conclude: 是否因轮次上限强制收敛
    """
    # 1. 轮次上限（强制收敛）
    if state.total_rounds >= max_rounds:
        return True, True

    # 2. 用户主动要求结论（预留字段，当前未实现）
    if hasattr(state, 'wants_conclude') and state.wants_conclude:
        return True, False

    # 3. 置信度判断
    tier = state.confidence_tier
    fact_confidence = state.confidence_score

    # HIGH 档：法律事实清晰 + 有一定证据 → 可以收敛
    if tier == "HIGH" and len(state.evidence_confirmed) >= 1:
        return True, False

    # MEDIUM 档：法律事实基本清楚，证据覆盖 40% 以上 → 可以收敛
    if tier == "MEDIUM":
        evidence_tpl = resolve_state_evidence_checklist(state).items
        evidence_ratio = len(state.evidence_confirmed) / max(len(evidence_tpl), 3)
        if evidence_ratio >= 0.4:
            return True, False

    # LOW 档：继续追问（除非达到轮次上限）
    return False, False
