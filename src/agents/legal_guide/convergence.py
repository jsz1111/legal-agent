"""Decision-sufficiency convergence for the legal guide."""
from __future__ import annotations

from src.agents.legal_guide.decision_sufficiency import (
    DecisionSufficiencyReport,
    assess_decision_sufficiency,
)

def should_conclude(state, max_rounds: int = 12) -> tuple[bool, bool]:
    """Decide convergence from legal decision sufficiency and hard stops.

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

    raw_report = getattr(state, "decision_sufficiency", {}) or {}
    report = (
        DecisionSufficiencyReport.model_validate(raw_report)
        if raw_report
        else assess_decision_sufficiency(state)
    )
    if report.sufficient_for_definitive_plan:
        return True, False

    return False, False
