"""check_convergence：判断法律指引是否可以收敛输出结论。"""
from __future__ import annotations


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
