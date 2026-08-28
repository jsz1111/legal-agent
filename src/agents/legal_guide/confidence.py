"""法律维权置信度打分：三维加权（打分前置、零 I/O）→ 0~1 分数 → HIGH/MEDIUM/LOW 三档。

用于维权助手（legal_guide）的分级收敛输出：分数越高，输出越明确、可直接执行；
分数低则转为谨慎建议并强烈引导咨询专业律师。规则式打分（确定性、零额外 LLM 调用、可单测）。

改造说明：打分挪到 RAG 检索之前，去掉 milvus_hit/case_hit 依赖，改为纯前置三维度。
"""
from __future__ import annotations

from src.core.config import get_settings

settings = get_settings()

# ── 各维度权重（合计 1.0）────────────────────────────────────────────────
W_EVIDENCE = 0.40      # 证据完整度（最核心）
W_FACT_CLARITY = 0.30  # 事实清晰度
W_RIGHTS_CLARITY = 0.30 # 权责清晰度

# ── 分档阈值（从配置读取）───────────────────────────────────────────────
TIER_HIGH = settings.GUIDE_CONFIDENCE_HIGH
TIER_MEDIUM = settings.GUIDE_CONFIDENCE_MID


def score_confidence(
    confirmed_issues: list[str],
    evidence_confirmed: list[str],
    evidence_total: int,
    domain_locked: bool,
    region_known: bool,
    time_known: bool,
    effective_evidence_count: float | None = None,
) -> dict:
    """计算维权方案置信度（纯前置、零 I/O）。

    Args:
        confirmed_issues: 已标准化的法律问题列表
        evidence_confirmed: 用户已确认的证据列表
        evidence_total: 该领域证据清单要求的总数（调用方传入，避免循环 import）
        effective_evidence_count: 按“上传副本/用户称持有/冲突”折算的方案准备度。
            为空时沿用 evidence_confirmed 数量，兼容旧调用；不代表法律证明力。
        domain_locked: 是否已锁定法律领域
        region_known: 是否已知用户地区
        time_known: 是否已知时间信息（影响时效判断）

    Returns:
        {
            "score": float,      # 0~1 总分
            "tier": str,         # HIGH / MEDIUM / LOW
            "breakdown": dict,   # 各维度得分（便于日志与调试）
        }
    """
    # 1. 证据完整度：用户已提供 / 清单要求总数
    if evidence_total > 0:
        count = len(evidence_confirmed) if effective_evidence_count is None else effective_evidence_count
        evidence_ratio = max(count, 0.0) / evidence_total
        evidence = min(evidence_ratio, 1.0) * W_EVIDENCE
    else:
        evidence = 0.0

    # 2. 事实清晰度：领域 + 地区 + 时间
    fact = 0.0
    if domain_locked:
        fact += 0.10
    if region_known:
        fact += 0.10
    if time_known:
        fact += 0.10
    fact = min(fact, W_FACT_CLARITY)

    # 3. 权责清晰度：标准化问题数量（反映法律关系是否明确）
    rights = 0.0
    if len(confirmed_issues) >= 1:
        rights += 0.20
    if len(confirmed_issues) >= 2:
        rights += 0.10
    rights = min(rights, W_RIGHTS_CLARITY)

    score = round(evidence + fact + rights, 3)

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
            "evidence": round(evidence, 3),
            "fact_clarity": round(fact, 3),
            "rights_clarity": round(rights, 3),
        },
    }


# ── 各档位对结论生成的引导语 ────────────────────────────────────────────
_TIER_GUIDANCE = {
    "HIGH": (
        "【置信度：高 - 方案可直接参考】\n"
        "法律依据与事实充分。请严格按以下顺序输出完整五段式维权方案：\n"
        "1. 【理解您的情况】一句共情\n"
        "2. 【法律依据】从检索结果中逐字引用，格式：《法律全称》第X条：[原文关键内容]\n"
        "3. 【类似案例参考】有则列1-2条，无则省略\n"
        "4. 【维权路径比较】2-3种方案，标注[免费]/[收费]、时长、优缺点，注明推荐方案\n"
        "5. 【优势与劣势】（必须输出，只列有利与不利两方面，不估计胜算等级、不输出百分比）：\n"
        "   - 有利因素：检索法条中哪些条文支持用户（引用具体条文名）\n"
        "   - 不利因素：用户自身行为或缺失证据中哪些可能被对方援引反驳，以及举证责任分析；"
        "若方案依赖未证实前提（如对方身份未核实、仅凭车牌/账号线索），必须写明该前提风险\n"
        "6. 【行动清单】证据清单 + 具体步骤 + 渠道\n"
        "语气明确可执行，在方案末尾注明：**本方案基于您提供的信息，可直接参考执行。个案有差异，关键决策前建议咨询专业律师确认细节。**"
    ),
    "MEDIUM": (
        "【置信度：中 - 方案可参考，需补充信息】\n"
        "基本事实清楚，可给出维权方案，但信息有缺口。请严格按以下顺序输出：\n"
        "1. 【理解您的情况】一句共情\n"
        "2. 【法律依据】从检索结果中逐字引用，格式：《法律全称》第X条：[原文关键内容]；"
        "检索结果不足时，说明'当前未检索到完整适用条文，以下为相关原则'\n"
        "3. 【类似案例参考】有则列，无则省略\n"
        "4. 【维权路径比较】2-3种方案，标注[免费]/[收费]、时长、优缺点\n"
        "5. 【优势与劣势】（必须输出，只列有利与不利两方面，不估计胜算等级、不输出百分比）：\n"
        "   - 有利因素：检索法条中哪些条文支持用户\n"
        "   - 不利因素：用户自身行为或证据缺口可能被对方援引的地方，以及举证责任；"
        "信息不足处注明'需补充XXX才能判断'\n"
        "6. 【行动清单】证据清单 + 步骤 + 渠道\n"
        "在方案开头告知：**本方案基于当前信息可供参考，但以下关键信息仍需补充：[列出缺失项]，补充后方案会更准确。**"
    ),
    "LOW": (
        "【置信度：低 - 仅供初步参考，不建议直接执行】\n"
        "信息严重不足，无法给出可执行方案。请严格按以下顺序输出：\n"
        "1. **在开头明确警示**：⚠️ **重要提示：当前信息有限，以下仅为初步指引，不建议直接执行。**\n"
        "2. 检索到的相关法律依据（让用户了解涉及哪些法律）\n"
        "3. **【维权路径比较】**（列出1-2条可执行路径，如劳动仲裁/行政投诉）\n"
        "4. **【优势与劣势】**（只列有利与不利两方面，不估计胜算等级、不输出百分比）：\n"
        "   - 有利因素：哪些法条原则上支持用户\n"
        "   - 不利因素：用户自身存在哪些风险点（如证据缺失、可能超时效等），以及举证责任\n"
        "5. **关键缺失信息清单**：明确列出还需补充哪些信息才能给出可执行方案\n"
        "6. **强烈建议**：\n"
        "   - 请补充上述关键信息，我将重新为您分析\n"
        "   - 或拨打 **12348** 法律援助热线咨询专业律师\n"
        "   - 切勿仅凭此初步指引直接行动，以免影响维权效果"
    ),
}


def tier_guidance(tier: str) -> str:
    """返回对应档位、注入 CONCLUDE_PROMPT 的引导语。"""
    return _TIER_GUIDANCE.get(tier, _TIER_GUIDANCE["MEDIUM"])
