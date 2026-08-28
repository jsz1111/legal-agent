"""结论前"用户处境多角度审视"：把用户从"维权方"默认叙事中解放出来。

整个 pipeline 默认叙事是"用户=受害维权方"。当用户本人也可能成为被追责方
（互殴/防卫过当/自己撞人/违约/有过错）时，需要切换到"涉案当事人"框架。
本模块通过一次小 LLM 调用产出结构化判定（UserSituationVerdict），命中则
在结论提示词中强制切换叙事框架，并用确定性守卫兜底注入 ⚠️ 追责警示。

与 ``_adversarial_gap_scan``（followup_planner.py）同模式：
小 LLM 调用 → Pydantic 结构化判定 → 驱动行为。无关键词、无场景库，纯推理。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from src.agents.legal_guide.case_model import format_case_context
from src.agents.legal_guide.followup_planner import _json_content
from src.agents.legal_guide.llm_runtime import llm_for_stage
from src.agents.legal_guide.state import GuideState


USER_SITUATION_PROMPT = """你是法律咨询工作流中的"用户处境审视员"。

主流程默认把用户当作受害维权方。请你换几个独立立场——**办案/受理机关**、**对方当事人**、
**第三方旁观者**——审视用户本人的处境，判断用户是否也可能成为被追责方，或方案是否
依赖了未证实前提。你的职责不是重复维权分析，而是发现"用户本人会吃亏"的侧面。

## 结构化案情（每项都带用户原话；不得补写未出现的事实）
{case_context}

## 用户确认的法律问题
{confirmed_issues}

## 用户陈述的案情事实
{collected_facts}

## 已被识别的不利因素
{adverse_facts}

## 用户明确没有的证据
{evidence_unavailable}

请从四个角度审视：

1. **自身风险**：用户自己的行为是否可能使其成为被追责方——是否也动了手、谁先动手、
   防卫是否明显超过必要限度、是否参与、是否违约、是否侵权、是否构成犯罪要件。
   若存在，判定其性质（criminal 刑事追诉 / administrative 行政处罚 / civil_counter 民事反索赔），
   并给出触发依据（必须来自用户陈述的客观描述，不得臆测动机）。
2. **对方反索赔**：对方是否可能反过来向用户主张权利（赔偿、违约金、返还等）。
3. **前提风险**：用户的乐观预期或本方案依赖哪些未证实前提（"有车牌就能找到人"
   "对方会赔""监控拍到了""对方不会追究"等）。
4. **时间敏感**：是否涉及会消失或失效的证据（监控覆盖、聊天记录可删、伤情自愈）或程序期限。

只输出 JSON，不要输出任何其他内容：
{{
  "own_risk_level": "none或warning或high",
  "own_risk_kinds": ["criminal", "administrative", "civil_counter"],
  "reasons": ["触发依据，来自用户陈述的客观描述"],
  "counter_claim": true或false,
  "time_sensitive": true或false,
  "premise_risks": ["未证实前提"]
}}"""


class UserSituationVerdict(BaseModel):
    """结构化处境判定；解析失败时不降级，直接让调用方感知错误。"""

    own_risk_level: str = "none"  # none | warning | high
    own_risk_kinds: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    counter_claim: bool = False
    time_sensitive: bool = False
    premise_risks: list[str] = Field(default_factory=list)


_RISK_KIND_LABELS = {
    "criminal": "刑事追诉",
    "administrative": "行政处罚",
    "civil_counter": "对方民事反索赔",
}


def _risk_kinds_label(verdict: UserSituationVerdict) -> str:
    labels = [
        _RISK_KIND_LABELS[item]
        for item in verdict.own_risk_kinds
        if item in _RISK_KIND_LABELS
    ]
    return "、".join(dict.fromkeys(labels)) or "法律追责"


async def assess_user_situation(state: GuideState, llm: Any) -> UserSituationVerdict:
    """结论前评估用户自身处境；不设置超时，也不使用默认兜底。"""
    adverse = "\n".join(f"- {item}" for item in (state.adverse_facts or [])) or "（暂未识别）"
    prompt = USER_SITUATION_PROMPT.format(
        case_context=format_case_context(state.case_facts),
        confirmed_issues="、".join(state.confirmed_issues) or "暂未确认",
        collected_facts="；".join(state.collected_facts) or "暂未确认",
        adverse_facts=adverse,
        evidence_unavailable="、".join(state.evidence_unavailable) or "（无）",
    )
    response = await llm_for_stage(llm, max_tokens=400).ainvoke(
        [SystemMessage(content=prompt)]
    )
    return UserSituationVerdict.model_validate(_json_content(response.content))


_TIME_SENSITIVE_HINT = (
    "本案涉及会随时间消失的证据或程序期限。【行动清单】必须提示用户尽快调取/备份相关证据，"
    "电子记录类通常只保留数天至数周（以运营方为准），不得只依赖对方或办案机关。"
)


def situation_guidance(verdict: UserSituationVerdict) -> str:
    """渲染注入 CONCLUDE_PROMPT 的 {situation_guidance}，驱动结论 LLM 切换叙事框架。"""
    if verdict.own_risk_level == "none":
        return _normal_situation_guidance(verdict)
    guidance = _party_framework_guidance(verdict)
    if verdict.time_sensitive:
        # 时效提示是通用侧面，不随叙事框架分叉：涉案框架同样必须提示用户尽快
        # 调取/备份易消失证据（修掉此前只对普通框架提示的分支不一致）。
        guidance += "\n- " + _TIME_SENSITIVE_HINT
    return guidance


def _normal_situation_guidance(verdict: UserSituationVerdict) -> str:
    """普通维权方框架：仅当存在反索赔/前提/时间敏感侧面时追加显式处理要求。"""
    parts: list[str] = []
    if verdict.counter_claim:
        parts.append(
            "对方可能反过来向用户主张权利（反索赔）。【优势与劣势】的不利因素必须写明这一点，"
            "并在【行动清单】中提示用户保留能证明自身责任范围的材料。"
        )
    if verdict.time_sensitive:
        parts.append(_TIME_SENSITIVE_HINT)
    if verdict.premise_risks:
        parts.append(
            "方案依赖以下未证实前提：{premises}。每个前提都必须写进【优势与劣势】的不利因素，"
            "不得把乐观预期当作既定事实。".format(
                premises="、".join(verdict.premise_risks) or "见上文"
            )
        )
    if not parts:
        parts.append(
            "用户本人暂未被识别为追责对象，按普通维权方框架撰写；"
            "仍须按上方程序性准确性与证据规则执行。"
        )
    return "\n".join(f"- {part}" for part in parts)


def _party_framework_guidance(verdict: UserSituationVerdict) -> str:
    """涉案当事人框架：用户本人可能被追责时，强制以下硬性要求。"""
    kinds = _risk_kinds_label(verdict)
    reasons = "；".join(verdict.reasons) or "依据您的陈述"
    framework_rules = [
        "用 ⚠️ 醒目警示用户本人当前也可能处于法律风险中（{kinds}）：若办案机关不认定免责情形，"
        "用户可能面临不利后果。该警示必须放在方案最显著位置，不得弱化为一句轻描淡写的“不利因素”。",
        "【行动清单】第一步必须是：立即联系专业刑事/行政律师，在接受询问或讯问前先获得个案法律意见；"
        "不得只写拨打12348法律援助热线。",
        "明确提示：接受询问/讯问前谨慎陈述，用户的每一句话都可能成为对己不利的证据，"
        "应先与律师沟通再作陈述。",
        "提示用户尽快以书面形式向办案机关提交对自己有利的陈述与证据（如事发时间线、对方先动手的"
        "证据、伤情、证人联系方式等），抢占叙事先机。",
        "若本案伤情等已可能达到刑事立案标准，调解/和解不得写成用户可自主选择的路径：它属于刑事"
        "程序中的和解环节，是否启动由办案机关依程序决定。",
        "【优势与劣势】必须把“用户本人被追责风险”列为最突出的不利因素，不得遗漏。",
        "全文语气偏保守，禁止乐观承诺（如“问题不大”“正常不会被追究”“只要……就不会有事”）。",
    ]
    body = "\n".join(
        f"{index}. {rule.format(kinds=kinds)}"
        for index, rule in enumerate(framework_rules, start=1)
    )
    return (
        "⚠️ 用户本人可能成为本案被追责对象（{kinds}）。本方案必须切换到**涉案当事人框架**，"
        "严格遵守以下硬性要求：\n{body}\n触发依据：{reasons}"
    ).format(kinds=kinds, body=body, reasons=reasons)


def _build_liability_warning_block(verdict: UserSituationVerdict) -> str:
    """确定性⚠️块：一次覆盖三个硬伤（本人追责警示/联系律师/谨慎陈述/书面陈述/调解纠偏）。"""
    kinds = _risk_kinds_label(verdict)
    reasons = "；".join(verdict.reasons) or "依据您的陈述"
    return (
        "> ⚠️ **重要风险提示：您本人也可能面临追责**\n"
        "> 结合您的陈述，您本人目前也可能处于法律风险之中（{kinds}）——若办案机关不认定免责情形，\n"
        "> 您可能面临行政处罚甚至刑事追诉。\n"
        "> 触发依据：{reasons}\n"
        "> - **立即联系专业刑事/行政律师**（不要只依赖法律援助热线），在接受询问/讯问前先获得个案意见。\n"
        "> - **接受询问/讯问前谨慎陈述**：您的每一句话都可能成为对您不利的证据，请先与律师沟通再作陈述。\n"
        "> - **尽快以书面形式向办案机关提交对自己有利的陈述和证据**（事发时间线、对方先动手的证据、您的伤情等），抢占叙事先机。\n"
        "> - **关于“调解/和解”**：若伤情等已可能达到刑事立案标准，调解不是您可自主选择的路径，\n"
        ">   而是刑事程序中的和解环节，是否启动由办案机关依程序决定。"
    ).format(kinds=kinds, reasons=reasons)


def _ensure_risk_insights(reply: str, verdict: UserSituationVerdict) -> str:
    """确定性守卫：追责场景且回复未覆盖本人追责+律师建议时，顶部前置⚠️块。

    普通案件（own_risk_level=none）原样返回，避免误伤正常维权方框架。
    """
    if verdict.own_risk_level == "none":
        return reply
    if ("追责" in reply and "律师" in reply) or "被追诉" in reply:
        return reply
    return _build_liability_warning_block(verdict) + "\n\n---\n\n" + reply
