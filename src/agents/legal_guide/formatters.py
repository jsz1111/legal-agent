"""纯格式化工具函数（无 IO / LLM 依赖，便于单测）。"""
from __future__ import annotations

from src.agents.legal_guide.evidence_rules import (
    EvidenceChecklist,
    resolve_evidence_checklist,
)


def fmt_channels(channels: list[dict]) -> str:
    """将渠道格式化为有顺序、理由、材料和来源的行动清单。"""
    if not channels:
        return "（暂无检索到具体渠道，建议拨打12348法律援助热线）"
    lines = []
    for index, c in enumerate(channels[:6], start=1):
        name = c.get("name", "")
        phone = c.get("phone", "")
        url = c.get("url", "")
        stage = c.get("route_stage", "办理渠道")
        lines.append(f"{index}. **{name}**（{stage}）")
        if reason := c.get("recommendation_reason"):
            lines.append(f"   - 推荐理由：{reason}")
        contacts = []
        if phone:
            contacts.append(f"电话：{phone}")
        if url:
            contacts.append(f"官方入口：{url}")
        if contacts:
            lines.append("   - " + "；".join(contacts))
        matters = [str(item) for item in (c.get("applicable_matters") or []) if item]
        if matters:
            lines.append("   - 适用事项：" + "；".join(matters[:3]))
        materials = [str(item) for item in (c.get("required_materials") or []) if item]
        if materials:
            lines.append("   - 先准备：" + "；".join(materials[:4]))
        if hours := c.get("service_hours"):
            lines.append(f"   - 办理时间：{hours}")
        source = c.get("source_org") or ""
        source_url = c.get("source_url") or ""
        verified = c.get("last_verified_on") or ""
        if source or source_url:
            source_text = source
            if source_url and source_url != url:
                source_text += f"（{source_url}）"
            if verified:
                source_text += f"，核验于{verified}"
            lines.append(f"   - 信息来源：{source_text}")
    return "\n".join(lines)


def fmt_evidence_checklist(
    checklist_or_domain: EvidenceChecklist | str,
    *context_parts: str | list[str],
) -> str:
    """Format one resolved checklist while keeping legacy domain calls valid."""
    checklist = (
        checklist_or_domain
        if isinstance(checklist_or_domain, EvidenceChecklist)
        else resolve_evidence_checklist(checklist_or_domain, *context_parts)
    )
    return "\n".join(f"  - {item}" for item in checklist.items)


_DOC_REQUEST_KEYWORDS = frozenset([
    "生成文书", "写文书", "帮我写", "申请书", "投诉信", "律师函",
    "起草", "文书", "起诉状", "协议书",
    "导出方案", "导出",
])

# 对「需要参考文书？」邀请的肯定性短回复
_DOC_AFFIRMATIVES = frozenset([
    "需要", "要", "好", "好的", "可以", "是", "是的",
    "帮我生成", "帮我", "生成", "生成吧", "生成一下", "麻烦了", "请生成",
])


def is_doc_request(msg: str) -> bool:
    """判断用户消息是否在请求生成参考文书（含关键词匹配或短肯定词）。"""
    msg = msg.strip()
    if any(kw in msg for kw in _DOC_REQUEST_KEYWORDS):
        return True
    # 短消息（≤10字）且是常见肯定词，视为对文书生成邀请的确认
    if len(msg) <= 10 and msg in _DOC_AFFIRMATIVES:
        return True
    return False


def requested_doc_type(msg: str, default: str) -> str:
    """从用户原话识别明确指定的文书类型；普通“生成文书”沿用阶段默认值。"""
    text = str(msg or "").strip()
    mappings = (
        (("劳动仲裁申请书",), "劳动仲裁申请书"),
        (("民事起诉状", "起诉状", "起诉书"), "民事起诉状"),
        (("催告函", "催款函"), "催告函"),
        (("律师函",), "律师函参考稿"),
        (("消费者投诉信", "投诉信"), "消费者投诉信"),
        (("行政复议申请书", "行政复议"), "行政复议申请书"),
        (("报案材料", "报案书"), "报案材料"),
        (("离婚协议书",), "离婚协议书（草稿）"),
        (("侵权警告函", "警告函"), "侵权警告函"),
    )
    for keywords, doc_type in mappings:
        if any(keyword in text for keyword in keywords):
            return doc_type
    if "仲裁申请书" in text:
        # “仲裁申请书”既可能指劳动仲裁，也可能指商事仲裁；由当前领域默认值消歧。
        return default if "仲裁申请书" in default else "仲裁申请书"
    return default
