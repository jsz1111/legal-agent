"""多模态工具：图片理解、OCR 等（可选功能，需要 API key）"""
from __future__ import annotations

import base64
from pathlib import Path
from loguru import logger
from langchain_core.messages import HumanMessage

from src.core.config import get_settings

settings = get_settings()


def is_multimodal_enabled() -> bool:
    """检查是否启用多模态功能"""
    return settings.ENABLE_MULTIMODAL and bool(settings.VL_API_KEY)


def build_context_aware_question(
    legal_domain: str = "",
    confirmed_issues: list[str] = None,
    evidence_confirmed: list[str] = None,
    evidence_unavailable: list[str] = None,
    recent_assistant_message: str = ""
) -> str:
    """
    根据对话上下文构建针对性的图片分析提示词。

    Args:
        legal_domain: 法律领域（如 consumer_market, labor_social_security）
        confirmed_issues: 已识别的法律问题
        evidence_confirmed: 已确认的证据
        evidence_unavailable: 明确缺失的证据
        recent_assistant_message: 助手最近一条消息（可能包含追问内容）

    Returns:
        针对性的图片分析提示词
    """
    confirmed_issues = confirmed_issues or []
    evidence_confirmed = evidence_confirmed or []
    evidence_unavailable = evidence_unavailable or []

    # 基础提示词
    base_prompt = "请详细分析这张图片中与法律证据相关的内容。\n\n"

    # 根据领域添加针对性提示
    domain_hints = {
        "consumer_market": "重点提取：订单信息（订单号、商品名称、价格、商家名称）、商品瑕疵描述、与描述不符之处、防伪码、聊天记录内容、退换货政策。",
        "labor_social_security": "重点提取：劳动合同条款、工资金额、工作时间、公司名称、签订日期、工资条明细、打卡记录、解除通知内容。",
        "contracts_property_housing": "重点提取：合同条款、违约条款、金额、甲乙双方信息、签订日期、房屋地址、租金、押金、物业费。",
        "traffic_personal_injury": "重点提取：事故认定书内容、责任划分、车辆信息、事故时间地点、医疗费用、诊断证明内容。",
    }

    if legal_domain in domain_hints:
        base_prompt += domain_hints[legal_domain] + "\n\n"

    # 根据已识别问题添加针对性提示
    if confirmed_issues:
        issues_str = "、".join(confirmed_issues[:3])
        base_prompt += f"用户当前遇到的法律问题：{issues_str}。请重点提取与这些问题相关的证据信息。\n\n"

    # 根据缺失证据添加针对性提示
    if evidence_unavailable:
        missing_str = "、".join(evidence_unavailable[:3])
        base_prompt += f"用户明确缺少的证据：{missing_str}。如果图片中包含这些信息，请特别标注。\n\n"

    # 根据助手最近的追问内容添加针对性提示
    if recent_assistant_message and len(recent_assistant_message) < 500:
        if "订单" in recent_assistant_message or "购买凭证" in recent_assistant_message:
            base_prompt += "重点：这可能是订单截图或购买凭证，请提取订单号、商品信息、价格、商家信息。\n\n"
        elif "聊天" in recent_assistant_message or "沟通" in recent_assistant_message:
            base_prompt += "重点：这可能是聊天记录截图，请提取对话内容、商家回复、沟通时间。\n\n"
        elif "商品" in recent_assistant_message or "照片" in recent_assistant_message:
            base_prompt += "重点：这可能是商品照片，请描述商品状态、瑕疵细节、与描述不符之处。\n\n"
        elif "合同" in recent_assistant_message or "协议" in recent_assistant_message:
            base_prompt += "重点：这可能是合同文件，请提取关键条款、双方信息、签订日期、违约责任。\n\n"

    base_prompt += """输出格式：
1. 图片类型：[订单截图/商品照片/聊天记录/合同文件/支付凭证/其他]
2. 关键信息：[逐条列出提取到的关键证据]
3. 法律相关性：[说明这些信息如何支持用户的维权诉求]

保持客观，只陈述图片中能看到的内容，不要推测或添加主观判断。"""

    return base_prompt


async def analyze_image(
    image_path: str,
    question: str = None,
    legal_domain: str = "",
    confirmed_issues: list[str] = None,
    evidence_confirmed: list[str] = None,
    evidence_unavailable: list[str] = None,
    recent_assistant_message: str = ""
) -> str:
    """
    分析图片内容（需要阿里云 qwen-vl API key）。

    Args:
        image_path: 图片文件路径
        question: 对图片的提问（如果为 None，则根据上下文自动生成）
        legal_domain: 法律领域
        confirmed_issues: 已识别的法律问题
        evidence_confirmed: 已确认的证据
        evidence_unavailable: 缺失的证据
        recent_assistant_message: 助手最近一条消息

    Returns:
        图片内容描述，如果多模态未启用或失败则返回友好提示
    """
    if not is_multimodal_enabled():
        return "⚠️ 图片理解功能未启用。请在配置文件中设置 VL_API_KEY 和 ENABLE_MULTIMODAL=true"

    # 如果没有提供 question，根据上下文自动生成
    if question is None:
        question = build_context_aware_question(
            legal_domain=legal_domain,
            confirmed_issues=confirmed_issues,
            evidence_confirmed=evidence_confirmed,
            evidence_unavailable=evidence_unavailable,
            recent_assistant_message=recent_assistant_message
        )

    try:
        # 阿里云 DashScope SDK
        from dashscope import MultiModalConversation

        # 读取图片并转为 base64
        image_file = Path(image_path)
        if not image_file.exists():
            return f"❌ 图片文件不存在：{image_path}"

        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 调用阿里云多模态 API
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_data}"},
                    {"text": question}
                ]
            }
        ]

        response = MultiModalConversation.call(
            model=settings.VL_MODEL,
            messages=messages,
            api_key=settings.VL_API_KEY,
        )

        if response.status_code == 200:
            result = response.output.choices[0].message.content
            logger.info(f"图片分析成功: {image_path}, 使用上下文={bool(legal_domain or confirmed_issues)}")
            return result
        else:
            logger.error(f"图片分析失败: {response.message}")
            return f"❌ 图片分析失败：{response.message}"

    except ImportError:
        logger.warning("dashscope 库未安装，请运行: pip install dashscope")
        return "⚠️ 多模态功能需要安装 dashscope 库：pip install dashscope"
    except Exception as e:
        logger.error(f"图片分析异常: {e}")
        return f"❌ 图片分析失败：{str(e)}"


async def extract_text_from_image(image_path: str) -> str:
    """
    从图片中提取文字（OCR）。

    Args:
        image_path: 图片文件路径

    Returns:
        提取的文字内容
    """
    return await analyze_image(
        image_path,
        question="请提取图片中的所有文字内容，保持原有格式和顺序。"
    )
