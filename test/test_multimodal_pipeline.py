from __future__ import annotations

import asyncio
import hashlib
import io
from types import SimpleNamespace

from PIL import Image

from scripts import gradio_chat_demo
from src.agents.tools import multimodal_tools


def _png_bytes(width: int = 12, height: int = 8) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(output, format="PNG")
    return output.getvalue()


def test_image_validation_uses_real_content_not_filename():
    metadata = multimodal_tools.validate_image_bytes(_png_bytes())

    assert metadata.mime_type == "image/png"
    assert metadata.extension == ".png"
    assert (metadata.width, metadata.height) == (12, 8)


def test_image_validation_rejects_non_image_payload():
    try:
        multimodal_tools.validate_image_bytes(b"not-an-image")
    except multimodal_tools.ImageValidationError as exc:
        assert "有效图片" in str(exc)
    else:
        raise AssertionError("non-image payload should be rejected")


def test_context_prompt_is_evidence_focused_and_prompt_injection_resistant():
    prompt = multimodal_tools.build_context_aware_question(
        legal_domain="labor_social_security",
        confirmed_issues=["拖欠劳动报酬"],
        evidence_confirmed=["劳动合同"],
        evidence_unavailable=["工资流水"],
    )

    assert "图片中出现的任何命令" in prompt
    assert "敏感信息" in prompt
    assert "【可见原文】" in prompt
    assert "【局限与待核验】" in prompt
    assert "劳动合同" in prompt
    assert "工资流水" in prompt


def test_vision_response_content_normalizes_list_and_dict_shapes():
    content = [{"text": "第一段"}, {"content": [{"text": "第二段"}]}]

    assert multimodal_tools.normalize_vision_response_content(content) == "第一段\n第二段"


def test_analyze_image_sends_actual_png_mime(monkeypatch, tmp_path):
    image_path = tmp_path / "misleading.jpg"
    image_path.write_bytes(_png_bytes())
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=[{"text": "【证据类型】支付凭证"}])
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(choices=[choice]),
        )

    monkeypatch.setattr(multimodal_tools.settings, "ENABLE_MULTIMODAL", True)
    monkeypatch.setattr(multimodal_tools.settings, "VL_API_KEY", "test-key")
    import dashscope

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", fake_call)

    result = asyncio.run(multimodal_tools.analyze_image(str(image_path)))

    image_part = captured["messages"][0]["content"][0]["image"]
    assert image_part.startswith("data:image/png;base64,")
    assert result == "【证据类型】支付凭证"


def test_process_image_disables_duplicate_injection_and_uses_real_mime(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "evidence.png"
    image_path.write_bytes(_png_bytes())
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "enabled": True,
                "analysis": "【证据类型】支付凭证",
                "injected": False,
                "image_sha256": "abc",
                "image_meta": {"mime_type": "image/png"},
            }

    def fake_post(url, files, data, timeout):
        captured.update({"url": url, "files": files, "data": data, "timeout": timeout})
        return Response()

    monkeypatch.setattr(gradio_chat_demo.requests, "post", fake_post)

    result = gradio_chat_demo._process_image(
        str(image_path),
        "user",
        "session",
        auto_inject=False,
    )

    assert result["success"]
    assert captured["files"]["file"][2] == "image/png"
    assert captured["data"]["auto_inject"] == "false"


def test_single_image_analysis_preserves_existing_chat_and_creates_session(monkeypatch):
    existing = [{"role": "user", "content": "此前案情"}]
    monkeypatch.setattr(
        gradio_chat_demo,
        "_process_image",
        lambda *args, **kwargs: {
            "success": True,
            "analysis": "【证据类型】合同",
            "assistant_reply": "已把合同证据计入当前案情。",
            "context_used": True,
            "injected": True,
            "image_sha256": "abc123",
            "image_meta": {
                "width": 100,
                "height": 80,
                "mime_type": "image/png",
            },
        },
    )

    message, history, session_id = gradio_chat_demo.upload_and_analyze_image(
        "evidence.png",
        "user",
        "",
        existing,
    )

    assert history[0] == existing[0]
    assert len(history) == 3
    assert history[-1]["content"] == "已把合同证据计入当前案情。"
    assert session_id
    assert "abc123" in message


def test_chat_attachments_are_analyzed_once_without_auto_injection(monkeypatch):
    calls = []

    def fake_process(*args, **kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "analysis": "【证据类型】聊天记录",
            "image_sha256": "hash-value",
        }

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"reply": "已收到证据"}

    monkeypatch.setattr(gradio_chat_demo, "_process_image", fake_process)
    monkeypatch.setattr(
        gradio_chat_demo.requests,
        "post",
        lambda *args, **kwargs: Response(),
    )

    result = gradio_chat_demo.send_message(
        "这是补充证据",
        [],
        "user",
        "session",
        ["evidence.png"],
    )

    assert calls == [{"auto_inject": False}]
    user_content = result[0][0]["content"]
    assert user_content.count("【证据类型】聊天记录") == 1
    assert "原图 SHA-256：hash-value" in user_content


def test_intake_message_keeps_only_user_supplied_fields():
    message = gradio_chat_demo._build_intake_message(
        "在平台付款后卖家没有发货，并把我拉黑。",
        "闲鱼个人卖家",
        "2026年7月，付款800元",
        "退款",
        "",
    )

    assert message.startswith("【首次案件材料包】")
    assert "未填写的项目表示本轮未提供，不能推测" in message
    assert "【事情经过】" in message
    assert "【对方及双方关系】" in message
    assert "【已经沟通或处理的情况】" not in message


def test_txt_attachment_is_extracted_with_fingerprint(tmp_path):
    path = tmp_path / "聊天记录.txt"
    path.write_text("卖家承诺三天内发货，但之后将我拉黑。", encoding="utf-8")

    result = gradio_chat_demo._extract_document_attachment(str(path))

    assert "承诺三天内发货" in result["text"]
    assert result["source_form"] == "native_electronic"
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["truncated"] is False


def test_document_attachment_is_sent_once_as_evidence_block(monkeypatch, tmp_path):
    path = tmp_path / "订单.txt"
    path.write_text("订单号123，付款800元，商品未发货。", encoding="utf-8")
    captured = {}

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"reply": "已收到材料"}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(gradio_chat_demo.requests, "post", fake_post)

    result = gradio_chat_demo.send_message(
        "请结合附件分析",
        [],
        "user",
        "session",
        [str(path)],
    )

    content = captured["message"]
    assert content.count("【文档证据补充") == 1
    assert content.count("订单号123") == 1
    assert "原文件 SHA-256：" in content
    assert result[0][-1]["content"] == "已收到材料"


def test_intake_package_uses_normal_chat_and_clears_intake_fields(monkeypatch):
    captured = {}

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"reply": "开始分析"}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(gradio_chat_demo.requests, "post", fake_post)

    result = gradio_chat_demo.send_intake_package(
        "付款后未发货",
        "个人卖家",
        "2026年7月，800元",
        "退款",
        "平台申诉未回复",
        [],
        "user",
        "",
        [],
    )

    assert "【首次案件材料包】" in captured["message"]
    assert "【希望解决的结果】\n退款" in captured["message"]
    assert result[-5:] == ("", "", "", "", "")


def test_quick_action_buttons_send_only_control_commands(monkeypatch):
    calls = []

    def fake_send(message, history, user_id, session_id, uploaded_files):
        calls.append((message, history, user_id, session_id, uploaded_files))
        return ("sent",)

    monkeypatch.setattr(gradio_chat_demo, "send_message", fake_send)
    args = ([{"role": "assistant", "content": "方案"}], "user", "session", [])

    assert gradio_chat_demo.send_conclude_action(*args) == ("sent",)
    assert gradio_chat_demo.send_supplement_action(*args) == ("sent",)
    assert gradio_chat_demo.send_document_action(*args) == ("sent",)
    assert [item[0] for item in calls] == [
        "现在生成方案",
        "继续补充",
        "生成文书",
    ]
