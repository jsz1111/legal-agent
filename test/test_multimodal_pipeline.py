from __future__ import annotations

import asyncio
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
