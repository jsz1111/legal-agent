"""
法律多智能体平台 — Gradio 对话测试台

使用方式：
1. 先启动后端：uvicorn src.main:app --port 8080 --reload
2. 再启动本脚本：python scripts/gradio_chat_demo.py
"""

import uuid
import requests
import gradio as gr

API_BASE = "http://localhost:8080"
CHAT_URL = f"{API_BASE}/api/v1/chat"
HEALTH_URL = f"{API_BASE}/health/deps"


# ── 工具函数 ──────────────────────────────────────────────────────────────

def check_health() -> str:
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        data = r.json()
        deps = data.get("dependencies", {})
        lines = [f"总状态：{'✅ 正常' if data.get('status') == 'ok' else '⚠️ 异常'}"]
        icons = {"postgres": "🗄️", "redis": "⚡", "minio": "📦", "milvus": "🔍", "neo4j": "🕸️"}
        for name, info in deps.items():
            icon = icons.get(name, "")
            status = "✅" if info.get("ok") else f"❌ {info.get('error', '')[:60]}"
            lines.append(f"{icon} {name}: {status}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 无法连接后端：{e}\n请先运行：uvicorn src.main:app --port 8080 --reload"


def send_message(
    user_message: str,
    history: list,
    user_id: str,
    session_id: str,
) -> tuple[list, str, str]:
    """发送消息，返回更新后的 history。"""
    if not user_message.strip():
        return history, session_id, ""

    # 自动生成 session_id（首轮）
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    history = history or []
    history.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            CHAT_URL,
            json={"user_id": user_id, "session_id": session_id, "message": user_message},
            timeout=60,
        )
        resp.raise_for_status()
        reply = resp.json().get("reply", "（无回复）")
    except requests.exceptions.ConnectionError:
        reply = "❌ 无法连接后端，请确认服务已启动（uvicorn src.main:app --port 8080）"
    except requests.exceptions.Timeout:
        reply = "⏳ 请求超时（60s），LLM 可能正在处理，请稍候重试"
    except Exception as e:
        reply = f"❌ 请求失败：{e}"

    history.append({"role": "assistant", "content": reply})
    return history, session_id, ""


def new_session() -> tuple[list, str]:
    """开始新会话，生成新 session_id。"""
    return [], str(uuid.uuid4())[:8]


# ── 预设场景 ──────────────────────────────────────────────────────────────

SCENARIOS = {
    "劳动纠纷 — 拖欠工资": "公司已经3个月没发工资了，我该怎么办？",
    "消费维权 — 网购假货": "我在某平台买了一件商品，收到后发现是假货，商家拒绝退款",
    "房屋租赁 — 押金纠纷": "房东以房屋有损坏为由不退押金，但损坏不是我造成的",
    "法律知识问答": "劳动仲裁和劳动诉讼有什么区别？分别需要多长时间？",
    "紧急情形": "我现在正在遭受家庭暴力，不知道怎么办",
}


# ── Gradio 界面 ───────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="法律多智能体平台 — 测试台") as demo:

        gr.Markdown("# 法律多智能体平台 测试台\n测试公民法律指引 / 法律知识问答多智能体对话流程。")

        with gr.Row():
            # ── 左侧控制面板 ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=240):

                gr.Markdown("### 配置")
                user_id_box = gr.Textbox(
                    label="用户ID", value="test_user_01",
                )
                session_id_box = gr.Textbox(
                    label="会话ID（首轮自动生成）", value="",
                )
                new_btn = gr.Button("🔄 新建会话", variant="secondary")

                gr.Markdown("### 快速场景")
                # 场景按钮：点击后填入输入框
                scenario_btns = []
                for label in SCENARIOS:
                    b = gr.Button(label, size="sm")
                    scenario_btns.append(b)

                gr.Markdown("### 后端状态")
                health_box = gr.Textbox(
                    label="", lines=7, interactive=False, value="点击「检查」刷新"
                )
                check_btn = gr.Button("🔍 检查后端", size="sm")

            # ── 右侧对话区 ────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话", height=560)
                with gr.Row():
                    msg_box = gr.Textbox(
                        label="", placeholder="描述您的法律问题……",
                        scale=5, container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        # ── 事件绑定 ──────────────────────────────────────────────────
        send_kwargs = dict(
            fn=send_message,
            inputs=[msg_box, chatbot, user_id_box, session_id_box],
            outputs=[chatbot, session_id_box, msg_box],
        )
        send_btn.click(**send_kwargs)
        msg_box.submit(**send_kwargs)

        new_btn.click(fn=new_session, outputs=[chatbot, session_id_box])
        check_btn.click(fn=check_health, outputs=health_box)
        demo.load(fn=check_health, outputs=health_box)

        # 场景按钮：填入输入框（不自动发送，让用户可以修改后再发）
        for btn, (_, text) in zip(scenario_btns, SCENARIOS.items()):
            btn.click(fn=lambda t=text: t, outputs=msg_box)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
