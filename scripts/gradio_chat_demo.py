"""
法律多智能体平台 — Gradio 对话测试台

【2024-07 改造版本】
- 澄清/追问分离：澄清定性案情（2轮上限），追问补证据（10轮上限）
- 打分前置：证据完整度0.4+事实清晰度0.3+权责清晰度0.3 → HIGH/MID/LOW
- 所有档位都检索：区别在语气确定程度，不再"低分不给说法"
- 高分自省降档：HIGH档检索后LLM判断法条适用性/时效/管辖

使用方式：
1. 先启动后端：uvicorn src.main:app --port 8080 --reload
2. 再启动本脚本：python scripts/gradio_chat_demo.py
"""

import uuid
import requests
import gradio as gr
import os

API_BASE = "http://localhost:8001"
BASE_URL = API_BASE
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
) -> tuple[list, str, str, str, str, str, str, str]:
    """发送消息，返回更新后的 history 以及检索调试信息。"""
    _empty_debug = ("", "", "", "", "")
    if not user_message.strip():
        return history, session_id, "", *_empty_debug

    # 自动生成 session_id（首轮）
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    history = history or []
    history.append({"role": "user", "content": user_message})

    debug_meta = debug_statute = debug_case = debug_graph = fallback_html = ""

    try:
        resp = requests.post(
            CHAT_URL,
            json={"user_id": user_id, "session_id": session_id, "message": user_message},
            timeout=120,  # 增加到120秒，知识检索需要更长时间
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "（无回复）")

        dbg = data.get("debug") or {}
        if dbg:
            debug_meta = (
                f"领域：{dbg.get('domain', '—')}\n"
                f"置信档位：{dbg.get('confidence_tier', '—')}"
            )
            debug_statute = dbg.get("statute_hits", "") or "（无命中）"
            debug_case    = dbg.get("case_hits", "")    or "（无命中）"
            laws_list = dbg.get("graph_laws", [])
            chs_list  = dbg.get("graph_channels", [])
            graph_lines = []
            if laws_list:
                graph_lines.append("【命中法律】")
                graph_lines.extend(f"  · {x}" for x in laws_list)
            if chs_list:
                graph_lines.append("【维权渠道】")
                graph_lines.extend(f"  · {x}" for x in chs_list)
            debug_graph = "\n".join(graph_lines) or "（无命中）"

            # 解析 fallback_guide 并渲染为 HTML
            fb = dbg.get("fallback_guide")
            if fb and isinstance(fb, dict):
                platform = fb.get("platform", "")
                url = fb.get("url", "")
                tips = fb.get("search_tips", "")
                fallback_html = f"""
<div style="border: 2px solid #ff9800; border-radius: 8px; padding: 16px; background: #fff3e0; margin: 12px 0;">
    <div style="font-size: 16px; font-weight: bold; color: #e65100; margin-bottom: 8px;">
        📋 案例查询指引
    </div>
    <div style="margin-bottom: 12px; color: #333;">
        <strong>平台：</strong>{platform}
    </div>
    <div style="margin-bottom: 12px;">
        <a href="{url}" target="_blank" style="display: inline-block; padding: 8px 16px; background: #ff5722; color: white; text-decoration: none; border-radius: 4px; font-weight: bold;">
            🔗 前往查询
        </a>
    </div>
    <div style="color: #555; line-height: 1.6;">
        <strong>搜索建议：</strong><br>{tips}
    </div>
</div>
"""

    except requests.exceptions.ConnectionError:
        reply = "❌ 无法连接后端，请确认服务已启动（uvicorn src.main:app --port 8080）"
    except requests.exceptions.Timeout:
        reply = "⏳ 请求超时（60s），LLM 可能正在处理，请稍候重试"
    except Exception as e:
        reply = f"❌ 请求失败：{e}"

    history.append({"role": "assistant", "content": reply})
    return history, session_id, "", debug_meta, debug_statute, debug_case, debug_graph, fallback_html


def new_session() -> tuple[list, str]:
    """开始新会话，生成新 session_id。"""
    return [], str(uuid.uuid4())[:8]


def upload_and_analyze_image(image_file, user_id: str, session_id: str) -> tuple[str, list]:
    """上传图片并分析内容（多模态功能），自动注入对话流"""
    if image_file is None:
        return "❌ 请先上传图片", []

    try:
        # 构建 multipart form data
        with open(image_file, 'rb') as f:
            files = {'file': (os.path.basename(image_file), f, 'image/jpeg')}
            data = {
                'user_id': user_id,
                'session_id': session_id,
                'question': '请详细描述这张图片中与法律证据相关的内容（如订单信息、商品描述、价格、瑕疵细节、聊天记录内容、商家回复等）',
                'auto_inject': 'true'  # 启用自动注入
            }

            resp = requests.post(
                f"{BASE_URL}/api/v1/chat/upload-image",
                files=files,
                data=data,
                timeout=30
            )

            if resp.status_code == 200:
                result = resp.json()
                if not result.get("enabled"):
                    return "⚠️ 多模态功能未启用。请在 .env 中配置 VL_API_KEY 和 ENABLE_MULTIMODAL=true", []

                analysis = result.get("analysis", "")
                injected = result.get("injected", False)
                assistant_reply = result.get("assistant_reply", "")

                # 构建返回消息
                message = f"✅ 图片分析成功\n\n📋 **提取的证据信息：**\n{analysis}\n\n"

                if injected and assistant_reply:
                    message += f"🤖 **助手已自动处理：**\n{assistant_reply[:300]}{'...' if len(assistant_reply) > 300 else ''}\n\n"
                    message += "💡 图片内容已自动注入对话，维权方案已更新。您可以继续补充其他信息。"

                    # 更新对话历史
                    new_history = [
                        {"role": "user", "content": f"【上传图片证据】\n{analysis[:200]}..."},
                        {"role": "assistant", "content": assistant_reply}
                    ]
                else:
                    message += "💡 您可以将这些信息复制到对话框中继续咨询"
                    new_history = []

                return message, new_history
            else:
                return f"❌ 上传失败：{resp.text}", []

    except requests.exceptions.ConnectionError:
        return "❌ 无法连接后端，请确认服务已启动", []
    except Exception as e:
        return f"❌ 上传异常：{str(e)}", []


# ── 预设场景 ──────────────────────────────────────────────────────────────

SCENARIOS = {
    "🔥 紧急情形（高危熔断）": "我现在正在遭受家庭暴力，对方威胁我不让报警",
    "❓ 模糊描述（触发澄清）": "房东不退钱",
    "📋 清晰案情（触发追问）": "退房后房东以房屋有损坏为由不退押金，但损坏不是我造成的，我有交房时的照片",
    "💼 劳动纠纷 — 拖欠工资": "公司已经3个月没发工资了，我有劳动合同、工资流水和考勤记录",
    "🛒 消费维权 — 网购假货": "我在某平台买了一件商品，收到后发现是假货，有订单截图和聊天记录，商家拒绝退款",
    "📚 法律知识问答": "劳动仲裁和劳动诉讼有什么区别？分别需要多长时间？",
    "🎓 测试LOW档（信息少）": "我被人打了",
    "✅ 测试HIGH档（信息全）": "我在北京工作，公司拖欠我3个月工资共2万元，我有劳动合同、银行流水、打卡记录和微信催款截图，事情发生在半年前",
}


# ── Gradio 界面 ───────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="法律多智能体平台 — 测试台") as demo:

        gr.Markdown("""
# 法律多智能体平台 测试台 🏛️

**改造版本（2024-07）**：澄清/追问分离 + 打分前置 + 档位分级输出

测试要点：
- **澄清上限**：模糊描述 → 最多澄清2轮 → 仍模糊降级LOW档
- **追问上限**：清晰案情 → 按证据清单追问 → 最多10轮
- **打分分档**：HIGH≥0.8（证据齐全）→ 深度检索+自省 → 笃定完整方案；MID/LOW → 谨慎/保守语气
- **紧急熔断**：每轮检测高危 → CRITICAL立即终止推送110/12348
        """)

        with gr.Row():
            # ── 左侧控制面板 ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=240):

                gr.Markdown("### ⚙️ 配置")
                user_id_box = gr.Textbox(
                    label="用户ID", value="test_user_01", placeholder="用于记忆关联"
                )
                session_id_box = gr.Textbox(
                    label="会话ID（首轮自动生成）", value="", placeholder="多轮对话标识"
                )
                new_btn = gr.Button("🔄 新建会话", variant="secondary")

                gr.Markdown("### 🎯 快速测试场景\n点击填入输入框，可修改后发送")
                # 场景按钮：点击后填入输入框
                scenario_btns = []
                for label in SCENARIOS:
                    b = gr.Button(label, size="sm")
                    scenario_btns.append(b)

                gr.Markdown("### 🔍 后端健康状态")
                health_box = gr.Textbox(
                    label="", lines=8, interactive=False, value="点击「检查」刷新"
                )
                check_btn = gr.Button("🔍 检查后端", size="sm")

                gr.Markdown("### 📸 图片证据上传（可选）")
                image_upload = gr.Image(
                    label="上传图片证据",
                    type="filepath",
                    height=200
                )
                upload_btn = gr.Button("🔍 分析图片", size="sm", variant="secondary")
                image_result = gr.Textbox(
                    label="图片分析结果",
                    lines=6,
                    interactive=False,
                    placeholder="上传图片后点击「分析图片」查看结果"
                )

            # ── 右侧对话区 ────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 对话记录",
                    height=500,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        label="",
                        placeholder="💬 描述您的法律问题（支持回车发送）……",
                        scale=5,
                        container=False,
                        lines=2
                    )
                    send_btn = gr.Button("📤 发送", variant="primary", scale=1)

                with gr.Accordion("🔍 检索详情（法律指引流程专用）", open=False):
                    debug_meta_box = gr.Textbox(
                        label="领域 / 置信档位",
                        lines=2,
                        interactive=False,
                    )
                    debug_statute_box = gr.Textbox(
                        label="Milvus 法条命中 (statute_index)",
                        lines=6,
                        max_lines=12,
                        interactive=False,
                    )
                    debug_case_box = gr.Textbox(
                        label="Milvus 案例命中 (case_index)",
                        lines=4,
                        max_lines=8,
                        interactive=False,
                    )
                    debug_graph_box = gr.Textbox(
                        label="Neo4j 图谱命中 (法律 + 维权渠道)",
                        lines=4,
                        max_lines=8,
                        interactive=False,
                    )
                    fallback_guide_box = gr.HTML(
                        label="案例查询指引",
                        visible=True,
                    )

        # ── 事件绑定 ──────────────────────────────────────────────────
        _debug_outputs = [debug_meta_box, debug_statute_box, debug_case_box, debug_graph_box, fallback_guide_box]
        send_kwargs = dict(
            fn=send_message,
            inputs=[msg_box, chatbot, user_id_box, session_id_box],
            outputs=[chatbot, session_id_box, msg_box] + _debug_outputs,
        )
        send_btn.click(**send_kwargs)
        msg_box.submit(**send_kwargs)

        new_btn.click(fn=new_session, outputs=[chatbot, session_id_box])
        check_btn.click(fn=check_health, outputs=health_box)
        demo.load(fn=check_health, outputs=health_box)

        # 图片上传分析（自动注入对话流）
        upload_btn.click(
            fn=upload_and_analyze_image,
            inputs=[image_upload, user_id_box, session_id_box],
            outputs=[image_result, chatbot]  # 更新分析结果和对话历史
        )

        # 场景按钮：填入输入框（不自动发送，让用户可以修改后再发）
        for btn, (_, text) in zip(scenario_btns, SCENARIOS.items()):
            btn.click(fn=lambda t=text: t, outputs=msg_box)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True,
    )
