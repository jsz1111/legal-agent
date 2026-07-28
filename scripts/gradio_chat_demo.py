"""
法律多智能体平台 — Gradio 对话测试台

【2024-07 改造版本】
- 澄清/追问分离：澄清定性案情（2轮上限），事实与证据追问合计最多6轮
- 打分前置：证据完整度0.4+事实清晰度0.3+权责清晰度0.3 → HIGH/MID/LOW
- 所有档位都检索：区别在语气确定程度，不再"低分不给说法"
- 高分自省降档：HIGH档检索后LLM判断法条适用性/时效/管辖

使用方式：
1. 先启动后端：uvicorn src.main:app --port 8080 --reload
2. 再启动本脚本：python scripts/gradio_chat_demo.py
"""

import uuid
import mimetypes
import requests
import gradio as gr
import os
import pandas as pd
import plotly.graph_objects as go

API_BASE = os.getenv("LEGAL_AGENT_API_BASE", "http://127.0.0.1:8080")
BASE_URL = API_BASE
CHAT_URL = f"{API_BASE}/api/v1/chat"
HEALTH_URL = f"{API_BASE}/health/deps"


_CHART_COLORS = ["#176B87", "#C8553D", "#3A7D44", "#6B5B95", "#B7791F"]


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


def _build_statistics_figure(statistics: dict):
    chart = statistics.get("chart") or {}
    chart_type = chart.get("type", "table")
    if chart_type == "table":
        return None

    figure = go.Figure()
    x_values = chart.get("x_values") or []
    series = chart.get("series") or []

    if chart_type in {"line", "bar"}:
        for index, item in enumerate(series):
            color = _CHART_COLORS[index % len(_CHART_COLORS)]
            if chart_type == "line":
                figure.add_trace(go.Scatter(
                    x=x_values,
                    y=item.get("data") or [],
                    name=item.get("name") or "数值",
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                    marker={"size": 8},
                ))
            else:
                figure.add_trace(go.Bar(
                    x=x_values,
                    y=item.get("data") or [],
                    name=item.get("name") or "数值",
                    marker_color=color,
                ))
    elif chart_type == "pie" and series:
        figure.add_trace(go.Pie(
            labels=x_values,
            values=series[0].get("data") or [],
            hole=0.42,
            textinfo="label+percent",
        ))
    elif chart_type == "scatter":
        for index, item in enumerate(series):
            points = item.get("data") or []
            figure.add_trace(go.Scatter(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                text=[point[2] if len(point) > 2 else "" for point in points],
                name=item.get("name") or "指标关系",
                mode="markers",
                marker={
                    "size": 11,
                    "color": _CHART_COLORS[index % len(_CHART_COLORS)],
                },
            ))
    elif chart_type == "heatmap" and series:
        y_values = chart.get("y_values") or []
        matrix = [[None for _ in x_values] for _ in y_values]
        for point in series[0].get("data") or []:
            if len(point) < 3:
                continue
            x_index, y_index, value = point[:3]
            if 0 <= y_index < len(matrix) and 0 <= x_index < len(x_values):
                matrix[y_index][x_index] = value
        figure.add_trace(go.Heatmap(
            x=x_values,
            y=y_values,
            z=matrix,
            colorscale="Blues",
            colorbar={"title": chart.get("y_label") or "数值"},
        ))
    else:
        return None

    figure.update_layout(
        title={"text": chart.get("title") or "法律统计分析", "x": 0.5},
        xaxis_title=chart.get("x_label") or None,
        yaxis_title=chart.get("y_label") or None,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F7F8FA",
        font={"family": "Microsoft YaHei, sans-serif", "color": "#202124"},
        margin={"l": 56, "r": 24, "t": 64, "b": 52},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified" if chart_type in {"line", "bar"} else "closest",
    )
    return figure


def _statistics_updates(statistics: dict | None):
    if not statistics:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    rows = statistics.get("rows") or []
    columns = statistics.get("columns") or []
    table = pd.DataFrame(rows)
    if columns:
        table = table.reindex(columns=columns)

    chart = statistics.get("chart") or {}
    source_lines = [
        "**数据摘要：**",
        "",
        statistics.get("summary") or "本次查询未生成摘要。",
        "",
        f"**图表建议：** {chart.get('reason', '根据查询结果自动选择。')}",
        "",
        "**年鉴来源：**",
    ]
    for source in statistics.get("sources") or []:
        years = "、".join(str(year) for year in source.get("years") or [])
        institution = source.get("institution") or "未注明机构"
        quality = "、".join(source.get("quality_flags") or []) or "未标记"
        source_lines.append(
            f"- {source.get('title') or '未命名统计表'}；{institution}；"
            f"统计年份：{years or '未注明'}；质量标记：`{quality}`"
        )
    if statistics.get("sql"):
        source_lines.extend(
            [
                "",
                "**已通过安全校验的 SQL：**",
                "",
                "```sql",
                statistics["sql"],
                "```",
            ]
        )

    figure = _build_statistics_figure(statistics)
    return (
        gr.update(value=figure, visible=figure is not None),
        gr.update(value=table, visible=not table.empty),
        gr.update(value="\n".join(source_lines), visible=True),
    )


def _document_update(document: dict | None):
    if not document:
        return gr.update(value="", visible=False)

    def absolute_url(value: str | None) -> str:
        if not value:
            return ""
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"{API_BASE.rstrip('/')}/{value.lstrip('/')}"

    lines = [f"**{document.get('doc_type') or '参考文书'}**", ""]
    generated_url = absolute_url(document.get("generated_docx_url"))
    if generated_url:
        lines.append(f"[下载智能填写参考稿 DOCX]({generated_url})")

    official_url = absolute_url(document.get("official_blank_url"))
    source = document.get("source") or {}
    if official_url:
        lines.extend(["", f"[下载官方空白模板 PDF]({official_url})"])
    if source:
        issuers = "、".join(source.get("issuers") or [])
        source_page = source.get("source_page_url") or ""
        source_text = (
            f"模板来源：{issuers or '未注明发布机关'}，"
            f"{source.get('document_no') or '未注明文号'}，"
            f"自 {source.get('effective_at') or '未注明日期'} 起推广使用。"
        )
        lines.extend(["", source_text])
        if source_page:
            lines.append(f"[查看发布机关原文]({source_page})")
    else:
        lines.extend(
            ["", "本次未匹配到全国统一官方空白模板，DOCX 使用系统通用参考格式。"]
        )

    missing = document.get("missing_fields") or []
    if missing:
        preview = "、".join(str(item) for item in missing[:10])
        suffix = "等" if len(missing) > 10 else ""
        lines.extend(["", f"提交前需补充：{preview}{suffix}"])
    lines.extend(
        ["", "> 智能填写稿为系统生成的可编辑参考稿，非发布机关出具。"]
    )
    return gr.update(value="\n".join(lines), visible=True)


def send_message(
    user_message: str,
    history: list,
    user_id: str,
    session_id: str,
    uploaded_files: list,  # 新增：暂存的文件列表
) -> tuple:
    """发送消息，返回更新后的 history 以及检索调试信息。"""
    _empty_debug = ("", "", "", "", "")

    # 如果既没有消息也没有文件，直接返回
    if not user_message.strip() and not uploaded_files:
        return (
            history,
            session_id,
            "",
            *_empty_debug,
            [],
            *_statistics_updates(None),
            _document_update(None),
        )

    # 自动生成 session_id（首轮）
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    history = history or []

    # 处理上传的文件（图片优先分析）
    files_info = []
    if uploaded_files:
        for file_path in uploaded_files:
            try:
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()

                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    # 调用图片分析
                    # 本轮会把分析结果随文字统一发送，禁止图片接口再次注入，
                    # 否则同一张图片会被状态机处理两遍。
                    result = _process_image(
                        file_path,
                        user_id,
                        session_id,
                        auto_inject=False,
                    )
                    if result.get("success"):
                        analysis = result.get("analysis", "")
                        digest = result.get("image_sha256", "")
                        files_info.append(
                            "【图片证据补充（视觉模型识别，需与原图核对）】\n"
                            f"文件：{file_name}\n原图 SHA-256：{digest}\n{analysis}"
                        )
                    else:
                        files_info.append(
                            f"📷 {file_name}: {result.get('message') or '分析失败'}"
                        )
                else:
                    files_info.append(f"📎 {file_name}: 暂不支持此格式")
            except Exception as e:
                files_info.append(f"❌ {file_name}: {str(e)}")

    # 组合用户消息：文字 + 图片分析结果
    combined_message = user_message
    if files_info:
        combined_message = user_message + "\n\n" + "\n\n".join(files_info)

    history.append({"role": "user", "content": combined_message})

    debug_meta = debug_statute = debug_case = debug_graph = fallback_html = ""
    statistics = None
    document = None

    try:
        resp = requests.post(
            CHAT_URL,
            json={"user_id": user_id, "session_id": session_id, "message": combined_message},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "（无回复）")
        statistics = data.get("statistics")
        document = data.get("document")

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
        reply = "⏳ 请求超时（120s），LLM 可能正在处理，请稍候重试"
    except Exception as e:
        reply = f"❌ 请求失败：{e}"

    history.append({"role": "assistant", "content": reply})

    # 发送后清空文件列表
    return (
        history,
        session_id,
        "",
        debug_meta,
        debug_statute,
        debug_case,
        debug_graph,
        fallback_html,
        [],
        *_statistics_updates(statistics),
        _document_update(document),
    )


def new_session() -> tuple[list, str]:
    """开始新会话，生成新 session_id。"""
    return [], str(uuid.uuid4())[:8]


def handle_file_upload(files, user_id: str, session_id: str, history: list) -> tuple[list, str]:
    """
    处理文件上传（图片/PDF/DOCX等），自动分析并注入对话流。

    Args:
        files: 上传的文件列表
        user_id: 用户ID
        session_id: 会话ID
        history: 当前对话历史

    Returns:
        (更新后的history, 状态消息)
    """
    if not files:
        return history, ""

    # 确保files是列表
    if not isinstance(files, list):
        files = [files]

    results = []
    new_history = history or []

    for file_path in files:
        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()

            # 判断文件类型
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                # 图片文件：调用多模态分析
                result = _process_image(file_path, user_id, session_id)
                results.append(result)

                if result.get("success"):
                    # 添加到对话历史
                    analysis = result.get("analysis", "")
                    assistant_reply = result.get("assistant_reply", "")

                    new_history.append({
                        "role": "user",
                        "content": f"📎 上传图片：{file_name}\n\n【提取的信息】\n{analysis[:300]}..."
                    })

                    if assistant_reply:
                        new_history.append({
                            "role": "assistant",
                            "content": assistant_reply
                        })

            elif file_ext in ['.pdf', '.docx', '.doc', '.txt']:
                # 文档文件：调用文档解析（暂时提示功能）
                new_history.append({
                    "role": "user",
                    "content": f"📎 上传文档：{file_name}\n\n⚠️ 文档解析功能开发中，请暂时手动输入关键信息。"
                })

            else:
                new_history.append({
                    "role": "user",
                    "content": f"📎 上传文件：{file_name}\n\n⚠️ 不支持的文件格式"
                })

        except Exception as e:
            new_history.append({
                "role": "user",
                "content": f"📎 文件处理失败：{file_name}\n错误：{str(e)}"
            })

    # 生成状态消息
    success_count = sum(1 for r in results if r.get("success"))
    status_msg = f"已处理 {len(files)} 个文件"
    if success_count > 0:
        status_msg += f"，{success_count} 个成功"

    return new_history, status_msg


def _process_image(
    image_file,
    user_id: str,
    session_id: str,
    *,
    auto_inject: bool = True,
) -> dict:
    """处理单张图片上传"""
    try:
        mime_type = mimetypes.guess_type(str(image_file))[0] or "application/octet-stream"
        with open(image_file, 'rb') as f:
            files = {'file': (os.path.basename(image_file), f, mime_type)}
            data = {
                'user_id': user_id,
                'session_id': session_id,
                'auto_inject': 'true' if auto_inject else 'false'
            }

            resp = requests.post(
                f"{BASE_URL}/api/v1/chat/upload-image",
                files=files,
                data=data,
                timeout=150 if auto_inject else 75,
            )

            if resp.status_code == 200:
                result = resp.json()
                if not result.get("enabled"):
                    return {
                        "success": False,
                        "message": result.get("message") or "多模态功能未启用",
                    }

                analysis = str(result.get("analysis") or "").strip()
                if not analysis or analysis.startswith(("❌", "⚠️")):
                    return {
                        "success": False,
                        "message": analysis or "图片分析未返回有效内容",
                    }

                return {
                    "success": True,
                    "analysis": analysis,
                    "assistant_reply": result.get("assistant_reply", ""),
                    "context_used": result.get("context_used", False),
                    "injected": result.get("injected", False),
                    "needs_case_context": result.get("needs_case_context", False),
                    "image_sha256": result.get("image_sha256", ""),
                    "image_meta": result.get("image_meta") or {},
                }
            else:
                try:
                    detail = resp.json().get("detail")
                except ValueError:
                    detail = None
                return {
                    "success": False,
                    "message": detail or f"上传失败（HTTP {resp.status_code}）",
                }

    except requests.exceptions.Timeout:
        return {"success": False, "message": "图片处理超时，请压缩图片后重试"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "无法连接后端服务"}
    except Exception:
        return {"success": False, "message": "图片处理失败，请重试"}


def upload_and_analyze_image(
    image_file,
    user_id: str,
    session_id: str,
    history: list,
) -> tuple[str, list, str]:
    """上传图片并分析内容（多模态功能），根据对话上下文自动生成针对性提示词"""
    current_history = list(history or [])
    if image_file is None:
        return "❌ 请先上传图片", current_history, session_id

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    try:
        result = _process_image(image_file, user_id, session_id, auto_inject=True)
        if not result.get("success"):
            return (
                f"❌ {result.get('message') or '图片分析失败'}",
                current_history,
                session_id,
            )

        analysis = result.get("analysis", "")
        assistant_reply = result.get("assistant_reply", "")
        context_used = result.get("context_used", False)
        injected = result.get("injected", False)
        needs_case_context = result.get("needs_case_context", False)
        image_sha256 = result.get("image_sha256", "")
        image_meta = result.get("image_meta") or {}

        message = "✅ 图片分析成功\n\n"
        if context_used:
            message += "**已结合当前案情进行识别**\n\n"
        message += f"**原图 SHA-256：** `{image_sha256}`\n\n"
        if image_meta:
            message += (
                f"**图片信息：** {image_meta.get('width', '?')} × "
                f"{image_meta.get('height', '?')}，{image_meta.get('mime_type', '')}\n\n"
            )
        message += f"**提取的证据信息：**\n{analysis}"

        current_history.append({
            "role": "user",
            "content": (
                "【上传图片证据】\n"
                f"原图 SHA-256：{image_sha256}\n{analysis}"
            ),
        })
        if injected and assistant_reply:
            current_history.append({"role": "assistant", "content": assistant_reply})
        else:
            current_history.append({
                "role": "assistant",
                "content": (
                    "图片已识别，但当前还没有可关联的维权案情。请先描述纠纷经过，"
                    "再上传图片；或者使用对话框旁的 📎 将图片和案情一次发送。"
                    if needs_case_context
                    else "图片已识别，但尚未写入当前维权流程，请随下一条案情描述一并发送。"
                ),
            })

        return message, current_history, session_id
    except Exception:
        return "❌ 图片处理失败，请重试", current_history, session_id


# ── 预设场景 ──────────────────────────────────────────────────────────────

SCENARIOS = {
    "🔥 紧急情形（高危熔断）": "我现在正在遭受家庭暴力，对方威胁我不让报警",
    "❓ 模糊描述（触发澄清）": "房东不退钱",
    "📋 清晰案情（触发追问）": "退房后房东以房屋有损坏为由不退押金，但损坏不是我造成的，我有交房时的照片",
    "💼 劳动纠纷 — 拖欠工资": "公司已经3个月没发工资了，我有劳动合同、工资流水和考勤记录",
    "🛒 消费维权 — 网购假货": "我在某平台买了一件商品，收到后发现是假货，有订单截图和聊天记录，商家拒绝退款",
    "📚 法律知识问答": "劳动仲裁和劳动诉讼有什么区别？分别需要多长时间？",
    "📈 法律统计趋势": "2018到2020年劳动争议一审收案变化趋势？",
    "📊 法律统计对比": "2020年全国法院民事一审收案和结案分别有多少？",
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
- **追问上限**：清晰案情 → 事实与证据追问合计最多6轮；用户消息总轮次硬上限12轮
- **打分分档**：HIGH≥0.65，MEDIUM≥0.50，LOW<0.50；所有档位均检索，证据不足时采用审慎语气
- **紧急熔断**：每轮检测高危 → CRITICAL立即终止推送110/12348
        """)

        # 新增：暂存上传的文件列表
        uploaded_files_state = gr.State([])

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
                # 文件暂存提示
                files_status = gr.Markdown("", visible=True)

                with gr.Row():
                    file_upload_btn = gr.UploadButton(
                        "📎",
                        file_types=["image", ".pdf", ".docx", ".doc", ".txt"],
                        file_count="multiple",
                        size="sm",
                        scale=0,
                        min_width=50
                    )
                    msg_box = gr.Textbox(
                        label="",
                        placeholder="💬 描述您的法律问题（支持回车发送）……",
                        scale=5,
                        container=False,
                        lines=2
                    )
                    send_btn = gr.Button("📤 发送", variant="primary", scale=1)

                with gr.Accordion("法律统计分析", open=True):
                    statistics_plot = gr.Plot(
                        label="自动推荐图表",
                        visible=False,
                    )
                    statistics_table = gr.Dataframe(
                        label="查询结果",
                        interactive=False,
                        visible=False,
                    )
                    statistics_source = gr.Markdown(
                        value="",
                        visible=False,
                    )

                with gr.Accordion("参考文书下载", open=True):
                    document_download = gr.Markdown(
                        value="",
                        visible=False,
                    )

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
            inputs=[msg_box, chatbot, user_id_box, session_id_box, uploaded_files_state],
            outputs=[
                chatbot,
                session_id_box,
                msg_box,
                *_debug_outputs,
                uploaded_files_state,
                statistics_plot,
                statistics_table,
                statistics_source,
                document_download,
            ],
        )
        send_btn.click(**send_kwargs)
        msg_box.submit(**send_kwargs)

        # 文件上传按钮：只暂存文件，不发送
        def store_files(files, current_files):
            """暂存上传的文件，返回文件列表和状态提示"""
            if not files:
                return current_files or [], ""

            # 确保files是列表
            if not isinstance(files, list):
                files = [files]

            # 合并到现有文件列表
            stored = current_files or []
            stored.extend(files)

            # 生成状态提示
            file_names = [os.path.basename(f) for f in stored]
            status = f"📎 已选择 {len(stored)} 个文件：{', '.join(file_names[:3])}"
            if len(stored) > 3:
                status += f" 等{len(stored)}个文件"

            return stored, status

        file_upload_btn.upload(
            fn=store_files,
            inputs=[file_upload_btn, uploaded_files_state],
            outputs=[uploaded_files_state, files_status]
        )

        # 发送后清空文件状态提示
        def clear_files_status():
            return ""

        send_btn.click(
            fn=clear_files_status,
            outputs=[files_status]
        )
        msg_box.submit(
            fn=clear_files_status,
            outputs=[files_status]
        )

        new_btn.click(
            fn=lambda: (
                [],
                str(uuid.uuid4())[:8],
                [],
                "",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
            ),
            outputs=[
                chatbot,
                session_id_box,
                uploaded_files_state,
                files_status,
                statistics_plot,
                statistics_table,
                statistics_source,
                document_download,
            ]
        )
        check_btn.click(fn=check_health, outputs=health_box)
        demo.load(fn=check_health, outputs=health_box)

        # 图片上传分析（自动注入对话流）
        upload_btn.click(
            fn=upload_and_analyze_image,
            inputs=[image_upload, user_id_box, session_id_box, chatbot],
            outputs=[image_result, chatbot, session_id_box]
        )

        # 场景按钮：填入输入框（不自动发送，让用户可以修改后再发）
        for btn, (_, text) in zip(scenario_btns, SCENARIOS.items()):
            btn.click(fn=lambda t=text: t, outputs=msg_box)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("LEGAL_AGENT_GRADIO_PORT", "7862")),
        share=False,
        inbrowser=False,
    )
