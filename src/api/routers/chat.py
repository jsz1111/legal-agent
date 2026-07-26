# src/api/routers/chat.py

from __future__ import annotations
import json
import traceback
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from pathlib import Path
import uuid

from src.core.config import get_settings
from src.infra.database import get_db
from src.infra.redis_cache import get_checkpointer_redis
from src.agents.supervisor_agent import get_supervisor_agent, UserContext
from src.agents.legal_guide.graph import run_guide, build_guide_deps
from src.agents.legal_guide.state import GuideState, GuidePhase

settings = get_settings()

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class DebugInfo(BaseModel):
    domain: str = ""
    confidence_tier: str = ""
    statute_hits: str = ""
    case_hits: str = ""
    graph_laws: list = []
    graph_channels: list = []
    fallback_guide: dict | None = None  # 案例检索兜底指引

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    debug: DebugInfo | None = None


def _make_keys(user_id: str, session_id: str) -> tuple[str, str]:
    """生成 Redis 键名。thread_id 与 Supervisor checkpointer 保持一致。"""
    thread_id = f"{user_id}:{session_id}"
    return f"guide_active:{thread_id}", f"guide_state:{thread_id}"


async def _run_guide_turn(
    message: str,
    thread_id: str,
    redis,
    db,
) -> tuple[str, DebugInfo]:
    """
    执行一轮法律指引对话（路由层直接调用，绕过 Supervisor）。
    从 Redis 恢复状态 → 执行 GuideGraph → 保存新状态 → 返回回复+调试信息。
    """
    from src.agents.legal_guide.formatters import is_doc_request
    from src.agents.legal_guide.doc_generator import generate_legal_doc
    from langchain_core.messages import HumanMessage, AIMessage

    active_key = f"guide_active:{thread_id}"
    state_key  = f"guide_state:{thread_id}"

    raw = await redis.get(state_key)
    existing_state = GuideState.model_validate_json(raw) if raw else None

    # 从 thread_id 提取 user_id（格式：user_id:session_id）
    user_id = thread_id.split(":")[0] if ":" in thread_id else None

    deps = build_guide_deps(db_session=db)

    # 特殊处理：phase=END 且用户请求文书 → 直接生成文书，不走完整状态机
    if existing_state and existing_state.phase == GuidePhase.END:
        if is_doc_request(message) and existing_state.confirmed_issues:
            logger.info("检测到文书生成请求，直接调用 generate_doc")
            # 添加用户消息到历史
            existing_state.messages.append(HumanMessage(content=message))
            # 直接调用文书生成函数
            doc_type, doc = await generate_legal_doc(
                legal_domain=existing_state.legal_domain,
                confirmed_issues=existing_state.confirmed_issues,
                region=existing_state.region,
                evidence_confirmed=existing_state.evidence_confirmed,
                law_context_str=existing_state.law_context_str,
                llm=deps.llm,
            )
            existing_state.doc_draft = doc
            existing_state.messages.append(AIMessage(content=doc))

            # 文书生成完成，删除 Redis 状态
            await redis.delete(active_key, state_key)

            debug = DebugInfo(
                domain=existing_state.legal_domain or "",
                confidence_tier=existing_state.confidence_tier or "",
                statute_hits=existing_state.law_context_str or "",
                case_hits=existing_state.case_context_str or "",
                graph_laws=existing_state.candidate_laws or [],
                graph_channels=existing_state.relevant_channels or [],
                fallback_guide=existing_state.fallback_guide,
            )
            return doc, debug

    reply, new_state = await run_guide(
        user_message=message,
        thread_id=thread_id,
        deps=deps,
        existing_state=existing_state,
        user_id=user_id,
    )

    debug = DebugInfo(
        domain=new_state.legal_domain or "",
        confidence_tier=new_state.confidence_tier or "GATHERING",  # 未打分时显示 GATHERING
        statute_hits=new_state.law_context_str or "",
        case_hits=new_state.case_context_str or "",
        graph_laws=new_state.candidate_laws or [],
        graph_channels=new_state.relevant_channels or [],
        fallback_guide=new_state.fallback_guide,  # 透传案例检索兜底指引
    )

    # 指引结束：检查是否有文书生成邀请，若有则保留 Redis 一轮等待用户确认
    if new_state.phase == GuidePhase.END:
        if not new_state.doc_draft and "生成文书" in reply:
            # conclude 提供了文书邀请，保留状态让下一轮路由到 generate_doc
            ttl = settings.REDIS_SESSION_TTL
            await redis.set(state_key, new_state.model_dump_json(), ex=ttl)
            await redis.set(active_key, "1", ex=ttl)
        else:
            await redis.delete(active_key, state_key)
        return reply, debug

    # 指引继续：更新 Redis 状态，重置 TTL
    ttl = settings.REDIS_SESSION_TTL
    await redis.set(state_key,  new_state.model_dump_json(), ex=ttl)
    await redis.set(active_key, "1",                         ex=ttl)
    return reply, debug


# ── 非流式接口 ────────────────────────────────────────────────────────────
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    非流式对话接口。
    路由层判断是否有活跃法律指引：有则直接走 GuideGraph，无则走 Supervisor。
    """
    try:
        redis = get_checkpointer_redis()
        thread_id = f"{req.user_id}:{req.session_id}"
        active_key = f"guide_active:{thread_id}"

        # ── 指引进行中：直接走 GuideGraph ──
        if await redis.exists(active_key):
            reply, debug = await _run_guide_turn(req.message, thread_id, redis, db)
            return ChatResponse(reply=reply, session_id=req.session_id, debug=debug)

        # ── 无活跃指引：走 Supervisor ──
        agent = await get_supervisor_agent()
        config = {"configurable": {"thread_id": thread_id}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            context=UserContext(user_id=req.user_id, session_id=req.session_id),
        )
        supervisor_reply = result["messages"][-1].content

        # 若本轮调用了 call_guide_agent，直接取 guide_agent 原始回复，
        # 绕过 Supervisor 可能的摘要/改写。
        import json as _json
        debug = None
        reply = supervisor_reply  # 默认用 Supervisor 回复（非指引场景）
        try:
            reply_key = f"guide_last_reply:{req.user_id}:{req.session_id}"
            debug_key = f"guide_last_debug:{req.user_id}:{req.session_id}"
            raw_reply = await redis.get(reply_key)
            raw_debug = await redis.get(debug_key)
            if raw_reply:
                # guide_agent 原始回复存在 → 直接用，忽略 Supervisor 重写
                reply = raw_reply.decode("utf-8") if isinstance(raw_reply, bytes) else raw_reply
                await redis.delete(reply_key)
            if raw_debug:
                d = _json.loads(raw_debug)
                debug = DebugInfo(
                    domain=d.get("domain", ""),
                    confidence_tier=d.get("confidence_tier", "") or "GATHERING",
                    statute_hits=d.get("statute_hits", ""),
                    case_hits=d.get("case_hits", ""),
                    graph_laws=d.get("graph_laws", []),
                    graph_channels=d.get("graph_channels", []),
                    fallback_guide=d.get("fallback_guide"),
                )
                await redis.delete(debug_key)
        except Exception:
            pass

        return ChatResponse(reply=reply, session_id=req.session_id, debug=debug)

    except Exception as e:
        logger.exception("chat 接口异常")
        user_msg = "服务暂时不可用，请稍后重试。如持续出现，请联系客服。"
        if settings.APP_DEBUG:  # 仅调试模式返回详情
            user_msg += f"\n调试信息：{str(e)}"
        raise HTTPException(status_code=500, detail=user_msg)


# ── 流式接口（SSE） ───────────────────────────────────────────────────────
@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    流式对话接口（Server-Sent Events）。
    法律指引进行中时，GuideGraph 的回复以整体推送；
    Supervisor 回复以流式推送。

    客户端接收格式：
        data: {"type": "token",  "content": "..."}
        data: {"type": "done",   "session_id": "..."}
        data: {"type": "error",  "message": "..."}
    """
    async def event_generator():
        try:
            redis = get_checkpointer_redis()
            thread_id = f"{req.user_id}:{req.session_id}"
            active_key = f"guide_active:{thread_id}"

            # ── 指引进行中：GuideGraph 非流式执行，结果整体推送 ──
            if await redis.exists(active_key):
                reply, debug = await _run_guide_turn(req.message, thread_id, redis, db)
                data = json.dumps({"type": "token", "content": reply}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                done_data = json.dumps(
                    {"type": "done", "session_id": req.session_id,
                     "debug": debug.model_dump()},
                    ensure_ascii=False,
                )
                yield f"data: {done_data}\n\n"
                return

            else:
                # ── 无活跃指引：Supervisor 流式推送 ──
                agent = await get_supervisor_agent()
                config = {"configurable": {"thread_id": thread_id}}
                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": req.message}]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, tuple):
                        msg_chunk, _ = chunk
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            data = json.dumps(
                                {"type": "token", "content": msg_chunk.content},
                                ensure_ascii=False,
                            )
                            yield f"data: {data}\n\n"

            done_data = json.dumps(
                {"type": "done", "session_id": req.session_id}, ensure_ascii=False
            )
            yield f"data: {done_data}\n\n"

        except Exception as e:
            logger.exception("chat/stream 接口异常")
            user_msg = "服务异常，连接已中断。"
            if settings.APP_DEBUG:
                user_msg += f" 调试信息：{str(e)}"
            error_data = json.dumps({"type": "error", "message": user_msg}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 图片上传与分析接口（多模态支持，可选）───────────────────────────────
@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session_id: str = Form(...),
    question: str = Form(default=None),  # 如果为 None，根据上下文自动生成
    auto_inject: bool = Form(default=True),  # 是否自动注入对话流
    db: AsyncSession = Depends(get_db_session),
):
    """
    上传图片并进行内容分析（需要启用多模态功能）。

    参数：
    - file: 图片文件
    - user_id: 用户ID
    - session_id: 会话ID（用于关联对话上下文）
    - question: 分析提示词（如果为 None，根据对话上下文自动生成）
    - auto_inject: 是否自动将分析结果注入对话流（默认true）

    返回：
    - image_url: 图片保存路径
    - analysis: 图片分析结果
    - enabled: 多模态功能是否启用
    - injected: 是否已注入对话流
    - assistant_reply: 如果auto_inject=true，返回助手的回复
    """
    from src.agents.tools.multimodal_tools import is_multimodal_enabled, analyze_image
    from src.core.config import get_settings

    settings = get_settings()

    # 检查是否启用多模态
    if not is_multimodal_enabled():
        return {
            "enabled": False,
            "message": "多模态功能未启用。请在 .env 中配置 VL_API_KEY 和 ENABLE_MULTIMODAL=true"
        }

    try:
        # 保存上传的图片
        upload_dir = Path("uploads") / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_ext = Path(file.filename).suffix or ".jpg"
        file_id = str(uuid.uuid4())
        save_path = upload_dir / f"{file_id}{file_ext}"

        # 写入文件
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        logger.info(f"图片已保存: {save_path}")

        # 获取当前对话上下文（如果存在）
        redis = get_checkpointer_redis()
        thread_id = f"{user_id}:{session_id}"
        state_key = f"guide_state:{thread_id}"

        legal_domain = ""
        confirmed_issues = []
        evidence_confirmed = []
        evidence_unavailable = []
        recent_assistant_message = ""

        # 尝试从 Redis 恢复 GuideState
        try:
            raw = await redis.get(state_key)
            if raw:
                existing_state = GuideState.model_validate_json(raw)
                legal_domain = existing_state.legal_domain or ""
                confirmed_issues = existing_state.confirmed_issues or []
                evidence_confirmed = existing_state.evidence_confirmed or []
                evidence_unavailable = existing_state.evidence_unavailable or []

                # 获取最近一条助手消息（可能包含追问内容）
                from langchain_core.messages import AIMessage
                for msg in reversed(existing_state.messages):
                    if isinstance(msg, AIMessage):
                        recent_assistant_message = msg.content[:500]  # 最多取500字
                        break

                logger.info(f"图片分析使用上下文: domain={legal_domain}, issues={confirmed_issues}")
        except Exception as ctx_err:
            logger.warning(f"获取对话上下文失败，使用通用分析: {ctx_err}")

        # 分析图片内容（根据上下文生成针对性提示词）
        analysis = await analyze_image(
            str(save_path),
            question=question,  # 如果为 None，会根据上下文自动生成
            legal_domain=legal_domain,
            confirmed_issues=confirmed_issues,
            evidence_confirmed=evidence_confirmed,
            evidence_unavailable=evidence_unavailable,
            recent_assistant_message=recent_assistant_message
        )

        response = {
            "enabled": True,
            "image_url": str(save_path),
            "analysis": analysis,
            "injected": False,
            "context_used": bool(legal_domain or confirmed_issues),  # 标记是否使用了上下文
        }

        # 如果启用自动注入，将分析结果作为用户消息注入对话流
        if auto_inject and analysis and not analysis.startswith("❌") and not analysis.startswith("⚠️"):
            try:
                # 构造结构化的证据消息
                evidence_message = f"【图片证据补充】\n{analysis}"

                # 调用对话接口，将分析结果注入
                active_key = f"guide_active:{thread_id}"

                # 检查是否有活跃的指引会话
                if await redis.exists(active_key):
                    # 直接调用 guide_agent
                    reply, debug = await _run_guide_turn(evidence_message, thread_id, redis, db)
                    response["injected"] = True
                    response["assistant_reply"] = reply
                    response["debug"] = debug.model_dump()
                else:
                    # 通过 Supervisor 路由
                    agent = await get_supervisor_agent()
                    config = {"configurable": {"thread_id": thread_id}}
                    result = await agent.ainvoke(
                        {"messages": [{"role": "user", "content": evidence_message}]},
                        config=config,
                    )
                    # 提取最后一条 AI 消息
                    ai_messages = [m for m in result.get("messages", []) if hasattr(m, "type") and m.type == "ai"]
                    reply = ai_messages[-1].content if ai_messages else "图片内容已记录"
                    response["injected"] = True
                    response["assistant_reply"] = reply

                logger.info(f"图片分析结果已自动注入对话流: session={session_id}, context_aware={response['context_used']}")

            except Exception as inject_err:
                logger.warning(f"自动注入对话流失败: {inject_err}")
                response["injected"] = False
                response["inject_error"] = str(inject_err)

        return response

    except Exception as e:
        logger.exception("图片上传分析失败")
        raise HTTPException(status_code=500, detail=str(e))
