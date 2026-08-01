"""公民法律指引 Worker — 调用 legal_guide/graph.py 状态机。"""
import json

from src.agents.legal_guide.graph import run_guide, build_guide_deps
from src.agents.legal_guide.state import GuidePhase
from src.core.config import get_settings
from src.infra.redis_cache import get_checkpointer_redis, set_with_optional_ttl
from src.infra.database import AsyncSessionLocal

_DEBUG_TTL = 120  # 调试信息保留 2 分钟，供路由层读取后展示
settings = get_settings()


def _save_debug_key(user_id: str, session_id: str) -> str:
    return f"guide_last_debug:{user_id}:{session_id}"


def _save_reply_key(user_id: str, session_id: str) -> str:
    return f"guide_last_reply:{user_id}:{session_id}"


async def call_guide_agent_impl(
    message: str,
    user_id: str,
    session_id: str,
    long_term_memories: list[str] | None = None,
) -> str:
    """
    执行公民法律指引首轮对话，保存状态并设置活跃标记。
    供 worker_tools.call_guide_agent 直接调用。
    """
    thread_id = f"{user_id}:{session_id}"
    redis = get_checkpointer_redis()
    request_context_key = f"current_guide_request_context:{user_id}:{session_id}"
    raw_request_context = (
        await redis.get(request_context_key) if hasattr(redis, "get") else None
    )
    if isinstance(raw_request_context, bytes):
        raw_request_context = raw_request_context.decode("utf-8")
    try:
        request_context = (
            json.loads(raw_request_context) if raw_request_context else {}
        )
    except (TypeError, json.JSONDecodeError):
        request_context = {}

    async with AsyncSessionLocal() as db_session:
        deps = build_guide_deps(db_session=db_session)
        reply, new_state = await run_guide(
            user_message=message,
            thread_id=thread_id,
            deps=deps,
            existing_state=None,
            user_id=user_id,
            long_term_memories=long_term_memories or [],
            request_context=request_context,
        )

    # 保存调试信息 + guide_agent原始回复（供路由层透传，短TTL）
    try:
        debug_data = {
            "case_id":          new_state.case_id,
            "case_generation":  new_state.case_generation,
            "domain":           new_state.legal_domain or "",
            "confidence_tier":  new_state.confidence_tier or "",
            "statute_hits":     new_state.law_context_str or "",
            "case_hits":        new_state.case_context_str or "",
            "graph_laws":       new_state.candidate_laws or [],
            "graph_channels":   new_state.relevant_channels or [],
            "fallback_guide":   new_state.fallback_guide,
            "workflow_stage":   new_state.workflow_stage,
            "state_version":    new_state.state_version,
            "event_sequence":   new_state.event_sequence,
            "input_event_type": new_state.input_event_type,
            "requested_route":  new_state.requested_route,
            "guard_status":     new_state.guard_status,
            "guard_report":     new_state.guard_report or None,
            "fact_blackboard_version": new_state.fact_blackboard_version,
            "fact_snapshot_version": new_state.fact_snapshot_version,
            "fact_change_count": len(new_state.fact_changes),
            "fact_conflict_count": len(new_state.fact_conflict_groups),
            "evidence_name_inventory_version": new_state.evidence_name_inventory_version,
            "decision_status": new_state.decision_status,
            "next_route": new_state.next_route,
            "fact_sufficiency": new_state.fact_sufficiency or {},
            "question_batch": new_state.question_batch or {},
            "fact_snapshot_draft": new_state.fact_snapshot_draft,
            "pause_state": new_state.pause_state,
            "internal_evidence_requirements": new_state.internal_evidence_requirements or [],
            "evidence_requirement_changes": new_state.evidence_requirement_changes or [],
            "formal_evidence_requirements": new_state.formal_evidence_requirements or [],
            "delivery_entries": new_state.delivery_entries or [],
            "evidence_plan_version": new_state.evidence_plan_version,
            "evidence_collection_status": new_state.evidence_collection_status or "not_open",
            "evidence_batch_id": new_state.evidence_batch_id or "",
            "evidence_batch_version": new_state.evidence_batch_version,
            "evidence_review_version": new_state.evidence_review_version,
            "evidence_review_id": new_state.evidence_review_id or "",
            "evidence_review_status": new_state.evidence_review_status or "not_started",
            "evidence_reviewed_at": new_state.evidence_reviewed_at or "",
            "evidence_observations": new_state.evidence_observations or [],
            "evidence_basis_refs": new_state.evidence_basis_refs or [],
            "evidence_basis_missing": new_state.evidence_basis_missing or [],
            "pending_evidence_verification": new_state.pending_evidence_verification or [],
            "verification_round_count": new_state.verification_round_count,
            "new_fact_candidates_from_evidence": new_state.new_fact_candidates_from_evidence or [],
            "content_conflicts": new_state.content_conflicts or [],
            "quality_gaps": new_state.quality_gaps or [],
            "unclassified_materials": new_state.unclassified_materials or [],
            "assessment_change_summary": new_state.assessment_change_summary or {},
            "evidence_review_report": new_state.evidence_review_report or {},
            "solution_draft": new_state.solution_draft or {},
            "solution_draft_status": new_state.solution_draft_status or "not_started",
            "solution_generation_id": new_state.solution_generation_id or "",
            "solution_generated_at": new_state.solution_generated_at or "",
            "plan_version_candidate": new_state.plan_version_candidate or "",
            "solution_based_on_fact_snapshot_version": (
                new_state.solution_based_on_fact_snapshot_version
            ),
            "solution_based_on_legal_model_version": (
                new_state.solution_based_on_legal_model_version
            ),
            "solution_based_on_evidence_plan_version": (
                new_state.solution_based_on_evidence_plan_version
            ),
            "solution_based_on_evidence_review_version": (
                new_state.solution_based_on_evidence_review_version
            ),
            "likelihood_assessment": new_state.likelihood_assessment or {},
            "likelihood_tier": new_state.likelihood_tier or "",
            "likelihood_change": new_state.likelihood_change or "",
            "solution_change_summary": new_state.solution_change_summary or {},
            "recommended_routes": new_state.recommended_routes or [],
            "alternative_routes": new_state.alternative_routes or [],
            "immediate_actions": new_state.immediate_actions or [],
            "case_tasks": new_state.case_tasks or [],
            "document_suggestions": new_state.document_suggestions or [],
            "action_basis_refs": new_state.action_basis_refs or [],
            "action_basis_gaps": new_state.action_basis_gaps or [],
            "conditional_plan": new_state.conditional_plan,
            "pending_solution_audit": new_state.pending_solution_audit,
            "solution_audit_status": new_state.solution_audit_status or "not_started",
            "solution_audit_id": new_state.solution_audit_id or "",
            "solution_reviewed_at": new_state.solution_reviewed_at or "",
            "solution_audit_report": new_state.solution_audit_report or {},
            "published_solution": new_state.published_solution or {},
            "plan_version": new_state.plan_version,
            "previous_plan_version": new_state.previous_plan_version,
            "plan_published_at": new_state.plan_published_at or "",
            "solution_version_summaries": [
                {
                    "plan_version": item.get("plan_version"),
                    "previous_plan_version": item.get("previous_plan_version"),
                    "published_at": item.get("published_at"),
                    "reviewed_at": item.get("reviewed_at"),
                    "likelihood_tier": (
                        item.get("solution", {})
                        .get("likelihood_assessment", {})
                        .get("tier")
                        if isinstance(item.get("solution"), dict)
                        else ""
                    ),
                    "change_summary": item.get("change_summary") or {},
                    "published_fingerprint": item.get("published_fingerprint"),
                }
                for item in (new_state.solution_versions or [])
                if isinstance(item, dict)
            ],
            "solution_persistence_status": (
                new_state.solution_persistence_status or "not_saved"
            ),
            "decision_trace_id": new_state.decision_trace_id,
            "retrieval_summary": new_state.retrieval_summary or {},
        }
        await redis.set(
            _save_debug_key(user_id, session_id),
            json.dumps(debug_data, ensure_ascii=False),
            ex=_DEBUG_TTL,
        )
        # 原始回复存 Redis，让 chat.py 直接取用，绕过 Supervisor 重写
        await redis.set(
            _save_reply_key(user_id, session_id),
            reply,
            ex=_DEBUG_TTL,
        )
    except Exception:
        pass

    state_key = f"guide_state:{user_id}:{session_id}"
    active_key = f"guide_active:{user_id}:{session_id}"
    ttl = settings.GUIDE_SESSION_TTL

    # 即使首轮模型降级后尚未形成标准问题，只要已有用户案情也必须保留，
    # 否则下一条短回答会脱离上下文重新开始。
    if new_state.phase == GuidePhase.END:
        if (
            new_state.confirmed_issues
            or new_state.unmatched_issues
            or new_state.case_facts
            or new_state.fact_blackboard
            or new_state.evidence_name_inventory
            or new_state.material_fact_observations
            or new_state.safety_pause_active
        ):
            await set_with_optional_ttl(
                redis,
                state_key,
                new_state.model_dump_json(),
                ttl,
            )
            await set_with_optional_ttl(redis, active_key, "1", ttl)
        return reply

    # 指引未结束：保存状态，设置活跃标记，等待后续轮次
    await set_with_optional_ttl(redis, state_key, new_state.model_dump_json(), ttl)
    await set_with_optional_ttl(redis, active_key, "1", ttl)

    return reply
