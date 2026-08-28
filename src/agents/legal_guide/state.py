"""GuideState / GuidePhase：法律指引状态机的分组黑板对象。

业务代码仍可用 ``state.case_facts`` 等旧的平铺属性访问状态；实际持久化和
LangGraph 通道按职责分组，避免把案件事实、追问、证据和检索缓存混成九十余
个顶层字段。旧版 Redis 中的平铺 JSON 会在加载时自动迁移。
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Mapping
import uuid

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GuidePhase(str, Enum):
    CLARIFY = "CLARIFY"
    ISSUE_SEARCH = "ISSUE_SEARCH"
    DETAIL_GATHER = "DETAIL_GATHER"
    CONCLUDE = "CONCLUDE"
    END = "__end__"


class _GuideSubstate(BaseModel):
    """Mutable, assignment-validated state owned by one workflow concern."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class CaseState(_GuideSubstate):
    """Case identity, conversation ownership and stable user context."""

    round: int = 0
    session_id: str = ""
    total_rounds: int = 0
    case_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_generation: int = 1
    awaiting_case_boundary: bool = False
    pending_case_message: str = ""
    case_boundary_audit: list[dict] = Field(default_factory=list)
    turn_control_intent: str = ""
    turn_contains_case_details: bool = False
    user_context: dict = Field(default_factory=dict)
    legal_domain: str = ""
    region: str = ""
    time_info: str = ""


class IssueState(_GuideSubstate):
    """Normalized legal issues and query vocabulary."""

    confirmed_issues: list[str] = Field(default_factory=list)
    unmatched_issues: list[str] = Field(default_factory=list)
    term_map: dict[str, str] = Field(default_factory=dict)
    case_frame: str = ""
    frame_confidence: float = 0.0
    issue_refresh_needed: bool = False
    last_confirmed_count: int = 0


class FactState(_GuideSubstate):
    """Case fact ledger and compatibility projections used by older nodes."""

    collected_facts: list[str] = Field(default_factory=list)
    draftable_facts: list[str] = Field(default_factory=list)
    case_facts: list[dict] = Field(default_factory=list)
    fact_records: dict[str, dict] = Field(default_factory=dict)
    adverse_facts: list[str] = Field(default_factory=list)


class FollowupState(_GuideSubstate):
    """Dynamic follow-up plan, pending questions and deduplication ledger."""

    scenario_analysis: dict = Field(default_factory=dict)
    scenario_confirmation_offered: bool = False
    asked_details: list[str] = Field(default_factory=list)
    pending_ask_details: list[str] = Field(default_factory=list)
    pending_ask_type: str = ""
    asked_followup_ids: list[str] = Field(default_factory=list)
    pending_followup_ids: list[str] = Field(default_factory=list)
    followup_plan: dict = Field(default_factory=dict)
    followup_decision_trace: list[dict] = Field(default_factory=list)
    decision_sufficiency: dict = Field(default_factory=dict)
    followup_basis_refs: list[dict] = Field(default_factory=list)
    followup_basis_graph: list[dict] = Field(default_factory=list)
    followup_basis_fingerprint: str = ""
    followup_basis_error: str = ""
    evidence_collection_offered: bool = False
    asked_decision_keys: list[str] = Field(default_factory=list)
    deferred_questions: list[str] = Field(default_factory=list)
    consecutive_counter_questions: int = 0
    consecutive_low_info_answers: int = 0


class EvidenceState(_GuideSubstate):
    """Evidence checklist, uploaded material assessment and solution versions."""

    evidence_assessments: dict[str, dict] = Field(default_factory=dict)
    evidence_items: list[dict] = Field(default_factory=list)
    proof_targets: list[dict] = Field(default_factory=list)
    evidence_requirements: list[dict] = Field(default_factory=list)
    evidence_requirement_version: int = 0
    evidence_evaluation_version: int = 0
    solution_version: int = 0
    solution_evidence_version: int = 0
    evidence_links: list[dict] = Field(default_factory=list)
    evidence_coverage: dict = Field(default_factory=dict)
    evidence_unavailable: list[str] = Field(default_factory=list)
    evidence_unverified: list[str] = Field(default_factory=list)
    evidence_confirmed: list[str] = Field(default_factory=list)


class RetrievalState(_GuideSubstate):
    """Read-only knowledge retrieval snapshot reused across workflow turns."""

    candidate_laws: list[dict] = Field(default_factory=list)
    retrieved_law_refs: list[dict] = Field(default_factory=list)
    similar_cases: list[dict] = Field(default_factory=list)
    relevant_channels: list[dict] = Field(default_factory=list)
    law_context_str: str = ""
    case_context_str: str = ""
    retrieval_error_note: str = ""
    retrieval_completed: bool = False
    retrieval_fingerprint: str = ""
    fallback_guide: dict | None = None


class ControlState(_GuideSubstate):
    """Convergence, confidence and legacy continuation controls."""

    force_conclude: bool = False
    wants_conclude: bool = False
    awaiting_supplement_choice: bool = False
    supplement_choice_offered: bool = False
    supplement_choice: str = ""
    supplement_has_details: bool = False
    allow_extra_followups: bool = False
    clarify_rounds: int = 0
    ask_rounds: int = 0
    facts_rounds: int = 0
    evidence_rounds: int = 0
    confidence_score: float = 0.0
    confidence_tier: str = "LOW"
    self_review_note: str = ""


class SafetyState(_GuideSubstate):
    """Urgency, personal safety pause and fraud stop-loss state."""

    urgency_level: str = "normal"
    safety_relevant: bool = False
    current_safety_status: str = "not_applicable"
    safety_pause_active: bool = False
    safety_pause_case_message: str = ""
    fraud_stop_loss_relevant: bool = False
    fraud_stop_loss_warning: str = ""
    fraud_stop_loss_offered: bool = False
    time_warning: str = ""


class OutputState(_GuideSubstate):
    """Generated artifacts that must remain attached to the current case."""

    doc_draft: str = ""
    requested_doc_type: str = ""
    latest_plan_text: str = ""  # node_conclude 产出的最终方案；导出 Word 时优先取此字段
    # Final-plan analysis artifacts.  They are intentionally optional so old
    # Redis snapshots and callers can continue to use the legacy fields.
    case_analysis_packet: dict = Field(default_factory=dict)
    issue_map: list[dict] = Field(default_factory=list)
    legal_basis_packet: dict = Field(default_factory=dict)
    issue_analyses: list[dict] = Field(default_factory=list)
    analysis_validation: dict = Field(default_factory=dict)
    analysis_stage: str = ""


class StrategyState(_GuideSubstate):
    """AI strategy-center output for the current case."""

    strategy_plan: dict = Field(default_factory=dict)
    strategy_plan_version: int = 0
    strategy_stage: str = ""
    strategy_validation: dict = Field(default_factory=dict)


_GROUP_MODELS: dict[str, type[_GuideSubstate]] = {
    "case": CaseState,
    "issues": IssueState,
    "facts": FactState,
    "followup": FollowupState,
    "evidence": EvidenceState,
    "retrieval": RetrievalState,
    "control": ControlState,
    "safety": SafetyState,
    "output": OutputState,
    "strategy": StrategyState,
}

_FLAT_FIELD_GROUP: dict[str, str] = {
    field_name: group_name
    for group_name, model in _GROUP_MODELS.items()
    for field_name in model.model_fields
}


class GuideState(BaseModel):
    """Legal-guide graph state with a stable flat compatibility facade.

    Top-level LangGraph channels are intentionally small. Existing callers can
    continue constructing, reading and assigning legacy flat field names.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    phase: GuidePhase = GuidePhase.CLARIFY
    case: CaseState = Field(default_factory=CaseState)
    issues: IssueState = Field(default_factory=IssueState)
    facts: FactState = Field(default_factory=FactState)
    followup: FollowupState = Field(default_factory=FollowupState)
    evidence: EvidenceState = Field(default_factory=EvidenceState)
    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    control: ControlState = Field(default_factory=ControlState)
    safety: SafetyState = Field(default_factory=SafetyState)
    output: OutputState = Field(default_factory=OutputState)
    strategy: StrategyState = Field(default_factory=StrategyState)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_state(cls, value: Any) -> Any:
        """Accept all pre-refactor constructors and persisted flat JSON."""

        if isinstance(value, cls) or not isinstance(value, Mapping):
            return value

        migrated = dict(value)
        for group_name, group_model in _GROUP_MODELS.items():
            raw_group = migrated.get(group_name)
            flat_values = {
                field_name: migrated.pop(field_name)
                for field_name in group_model.model_fields
                if field_name in migrated
            }
            if not flat_values:
                # During validate_assignment Pydantic passes the existing
                # submodel instances back through this validator. Preserve
                # those instances so an unrelated phase assignment cannot
                # turn every nested state into an unvalidated plain dict.
                continue
            if isinstance(raw_group, BaseModel):
                nested = raw_group.model_dump()
            elif isinstance(raw_group, Mapping):
                nested = dict(raw_group)
            else:
                nested = {}
            nested.update(flat_values)
            migrated[group_name] = nested
        return migrated

    def __getattr__(self, name: str) -> Any:
        group_name = _FLAT_FIELD_GROUP.get(name)
        if group_name is not None:
            group = object.__getattribute__(self, group_name)
            return getattr(group, name)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        group_name = _FLAT_FIELD_GROUP.get(name)
        if group_name is not None and group_name in self.__dict__:
            setattr(self.__dict__[group_name], name, value)
            return
        super().__setattr__(name, value)

    def group_updates(self, updates: Mapping[str, Any] | None) -> dict[str, Any]:
        """Translate a legacy node update into grouped LangGraph channels."""

        if not updates:
            return {}

        grouped: dict[str, Any] = {}
        group_copies: dict[str, _GuideSubstate] = {}
        for name, value in updates.items():
            group_name = _FLAT_FIELD_GROUP.get(name)
            if group_name is None:
                grouped[name] = value
                continue
            if group_name not in group_copies:
                group_copies[group_name] = getattr(self, group_name).model_copy()
            setattr(group_copies[group_name], name, value)

        grouped.update(group_copies)
        return grouped

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "GuideState":
        """Keep Pydantic's copy API compatible with flat update dictionaries."""

        return super().model_copy(
            update=self.group_updates(update),
            deep=deep,
        )

    def flat_snapshot(self) -> dict[str, Any]:
        """Return a flat diagnostic view without changing grouped persistence."""

        snapshot: dict[str, Any] = {
            "messages": list(self.messages),
            "phase": self.phase,
        }
        for group_name in _GROUP_MODELS:
            snapshot.update(getattr(self, group_name).model_dump())
        return snapshot
