"""GuideState / GuidePhase：法律指引状态机的黑板对象。"""
from __future__ import annotations

from enum import Enum
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class GuidePhase(str, Enum):
    CLARIFY       = "CLARIFY"       # 描述模糊，引导澄清
    ISSUE_SEARCH  = "ISSUE_SEARCH"  # 已提取法律问题，开始检索
    DETAIL_GATHER = "DETAIL_GATHER" # 追问关键细节/证据
    CONCLUDE      = "CONCLUDE"      # 生成行动方案
    END           = "__end__"


class GuideState(BaseModel):
    # 对话消息历史
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # 流程控制
    phase: GuidePhase = GuidePhase.CLARIFY
    round: int = 0
    session_id: str = ""

    # 法律问题追踪（对标症状四元组）
    confirmed_issues: list[str] = Field(default_factory=list)   # 已标准化的法律问题
    unmatched_issues: list[str] = Field(default_factory=list)   # 无法标准化的描述
    asked_details: list[str] = Field(default_factory=list)      # 已追问过的细节（防重复）
    pending_ask_details: list[str] = Field(default_factory=list) # 本轮追问内容，供 parse_details 解析

    # 检索结果（issue_search节点写入，conclude节点读取）
    candidate_laws: list[dict] = Field(default_factory=list)      # graph查询到的适用法律元信息
    similar_cases: list[dict] = Field(default_factory=list)       # graph查询到的类案列表
    relevant_channels: list[dict] = Field(default_factory=list)   # channels 查询结果
    law_context_str: str = ""    # statute_rag 格式化文本（直接传入 CONCLUDE_PROMPT）
    case_context_str: str = ""   # case_rag 格式化文本

    # 用户上下文
    user_context: dict = Field(default_factory=dict)   # 历史咨询记录（从PG加载）
    legal_domain: str = ""                             # 锁定的法律领域
    region: str = ""                                   # 用户所在地区（影响渠道推荐）
    evidence_confirmed: list[str] = Field(default_factory=list)  # 用户确认已有的证据

    # 控制字段
    force_conclude: bool = False       # True = 达到轮次上限，强制收敛
    urgency_level: str = "normal"      # critical / time / normal
    time_warning: str = ""             # 时效提醒文案（由 check_urgency 生成）

    # 置信度打分（issue_search 节点写入，conclude 节点读取，决定输出档次）
    confidence_score: float = 0.0      # 0~1 维权方案置信度
    confidence_tier: str = "LOW"       # HIGH / MEDIUM / LOW
