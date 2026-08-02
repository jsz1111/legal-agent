"""GuideState / GuidePhase：法律指引状态机的黑板对象。"""
from __future__ import annotations

from enum import Enum
from typing import Annotated
import uuid
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
    round: int = 0  # 用户消息轮次，只由 prepare_turn 每轮递增一次
    session_id: str = ""
    total_rounds: int = 0  # 兼容旧状态的总轮次计数，用于强制收敛
    case_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_generation: int = 1
    awaiting_case_boundary: bool = False
    pending_case_message: str = ""
    case_boundary_audit: list[dict] = Field(default_factory=list)
    turn_control_intent: str = ""
    turn_contains_case_details: bool = False

    # 法律问题追踪（对标症状四元组）
    # confirmed_issues / unmatched_issues 是两个互不合并的池：
    #   confirmed_issues = 法条正文用语 → 喂 BM25 sparse_query + PG ilike + Dense
    #   unmatched_issues = 未能标准化的口语 → 只喂 Dense/HyDE
    # 合并会让口语词流进字面匹配通道（BM25/LIKE），那两个通道对口语零召回。
    confirmed_issues: list[str] = Field(default_factory=list)   # 标准法律术语（唯一可用于字面匹配）
    unmatched_issues: list[str] = Field(default_factory=list)   # 无法标准化的口语描述
    term_map: dict[str, str] = Field(default_factory=dict)      # {口语原词: 标准术语}，调试面板展示
    collected_facts: list[str] = Field(default_factory=list)    # 跨轮累积的金额、时间、关系、行为等案情事实
    draftable_facts: list[str] = Field(default_factory=list)    # 用户清晰陈述、可安全用于文书的事实；疑问/冲突/推测不得进入
    case_facts: list[dict] = Field(default_factory=list)        # 通用原子案情：语义键、原文、确定性、轮次和修订状态
    asked_details: list[str] = Field(default_factory=list)      # 已追问过的细节（防重复）
    pending_ask_details: list[str] = Field(default_factory=list) # 本轮追问内容，供 parse_details 解析
    pending_ask_type: str = ""                                  # facts / evidence，避免跨轮追问类型串线
    asked_followup_ids: list[str] = Field(default_factory=list)  # 结构化题库 ID，跨文案稳定防重复
    pending_followup_ids: list[str] = Field(default_factory=list) # 当前待答题库 ID（通常仅1项）
    followup_plan: dict = Field(default_factory=dict)             # assess_retrieve 动态生成，ask_followup 负责展示
    followup_decision_trace: list[dict] = Field(default_factory=list)
    decision_sufficiency: dict = Field(default_factory=dict)
    followup_basis_refs: list[dict] = Field(default_factory=list)  # 追问阶段轻量检索依据；明确不含类案
    followup_basis_graph: list[dict] = Field(default_factory=list)
    followup_basis_fingerprint: str = ""
    followup_basis_error: str = ""
    evidence_collection_offered: bool = False
    issue_refresh_needed: bool = False                            # 主动补充超出原追问范围时重跑问题/领域识别
    asked_decision_keys: list[str] = Field(default_factory=list)  # 已追问的法律决策点，跨问法去重
    fact_records: dict[str, dict] = Field(default_factory=dict)   # 用户陈述的清晰度/冲突状态，不代表已查证
    evidence_assessments: dict[str, dict] = Field(default_factory=dict) # 存在性、相关性、真实性和可采性分层
    evidence_items: list[dict] = Field(default_factory=list)      # 结构化证据项及其来源、完整性等基础属性
    proof_targets: list[dict] = Field(default_factory=list)       # 当前领域需要覆盖的证明目标
    evidence_requirements: list[dict] = Field(default_factory=list) # 随事实增量维护的证据准备清单
    evidence_requirement_version: int = 0
    evidence_evaluation_version: int = 0  # 每次收到附件后递增；重复提交也形成新评估版本
    solution_version: int = 0              # 每次生成或更新维权方案后递增
    solution_evidence_version: int = 0     # 当前方案已纳入的证据评估版本
    evidence_links: list[dict] = Field(default_factory=list)      # 证据项与证明目标之间的可解释关联
    evidence_coverage: dict = Field(default_factory=dict)         # 证明目标覆盖、缺口和补强建议
    evidence_unavailable: list[str] = Field(default_factory=list) # 用户明确表示没有的证据
    evidence_unverified: list[str] = Field(default_factory=list)  # 图片/转述中提到但本次未直接核验的材料
    deferred_questions: list[str] = Field(default_factory=list)  # 追问期间用户反问、尚未答复的问题
    consecutive_counter_questions: int = 0                       # 连续未回答待确认项的反问次数
    consecutive_low_info_answers: int = 0                        # 连续真正未推进案情或决策的回答

    # 检索结果（issue_search节点写入，conclude节点读取）
    candidate_laws: list[dict] = Field(default_factory=list)      # graph查询到的适用法律元信息
    retrieved_law_refs: list[dict] = Field(default_factory=list)  # 本轮RAG真实法条，供动态追问依据引用
    similar_cases: list[dict] = Field(default_factory=list)       # graph查询到的类案列表
    relevant_channels: list[dict] = Field(default_factory=list)   # channels 查询结果
    law_context_str: str = ""    # statute_rag 格式化文本（直接传入 CONCLUDE_PROMPT）
    case_context_str: str = ""   # case_rag 格式化文本

    # 用户上下文
    user_context: dict = Field(default_factory=dict)   # 历史咨询记录（从PG加载）
    legal_domain: str = ""                             # 锁定的法律领域
    region: str = ""                                   # 用户所在地区（影响渠道推荐）
    time_info: str = ""                                # 用户已确认的时间信息
    evidence_confirmed: list[str] = Field(default_factory=list)  # 用户称持有/已上传材料，不等于真实性或效力已确认

    # 控制字段
    force_conclude: bool = False       # True = 达到轮次上限，强制收敛
    wants_conclude: bool = False       # 用户明确要求停止追问并按现有信息给方案
    awaiting_supplement_choice: bool = False  # 兼容旧会话；新流程不再展示二次选择菜单
    supplement_choice_offered: bool = False   # 兼容旧会话持久化字段
    supplement_choice: str = ""              # 兼容旧会话中的 continue / conclude
    supplement_has_details: bool = False      # 选择语句中是否同时包含需要先入库的新案情或材料
    allow_extra_followups: bool = False       # 兼容旧会话；新流程由信息增益直接决定追问或收敛
    urgency_level: str = "normal"      # critical / time / normal
    safety_relevant: bool = False        # 当前纠纷是否涉及用户或他人的人身安全
    current_safety_status: str = "not_applicable"  # danger / safe / unknown / not_applicable
    safety_pause_active: bool = False     # 现实危险处理中暂停普通流程；确认安全后恢复同一案件
    safety_pause_case_message: str = ""   # 暂停前的原始危险陈述，恢复后归入同一案件
    fraud_stop_loss_relevant: bool = False  # 高确定性诈骗表述需要展示紧急止损提示
    fraud_stop_loss_warning: str = ""       # 尚未向用户展示的止付、冻结、报警提示
    fraud_stop_loss_offered: bool = False   # 同一案件已展示过止损提示，避免每轮重复
    time_warning: str = ""             # 时效提醒文案（由 check_urgency 生成）
    clarify_rounds: int = 0            # 澄清轮数（上限 2 轮）
    ask_rounds: int = 0                # 事实+证据追问总轮数
    facts_rounds: int = 0              # 追问法律细节轮数
    evidence_rounds: int = 0           # 追问证据轮数
    last_confirmed_count: int = 0      # 上次检索时的 confirmed_issues 数量

    # 置信度打分（score 节点写入，conclude 节点读取，决定输出档次）
    confidence_score: float = 0.0      # 0~1 维权方案置信度
    confidence_tier: str = "LOW"       # HIGH / MEDIUM / LOW
    self_review_note: str = ""         # 自省降档原因（HIGH→MID 时记录）
    retrieval_error_note: str = ""     # 检索服务异常降级提示
    retrieval_completed: bool = False   # 已完成过一次检索；收敛指令可复用上一轮结果
    retrieval_fingerprint: str = ""     # 生成检索快照时的案情指纹，避免无实质变化时重复检索

    # 案例检索兜底指引（case_rag 返回空时生成）
    fallback_guide: dict | None = None  # {"platform": str, "url": str, "search_tips": str}

    # 胜算评估辅助：不利于用户的事实（由 node_parse_details 跨轮积累）
    adverse_facts: list[str] = Field(default_factory=list)
    # 例：['用户为违约方（提前退租）', '合同明确约定押金不退', '未提前通知房东']

    # 文书生成
    doc_draft: str = ""  # API 文书生成入口写入，供前端展示或下载
    requested_doc_type: str = ""  # 用户明确指定的文书类型；为空时按当前维权阶段选择默认类型
