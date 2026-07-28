# ============================================================
# 法律数据 SQLAlchemy Model 定义
#
# 包含法律法规、法条、案例、维权渠道、用户档案和咨询记录等领域模型。
# ============================================================

from sqlalchemy import JSON, String, Text, Integer, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base_model import BaseModel


# ----------------------------------------------------------
# 维权渠道表
# 来源：batch-06-citizen-actions-all-regions/download_report.json
# 用途：咨询结束后推送相关投诉/求助渠道（如 12315、劳动仲裁委等）
# ----------------------------------------------------------
class Channel(BaseModel):
    __tablename__ = "channels"

    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="渠道名称，如：12315消费者投诉热线")
    domain: Mapped[str] = mapped_column(String(100), nullable=False, comment="主要适用法律领域，如：consumer_market")
    channel_type: Mapped[str] = mapped_column(String(50), nullable=False, comment="渠道类型：hotline/website/app")
    phone: Mapped[str | None] = mapped_column(String(50), comment="联系电话（contacts 中 kind=phone 的第一条）")
    url: Mapped[str | None] = mapped_column(String(500), comment="官方网址（contacts 中 kind=website 的第一条）")
    region_code: Mapped[str] = mapped_column(String(20), nullable=False, comment="地区代码：CN=全国，其他使用行政区划代码")
    channel_code: Mapped[str | None] = mapped_column(String(100), comment="稳定渠道编号，用于更新和去重")
    service_level: Mapped[str] = mapped_column(
        String(20), default="national", comment="服务层级：national/province/city"
    )
    description: Mapped[str | None] = mapped_column(Text, comment="渠道职责和使用说明")
    applicable_matters: Mapped[list] = mapped_column(JSON, default=list, comment="适用事项列表")
    required_materials: Mapped[list] = mapped_column(JSON, default=list, comment="办理前建议准备的材料")
    service_hours: Mapped[str | None] = mapped_column(String(200), comment="服务时间或查询提示")
    source_org: Mapped[str | None] = mapped_column(String(200), comment="渠道信息发布机关")
    source_url: Mapped[str | None] = mapped_column(String(1000), comment="官方来源页面")
    last_verified_on: Mapped[str | None] = mapped_column(String(10), comment="最近人工核验日期 YYYY-MM-DD")
    status: Mapped[str] = mapped_column(String(20), default="active", comment="active/outdated/unverified")
    priority: Mapped[int] = mapped_column(Integer, default=100, comment="同层级推荐优先级，数值越小越优先")

    __table_args__ = (
        Index("ix_channels_domain", "domain"),
        Index("ix_channels_region", "region_code"),
        Index("uq_channels_channel_code", "channel_code", unique=True),
        Index("ix_channels_lookup", "domain", "region_code", "status", "priority"),
    )


# ----------------------------------------------------------
# 法律法规表
# 来源：batch-04/download_report.json（72部法律/行政法规/司法解释）
# 用途：
#   1. 法条结构化查询（法律名→法条内容）
#   2. 法律图谱节点（Neo4j Law 节点）
# ----------------------------------------------------------
class Law(BaseModel):
    __tablename__ = "laws"

    title: Mapped[str] = mapped_column(String(300), nullable=False, comment="法律全称，如：中华人民共和国劳动法")
    category: Mapped[str] = mapped_column(String(50), nullable=False, comment="类别：法律/行政法规/司法解释")
    authority: Mapped[str | None] = mapped_column(String(200), comment="发布机关，如：全国人民代表大会常务委员会")
    domain: Mapped[str] = mapped_column(String(100), nullable=False, comment="法律领域，如：labor_social_security")
    effective_from: Mapped[str | None] = mapped_column(String(20), comment="施行日期，格式 YYYY-MM-DD")
    file_path: Mapped[str | None] = mapped_column(String(500), comment="本地 docx/pdf 相对路径")

    __table_args__ = (
        Index("ix_laws_domain", "domain"),
        Index("ix_laws_title", "title"),
    )


# ----------------------------------------------------------
# 法条表
# 来源：batch-04 docx 文件（python-docx 解析，正则提取"第X条"）
# 用途：
#   1. 法条精确查询
#   2. 法条向量化后入 Milvus statute_index（语义检索）
#   3. Neo4j Law-Article 关系
# ----------------------------------------------------------
class Article(BaseModel):
    __tablename__ = "articles"

    law_id: Mapped[int] = mapped_column(
        ForeignKey("laws.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属法律 ID"
    )
    article_no: Mapped[str] = mapped_column(String(50), nullable=False, comment="条号，如：第一条 / 第二十三条")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="条文正文")

    __table_args__ = (
        Index("ix_articles_law", "law_id"),
        Index("ix_articles_no", "article_no"),
    )


# ----------------------------------------------------------
# 案例表
# 来源：
#   1. data/sources/cail2019_scm/ — 刑事类案三元组（criminal_public_security）
#   2. data/sources/formal/concrete/ — 劳动/消费 HTML 案例
# 用途：类案推送（暂仅覆盖刑事/劳动/消费三域，其余域查询返回空）
# ----------------------------------------------------------
class LegalCase(BaseModel):
    __tablename__ = "legal_cases"

    facts: Mapped[str] = mapped_column(Text, nullable=False, comment="案件事实或精简检索文本")
    domain: Mapped[str] = mapped_column(String(100), nullable=False, comment="法律领域，如：criminal_public_security")
    source: Mapped[str] = mapped_column(String(50), nullable=False, comment="数据来源标签：cail2019_scm / concrete")
    title: Mapped[str | None] = mapped_column(String(300), comment="案件名称（concrete 有，CAIL 无）")
    cause: Mapped[str | None] = mapped_column(String(200), comment="案由，如：劳动争议 / 买卖合同纠纷")
    court: Mapped[str | None] = mapped_column(String(200), comment="审理法院")
    gist: Mapped[str | None] = mapped_column(Text, comment="裁判要旨（concrete 有，CAIL 无）")

    # 通用案例库字段。旧数据允许为空，新数据以 case_id 作为跨系统稳定标识。
    case_id: Mapped[str | None] = mapped_column(String(64), comment="稳定案例 ID")
    original_url: Mapped[str | None] = mapped_column(String(1000), comment="裁判文书原始链接")
    case_number: Mapped[str | None] = mapped_column(String(200), comment="案号")
    region: Mapped[str | None] = mapped_column(String(100), comment="所属地区")
    case_type: Mapped[str | None] = mapped_column(String(50), comment="案件类型")
    case_type_code: Mapped[str | None] = mapped_column(String(20), comment="案件类型编码")
    procedure: Mapped[str | None] = mapped_column(String(50), comment="审理程序")
    judgment_date: Mapped[str | None] = mapped_column(String(20), comment="裁判日期")
    publication_date: Mapped[str | None] = mapped_column(String(20), comment="公开日期")
    parties: Mapped[str | None] = mapped_column(Text, comment="当事人")
    legal_basis: Mapped[str | None] = mapped_column(Text, comment="法律依据")
    full_text: Mapped[str | None] = mapped_column(Text, comment="清洗后的裁判文书全文")
    full_text_length: Mapped[int | None] = mapped_column(Integer, comment="清洗后全文字数")
    retrieval_text: Mapped[str | None] = mapped_column(Text, comment="用于类案召回的精简文本")
    selection_tags: Mapped[str | None] = mapped_column(Text, comment="质量筛选标签")

    __table_args__ = (
        Index("ix_legal_cases_domain", "domain"),
        Index("ix_legal_cases_source", "source"),
        Index("uq_legal_cases_case_id", "case_id", unique=True),
        Index("ix_legal_cases_case_number", "case_number"),
        Index("ix_legal_cases_region", "region"),
        Index("ix_legal_cases_procedure", "procedure"),
        Index("ix_legal_cases_judgment_date", "judgment_date"),
    )


# ----------------------------------------------------------
# 法条-案例关联表
# 用途：记录哪些法条被哪些案例引用（阶段二导入时按需构建）
# ----------------------------------------------------------
class LawCase(BaseModel):
    __tablename__ = "law_cases"

    law_id: Mapped[int] = mapped_column(
        ForeignKey("laws.id", ondelete="CASCADE"),
        nullable=False,
        comment="法律 ID"
    )
    case_id: Mapped[int] = mapped_column(
        ForeignKey("legal_cases.id", ondelete="CASCADE"),
        nullable=False,
        comment="案例 ID"
    )

    __table_args__ = (
        Index("ix_law_cases_law", "law_id"),
        Index("ix_law_cases_case", "case_id"),
    )


# ----------------------------------------------------------
# 权威依据来源及追问规则引用
# PostgreSQL 只保存可审计的来源、版本和映射；规则执行仍以 JSON 题库为准。
# ----------------------------------------------------------
class AuthoritySource(BaseModel):
    __tablename__ = "authority_sources"

    source_key: Mapped[str] = mapped_column(String(120), nullable=False, comment="跨版本稳定来源编号")
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="官方文件或系统规则名称")
    issuer: Mapped[str] = mapped_column(String(500), nullable=False, comment="发布机关")
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="official_law/official_form/official_guide/system_rule"
    )
    authority_level: Mapped[str] = mapped_column(String(50), nullable=False, comment="权威依据等级")
    official_url: Mapped[str | None] = mapped_column(String(1200), comment="官方发布页")
    domains: Mapped[list] = mapped_column(JSON, default=list, comment="适用法律领域")
    usage_note: Mapped[str | None] = mapped_column(Text, comment="适用边界和免责声明")
    status: Mapped[str] = mapped_column(String(30), default="active", comment="active/outdated/system_only")

    __table_args__ = (
        Index("uq_authority_sources_source_key", "source_key", unique=True),
        Index("ix_authority_sources_status", "status"),
    )


class AuthorityVersion(BaseModel):
    __tablename__ = "authority_versions"

    source_id: Mapped[int] = mapped_column(
        ForeignKey("authority_sources.id", ondelete="CASCADE"), nullable=False
    )
    version_key: Mapped[str] = mapped_column(String(180), nullable=False, comment="来源版本稳定编号")
    document_no: Mapped[str | None] = mapped_column(String(150), comment="官方文号")
    published_at: Mapped[str | None] = mapped_column(String(30), comment="发布日期")
    effective_from: Mapped[str | None] = mapped_column(String(30), comment="生效日期")
    effective_to: Mapped[str | None] = mapped_column(String(30), comment="失效日期")
    official_file_url: Mapped[str | None] = mapped_column(String(1500), comment="官方原件下载地址")
    local_path: Mapped[str | None] = mapped_column(String(1000), comment="本地官方原件相对路径")
    sha256: Mapped[str | None] = mapped_column(String(64), comment="原件 SHA256")
    content_type: Mapped[str | None] = mapped_column(String(100), comment="原件 MIME 类型")
    review_status: Mapped[str] = mapped_column(
        String(40), default="pending_legal_review", comment="完整性和法律审校状态"
    )
    verified_at: Mapped[str | None] = mapped_column(String(40), comment="最近完整性核验时间")
    source_metadata: Mapped[dict] = mapped_column(JSON, default=dict, comment="下载报告和版本补充元数据")

    __table_args__ = (
        Index("uq_authority_versions_version_key", "version_key", unique=True),
        Index("ix_authority_versions_source", "source_id"),
        Index("ix_authority_versions_review", "review_status"),
    )


class FollowupRuleCitation(BaseModel):
    __tablename__ = "followup_rule_citations"

    rule_id: Mapped[str] = mapped_column(String(120), nullable=False, comment="followup_catalog 规则 ID")
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    rule_type: Mapped[str] = mapped_column(String(20), nullable=False, comment="fact/evidence")
    rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_version_id: Mapped[int] = mapped_column(
        ForeignKey("authority_versions.id", ondelete="CASCADE"), nullable=False
    )
    locator: Mapped[str | None] = mapped_column(Text, comment="条款、页码或栏目定位")
    source_excerpt: Mapped[str | None] = mapped_column(Text, comment="经审校的官方原文摘录")
    derivation_note: Mapped[str | None] = mapped_column(Text, comment="从依据到追问点的转化说明")
    mapping_status: Mapped[str] = mapped_column(
        String(40), default="needs_pinpoint", comment="pinpointed/source_located/needs_pinpoint/system_only"
    )

    __table_args__ = (
        UniqueConstraint("rule_id", "source_version_id", name="uq_followup_rule_source_version"),
        Index("ix_followup_citations_domain", "domain"),
        Index("ix_followup_citations_rule", "rule_id"),
        Index("ix_followup_citations_status", "mapping_status"),
    )


# ----------------------------------------------------------
# 用户档案表
# 用途：记录咨询用户基本信息，支持历史咨询关联
# ----------------------------------------------------------
class User(BaseModel):
    __tablename__ = "users"

    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="用户姓名")
    gender: Mapped[str | None] = mapped_column(String(10), comment="性别：男/女")
    age: Mapped[int | None] = mapped_column(Integer, comment="年龄")
    phone: Mapped[str | None] = mapped_column(String(20), comment="联系电话")

    __table_args__ = (
        Index("ix_users_name", "name"),
    )


# ----------------------------------------------------------
# 咨询记录表
# 用途：
#   1. 历史咨询语义检索（向量化后存 Milvus）
#   2. 运营数据查询 NL2SQL
# ----------------------------------------------------------
class Consultation(BaseModel):
    __tablename__ = "consultations"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="用户 ID"
    )
    channel_id: Mapped[int | None] = mapped_column(
        ForeignKey("channels.id", ondelete="SET NULL"),
        comment="推荐维权渠道 ID"
    )
    issue_description: Mapped[str | None] = mapped_column(Text, comment="用户诉求描述")
    legal_advice: Mapped[str | None] = mapped_column(Text, comment="法律建议结论")
    action_plan: Mapped[str | None] = mapped_column(Text, comment="行动方案 JSON")
    urgency_level: Mapped[str] = mapped_column(
        String(20), default="normal", comment="紧急程度：normal / urgent / emergency"
    )
    session_id: Mapped[str | None] = mapped_column(String(100), comment="关联 Redis 会话 ID")
    milvus_doc_id: Mapped[str | None] = mapped_column(String(100), comment="向量化后在 Milvus 中的文档 ID")

    __table_args__ = (
        Index("ix_consultations_user", "user_id"),
        Index("ix_consultations_channel", "channel_id"),
    )
