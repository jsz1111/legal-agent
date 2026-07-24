# ============================================================
# 法律数据 SQLAlchemy Model 定义
#
# 映射关系（对标 medical/model.py）：
#   Law         ← Disease       法律法规
#   Article     ← Symptom       法条
#   LegalCase   ← Drug          案例
#   Channel     ← Department    维权渠道
#   User        ← Patient       用户档案
#   Consultation← Consultation  咨询记录
# ============================================================

from sqlalchemy import String, Text, Integer, ForeignKey, Index
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
    region_code: Mapped[str] = mapped_column(String(20), nullable=False, comment="地区代码：CN=国家级 / BJ=北京 / SH=上海 等")

    __table_args__ = (
        Index("ix_channels_domain", "domain"),
        Index("ix_channels_region", "region_code"),
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

    facts: Mapped[str] = mapped_column(Text, nullable=False, comment="案件事实描述（CAIL: A/B/C 文本；concrete: 判决正文）")
    domain: Mapped[str] = mapped_column(String(100), nullable=False, comment="法律领域，如：criminal_public_security")
    source: Mapped[str] = mapped_column(String(50), nullable=False, comment="数据来源标签：cail2019_scm / concrete")
    title: Mapped[str | None] = mapped_column(String(300), comment="案件名称（concrete 有，CAIL 无）")
    cause: Mapped[str | None] = mapped_column(String(200), comment="案由，如：劳动争议 / 买卖合同纠纷")
    court: Mapped[str | None] = mapped_column(String(200), comment="审理法院")
    gist: Mapped[str | None] = mapped_column(Text, comment="裁判要旨（concrete 有，CAIL 无）")

    __table_args__ = (
        Index("ix_legal_cases_domain", "domain"),
        Index("ix_legal_cases_source", "source"),
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
# 用户档案表
# 对标 medical/model.py 中的 Patient
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
# 对标 medical/model.py 中的 Consultation
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
    issue_description: Mapped[str | None] = mapped_column(Text, comment="用户诉求描述（对标 chief_complaint）")
    legal_advice: Mapped[str | None] = mapped_column(Text, comment="法律建议结论（对标 diagnosis）")
    action_plan: Mapped[str | None] = mapped_column(Text, comment="行动方案 JSON（对标 prescription）")
    urgency_level: Mapped[str] = mapped_column(
        String(20), default="normal", comment="紧急程度：normal / urgent / emergency"
    )
    session_id: Mapped[str | None] = mapped_column(String(100), comment="关联 Redis 会话 ID")
    milvus_doc_id: Mapped[str | None] = mapped_column(String(100), comment="向量化后在 Milvus 中的文档 ID")

    __table_args__ = (
        Index("ix_consultations_user", "user_id"),
        Index("ix_consultations_channel", "channel_id"),
    )
