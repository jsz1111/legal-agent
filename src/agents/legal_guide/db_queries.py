"""法律指引 Agent 的 PostgreSQL 查询：用户上下文 + 咨询记录保存。"""
from __future__ import annotations

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from src.agents.legal_guide.channel_catalog import (
    channel_query_domains,
    channel_query_region_codes,
    fallback_channels,
    rank_channel_candidates,
)


def _channel_to_dict(channel) -> dict:
    return {
        "channel_code": channel.channel_code,
        "name": channel.name,
        "domain": channel.domain,
        "channel_type": channel.channel_type,
        "service_level": channel.service_level,
        "phone": channel.phone,
        "url": channel.url,
        "region_code": channel.region_code,
        "description": channel.description,
        "applicable_matters": channel.applicable_matters or [],
        "required_materials": channel.required_materials or [],
        "service_hours": channel.service_hours,
        "source_org": channel.source_org,
        "source_url": channel.source_url,
        "last_verified_on": channel.last_verified_on,
        "status": channel.status,
        "priority": channel.priority,
    }


async def query_recommended_channels(
    domain: str,
    region: str = "",
    db: AsyncSession | None = None,
    limit: int = 6,
) -> list[dict]:
    """从 PostgreSQL 确定性查询渠道，并按专属渠道、法律服务、政务兜底排序。"""
    from src.infra.database import AsyncSessionLocal
    from src.modules.legal.model import Channel

    own_session = db is None
    if own_session:
        db = AsyncSessionLocal()
    try:
        result = await db.execute(
            select(Channel).where(
                Channel.domain.in_(channel_query_domains(domain)),
                Channel.region_code.in_(channel_query_region_codes(region)),
                Channel.status == "active",
            )
        )
        candidates = [_channel_to_dict(item) for item in result.scalars().all()]
        ranked = rank_channel_candidates(candidates, domain, region, limit=limit)
        if ranked:
            logger.debug("渠道查询 domain={} region={} candidates={} selected={}", domain, region or "CN", len(candidates), len(ranked))
            return ranked
        logger.warning("渠道库无匹配结果，使用内置全国兜底 | domain={} region={}", domain, region)
        return fallback_channels(domain, region, limit=limit)
    except Exception as exc:
        await db.rollback()
        logger.warning("渠道数据库查询失败，使用内置全国兜底: {}", exc)
        return fallback_channels(domain, region, limit=limit)
    finally:
        if own_session:
            await db.close()


async def load_user_context(
    user_id: str | None,
    db: AsyncSession | None = None,
    recent_limit: int = 3,
) -> dict:
    """
    加载用户历史咨询记录，提取历史领域和地区偏好。
    user_id 为 None 时返回空上下文。
    """
    # Consultation.user_id 是整数 FK。字符串测试账号/外部标识只用于 Redis、Milvus，
    # 不应发起 bigint = varchar 查询，否则会让复用的 AsyncSession 事务进入 aborted 状态。
    if not user_id or not str(user_id).isdigit():
        if user_id:
            logger.debug(f"user_id '{user_id}' 非数字型，跳过 PostgreSQL 历史上下文加载")
        return {}

    from src.infra.database import AsyncSessionLocal
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        from src.modules.legal.model import Consultation
        result = await db.execute(
            select(Consultation)
            .where(Consultation.user_id == int(user_id))
            .order_by(desc(Consultation.created_at))
            .limit(recent_limit)
        )
        consultations = result.scalars().all()
        if not consultations:
            return {}

        domains = list({c.domain for c in consultations if getattr(c, "domain", None)})
        region = next(
            (getattr(c, "region", None) for c in consultations if getattr(c, "region", None)),
            "",
        )
        logger.debug(f"加载用户上下文 user_id={user_id}: domains={domains} region={region}")
        return {"prior_domains": domains, "region": region}
    except Exception as e:
        logger.warning(f"加载用户上下文失败: {e}")
        return {}
    finally:
        if _own_session:
            await db.close()


async def save_guide_record(
    user_id: str | None,
    session_id: str,
    domain: str,
    issues: list[str],
    db: AsyncSession | None = None,
) -> int | None:
    """
    将本次法律指引结果保存到 consultations 表。
    user_id 须为数字型（users 表 FK）；非数字型（如 test_user_01）时跳过保存。
    返回新记录 ID，失败返回 None。
    """
    # Consultation.user_id 是 users.id 的整数 FK，非数字 user_id 无法保存
    if not user_id or not str(user_id).isdigit():
        logger.debug(f"user_id '{user_id}' 非数字型，跳过咨询记录保存")
        return None

    from src.infra.database import AsyncSessionLocal

    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        from src.modules.legal.model import Consultation
        record = Consultation(
            user_id=int(user_id),
            session_id=session_id,
            issue_description=f"[{domain}] " + "；".join(issues[:5]),
            urgency_level="normal",
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        logger.info(f"法律指引记录保存成功 ID={record.id}")
        return record.id
    except Exception as e:
        await db.rollback()
        logger.error(f"保存指引记录失败: {e}")
        return None
    finally:
        if _own_session:
            await db.close()
