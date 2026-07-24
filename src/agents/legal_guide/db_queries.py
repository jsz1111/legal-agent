"""法律指引 Agent 的 PostgreSQL 查询：用户上下文 + 咨询记录保存。"""
from __future__ import annotations

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc


async def load_user_context(
    user_id: str | None,
    db: AsyncSession | None = None,
    recent_limit: int = 3,
) -> dict:
    """
    加载用户历史咨询记录，提取历史领域和地区偏好。
    user_id 为 None 时返回空上下文。
    """
    if not user_id:
        return {}

    from src.infra.database import AsyncSessionLocal
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        from src.modules.legal.model import Consultation
        result = await db.execute(
            select(Consultation)
            .where(Consultation.user_id == user_id)
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
    返回新记录 ID，失败返回 None。
    """
    from src.infra.database import AsyncSessionLocal

    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        from src.modules.legal.model import Consultation
        record = Consultation(
            user_id=user_id,
            session_id=session_id,
            domain=domain,
            chief_complaint="；".join(issues[:5]),
            status="completed",
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
