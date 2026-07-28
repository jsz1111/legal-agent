"""中国法律年鉴统计库的独立 PostgreSQL 连接池。"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings


settings = get_settings()

legal_statistics_engine = create_async_engine(
    settings.LEGAL_STATISTICS_DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=60 * 5,
    pool_pre_ping=True,
)

LegalStatisticsSessionLocal = async_sessionmaker(
    legal_statistics_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
