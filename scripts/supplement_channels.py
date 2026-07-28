"""建立渠道详情字段并导入全国、北京、上海试点数据（可重复执行）。"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select, func, text
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import Channel

CHANNELS_JSON = ROOT / "data/channels_pilot.json"


SCHEMA_STATEMENTS = [
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS channel_code VARCHAR(100)",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS service_level VARCHAR(20) NOT NULL DEFAULT 'national'",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS description TEXT",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS applicable_matters JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS required_materials JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS service_hours VARCHAR(200)",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS source_org VARCHAR(200)",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS source_url VARCHAR(1000)",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS last_verified_on VARCHAR(10)",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active'",
    "ALTER TABLE channels ADD COLUMN IF NOT EXISTS priority INTEGER NOT NULL DEFAULT 100",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_channels_channel_code ON channels(channel_code)",
    "CREATE INDEX IF NOT EXISTS ix_channels_lookup ON channels(domain, region_code, status, priority)",
]


async def ensure_channel_schema(session) -> None:
    for statement in SCHEMA_STATEMENTS:
        await session.execute(text(statement))
    # 仅规范试点地区的旧缩写；其他地区不在本轮改动范围内。
    await session.execute(text("UPDATE channels SET region_code='110000' WHERE region_code='BJ'"))
    await session.execute(text("UPDATE channels SET region_code='310000' WHERE region_code='SH'"))
    await session.commit()


async def supplement_channels():
    """补充试点渠道，并用 channel_code 保证幂等。"""
    with open(CHANNELS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    async with AsyncSessionLocal() as session:
        await ensure_channel_schema(session)
        added = 0
        updated = 0
        for ch_data in data["channels"]:
            result = await session.execute(
                select(Channel).where(Channel.channel_code == ch_data["channel_code"])
            )
            channel = result.scalar_one_or_none()
            if channel is None and ch_data.get("match_name"):
                result = await session.execute(
                    select(Channel).where(Channel.name == ch_data["match_name"]).limit(1)
                )
                channel = result.scalar_one_or_none()

            values = {k: v for k, v in ch_data.items() if k != "match_name"}
            if channel is None:
                channel = Channel(**values)
                session.add(channel)
                added += 1
                print(f"  [ADD] {ch_data['name']}")
            else:
                for key, value in values.items():
                    setattr(channel, key, value)
                updated += 1
                print(f"  [UPDATE] {ch_data['name']}")

        await session.commit()
        print(f"\n[SUCCESS] added={added}, updated={updated}")

        # 统计当前总数
        count_stmt = select(func.count()).select_from(Channel)
        result = await session.execute(count_stmt)
        total = result.scalar()
        print(f"[TOTAL] Total channels: {total}")


async def main():
    print("=" * 60)
    print("Supplement Channels Data")
    print("=" * 60)
    print(f"Data source: {CHANNELS_JSON}")
    print()

    await supplement_channels()

    print("\n" + "=" * 60)
    print("Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
