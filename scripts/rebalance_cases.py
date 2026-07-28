"""
补充民事案例数据 + 平衡刑事案例数量

策略：
1. 保留 200 条刑事案例（从 1649 条中随机采样）
2. 补充民事案例各领域 150-200 条：
   - 租房纠纷（housing）: 150 条
   - 劳动纠纷（labor_social_security）: 150 条
   - 消费维权（consumer_market）: 150 条（含原有 6 条）
   - 婚姻家庭（family_marriage）: 100 条
   - 合同商事（contract_commercial）: 100 条

目标总数：约 750 条，各领域相对平衡
"""
import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select, delete
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import LegalCase


async def rebalance_criminal_cases():
    """将刑事案例从 1649 条降采样到 200 条"""
    async with AsyncSessionLocal() as session:
        # 查询所有刑事案例 ID
        stmt = select(LegalCase.id).where(
            LegalCase.domain == "criminal_public_security"
        )
        result = await session.execute(stmt)
        all_ids = [row[0] for row in result.all()]

        print(f"Current criminal cases: {len(all_ids)}")

        if len(all_ids) <= 200:
            print("No need to reduce, already <= 200")
            return

        # 随机保留 200 条，删除其余
        keep_ids = set(random.sample(all_ids, 200))
        delete_ids = [id for id in all_ids if id not in keep_ids]

        print(f"Keeping: {len(keep_ids)}, Deleting: {len(delete_ids)}")

        # 批量删除
        delete_stmt = delete(LegalCase).where(LegalCase.id.in_(delete_ids))
        result = await session.execute(delete_stmt)
        await session.commit()

        print(f"[SUCCESS] Deleted {result.rowcount} criminal cases")
        print(f"[SUCCESS] Remaining criminal cases: 200")


async def main():
    print("=" * 60)
    print("Step 1: Rebalance Criminal Cases")
    print("=" * 60)

    await rebalance_criminal_cases()

    print("\n" + "=" * 60)
    print("Step 1 Completed!")
    print("=" * 60)
    print("\nNext: Supplement civil cases from wenshu.court.gov.cn")
    print("  - Housing disputes: 150 cases")
    print("  - Labor disputes: 150 cases")
    print("  - Consumer disputes: 150 cases")
    print("  - Marriage/family: 100 cases")
    print("  - Contract/commercial: 100 cases")


if __name__ == "__main__":
    asyncio.run(main())
