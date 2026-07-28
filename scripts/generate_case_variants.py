"""
批量生成民事案例变体（基于真实判例模板）

策略：
1. 使用13个种子案例作为模板
2. 通过变换关键要素生成变体（金额、地点、时间、具体情节）
3. 保持案例的真实性和合理性
4. 每个领域生成50-100条
"""
import asyncio
import random
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import LegalCase


# 案例模板和变体参数
CASE_TEMPLATES = {
    "real_estate_construction": [
        {
            "title_template": "{name}诉{company}房屋租赁合同纠纷案",
            "cause": "房屋租赁合同纠纷",
            "facts_template": "原告{name}于{year}年{month}月与被告{company}签订房屋租赁合同，约定租期{period}，押金{deposit}元。租期届满后，{name}按约定交还房屋，但被告以{reason}为由拒绝退还押金。经查，{finding}。",
            "gist_template": "法院认为，{judgment}。被告应退还原告押金{deposit}元及利息。",
            "court_template": "{city}人民法院",
            "variants": 30
        },
        {
            "title_template": "{name}诉{company}物业服务合同纠纷案",
            "cause": "物业服务合同纠纷",
            "facts_template": "原告{name}所在小区物业管理存在{problem}等问题。{name}拒绝缴纳物业费共计{amount}元，物业公司起诉要求支付欠费。",
            "gist_template": "法院认为，物业公司未按约定提供服务，构成违约，但业主仍需支付部分物业费，可相应减免{percent}%。",
            "court_template": "{city}人民法院",
            "variants": 25
        },
    ],
    "labor_social_security": [
        {
            "title_template": "{name}诉{company}追索劳动报酬纠纷案",
            "cause": "追索劳动报酬",
            "facts_template": "原告{name}在被告公司工作{months}个月，公司以{reason}为由拖欠工资共计{amount}元。{name}多次催讨未果，遂提起劳动仲裁。经查，公司确实拖欠工资。",
            "gist_template": "仲裁委裁决公司支付拖欠工资{amount}元、经济补偿金{compensation}元。",
            "court_template": "{city}劳动人事争议仲裁委员会",
            "variants": 30
        },
        {
            "title_template": "{name}诉{company}违法解除劳动合同纠纷案",
            "cause": "劳动合同纠纷",
            "facts_template": "原告{name}在被告{company}担任{position}，工作{years}年。公司以{reason}为由单方面解除劳动合同，未提前通知也未支付经济补偿。",
            "gist_template": "法院认为，公司解除劳动合同程序违法，应支付违法解除劳动合同赔偿金{amount}元。",
            "court_template": "{city}人民法院",
            "variants": 30
        },
    ],
    "consumer_market": [
        {
            "title_template": "{name}诉{company}网络购物合同纠纷案",
            "cause": "网络购物合同纠纷",
            "facts_template": "原告{name}在被告{platform}购买{product}，收到后发现{defect}。{name}要求退货退款并主张三倍赔偿。商家拒绝，称{excuse}。",
            "gist_template": "法院认为，商家{finding}构成虚假宣传，应承担退一赔三责任，退还货款{price}元，赔偿{compensation}元。",
            "court_template": "{city}人民法院",
            "variants": 30
        },
        {
            "title_template": "{name}诉{company}服务合同纠纷案",
            "cause": "服务合同纠纷",
            "facts_template": "原告{name}购买{company}{service}，费用{amount}元。使用{months}个月后{problem}。{name}要求退还剩余费用，{company}以合同约定不可退款为由拒绝。",
            "gist_template": "法院认为，{finding}，应按比例退还剩余费用{refund}元。格式条款中的不可退款条款因显失公平而无效。",
            "court_template": "{city}人民法院",
            "variants": 25
        },
    ],
    "family_marriage": [
        {
            "title_template": "{name1}诉{name2}离婚纠纷案",
            "cause": "离婚纠纷",
            "facts_template": "原告{name1}与被告{name2}结婚{years}年，育有{children}。因{reason}，{name1}起诉离婚。被告不同意离婚。",
            "gist_template": "法院认为，{finding}，准予离婚。{custody}由{parent}抚养，另一方每月支付抚养费{amount}元至子女成年。",
            "court_template": "{city}人民法院",
            "variants": 25
        },
        {
            "title_template": "{name}诉某某抚养费纠纷案",
            "cause": "抚养费纠纷",
            "facts_template": "原告{name}（未成年人）父母离婚后由{parent}抚养，另一方每月支付抚养费{old_amount}元。现{reason}，要求增加抚养费至每月{new_amount}元。",
            "gist_template": "法院认为，{finding}，判决另一方每月支付抚养费{amount}元。",
            "court_template": "{city}人民法院",
            "variants": 20
        },
    ],
    "contract_commercial": [
        {
            "title_template": "{company1}诉{company2}买卖合同纠纷案",
            "cause": "买卖合同纠纷",
            "facts_template": "原告{company1}向被告{company2}采购{product}，约定{payment}。{problem}。被告认为{defense}，要求支付货款{amount}元。",
            "gist_template": "法院{finding}，判决原告支付货款{amount}元及逾期付款利息。",
            "court_template": "{city}人民法院",
            "variants": 25
        },
        {
            "title_template": "{company1}诉{company2}借款合同纠纷案",
            "cause": "借款合同纠纷",
            "facts_template": "原告{company1}向被告{company2}出借资金{amount}元，约定年利率{rate}%，借期{period}。到期后被告仅偿还本金{repaid}元。",
            "gist_template": "法院认为，借款合同合法有效，判决被告偿还本金{principal}元、支付利息{interest}元及逾期利息。",
            "court_template": "{city}人民法院",
            "variants": 25
        },
    ],
}

# 变体参数库
NAMES = ["张某", "李某", "王某", "刘某", "陈某", "杨某", "赵某", "黄某", "周某", "吴某", "徐某", "孙某", "马某", "朱某", "胡某", "郭某", "林某", "何某", "高某", "罗某"]
COMPANIES = ["某房产公司", "某物业公司", "某科技公司", "某餐饮公司", "某制造公司", "某贸易公司", "某投资公司", "某建筑公司"]
CITIES = ["北京市朝阳区", "上海市浦东新区", "广州市天河区", "深圳市南山区", "成都市武侯区", "杭州市西湖区", "武汉市江汉区", "西安市雁塔区", "南京市鼓楼区", "天津市和平区", "重庆市渝中区", "苏州市工业园区"]
PLATFORMS = ["某电商平台", "某购物网站", "某在线商城"]
PRODUCTS = ["手机", "笔记本电脑", "运动鞋", "化妆品", "家具", "电视机", "空调", "冰箱"]
POSITIONS = ["销售经理", "技术工程师", "行政助理", "财务专员", "运营主管", "设计师", "厨师", "司机"]


def generate_case_variants(domain: str, template: Dict, count: int) -> List[Dict]:
    """根据模板生成案例变体"""
    cases = []

    for i in range(count):
        # 随机参数
        name = random.choice(NAMES)
        name1 = random.choice(NAMES)
        name2 = random.choice([n for n in NAMES if n != name1])
        company = random.choice(COMPANIES)
        company1 = random.choice(COMPANIES)
        company2 = random.choice([c for c in COMPANIES if c != company1])
        city = random.choice(CITIES)
        platform = random.choice(PLATFORMS)
        product = random.choice(PRODUCTS)
        position = random.choice(POSITIONS)

        year = random.randint(2021, 2023)
        month = random.randint(1, 12)
        months = random.randint(3, 24)
        years = random.randint(1, 8)
        deposit = random.choice([3000, 5000, 8000, 10000, 15000])
        amount = random.choice([15000, 25000, 35000, 50000, 80000, 120000])
        price = random.choice([800, 1200, 1500, 2000, 3000, 5000])

        # 填充模板
        title = template["title_template"].format(
            name=name, name1=name1, name2=name2,
            company=company, company1=company1, company2=company2,
            platform=platform
        )

        facts = template["facts_template"].format(
            name=name, name1=name1, name2=name2,
            company=company, company1=company1, company2=company2,
            platform=platform, product=product, position=position, city=city,
            year=year, month=month, months=months, years=years,
            deposit=deposit, amount=amount, price=price,
            old_amount=random.choice([800, 1000, 1500, 2000]),
            new_amount=random.choice([2500, 3000, 3500, 4000]),
            period=f"{random.choice([6, 12, 24])}个月",
            reason=random.choice(["经营困难", "业绩不佳", "组织调整", "岗位取消", "性格不合长期分居", "感情不和", "子女上大学后实际支出增加"]),
            problem=random.choice(["突然关门", "设施故障", "服务质量差"]),
            defect=random.choice(["质量不符", "描述不符", "存在瑕疵"]),
            excuse=random.choice(["已在详情页说明", "无质量问题", "过了退换期"]),
            finding=random.choice(["双方分居满2年，感情确已破裂", "长期分居，婚姻关系名存实亡", "子女上大学后实际支出增加，且另一方收入有所提高", "会所停止营业导致合同无法履行"]),
            children=random.choice(["一子", "一女", "一子一女"]),
            parent=random.choice(["母亲", "父亲"]),
            custody=random.choice(["婚生子", "婚生女"]),
            payment=random.choice(["货到付款", "预付30%", "款到发货"]),
            defense=random.choice(["产品符合约定", "已按合同履行", "质量合格"]),
            repaid=amount // 2,
            rate=random.choice([8, 10, 12, 15]),
            service=random.choice(["健身年卡", "美容套餐", "培训课程"])
        )

        gist = template["gist_template"].format(
            name=name, deposit=deposit, amount=amount,
            compensation=amount // 3, price=price,
            judgment=random.choice(["房屋的正常使用磨损不应由承租人承担责任", "出租方违约在先"]),
            finding=random.choice(["在宣传中突出虚假信息", "存在欺诈行为", "未履行告知义务"]),
            percent=random.choice([20, 30, 40]),
            principal=amount // 2,
            interest=int(amount * 0.12),
            refund=int(amount * 0.75)
        )

        court = template["court_template"].format(city=city)

        cases.append({
            "title": title,
            "cause": template["cause"],
            "facts": facts,
            "gist": gist,
            "court": court,
            "domain": domain,
            "source": "generated_variant"
        })

    return cases


async def batch_import_cases():
    """批量导入生成的案例"""
    all_cases = []

    print("Generating case variants...")
    for domain, templates in CASE_TEMPLATES.items():
        print(f"\n[{domain}]")
        for template in templates:
            variants = generate_case_variants(domain, template, template["variants"])
            all_cases.extend(variants)
            print(f"  Generated {len(variants)} variants for: {template['cause']}")

    print(f"\nTotal cases generated: {len(all_cases)}")

    # 导入数据库
    async with AsyncSessionLocal() as session:
        for case_data in all_cases:
            case = LegalCase(
                title=case_data["title"],
                cause=case_data["cause"],
                facts=case_data["facts"],
                gist=case_data["gist"],
                court=case_data["court"],
                domain=case_data["domain"],
                source=case_data["source"]
            )
            session.add(case)

        await session.commit()
        print(f"\n[SUCCESS] Imported {len(all_cases)} cases to database")

        # 统计
        from sqlalchemy import select, func
        stmt = select(
            LegalCase.domain,
            func.count(LegalCase.id).label('count')
        ).group_by(LegalCase.domain)
        result = await session.execute(stmt)

        print("\n[STATISTICS] Final case distribution:")
        total = 0
        for row in result:
            print(f"  {row.domain}: {row.count}")
            total += row.count
        print(f"  TOTAL: {total}")


async def main():
    print("=" * 60)
    print("Batch Generate Civil Case Variants")
    print("=" * 60)
    print()

    await batch_import_cases()

    print("\n" + "=" * 60)
    print("Generation Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
