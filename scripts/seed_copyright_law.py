"""
著作权法数据种子脚本。
向 PostgreSQL / Neo4j / Milvus 三端同步著作权法核心条文。

用法：
    python scripts/seed_copyright_law.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import Law, Article

settings = get_settings()

LAW_TITLE = "中华人民共和国著作权法"
LAW_DOMAIN = "intellectual_property"
LAW_CATEGORY = "法律"
LAW_AUTHORITY = "全国人民代表大会常务委员会"
LAW_EFFECTIVE = "2021-06-01"

# 著作权法（2020修正）核心条文
ARTICLES = [
    ("第二条",
     "中国公民、法人或者非法人组织的作品，不论是否发表，依照本法享有著作权。"
     "外国人、无国籍人的作品根据其作者所属国或者经常居住地国同中国签订的协议"
     "或者共同参加的国际条约享有的著作权，受本法保护。"),

    ("第三条",
     "本法所称的作品，是指文学、艺术和科学领域内具有独创性并能以一定形式表现的智力成果，包括："
     "（一）文字作品；（二）口述作品；（三）音乐、戏剧、曲艺、舞蹈、杂技艺术作品；"
     "（四）美术、建筑作品；（五）摄影作品；（六）视听作品；"
     "（七）工程设计图、产品设计图、地图、示意图等图形作品和模型作品；"
     "（八）计算机软件；（九）符合作品特征的其他智力成果。"),

    ("第十条",
     "著作权包括下列人身权和财产权："
     "（一）发表权；（二）署名权；（三）修改权；（四）保护作品完整权；"
     "（五）复制权；（六）发行权；（七）出租权；（八）展览权；"
     "（九）表演权；（十）放映权；（十一）广播权；（十二）信息网络传播权；"
     "（十三）摄制权；（十四）改编权；（十五）翻译权；（十六）汇编权；"
     "（十七）应当由著作权人享有的其他权利。"),

    ("第十一条",
     "著作权属于作者，本法另有规定的除外。创作作品的自然人是作者。"
     "由法人或者非法人组织主持，代表法人或者非法人组织意志创作，并由法人或者"
     "非法人组织承担责任的作品，法人或者非法人组织视为作者。"),

    ("第十七条",
     "受委托创作的作品，著作权的归属由委托人和受托人通过合同约定。"
     "合同未作明确约定或者没有订立合同的，著作权属于受托人。"),

    ("第二十三条",
     "在下列情况下使用作品，可以不经著作权人许可，不向其支付报酬，但应当指明作者姓名或者名称、"
     "作品名称，并且不得影响该作品的正常使用，也不得不合理地损害著作权人的合法权益："
     "（一）为个人学习、研究或者欣赏，使用他人已经发表的作品；"
     "（二）为介绍、评论某一作品或者说明某一问题，在作品中适当引用他人已经发表的作品；"
     "（六）将中国公民、法人或者非法人组织已经发表的以汉语言文字创作的作品翻译成少数民族语言"
     "文字作品在国内出版发行。"),

    ("第五十二条",
     "有下列侵权行为的，应当根据情况，承担停止侵害、消除影响、赔礼道歉、赔偿损失等民事责任："
     "（一）未经著作权人许可，发表其作品的；（二）未经合作作者许可，将与他人合作创作的"
     "作品当作自己单独创作的作品发表的；（三）没有参加创作，为谋取个人名利，在他人作品上署名的；"
     "（四）歪曲、篡改他人作品的；（五）剽窃他人作品的；"
     "（六）未经著作权人许可，以展览、摄制视听作品的方法使用作品，或者以改编、翻译、注释等"
     "方式使用作品的；（七）使用他人作品，应当支付报酬而未支付的；"
     "（十一）未经著作权人许可，故意删除或者改变作品的权利管理信息的。"),

    ("第五十四条",
     "侵犯著作权或者与著作权有关的权利的，侵权人应当按照权利人因此受到的实际损失"
     "或者侵权人的违法所得给予赔偿；权利人的实际损失或者侵权人的违法所得难以计算的，"
     "可以参照该权利使用费给予赔偿。对故意侵犯著作权或者与著作权有关的权利，"
     "情节严重的，可以在按照上述方法确定数额的一倍以上五倍以下给予赔偿。"
     "权利人的实际损失、侵权人的违法所得、权利使用费难以计算的，"
     "由人民法院根据侵权行为的情节，判决给予五百元以上五百万元以下的赔偿。"),

    ("第六十条",
     "著作权人或者与著作权有关的权利人有证据证明他人正在实施或者即将实施侵犯其权利的行为，"
     "如不及时制止将会使其合法权益受到难以弥补的损害的，可以依法在提起诉讼前"
     "向人民法院申请采取责令停止有关行为和财产保全的措施。"),
]


async def seed_postgres() -> int:
    """插入著作权法到 PostgreSQL，返回 law_id。"""
    async with AsyncSessionLocal() as session:
        # 幂等：已存在则跳过
        existing = (await session.execute(
            select(Law).where(Law.title == LAW_TITLE)
        )).scalar_one_or_none()

        if existing:
            print(f"[PG] 著作权法已存在 id={existing.id}，跳过插入")
            return existing.id

        law = Law(
            title=LAW_TITLE,
            domain=LAW_DOMAIN,
            category=LAW_CATEGORY,
            authority=LAW_AUTHORITY,
            effective_from=LAW_EFFECTIVE,
        )
        session.add(law)
        await session.flush()
        law_id = law.id

        for art_no, content in ARTICLES:
            session.add(Article(law_id=law_id, article_no=art_no, content=content))

        await session.commit()
        print(f"[PG] 著作权法插入成功 id={law_id}，{len(ARTICLES)} 条法条")
        return law_id


def sync_neo4j(law_id: int):
    """在 Neo4j 中创建 Law 节点和 APPLIES_TO Domain 关系。"""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    with driver.session() as s:
        s.run("""
            MERGE (d:Domain {name: $domain})
            MERGE (l:Law {pg_id: $pg_id})
            SET l.title=$title, l.domain=$domain, l.category=$category, l.authority=$authority
            MERGE (l)-[:APPLIES_TO]->(d)
        """, pg_id=law_id, title=LAW_TITLE, domain=LAW_DOMAIN,
             category=LAW_CATEGORY, authority=LAW_AUTHORITY)

        # Article 节点
        for art_no, _ in ARTICLES:
            s.run("""
                MATCH (l:Law {pg_id: $law_id})
                MERGE (a:Article {pg_id: $art_key})
                SET a.law_pg_id=$law_id, a.article_no=$art_no
                MERGE (l)-[:HAS_ARTICLE]->(a)
            """, law_id=law_id, art_key=f"{law_id}_{art_no}", art_no=art_no)

    driver.close()
    print(f"[Neo4j] 著作权法节点+关系写入完成")


async def sync_milvus(law_id: int):
    """将著作权法法条向量化并 upsert 到 Milvus statute_index。"""
    from pymilvus import connections, Collection
    from src.infra.embedding import get_embedding_model

    connections.connect(
        alias="seed",
        host=settings.MILVUS_HOST,
        port=str(settings.MILVUS_PORT),
    )
    col = Collection("statute_index", using="seed")
    col.load()

    embed_model = get_embedding_model()
    texts = [content for _, content in ARTICLES]
    print(f"[Milvus] 向量化 {len(texts)} 条法条...")
    embeddings = await embed_model.aembed_documents(texts)

    records = []
    for (art_no, content), emb in zip(ARTICLES, embeddings):
        records.append({
            "id":         f"{law_id}_{art_no}",
            "domain":     LAW_DOMAIN,
            "law_id":     str(law_id),
            "article_no": art_no,
            "text":       content[:3000],
            "embedding":  emb,
        })

    col.upsert(records)
    col.flush()
    connections.disconnect("seed")
    print(f"[Milvus] {len(records)} 条法条 upsert 完成")


async def main():
    print("=== 著作权法数据种子 ===")
    law_id = await seed_postgres()
    sync_neo4j(law_id)
    await sync_milvus(law_id)
    print("\n完成！著作权法已同步到 PostgreSQL + Neo4j + Milvus")


if __name__ == "__main__":
    asyncio.run(main())
