"""
法律术语知识图谱构建脚本。

从 PostgreSQL articles 表中批量提取核心法律术语，写入：
  - Neo4j: LegalConcept 节点 + BELONGS_TO(Domain) 关系
  - Milvus: legal_term_index（Dense，用于第三层语义兜底术语替换）

通过 Neo4j 精确术语节点和 Milvus 语义索引实现三层法律术语标准化。

用法：
    python scripts/build_legal_concepts.py
    python scripts/build_legal_concepts.py --rebuild   # 重建 Milvus collection
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from neo4j import GraphDatabase
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sqlalchemy import select
from langchain_core.messages import SystemMessage

from src.agents.legal_guide.llm_runtime import build_chat_llm
from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.infra.embedding import get_embedding_model
from src.modules.legal.model import Article, Law

settings = get_settings()
EMBEDDING_DIM = 1024
MILVUS_ALIAS  = "term_init"
BATCH_ARTICLES = 10   # 每批喂给 LLM 的法条数（控制 token 消耗）
BATCH_EMBED    = 50   # 每批向量化数量

# ── 提取 Prompt ───────────────────────────────────────────────────────────────
TERM_EXTRACT_PROMPT = """从以下法律条文中提取核心法律概念/术语。

要求：
1. 每条条文提取2-4个最核心的法律概念短语（4-15个汉字）
2. 优先提取"用户可能遭遇的法律问题类型"，即条文规制的具体行为或权利，例如：
   - 劳动领域：拖欠劳动报酬、违法解除劳动合同、未签劳动合同
   - 消费领域：经营者欺诈、消费者知情权、虚假宣传
   - 房产领域：违规收取押金、违法驱逐租客、非法占用宅基地
3. 不要提取纯程序性词语（"依法处理"、"经审批"、"予以通知"）
4. 不要提取孤立的主体名词（"用人单位"、"劳动者"单独出现时不算）
5. 整个列表去重，不重复

法律条文（共 {count} 条）：
{articles_text}

只输出 JSON 数组，格式：["术语1", "术语2", ...]，不加解释。"""


# ── 数据读取 ──────────────────────────────────────────────────────────────────

async def fetch_articles_by_domain() -> dict[str, list[dict]]:
    """从 PostgreSQL 读取全量法条，按领域分组。"""
    async with AsyncSessionLocal() as session:
        arts = (await session.execute(select(Article))).scalars().all()
        laws = {
            l.id: {"domain": l.domain or "other", "title": l.title}
            for l in (await session.execute(select(Law))).scalars().all()
        }
    by_domain: dict[str, list[dict]] = {}
    for a in arts:
        law_info = laws.get(a.law_id, {})
        domain = law_info.get("domain") or "other"
        by_domain.setdefault(domain, []).append({
            "law_id":     a.law_id,
            "article_no": a.article_no,
            "content":    a.content or "",
            "law_title":  law_info.get("title", ""),
        })
    logger.info("按领域分组: {}", ", ".join(f"{d}={len(v)}" for d, v in by_domain.items()))
    return by_domain


# ── LLM 术语提取 ──────────────────────────────────────────────────────────────

async def extract_terms_for_batch(articles: list[dict], llm) -> list[str]:
    """LLM 从一批法条中提取核心法律术语。"""
    articles_text = "\n---\n".join(
        f"[{a['article_no']}] {a['content'][:300]}" for a in articles
    )
    prompt = TERM_EXTRACT_PROMPT.format(count=len(articles), articles_text=articles_text)
    try:
        resp = await llm.ainvoke([SystemMessage(content=prompt)])
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        terms = json.loads(content)
        # 过滤长度不合理的术语
        return [t.strip() for t in terms if isinstance(t, str) and 4 <= len(t.strip()) <= 20]
    except Exception as e:
        logger.warning(f"术语提取失败（批次跳过）: {e}")
        return []


async def extract_all_terms(by_domain: dict, llm) -> dict[str, list[str]]:
    """对所有领域批量提取术语，返回 {domain: [去重后的术语列表]}。"""
    domain_terms: dict[str, list[str]] = {}
    for domain, articles in by_domain.items():
        logger.info("提取 {} 术语（{} 条法条）...", domain, len(articles))
        terms_set: set[str] = set()
        for i in range(0, len(articles), BATCH_ARTICLES):
            batch = articles[i: i + BATCH_ARTICLES]
            terms = await extract_terms_for_batch(batch, llm)
            terms_set.update(terms)
            logger.debug("  批次 {}/{}: +{} 术语，累计 {}",
                         i // BATCH_ARTICLES + 1,
                         (len(articles) + BATCH_ARTICLES - 1) // BATCH_ARTICLES,
                         len(terms), len(terms_set))
        domain_terms[domain] = sorted(terms_set)
        logger.info("   → {} 个唯一术语", domain, len(terms_set))
    total = sum(len(v) for v in domain_terms.values())
    logger.info("全部提取完成，共 {} 个术语", total)
    return domain_terms


# ── Neo4j 写入 ────────────────────────────────────────────────────────────────

def write_to_neo4j(domain_terms: dict[str, list[str]]):
    """将术语写入 Neo4j 作为 LegalConcept 节点，关联到 Domain。"""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    with driver.session() as s:
        # 添加唯一性约束
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:LegalConcept) REQUIRE n.name IS UNIQUE")
        logger.info("LegalConcept 约束就绪")

        total = 0
        for domain, terms in domain_terms.items():
            if not terms:
                continue
            params = [{"name": t, "domain": domain} for t in terms]

            # MERGE 节点（幂等，可重复执行）
            s.run("""
                UNWIND $params AS p
                MERGE (c:LegalConcept {name: p.name})
                SET c.domain = p.domain
            """, params=params)

            # 关联 Domain（Domain 节点由 init_legal_neo4j.py 已建）
            s.run("""
                UNWIND $params AS p
                MATCH (c:LegalConcept {name: p.name})
                MATCH (d:Domain {name: p.domain})
                MERGE (c)-[:BELONGS_TO]->(d)
            """, params=params)

            total += len(terms)
            logger.info("  {} → {} 个 LegalConcept 节点写入", domain, len(terms))

    driver.close()
    logger.info("Neo4j 写入完成，总计 {} 个 LegalConcept 节点", total)


# ── Milvus 写入 ───────────────────────────────────────────────────────────────

def ensure_term_collection(alias: str, rebuild: bool = False) -> Collection:
    """确保 legal_term_index collection 存在。"""
    name = "legal_term_index"
    if utility.has_collection(name, using=alias):
        if rebuild:
            logger.info("{} 已存在，删除重建", name)
            utility.drop_collection(name, using=alias)
        else:
            logger.info("{} 已存在，断点续传模式", name)
            col = Collection(name, using=alias)
            col.load()
            return col

    fields = [
        FieldSchema("id",        DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema("name",      DataType.VARCHAR, max_length=128),
        FieldSchema("domain",    DataType.VARCHAR, max_length=100),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="法律标准术语语义索引（供第三层语义兜底）")
    col = Collection(name, schema, using=alias)
    col.create_index("embedding", {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 64},
    })
    col.load()
    logger.info("{} 创建成功", name)
    return col


async def write_to_milvus(domain_terms: dict[str, list[str]], col: Collection, embed_model):
    """将术语向量化写入 legal_term_index（按名称全局去重）。"""
    seen: set[str] = set()
    unique_rows: list[dict] = []
    for domain, terms in domain_terms.items():
        for term in terms:
            if term not in seen:
                seen.add(term)
                # ID 截取前 60 字节（避免 VARCHAR 128 超长，中文3字节/字）
                term_id = term.encode("utf-8")[:120].decode("utf-8", errors="ignore")
                unique_rows.append({"id": term_id, "name": term, "domain": domain})

    # 断点续传：跳过已存在的 ID
    from pymilvus import MilvusClient
    client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    existing_ids: set[str] = set()
    try:
        offset = 0
        while True:
            results = client.query(collection_name="legal_term_index",
                                   filter="", output_fields=["id"],
                                   limit=1000, offset=offset)
            if not results:
                break
            existing_ids.update(r["id"] for r in results)
            offset += 1000
            if len(results) < 1000:
                break
        if existing_ids:
            logger.info("断点续传：跳过已有 {} 条", len(existing_ids))
    except Exception:
        pass

    rows_to_write = [r for r in unique_rows if r["id"] not in existing_ids]
    if not rows_to_write:
        logger.info("所有术语已索引，无需处理")
        return

    logger.info("向量化 {} 个术语 → legal_term_index...", len(rows_to_write))
    for i in range(0, len(rows_to_write), BATCH_EMBED):
        batch = rows_to_write[i: i + BATCH_EMBED]
        texts = [r["name"] for r in batch]
        for attempt in range(3):
            try:
                embeddings = await embed_model.aembed_documents(texts)
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise e
        records = [
            {"id": r["id"], "name": r["name"], "domain": r["domain"], "embedding": emb}
            for r, emb in zip(batch, embeddings)
        ]
        col.upsert(records)
        col.flush()
        logger.info("  {}/{}", min(i + BATCH_EMBED, len(rows_to_write)), len(rows_to_write))

    logger.info("legal_term_index 写入完成，共 {} 条", len(rows_to_write))


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main(rebuild: bool = False):
    llm = build_chat_llm(temperature=0.1)  # 低温度保证术语提取稳定
    embed_model = get_embedding_model()

    by_domain = await fetch_articles_by_domain()
    domain_terms = await extract_all_terms(by_domain, llm)

    write_to_neo4j(domain_terms)

    connections.connect(alias=MILVUS_ALIAS,
                        host=settings.MILVUS_HOST,
                        port=settings.MILVUS_PORT)
    col = ensure_term_collection(MILVUS_ALIAS, rebuild=rebuild)
    await write_to_milvus(domain_terms, col, embed_model)
    connections.disconnect(alias=MILVUS_ALIAS)

    logger.info("===== 法律术语知识图谱构建完成 =====")
    logger.info("验证命令：")
    logger.info("  Neo4j Browser: MATCH (c:LegalConcept)-[:BELONGS_TO]->(d:Domain)")
    logger.info("                 RETURN d.name, count(c) ORDER BY count(c) DESC")
    logger.info("  Milvus: legal_term_index 行数查询")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="删除重建 Milvus legal_term_index（默认断点续传）")
    args = parser.parse_args()
    asyncio.run(main(rebuild=args.rebuild))
