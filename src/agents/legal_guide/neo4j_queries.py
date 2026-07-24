"""法律指引 Agent 的 Neo4j 查询：领域→法律+渠道关系图。"""
from __future__ import annotations

from loguru import logger
from neo4j import AsyncDriver


async def query_laws_and_channels(domain: str, neo4j_driver: AsyncDriver) -> dict:
    """
    通过领域节点查询适用法律列表和对口维权渠道。
    返回 {"laws": [...], "channels": [...]}
    """
    if not domain:
        return {"laws": [], "channels": []}

    cypher = """
    MATCH (l:Law)-[:APPLIES_TO]->(d:Domain {name: $domain})
    OPTIONAL MATCH (c:Channel)-[:HANDLES]->(d)
    RETURN
        collect(DISTINCT {title: l.title, category: l.category}) AS laws,
        collect(DISTINCT {name: c.name, channel_type: c.channel_type,
                          phone: c.phone, url: c.url}) AS channels
    """
    try:
        async with neo4j_driver.session() as session:
            result = await session.run(cypher, domain=domain)
            record = await result.single()
            if not record:
                return {"laws": [], "channels": []}
            channels = [c for c in (record["channels"] or []) if c.get("name")]
            laws = [l for l in (record["laws"] or []) if l.get("title")]
            logger.debug(f"Neo4j查询 domain={domain}: {len(laws)}部法律, {len(channels)}个渠道")
            return {"laws": laws, "channels": channels}
    except Exception as e:
        logger.warning(f"Neo4j查询失败 domain={domain}: {e}")
        return {"laws": [], "channels": []}


async def query_channels_by_region(
    domain: str,
    region_code: str,
    neo4j_driver: AsyncDriver,
) -> list[dict]:
    """查询指定领域+地区的渠道（有地区节点时使用）。"""
    if not domain:
        return []
    cypher = """
    MATCH (c:Channel)-[:HANDLES]->(d:Domain {name: $domain})
    WHERE c.region_code = $region OR c.region_code = 'CN'
    RETURN c.name AS name, c.phone AS phone, c.url AS url,
           c.channel_type AS channel_type, c.region_code AS region_code
    ORDER BY c.region_code DESC
    LIMIT 10
    """
    try:
        async with neo4j_driver.session() as session:
            result = await session.run(cypher, domain=domain, region=region_code)
            records = await result.data()
            logger.debug(f"按地区查渠道 domain={domain} region={region_code}: {len(records)}条")
            return records
    except Exception as e:
        logger.warning(f"按地区查渠道失败: {e}")
        return []
