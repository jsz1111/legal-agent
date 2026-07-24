"""agents/legal_knowledge/ 所有提示词常量。"""

# ── 法条 RAG ──────────────────────────────────────────────────────────────────
STATUTE_QA_PROMPT = """你是专业的法律助手。根据以下法条内容回答用户的法律问题。

法条内容：
{context}

用户问题：{question}

回答要求：
1. 直接引用相关法条（注明法律名称和条号，如"《劳动合同法》第三十条"）
2. 用通俗语言解释法条含义
3. 说明该法条如何适用于用户的具体情况
4. 若法条内容不足以完整回答，请明确说明"""

# ── 类案 RAG ──────────────────────────────────────────────────────────────────
CASE_QA_PROMPT = """你是专业的法律助手。根据以下类似案例回答用户的问题。

类案信息：
{context}

用户问题：{question}

回答要求：
1. 列出最相关的类案及其裁判要旨
2. 说明类案与用户情况的相似之处
3. 参考类案裁判结果给出参考意见
4. 末尾注明"以上内容仅供参考，具体情况建议咨询专业律师"
"""

# ── 知识图谱 RAG ──────────────────────────────────────────────────────────────
LEGAL_ENTITY_EXTRACT_PROMPT = """从以下法律问题中提取关键实体，返回 JSON 格式。

问题：{question}

提取以下类型的实体：
- domain: 法律领域（如 labor_social_security / consumer_market / contract_commercial / criminal_public_security / real_estate / intellectual_property 等，可为空列表）
- law: 具体法律名称（如《劳动合同法》《消费者权益保护法》，可为空列表）
- legal_term: 法律术语（如 劳动仲裁 / 工伤赔偿 / 合同违约，可为空列表）
- channel: 维权渠道（如 劳动仲裁委 / 消费者协会 / 12315，可为空列表）

只返回 JSON，格式：
{{"domain": [], "law": [], "legal_term": [], "channel": []}}"""

LEGAL_NL2CYPHER_PROMPT = """你是 Neo4j Cypher 专家。根据以下图谱 Schema 和用户问题，生成 Cypher 查询语句。

图谱 Schema：
节点：
  Law   {{pg_id: int, title: str, domain: str, category: str, authority: str}}
  Article {{pg_id: int, law_pg_id: int, article_no: str}}
  Channel {{pg_id: int, name: str, domain: str, channel_type: str, region_code: str, phone: str, url: str}}
  Domain  {{name: str}}

关系：
  (Law)-[:HAS_ARTICLE]->(Article)
  (Law)-[:APPLIES_TO]->(Domain)
  (Channel)-[:HANDLES]->(Domain)

常用查询模式：
// 通过领域找适用法律和对口渠道
MATCH (l:Law)-[:APPLIES_TO]->(d:Domain {{name: $domain}})<-[:HANDLES]-(c:Channel)
RETURN l.title, d.name, c.name AS channel_name, c.phone, c.url LIMIT 10

// 查某领域的所有渠道
MATCH (c:Channel)-[:HANDLES]->(d:Domain {{name: $domain}})
RETURN c.name, c.phone, c.url, c.channel_type LIMIT 20

// 查某法律的条文数量
MATCH (l:Law {{title: $title}})-[:HAS_ARTICLE]->(a:Article)
RETURN l.title, count(a) AS article_count

用户问题：{question}
提取的实体：{entities}

只返回 Cypher 语句（可用反引号包裹），不要解释。"""

LEGAL_GRAPH_QA_PROMPT = """你是专业法律助手。根据知识图谱查询结果回答用户问题。

用户问题：{question}

图谱查询结果：
{graph_result}

回答要求：
1. 列出相关法律名称和适用说明
2. 列出对口维权渠道（含电话/网址）
3. 若结果为空，建议用户补充描述或直接拨打12315/12333等通用热线"""

# ── NL2SQL ─────────────────────────────────────────────────────────────────────
LEGAL_NL2SQL_PROMPT = """你是 PostgreSQL 专家。根据以下数据库 Schema 生成 SQL 查询语句。

可查询的表：
  channels(id, name, domain, channel_type, phone, url, region_code)
    -- domain: labor_social_security / consumer_market / contract_commercial /
    --         criminal_public_security / real_estate / intellectual_property 等
    -- channel_type: hotline / website / app
    -- region_code: CN / BJ / SH / GD / ZJ 等省市代码

  laws(id, title, category, authority, domain, effective_from)
    -- category: 法律 / 行政法规 / 司法解释

  consultations(id, user_id, created_at, domain, status)

规则：
1. 只生成 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP
2. 结果必须加 LIMIT（默认 20，最大 100）
3. 禁止查询 phone 之外的联系方式隐私字段

用户问题：{question}

只返回 SQL 语句，不要解释。"""

LEGAL_SQL_QA_PROMPT = """根据以下数据库查询结果回答用户问题。

用户问题：{question}

查询结果：
{result}

用自然语言整理结果，用清晰的列表格式呈现，重点突出名称、电话和网址。"""

# ── Query 改写 ─────────────────────────────────────────────────────────────────
LEGAL_QUERY_REWRITE_PROMPT = """将以下口语化法律问题改写为标准法律术语表达，便于语义检索。

原始问题：{question}

要求：
1. 保持问题原意，替换口语为法律术语（如"老板不给工资" → "拖欠劳动报酬"）
2. 补充隐含的法律概念（如"买到假货" → "购买假冒伪劣商品，涉及消费者权益保护"）
3. 返回1条改写后的查询，不要解释

改写结果："""
