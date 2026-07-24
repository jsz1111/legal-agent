"""法律指引 Agent 的所有提示词常量。"""

# ── 1. 法律问题提取（with_structured_output 用） ────────────────────────────
ISSUE_EXTRACT_PROMPT = """你是法律问题提取专家。

从用户描述中提取具体的法律问题，并标准化为法律术语。
同时判断所属法律领域（从以下11个中选一个）：
labor_social_security（劳动/社保）、consumer_market（消费/市场）、
contract_commercial（合同/商事）、criminal_public_security（刑事/治安）、
family_marriage（婚姻家庭）、real_estate_construction（房产/建筑）、
intellectual_property（知识产权）、administrative（行政）、
traffic_accident（交通事故）、medical_dispute（医疗纠纷）、other（其他）

标准化示例：
- "老板不给工资" → "拖欠劳动报酬"，domain: labor_social_security
- "买到假货" → "销售假冒伪劣商品"，domain: consumer_market
- "借钱不还" → "民间借贷纠纷"，domain: contract_commercial

用户描述：{user_input}

填入 issues（标准化法律问题列表）和 domain 字段。无明确问题时 issues 为空列表。"""


# ── 2. 紧急程度判断 ─────────────────────────────────────────────────────────
URGENCY_CHECK_PROMPT = """判断用户描述的法律情况紧急程度。

用户描述：{user_input}

三个级别：
CRITICAL：人身安全受威胁、正在被拘押/跟踪、家暴进行中、紧急财产被扣押冻结
TIME：用户提到了具体时间（如"3年前""去年"），可能接近或超过时效
  （劳动仲裁1年、消费/合同3年、人身损害1年）
NORMAL：一般维权，无紧迫时效压力

输出JSON（只输出JSON）：
{{"urgency": "CRITICAL"|"TIME"|"NORMAL", "reason": "原因", "time_clue": "提取到的时间信息或空字符串"}}"""


# ── 3. 澄清模糊描述 ─────────────────────────────────────────────────────────
CLARIFY_PROMPT = """你是法律指引助手，帮助普通市民描述法律情况。

用户说：{user_input}

描述较模糊，请用通俗语言询问2个最关键的问题（不超过2个）：
- 发生了什么事情
- 对方是谁（个人/公司/政府机构）
- 大概什么时候发生的

语气亲切，像朋友帮忙梳理。直接输出追问内容，不要开场白。"""


# ── 4. 追问领域细节 ─────────────────────────────────────────────────────────
ASK_DETAILS_PROMPT = """你是法律指引助手，正在帮用户梳理{domain_label}方面的情况。

已了解的问题：{confirmed_issues}
需要确认的细节：
{details_to_ask}

请将上述细节转化为自然的追问（最多3个问题），可以简短解释为什么需要这个信息。
语气像朋友帮忙整理，直接输出内容。"""


# ── 5. 解析用户对追问的回答 ────────────────────────────────────────────────
PARSE_DETAILS_PROMPT = """从用户回答中提取关键信息。

问了用户：
{asked_details}

用户的回答：
{user_answer}

输出JSON（只输出JSON）：
{{
  "new_issues": ["新发现的法律问题，如有"],
  "evidence": ["用户提到已有的证据，如有"],
  "region": "提取到的城市/地区，无则空字符串",
  "time_info": "提取到的时间信息，无则空字符串"
}}"""


# ── 6. 生成行动方案（conclude节点用） ──────────────────────────────────────
CONCLUDE_PROMPT = """请为用户生成一份实用的法律维权行动方案。语言通俗，不堆砌术语。

## 置信度指引（决定本次输出的确定程度）
{confidence_guidance}

## 用户情况
- 法律问题：{confirmed_issues}
- 领域：{legal_domain}
- 所在地区：{region}
- 已有证据：{evidence_confirmed}
{time_warning}

## 检索到的法律依据
{law_context}

## 类似案例
{case_context}

## 可用渠道
{channels}

## 输出格式（严格按以下五部分，简洁实用）

**【理解您的情况】**
一句话识别用户处境和情绪（如"工资被拖欠确实令人气愤"），不超过20字。

**【法律依据】**
列出1-3条最相关条文，格式：《法律名》第X条：摘要（不超过50字）
无具体条文时，说明适用的法律原则。

**【类似案例参考】**（有则列1-2条，无则整体省略此部分）

**【维权路径比较】**
列2-3种方案，每种标注 [免费] 或 [收费]、预计时长、优缺点。
最后注明推荐方案。

**【行动清单】**
□ 立即保存的证据：
{evidence_checklist}
□ 具体步骤（1. 2. 3.…）
□ 联系渠道：名称 / 电话 / 网址

（可选）**【常见误区】** 纠正本领域1条最常见误解。

{force_conclude_note}"""


# ── 7. 领域专项追问模板 ────────────────────────────────────────────────────
DOMAIN_DETAIL_TEMPLATES: dict[str, list[str]] = {
    "labor_social_security": [
        "距离事件发生多久了？（影响仲裁时效判断）",
        "是否签有书面劳动合同？",
        "现在还在该单位工作，还是已经离职？",
        "有工资单、银行流水或打卡记录吗？",
    ],
    "consumer_market": [
        "购买金额大概多少？",
        "商家是否已经回应过您的诉求？",
        "有订单截图或购买凭证吗？",
    ],
    "contract_commercial": [
        "合同是书面签订还是口头约定？",
        "对方的违约行为具体是什么？",
        "您方的损失大概是多少？",
    ],
    "criminal_public_security": [
        "当事人之间是什么关系？",
        "是否已经报警？有报警回执吗？",
        "有录音、录像或其他证据吗？",
    ],
    "family_marriage": [
        "是否已经尝试过协商或调解？",
        "主要争议是什么（财产/子女抚养/离婚）？",
    ],
    "real_estate_construction": [
        "涉及金额大概多少？",
        "是否有书面合同？",
        "问题类型是什么（延期交房/质量/中介纠纷）？",
    ],
    "traffic_accident": [
        "事故发生时有没有报警/做笔录？",
        "对方有保险吗？",
        "有伤亡或只是财产损失？",
    ],
}

# ── 8. 领域证据清单模板 ───────────────────────────────────────────────────
EVIDENCE_TEMPLATES: dict[str, list[str]] = {
    "labor_social_security": [
        "劳动合同（无合同则收集工牌/工作证/同事证言）",
        "工资单 / 银行流水",
        "打卡记录 / 考勤表",
        "与老板/HR的通讯记录（微信/短信截图）",
    ],
    "consumer_market": [
        "订单截图 / 购买凭证 / 发票",
        "商品照片或视频（记录问题）",
        "与商家的聊天记录",
        "物流信息截图",
    ],
    "contract_commercial": [
        "合同文本（或录音证明口头约定）",
        "付款凭证（转账记录）",
        "催告函 / 违约通知（保留发送记录）",
        "损失证明材料",
    ],
    "criminal_public_security": [
        "报警回执",
        "伤情鉴定报告",
        "录音 / 录像证据",
        "证人姓名和联系方式",
    ],
    "family_marriage": [
        "结婚证 / 离婚协议草稿",
        "财产证明（房产证/银行存款证明）",
        "子女户口本",
    ],
    "real_estate_construction": [
        "购房 / 租房合同",
        "付款记录",
        "房屋质量问题照片",
        "与开发商/中介的通讯记录",
    ],
    "traffic_accident": [
        "交警事故认定书",
        "现场照片 / 行车记录仪视频",
        "伤情诊断证明 / 医疗费票据",
        "对方保险信息",
    ],
}

DOMAIN_LABELS: dict[str, str] = {
    "labor_social_security":    "劳动/社保",
    "consumer_market":          "消费维权",
    "contract_commercial":      "合同纠纷",
    "criminal_public_security": "刑事/治安",
    "family_marriage":          "婚姻家庭",
    "real_estate_construction": "房产/建筑",
    "intellectual_property":    "知识产权",
    "administrative":           "行政",
    "traffic_accident":         "交通事故",
    "medical_dispute":          "医疗纠纷",
    "other":                    "其他",
}
