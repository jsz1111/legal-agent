"""法律指引 Agent 的所有提示词常量。"""

# ── 1. 法律问题提取（with_structured_output 用） ────────────────────────────
ISSUE_EXTRACT_PROMPT = """从对话中提取法律问题和当前用户新补充的原子事实，只输出 JSON。

领域 domain 只能选：
labor_social_security, consumer_market, contracts_property_housing,
criminal_public_security, family_vulnerable_groups, traffic_personal_injury,
medical_education_tax, administrative_remedies, intellectual_property,
environment_pollution, cyber_data_fraud, mediation_notary_arbitration, other。

输入：
{user_input}

输出结构：
{{
  "issues": ["简短、保守的标准法律问题"],
  "domain": "领域代码",
  "case_frame": "personal_safety|financial_loss|contract_service|family_dispute|labor_dispute|administrative_remedy|other",
  "frame_confidence": 0.0,
  "facts": ["当前用户明确说出的事实"],
  "case_updates": [{{
    "key": "稳定语义键",
    "category": "actor|relationship|event|claim|amount|time|location|evidence|procedure|harm|uncertainty",
    "statement": "不含法律结论的完整事实",
    "subject": "", "relation": "", "value": "",
    "certainty": "asserted|uncertain|denied",
    "operation": "add|replace|deny",
    "source_text": "当前用户消息中的逐字原文",
    "evidence_status": "仅 category=evidence 时填写 obtained|lead|unavailable|unknown"
  }}],
  "evidence_details": [{{
    "name": "材料名称",
    "source_form": "paper_original|native_electronic|exported_file|screenshot|copy|user_statement|unknown",
    "completeness": "complete|partial|unknown",
    "identity_visibility": "clear|unclear|not_applicable|unknown",
    "time_visibility": "clear|unclear|not_applicable|unknown",
    "acquisition_method": "user_created|received_from_counterparty|platform_or_institution_export|third_party|unknown",
    "proof_roles": ["relationship|transaction|agreement|payment|event|problem|identity|time|communication|procedure|harm|loss|liability|ownership|infringement"],
    "source_text": "当前用户消息中的逐字原文"
  }}],
  "region": "", "time_info": ""
}}

约束：
1. 结合近期对话理解短回答，但 case_updates 和 evidence_details 只记录“当前用户消息”的新增、更正或否定内容。
2. source_text 必须逐字出自当前用户消息；没有原文支持就不输出该项。
3. 对照已有语义键：重复事实不新增键；补充沿用原键；更正用 replace；明确否定用 deny。
4. 事实、诉求和法律判断分开。不得把用户陈述直接认定为违法、违约、侵权或犯罪。
5. 证据材料一份只生成一个 evidence_details；金额、日期、主体等材料内容不是新的证据项。
6. 不判断证据真实性、合法性、可采性、证明力或案件结果。信息不足时使用空数组或 unknown。
7. domain 和 case_frame 都来自模型对本案具体事件性质的推理，不得套固定模板或只看关键词。除非用户描述明确的身体伤害、被打、殴打、受伤、现实威胁、人身安全风险，否则不强制 personal_safety；其他场景按事实选择最接近的 domain 与 case_frame，信息不足时用 other 并把 frame_confidence 降到 0.5 以下。
8. 不输出解释、Markdown 或代码围栏。"""


INTAKE_CLASSIFY_PROMPT = """根据用户已经填写的案件信息，只完成法律问题和领域分类。

案件信息：
{case_summary}

domain 只能选：
labor_social_security, consumer_market, contracts_property_housing,
criminal_public_security, family_vulnerable_groups, traffic_personal_injury,
medical_education_tax, administrative_remedies, intellectual_property,
environment_pollution, cyber_data_fraud, mediation_notary_arbitration, other。

只输出JSON：
{{"issues":["简短、保守的法律问题"],"domain":"领域代码","case_frame":"personal_safety|financial_loss|contract_service|family_dispute|labor_dispute|administrative_remedy|other","frame_confidence":0.0,"region":"用户明确写出的地区或空字符串","time_info":"用户明确写出的时间或空字符串"}}

不得把用户陈述直接认定为违法、违约、侵权或犯罪；信息不足时 issues 使用空数组。domain 和 case_frame 根据事件性质判断，不要按关键词套领域；仅当用户明确描述身体伤害、被打、殴打、受伤、现实威胁时 case_frame 必须为 personal_safety，domain 不得选 cyber_data_fraud/consumer_market。"""


# ── 2. 紧急程度判断 ─────────────────────────────────────────────────────────
URGENCY_CHECK_PROMPT = """判断用户描述的法律情况紧急程度。重点判断用户当前此刻是否仍有危险，不要把既往受伤事实自动等同于当前危险。

最近对话（越靠后越新）：
{recent_dialogue}

当前用户消息：{user_input}

三个级别及触发示例：

CRITICAL（立即危险，优先处理）：当前人身安全受威胁、正在被拘押/跟踪、暴力正在发生或明确即将发生、紧急财产被扣押冻结
  - "正在打我""对方就在门外扬言伤害我" → CRITICAL
  - "有人跟踪我""被人盯上了" → CRITICAL
  - "被关在派出所/被带走了""现在被拘留" → CRITICAL
  - "银行卡/账户被冻结，钱取不出来" → CRITICAL
  - "孩子被对方强行带走了" → CRITICAL
  - "对方扬言要来伤害我" → CRITICAL

TIME（时效风险，需提醒）：用户提到具体时间节点，根据时效规则可能接近或超期
  时效参考：劳动仲裁1年 / 人身损害3年 / 消费合同一般民事3年
  - "3年前老板欠我工资""事情发生在2021年" → TIME（劳动1年时效，可能已超）
  - "去年买的东西有问题，一直没处理" → TIME
  - "距离离职快两年了" → TIME（劳动仲裁时效1年，已超）
  - "差不多快一年前的事" → TIME（接近时效临界）
  - "当时受伤，已经过了快三年" → TIME（人身损害3年时效，临近）

NORMAL：一般维权，用户未提及具体时间或事件刚发生不久，无时效紧迫压力
  - 既往发生过殴打、受伤等事件，但用户已明确现在安全，当前消息只是在补充身份、证据、时间或地点 → NORMAL
  - 最近一轮明确“现在安全”，后续没有出现新的正在发生或即将发生的危险信号 → 不得仅因再次提到“打伤、受伤”判为 CRITICAL

安全状态必须按最近对话区分：
- safety_relevant=true：对话涉及暴力、威胁、跟踪、拘束或其他人身安全风险；否则为 false。
- safety_status="danger"：明确正在发生或即将发生危险。
- safety_status="safe"：用户明确说现在安全或已脱离现场。
- safety_status="unknown"：涉及人身安全，但只知道既往发生过事件，当前是否安全没有说明。
- safety_status="not_applicable"：不涉及人身安全。
- 仅说“被打了、受伤了、发生过威胁”而没有说明当前状态时，必须是 unknown，不能猜测为 safe 或 danger。

输出JSON（只输出JSON）：
{{"urgency": "CRITICAL"|"TIME"|"NORMAL", "safety_relevant": true|false, "safety_status": "danger"|"safe"|"unknown"|"not_applicable", "reason": "原因", "time_clue": "提取到的时间信息或空字符串"}}"""


COUNTER_QUESTION_RESPONSE_PROMPT = """你在法律维权信息整理过程中，用户没有回答当前追问，而是提出了一个问题。
请先直接、简短回答用户的问题，再由程序恢复原追问。

用户问题：{user_question}
当前案情：
{case_context}
当前已检索法律依据：
{law_context}

规则：
1. 只回答用户这一个问题，使用2-4句通俗中文，不再提出新问题。
2. 只能依据当前案情和已检索内容；依据不足时明确说“根据目前信息还不能确定”。
3. 不得编造法律名称、条号、期限、金额、机构联系方式或案件结果。
4. 不得声称责任、违法、犯罪或胜诉已经确定。
5. 不输出标题、列表、JSON或内部工作流说明。"""


# ── 3. 澄清模糊描述 ─────────────────────────────────────────────────────────
CLARIFY_PROMPT = """你是法律指引助手，帮助普通市民描述法律情况。

截至当前的对话：
{recent_dialogue}

已保存的案情：
{case_context}

用户最新回复：{user_input}

描述较模糊，请用通俗语言只询问1个最关键的问题：
- 发生了什么事情
- 对方是谁（个人/公司/政府机构）
- 大概什么时候发生的

必须结合完整上下文理解用户最新回复，先判断哪些内容已经说清楚。
不得再次询问对话或已保存案情中已经回答的内容，也不得让用户重新讲一遍事情经过。
尽量给出二选一或简单示例，允许用户回答“不知道”。语气亲切，像朋友帮忙梳理。
直接输出一个问题，不要开场白，不要一次列出多个编号问题。"""


# ── 4. 追问领域细节 ─────────────────────────────────────────────────────────
ASK_DETAILS_PROMPT = """你是法律指引助手，正在帮用户梳理{domain_label}方面的情况。

已了解的问题：{confirmed_issues}
需要确认的细节：
{details_to_ask}

请将上述细节转化为自然的追问（最多3个问题），可以简短解释为什么需要这个信息。
语气像朋友帮忙整理，直接输出内容。"""


# ── 5. 解析用户对追问的回答 ────────────────────────────────────────────────
PARSE_DETAILS_PROMPT = """从用户回答中提取关键信息，并形成可跨轮合并的原子案情。

问了用户：
{asked_details}

用户的回答：
{user_answer}

已有结构化事实及语义键：
{case_context}

先判断用户这条消息**是否在回答上面的问题**：
- 用户提供了信息、说明有/没有某项证据、补充了时间地点 → is_answer = true
- 用户在反问、提出新问题、表达疑惑（如"什么是仲裁？"、"这个有什么用"、"我该先做什么"）
  且没有提供任何被问到的信息 → is_answer = false，把用户的问题原样写进 user_question
- 既回答了又反问 → is_answer = true，同时把反问写进 user_question
- answers_asked_question 只表示是否直接回答了原问题；用户没有直接回答、但主动补充了其他案情或材料时，is_answer=true、answers_asked_question=false，其他信息仍正常提取

输出JSON（只输出JSON）：
{{
  "is_answer": true,
  "answers_asked_question": true,
  "answered_question_ids": ["本轮确实得到回答的问题ID；普通单问题无ID时留空"],
  "user_question": "用户的反问原文，无则空字符串",
  "collected_facts": ["本轮回答中可用于案件分析的客观事实，如金额、时间、身份关系、行为经过"],
  "case_updates": [
    {{
      "key": "稳定语义键，例如 counterparty.response",
      "category": "actor/relationship/event/claim/amount/time/location/evidence/procedure/harm/uncertainty",
      "statement": "不添加法律判断的简明事实",
      "subject": "主体",
      "relation": "行为或关系",
      "value": "对象、金额或状态",
      "certainty": "asserted/uncertain/denied",
      "operation": "add/replace/deny",
      "source_text": "当前回答中的逐字原文片段",
      "evidence_status": "仅 category=evidence 时填写 obtained/lead/unavailable/unknown"
    }}
  ],
  "evidence": ["用户提到已有的证据，如有"],
  "evidence_unavailable": ["用户明确表示没有的证据"],
  "evidence_details": [
    {{
      "name": "材料名称",
      "source_form": "paper_original/native_electronic/exported_file/screenshot/copy/user_statement/unknown",
      "completeness": "complete/partial/unknown",
      "identity_visibility": "clear/unclear/not_applicable/unknown",
      "time_visibility": "clear/unclear/not_applicable/unknown",
      "acquisition_method": "user_created/received_from_counterparty/platform_or_institution_export/third_party/unknown",
      "proof_roles": ["该材料内容直接对应的通用证明角色"],
      "source_text": "能够支持上述属性的用户回答逐字片段"
    }}
  ],
  "region": "提取到的城市/地区，无则空字符串",
  "time_info": "提取到的时间信息，无则空字符串",
  "adverse_facts": ["用户自身存在的不利法律事实；无则空列表"]
}}

注意：
1. 只有用户既未回答、也未提供任何新案情或材料时才令 is_answer=false，此时其余抽取字段一律留空，不要从问句里猜测证据。
   如果用户没有直接回答原问题，但用陈述句补充了新的时间、地点、金额、对象、行为或材料，必须令 is_answer=true、answers_asked_question=false，并正常提取新增内容，不能把这种主动补充误判为反问。
2. 如果用户说"没有"、"无"、"不存在"等否定词，请提取到 evidence_unavailable 中。
3. adverse_facts 只记录客观不利事实，不做法律判断，用一句话简短描述。
4. collected_facts 和 case_updates 只提取用户明确说出的事实，不推测，不把事实直接写成法律结论。
5. case_updates.source_text 必须逐字来自“用户的回答”；找不到原文锚点的内容不要输出。
6. 同一语义事实沿用相同 key。用户明确更正时 operation=replace；明确否定旧事实时 operation=deny；不确定陈述 certainty=uncertain。
   如果用户只是重复已有事实且没有新增细节，不要再次输出该事实；如果新增细节属于已有事实的一部分，优先沿用已有 key 或使用同一 key 的下级 key。
7. 证据也写入 case_updates，category=evidence；用户称持有时 certainty=asserted，明确没有时 certainty=denied。必须填写 evidence_status：只有明确已取得、已保存、手里有、拍了/录了/上传了原始材料时才是 obtained；“附近有监控、可以调取、能联系证人、可能有记录”都是 lead，不能写成已持有。
8. adverse_facts 只记录用户明确陈述且可能影响责任、时效、请求或程序的客观内容，不能替用户作出“构成违约、超过时效、必然不能获赔”等法律判断。
9. 本节点只解析回答中的事实和证据，不新增或升级法律问题；违法、违约、侵权、犯罪及责任成立与否均交给检索后的法律判断。
10. 用户即使同时说“不要再问”“现在生成方案”，只要还提供了事实或证据，is_answer 仍为 true，并照常提取这些内容；流程控制指令不等于没有回答。
11. evidence_details 只结构化用户明确说出的材料属性和 proof_roles，不得根据材料类型自行猜测。proof_roles 只能从 relationship/transaction/agreement/payment/event/problem/identity/time/communication/procedure/harm/loss/liability/ownership/infringement 中选择。没有逐字 source_text 锚点的属性一律填 unknown；不要在这里判断真实性、合法性、可采性或证明力。
12. 每份材料只生成一条 evidence_details 和一条 category=evidence 的 case_update；材料里能看到的名称、金额、日期、是否裁剪等是材料属性，不得拆成新的证据项。
13. 批量问题带有方括号问题ID时，answered_question_ids 只列出用户本轮实际回答的ID；未回答、留空或只反问的ID不得列入。
14. collected_facts 和 case_updates.statement 必须是"自包含"的完整陈述，单独一条就能读懂完整语义，禁止只写用户回答的字面碎片词（如只写"有""没有""不知道"）。
    被问"是否持有转账凭证"而用户只答"没有"时，应写成"未持有转账凭证"或"是否持有转账凭证：没有"，不得只写"没有"；
    用户答"有"时，应写成"已持有转账凭证"或"是否持有转账凭证：有"。
    用户回答本身已完整（如"我上周买的手机"）时照原样保留，不要画蛇添足。"""


# ── 6. 结论前结构化分析（conclude节点用） ────────────────────────────────────

ISSUE_MAP_PROMPT = """你是法律案件争点识别员。请根据完整案件快照识别真正影响用户维权路径和法律判断的核心争点。

要求：
1. 使用全部有效事实，不得只关注最近消息；事实状态为未知或冲突时不得当作确定事实。
2. 不要为每个场景套用固定结论，应根据事件顺序、主体行为、损害、证据和程序状态动态判断。
3. 每个争点必须给出 supporting_fact_keys、检索问题和会改变判断的事实。
4. 同时识别核心争点、条件性争点、程序争点和证据争点；没有必要的类别可以省略。
5. 主动识别事实之间的矛盾：不能把相反陈述任选其一当作确定事实；要说明它影响哪一个判断、需要什么材料消解。
6. 不需要展示思维过程，只输出 JSON，不要输出 Markdown 或解释文字。

输出格式：
{{
  "fact_tensions": [
    {{
      "title": "存在冲突的事实问题",
      "side_a_fact_keys": ["一方陈述对应的事实 key"],
      "side_b_fact_keys": ["相反陈述对应的事实 key"],
      "why_it_matters": "该冲突会影响的法律判断",
      "resolution_action": "优先用什么事实或证据核实"
    }}
  ],
  "issues": [
    {{
      "issue_id": "issue_1",
      "title": "简洁的法律争点",
      "importance": "core|conditional|procedure|evidence",
      "reason": "为什么本案必须分析这一点",
      "supporting_fact_keys": ["事实key"],
      "retrieval_questions": ["围绕该争点检索的自然语言问题"],
      "facts_that_change_result": ["会改变判断的事实"]
    }}
  ]
}}

## 完整案件快照
{case_snapshot}
"""


ISSUE_APPLICATION_PROMPT = """你是法律维权方案的案件分析员。请将完整案件事实、争点和最终检索依据结合起来，逐项完成法律适用分析。

要求：
1. 每个核心争点都要说明：支持当前判断的事实、不利或相反事实、法律依据为何相关、当前倾向性判断、条件性分支、证据任务和行动建议。
2. 法律名称、条号和条文内容只能使用“最终检索法律依据”中的内容。没有检索到具体条号时，不得虚构条号。对每一条拟引用依据，必须先核对它自己的触发条件与本案行为的先后顺序、发生阶段、目的和后果；不能仅因出现相同关键词就引用。
3. 可以使用一般法律知识组织事实关系和争点，但最终结论必须区分已确认事实、未知事实和条件假设。
4. 不得仅因一个关键词（例如“打起来了”）直接认定用户违法或需要承担责任；必须分析先后顺序、行为持续阶段、使用手段和后果。对于仅在特定阶段才适用的转化、加重或例外规则，若阶段事实未被确认，只能作为条件分支说明，不能写成当前主判断。
5. 不要输出内部思维过程，只输出 JSON。
6. 对每一条真正进入分析的 legal_basis_ref，必须拆成该法律规则的实际构成要件或适用条件，不能只写“该法条支持用户”或“该法条相关”。每个要件都要单独判断：现有事实是否支持、现有证据能否证明、是否仍未知、是否会改变结果。
7. 对每个争点必须主动设想对方、平台或办案机关最可能提出的抗辩，并给出用户应如何回应、需要补什么材料。禁止用“需进一步核实”代替具体抗辩分析；只有说明“哪一步还不确定、为什么不确定、什么材料能改变”才能标为 unknown。
8. legal_element_matrix 中每个要件的 supporting_facts 和 evidence_items 必须来自案件快照或证据状态，并使用自然语言可读表述；禁止输出 user_、followup.、fraudster_、emergency_action_taken 等内部 key/字段 id；没有对应项时写“暂无”，并在 why 中说明缺少什么。
9. 如果同一要件存在相反事实或证据冲突，status 必须标为 conflicted，并说明两种结果分别会怎样影响争点判断。
10. 若“最终检索法律依据”中包含 issue_authorities，则每个 issue 应优先使用该映射中与自身 issue_id 对应的 authorities，再补充必要的通用法条；不要把其他争点的专属法条硬套到当前争点。
11. evidence_items 必须优先使用案件快照中 evidence_confirmed、evidence_unverified、evidence_unavailable 里的自然语言名称；用户已经明确说持有聊天记录、转账记录等材料时，不得写成“无对应证据”。

输出格式：
{{
  "analyses": [
    {{
      "issue_id": "issue_1",
      "title": "争点标题",
      "current_view": "基于当前事实的阶段性判断",
      "supporting_facts": ["自然语言事实表述"],
      "adverse_facts": ["不利或相反事实；没有则为空数组"],
      "legal_basis_refs": ["法律名称+条号"],
      "application_analysis": "解释事实如何满足或不满足法律条件",
      "conditional_branch": "如果哪些事实成立，判断会怎样变化",
      "facts_to_verify": ["仍需核实的关键事实"],
      "legal_element_matrix": [
        {{
          "legal_basis_ref": "法律名称+条号",
          "element": "该法律规则的构成要件或适用条件",
          "supporting_facts": ["事实key或事实表述"],
      "evidence_items": ["自然语言证据名称，例如：聊天记录、转账记录、报警回执"],
          "status": "satisfied|unknown|conflicted|not_met",
          "why": "为什么当前判断是这个状态",
          "what_would_change": "需要什么事实或证据才能改变该要件判断"
        }}
      ],
      "opponent_counterarguments": [
        {{
          "argument": "对方、平台或办案机关可能提出的抗辩",
          "response": "用户应如何回应",
          "evidence_needed": "需要补充什么材料"
        }}
      ],
      "evidence_actions": ["对应证据和获取动作"],
      "recommended_actions": ["立即执行、直接影响本案判断的行动"],
      "procedure_steps": ["与立即行动不重复的后续程序顺序，例如取得回执后跟进、在何种条件下转入下一救济"]
    }}
  ]
}}

## 完整案件快照
{case_snapshot}

## 识别出的法律争点
{issue_map}

## 最终检索法律依据
{legal_basis}
"""


# ── 7. 生成行动方案（conclude节点用） ──────────────────────────────────────
CONCLUDE_PROMPT = """请为用户生成一份实用的法律维权行动方案。语言通俗，不堆砌术语。

## 置信度指引（决定本次输出的确定程度）
{confidence_guidance}

## 用户处境审视（叙事框架与必须遵守的硬性要求）
{situation_guidance}

## 面向当前用户的表达要求
{audience_guidance}

## 结论前案件分析包（优先于近期对话片段）
{case_analysis_packet}

## AI识别的法律争点
{issue_map}

## 按争点完成的事实与法律适用分析
{issue_analyses}

## 法条要件核对（程序从结构化分析渲染，必须保留或吸收）
{legal_element_review}

## 与争点关联的最终法律依据包
{legal_basis_packet}

## AI策略中枢生成的统一行动策略
{strategy_plan}

## 案件对抗与执行推演审查
{adversarial_execution_review}

## 反方压力测试（程序从对抗推演渲染，必须保留或吸收）
{adversarial_review_block}

## 策略中枢优先原则
最终回复以 AI 策略中枢为主线。下面的固定栏目只是兼容性提示，不是思考边界；如果案件需要新的争点、反方观点、条件分支、程序节点或维权请求，可以直接增加自然语言段落。不要为了填满栏目重复事实，也不要把没有实际帮助的渠道、案例或通用清单硬塞进回复。先给出当前最重要的判断，再解释事实如何影响法律适用，最后给出按优先级排序的行动和会改变结论的条件。

## 对抗与执行推演优先原则
最终方案必须使用 adversarial_execution_review 中的不利点、程序风险、条件路径和可执行性检查：不得回避对用户不利的点；必须写明这些点成立时结论会怎样变化；行动清单中的每一步都要有对象、目的、所需材料和受阻后的下一步；重复动作必须合并。

## 最终呈现的推理纪律
1. 先处理事实冲突：案件分析包中的 fact_tensions 或 status=conflicted 不是“信息不足”的泛泛提示，而是必须说明其两种陈述、影响的判断和优先核实动作；不得任选一边写成事实。
2. 严格区分证据状态：obtained 才可写为“目前已持有”；lead、unknown 或未核验证据只能写为线索或待调取事项。
3. 法律依据只展示与某个争点存在直接或条件关联的条文；对于只因关键词命中、但无法说明与本案哪个争点有关的条文，不要展示。
3a. 条件关联的条文必须写清“只有在何种已核实的行为阶段或目的成立时才可能适用”；不能把它与当前主判断并列成已经适用的依据。
4. “现在最优先行动”只放立即要做且最影响结果的 3—5 步；“最优程序路径”只写后续程序节点，不重复前一节的动作；“行动清单”仅保留前两节未覆盖的补充事项。
5. “现在生成方案”“继续补充”“好的/收到”等流程指令或确认语不是案件事实，不得在案件还原、争点或法律适用中引用。

## 用户情况
- 法律问题：{confirmed_issues}
- 领域：{legal_domain}
- 所在地区：{region}
- 已确认时间信息：{time_info}
- 用户陈述的案情事实（仍以结构化状态判断能否采用）：{collected_facts}
- 相关长期记忆：{long_term_memories}
- 已有证据：{evidence_confirmed}
- 未核验的证据线索（只能提示用户核验，不得写成已经持有）：{evidence_unverified}
- 明确缺失的证据：{evidence_unavailable}
{time_warning}
{self_review_note}
{deferred_questions}

## 结构化事实状态（仅供内部判断，不要向用户输出英文状态码）
{fact_assessments}

## 结构化证据状态（方案准备度不等于法律证明力）
{evidence_assessments}

## 证明目标覆盖评估（必须区分“有材料”和“能证明什么”）
{evidence_coverage}

使用规则：长期记忆只在与本案直接相关时作为补充；若与本轮陈述冲突，以用户本轮及较新的陈述为准。

事实使用规则：结构化事实状态标记为“回答含义不明确”或“与前述信息不一致”的内容，不得当作确定事实写入方案；只能说明尚需确认，并分别提示不同答案可能带来的影响。不得把“一个月前发生”改写成“提前一个月通知”等不同事实。

深度分析规则：必须优先使用“结论前案件分析包”和“按争点完成的事实与法律适用分析”。最终回复不能只摘要事实或罗列法条，应解释每个核心争点中哪些事实支持判断、哪些事实可能改变判断，以及对应的具体行动。固定的是栏目结构，不是本案结论。

要件拆解规则：争点分析中引用每一条法律依据时，必须把该条拆成实际构成要件或适用条件，逐项写明“当前事实是否支持、现有证据能否证明、是否仍未知、需要什么材料才能改变”。不得只写“依据《××法》第×条支持/相关”。必须同时回应 legal_element_matrix 中标为 unknown、conflicted 或 not_met 的要件，以及 opponent_counterarguments 中对方可能提出的抗辩。若没有对应分析，不得用“需进一步核实”一笔带过；应说明具体缺口和补强材料。

证据与时间解释规则：
1. 只把用户实际上传、展示或明确确认持有的材料视为“已有证据”。截图/录音里某人声称另有实物、照片或文件，只能写成“待核验线索”，不得据此宣称证据链完整。
2. 图片中的后期标注、评论、圈画不是原始陈述，必须与原始聊天、录音或交易记录核验。
3. `00:01:20`、`01:32` 等音视频时间码是播放位置，不是事件发生日期，不得用于时效或案发日期判断。
4. 单张截图不得称为“铁证”，不得写“已经足够证明”“证据链完整”“胜诉希望很大”；应说明其能证明什么、不能证明什么，以及需要哪类原始载体印证。

## 用户不利事实（可能被对方援引的因素）
{adverse_facts_section}

## 长对话事实沉淀（按轮次整理的历史细节，优先作为事实边界）
{long_dialogue_memory}

## 近期对话片段（理解完整上下文）
{dialogue_snippet}

## 检索到的法律依据
{law_context}

## 类似案例
{case_context}

## 可用渠道
{channels}

## 证据清单性质与来源
{evidence_source}

## 本轮追问规则的依据来源
{followup_authority}

## 程序性准确性要求
1. 不得把劳动仲裁管辖只写成“公司注册地”。劳动争议通常可向劳动合同履行地或用人单位所在地的劳动争议仲裁委员会申请，具体以当地受理规则为准。
2. 不得把拖欠工资的仲裁时效一概写成“从欠薪日开始一年”。劳动关系存续期间因拖欠劳动报酬发生争议，申请仲裁不受一般一年期间限制；劳动关系终止的，应当自终止之日起一年内提出。
3. 不得把劳动行政部门“责令限期支付、逾期再加付”的职权，写成劳动者可直接通过仲裁当然取得50%-100%加付赔偿金。
4. 不得凭经验写死劳动监察或劳动仲裁的实际办结月份。劳动仲裁只可说明法定期限一般为受理后45日，复杂案件依法延期不超过15日，实际进度以受理机构为准。
5. 不得使用“撕破脸”“老板跑路”“老赖”“空头支票”等对抗性或贬损表达。申请劳动仲裁不以离职为前提。
6. 不得断言聊天记录单独就是有力证据、银行流水单独即可证明欠薪，或现有证据已经足够；必须提示核验主体、时间、完整性及与其他材料的相互印证。
7. 不得把欠薪、未缴社保或所谓“被迫离职”直接等同于必然获得经济补偿；须结合解除依据、通知程序、工作年限和工资基数判断。
8. 劳动监察不是申请劳动仲裁的必经前置程序；不得写成必须先投诉或调解失败才能仲裁。
9. 不得给劳动监察写固定15-30日、15-45日等经验期限；实际进度以当地承办机关告知为准。
10. 公司已经欠薪时，不得因公司口头承诺未来补发而建议用户继续等待；应说明现在即可保全证据并咨询、投诉或依法申请仲裁。
   未签书面劳动合同的二倍工资请求，也不得在事实和检索依据不足时直接计算可获支持的月份，或武断确定仲裁时效起算点。
4. 程序费用、办理期限、管辖和前置程序若未被检索内容或渠道数据直接覆盖，应使用“通常”“以当地机构确认为准”等审慎表达，不得给出绝对承诺。
   劳动争议仲裁依法不收费，劳动仲裁路径必须标注为[免费]，不得写“预收受理费”“由败诉方承担仲裁费”；律师、鉴定、复印等其他服务费用应另行区分。
5. 向 12315 或市场监督管理部门投诉不是提起民事诉讼的法定前置程序，不得写成“必须先投诉才能起诉”。
6. 食品安全法第一百四十八条中的“一千元”是增加赔偿的最低额规则，不是“最高赔偿”，也不是只要发现异物就自动获得；应明确以食品不符合安全标准、经营者责任和证据核验等适用条件为前提。
7. 不得倒置举证责任。除非检索到的法律明确规定，不得笼统声称“商家必须证明没有问题”；用户仍需先证明消费关系、异物与涉案食品的关联及其主张所依据的事实。
8. 市场监管部门可依法调查并作出行政处理，但消费争议调解本身不能强制商家支付民事赔偿；应区分“行政执法”和“民事赔偿调解”，不得笼统写成监管部门只能调解或一定能替用户追回赔偿。
9. 用户陈述只能表述为“用户称”“据您目前描述”；已上传的截图或复制件只能表述为“已查看副本，真实性、完整性和证明力仍需核验”。前后冲突或约数仅在影响方案时用通俗语言指出，不得输出内部状态码。
10. “追问规则的依据来源”用于解释问题设计和材料整理边界，不等于该来源已经对用户个案作出认定；标有仍待精确条款定位的来源不得冒充逐条人工审校结论。

## 输出格式（严格按以下部分，简洁实用）

0. **必输栏目**：必须输出以下栏目，顺序可调整，但不得缺失；模型可以增加更细栏目，但不能省略：
   - 【现在最优先行动】
   - 【理解您的情况】
   - 【案件完整还原】
   - 【核心争点】
   - 【法条要件核对】
   - 【反方压力测试】
   - 【条件分支】
   - 【维权路径比较】
   - 【证据作战图】
   - 【优势与劣势】
   - 【行动清单】
   - 【决策边界与条件】

格式规则：
1. 使用标准 Markdown；每个栏目标题必须单独成行，标题前后保留空行。
2. 并列条件、证据、路径、利弊和行动步骤必须使用列表，不得挤成连续长段落。
3. 每个自然段原则上不超过四句话；先给判断，再给理由和行动。
4. 只加粗影响判断的关键词、期限、金额和行动，不得整段加粗。
5. 不得把多个栏目标题或多个独立要点写在同一行。

**【现在最优先行动】**
先列出 3—5 项按紧急程度排序的行动。每一项都要写清对象、动作和目的；优先使用“按争点完成的事实与法律适用分析”里的对应行动，不得把所有案件都默认引向 12348 或 12345。

**【理解您的情况】**
一句话识别用户处境和情绪（如"工资被拖欠确实令人气愤"），不超过20字。

**【案件完整还原】**
按“经过、主体、时间地点、损失、已采取处理、已有证据、诉求”还原本案。必须使用完整案件快照中已确认的事实，不能只使用最近消息；避免把用户明确表示不清楚的内容写成事实。

**【核心争点分析】**
逐项使用上方的结构化争点分析：当前判断、结合本案、判断会变化的条件、仍需核实、对应行动。不得用笼统的“以机关认定为准”替代分析。
每条法律依据必须做要件拆解：该条文的构成要件/适用条件是什么，当前事实支持哪一项、现有证据能证明哪一项、哪一项仍未知、哪一项会因新的材料改变。同时写明对方可能提出的抗辩、用户如何回应、需要补什么证据。

**【法律依据】**
**关键要求**：
1. 只能引用上方「检索到的法律依据」中出现的条文，逐字照录法律名称、条号和原文
2. 格式：《法律全称》第X条：[原文关键内容]
3. 禁止引用检索结果中不存在的条文——即使你认为某条文适用，若未出现在检索结果中，绝对不能写出来
4. 不得改变法条中的适用主体和前置条件。例如法条规定由劳动行政部门责令限期支付后才可加付赔偿金，不能改写成仲裁机构可直接裁决加付。
5. ⚠️ **刑事法律限制**：民事纠纷（消费纠纷、合同纠纷、劳动纠纷等）中不得主动提及刑事犯罪相关法律。仅当用户描述涉及**人身安全严重威胁**（暴力伤害、拘禁、持械威胁等）时，才可援引刑事法律条文。
6. **选择策略**：
   - 从检索结果中选出与本案**高度相关**的法条（通常 3-5 条，复杂案情可达 6-8 条）
   - 优先选择直接支持用户诉求的核心条文（如惩罚性赔偿、举证责任、时效规定）
   - 「核心法条」部分的条文相关性通常高于「参考法条」，但不要机械地只选核心部分——如果参考法条中有关键条款（如消费者权益保护法第55条"假一赔三"），务必纳入
   - 若“与争点关联的最终法律依据包”中包含 issue_authorities，则每个争点必须优先使用该映射中与自身 issue_id 对应的条文，并把“争点 → 法条 → 要件”的对应关系写进分析，而不是从全局检索池随机挑选。
7. 若检索结果未覆盖案件最核心的条文（如明显缺少某个关键法律），在引用完已有条文后，补充说明：
   "注：当前检索到的条文未完全覆盖本案核心法律依据（例如「XXX」规则），建议拨打 12348 由专业律师确认完整法条"
8. 如检索结果为空，明确说明"当前未检索到具体适用条文，建议拨打 12348 咨询专业律师"
9. 本节点已经进入最终方案阶段。不得再输出“关键缺失信息清单”“请补充以下信息”“强烈建议补充”或成组问题。信息不足只能在【优势与劣势】中用一两句话说明其影响，并继续给出按现有信息可执行的行动步骤。

**【类似案例参考】**（有则列1-2条，无则整体省略此部分）
只能使用上方「类似案例」中真实出现的案件名称、案号和摘要。不得编造“案例1”“法院通常支持”等无来源结论；必须注明类案不等于本案结果。
若类案的案由、交易渠道或争议事实与本案不同，必须点明差异，不得称为“高度一致”。

**【维权路径比较】**
若本案需进入刑事/治安程序（报警、受案、立案、侦查、伤情鉴定等），**必须按程序先后撰写，不得把报警、民事诉讼、调解、伤情鉴定并列成可任选的方案**：
① 报案/配合调查 → ② 申请伤情鉴定 → ③ 等待/配合侦查 → ④ 依据程序结果再谈民事赔偿或刑事附带民事诉讼（刑附民需程序推进到相应阶段后另行启动）。
用①②③④编号，注明“按程序推进，不能跳步或任选”，并在每一步说明由谁办理、需要什么材料。
仅纯民事维权（协商/投诉/诉讼等）才允许并列比较2-3种方案，每种标注 [免费] 或 [收费]、预计时长、优缺点与启动条件，最后注明推荐方案。

**【优势与劣势】**
只客观列出有利与不利两方面，**不估计胜算等级（不写”较高/中等/较低”）**，不输出任何百分比数字。
1. **有利因素**：必须”具体法条（名称+条号）+ 本案具体事实/证据”**成对**列出，说明该条文的哪个要件被哪些事实支撑；禁止空泛的”有法律依据支持”。
2. **不利因素**：必须”具体法条要件 + 缺失/不利事实”**成对**列出，写明对方/办案机关可能如何利用、以及用户还有哪些补救动作——
   - 若对方身份尚未核实而仅凭车牌/账号等线索：必须写明”该线索可能并非行为人本人（套牌/借用/非本人驾驶），能否锁定取决于警方核验”。
   - 若关键事实、证据或前提尚未确认：必须明确写出这些不确定点，不得把乐观假设当成既定事实。
   - 信息不足时不得省略或一笔带过不利因素；不利因素必须与有利因素同等具体。
3. 涉及会随时间消失的证据（现场监控、聊天记录、现场痕迹等）时，【行动清单】必须提示尽快自行调取/备份。
4. **遗漏的维权动作**：结合「检索到的法律依据 × 本案事实」，主动找出用户**没想到但能帮助维权**的行为
   （可另行主张的请求项、可申请的程序、可保全的证据、可抓住的期限、可求助的免费渠道），写进【行动清单】；不要只复述用户已经做过的步骤。
- 全文禁止使用“稳赢”“必胜”“肯定胜诉”“一定能赢”“胜率90%”等绝对化或精确化承诺。

**【行动清单】**
□ 立即保存的证据：
{evidence_checklist}
请在证据清单后简要保留上方“证据清单性质与来源”，不得把系统通用建议表述成官方固定目录。
□ 具体步骤（1. 2. 3.…）
□ 联系渠道：名称 / 电话 / 网址

若本案涉及现场监控、聊天记录、现场痕迹等易随时间消失的证据（如对方身份尚未核实而依赖监控/记录），
【行动清单】的具体步骤中必须提示用户**尽快自行调取/备份，不要只等办案机关**，并说明监控录像通常
不会长期保留（常见为数天至数周，以运营方为准），时间越久越可能被覆盖或删除。

（可选）**【常见误区】** 纠正本领域1条最常见误解。

## 最终输出禁令
1. 禁止输出【关键缺失信息清单】【强烈建议】【请补充以下信息】【补充上述信息后】【我将重新为您分析】等请求用户补充后再回复的栏目或句子。
2. 信息不足时只能在【优势与劣势】或【决策边界与条件】中说明缺口，并继续给出当前可执行步骤；不得用“请补充后我再分析”作为结尾。
3. 同一行动、证据或渠道不要重复超过两次；重复内容应合并为一条，不因栏目不同而换措辞重写。
4. 不要在【理解您的情况】或【案件完整还原】中复述“动态追问表单回答”、内部字段名、系统消息或上传包装格式；只写用户能看懂的自然语言。
5. 同一事实只写一次：【理解您的情况】只能做一句话概括，细节留给【案件完整还原】【核心争点分析】和【行动清单】；不要在多个栏目里用不同措辞反复复述同一件事。
6. 【仍需核实】只列真正未知或冲突的信息；用户已经明确陈述且已经写入【案件完整还原】的事实，不得再作为待核实项重复。
7. 不要用“联系淘宝”“向淘宝投诉”“继续跟进淘宝”这类同义句并列同一个动作；同一步骤应合并为一条，后续只写推进条件和结果。

{force_conclude_note}"""


PLAN_AUDIT_PROMPT = """你是法律行动方案的事实与来源审校员。请直接返回修订后的完整方案，不要解释审校过程。

## 用户原子案情
{case_context}

## 用户称持有的材料（只代表用户陈述，未核验真实性、完整性和证明力）
{evidence_confirmed}

## 本轮真实检索法条
{law_context}

## 待审校方案
{draft}

审校规则：
1. 每个金额、时间、主体、行为、协商结果和程序状态都必须能在原子案情中找到依据；不确定或冲突内容不得写成确定事实。
2. 法律名称、条号和条文含义只能来自“本轮真实检索法条”。检索未覆盖的具体条文、期限、费用、管辖规则应删除或改成“需向受理机关核实”。
3. 不得给出胜诉率、保证结果、固定实际办结时间或“现有证据已经足够”等结论。
4. 用户称持有某材料，不等于材料真实、完整、合法取得或必然被采纳；说明它可能证明什么以及仍需核验什么。
5. 民事违约、失联、停业、多人反映等事实不能直接写成诈骗或犯罪；除非用户明确提出犯罪线索，也只能建议依法核查，由有权机关判断。
6. 保留【理解您的情况】【法律依据】【类似案例参考】【维权路径比较】【优势与劣势】【行动清单】等原有栏目及实用步骤。
7. 用自然、克制、尊重的中文，不使用贬损、煽动对抗或夸大措辞。
"""


# ── 6.5 自省提示词（retrieve 节点用，仅 HIGH 档触发）─────────────────────
SELF_REVIEW_PROMPT = """你是法律审查专家。请审查检索到的法条原文是否真正适用于本案。

## 用户案情
{case_summary}

## 检索到的法条原文
{law_context}

请判断以下三点（无结构化时效/管辖库，靠你的法律知识启发式判断）：
1. **法条适用性**：检索到的法条是否真正适用于本案的具体情形？
2. **时效风险**：根据用户提到的时间线索，是否**明显**已超过诉讼时效或仲裁时效？
3. **管辖疑问**：是否存在明显的管辖争议（如跨地区、特殊行业）？

⚠️ **降档标准**：只在发现**严重且明确**的问题时才建议降档，例如：
- 法条完全不适用（如把刑事法条用于民事纠纷）
- 明确超过时效（如劳动争议发生3年前，已远超1年仲裁时效）
- 管辖权明显不符（如涉外案件但未考虑涉外管辖）

**不应降档的情况**：
- 证据可能不足（证据评估不属于法条适用性判断）
- 时间信息不完整但未明显超时效
- 细节有待补充但大方向正确

如发现**严重且明确**的问题，输出 JSON：
{{"ok": false, "concern": "简要说明哪一点有严重问题、为什么"}}

若无严重问题，输出：
{{"ok": true, "concern": ""}}

只输出 JSON，不要其他内容。"""


# ── 6.8 领域名别名表：旧域码 → 实际 Neo4j/Milvus 域码 ────────────────────
# ISSUE_EXTRACT_PROMPT 已直接输出下方 13 个规范域码，本表只做历史别名兼容
# （早期 prompt 用过 contract_commercial / family_marriage 等旧名）。
# 本模块所有按领域取值的字典（DOMAIN_LABELS / EVIDENCE_TEMPLATES /
# DOMAIN_DETAIL_TEMPLATES / DOC_TYPE_MAP）统一以规范域码为 key。
DOMAIN_MAPPING: dict[str, str] = {
    "contract_commercial":      "contracts_property_housing",
    "real_estate_construction": "contracts_property_housing",
    "family_marriage":          "family_vulnerable_groups",
    "traffic_accident":         "traffic_personal_injury",
    "medical_dispute":          "medical_education_tax",
    "administrative":           "administrative_remedies",
}


# ── 7. 领域专项追问模板（已改为证据驱动，本模板仅作定性辅助备用）────────────
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
    "contracts_property_housing": [
        "是书面合同还是口头约定？",
        "纠纷类型是什么（借贷/买卖/租房/延期交房/房屋质量/中介）？",
        "对方的违约行为具体是什么？",
        "涉及金额或您方损失大概多少？",
    ],
    "criminal_public_security": [
        "当事人之间是什么关系？",
        "是否已经报警？有报警回执吗？",
        "有录音、录像或其他证据吗？",
    ],
    "family_vulnerable_groups": [
        "是否已经尝试过协商或调解？",
        "主要争议是什么（财产/子女抚养/离婚）？",
    ],
    "traffic_personal_injury": [
        "事故发生时有没有报警/做笔录？",
        "对方有保险吗？",
        "有伤亡或只是财产损失？",
    ],
    "medical_education_tax": [
        "就诊或办理的时间和机构是哪里？",
        "损害结果具体是什么？",
        "是否已经向机构或主管部门反映过？",
    ],
    "administrative_remedies": [
        "作出决定的是哪个机关？",
        "收到决定书是多久之前？（影响复议/诉讼时效）",
        "是否已经申请过行政复议？",
    ],
    "intellectual_property": [
        "涉及的权利类型是什么（商标/著作权/专利/商业秘密）？",
        "您是否已取得权属证明（注册证/登记证书）？",
        "侵权行为是在什么渠道发生的（电商/线下/网络平台）？",
    ],
    "cyber_data_fraud": [
        "是通过什么渠道发生的（网络平台/电话/App）？",
        "涉及金额大概多少？",
        "是否已经报警或向平台投诉？",
    ],
    "environment_pollution": [
        "污染类型是什么（噪声/水/大气/固废）？",
        "持续多久了？",
        "是否已向环保部门举报？",
    ],
    "mediation_notary_arbitration": [
        "当前处在哪个阶段（协商/调解/仲裁/已有裁决）？",
        "是否签有仲裁协议或调解协议？",
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
    # 合同/商事 与 房产/租房/建筑 合并为同一域（见 DOMAIN_MAPPING）
    "contracts_property_housing": [
        "合同文本（购房/租房/买卖合同；口头约定则用录音或聊天记录佐证）",
        "付款凭证（转账记录 / 收据 / 发票）",
        "催告函 / 违约通知（保留发送记录）",
        "违约事实证据（房屋质量问题照片 / 逾期交付证明 / 损失证明）",
        "与对方（开发商/中介/供货方）的通讯记录",
    ],
    "criminal_public_security": [
        "报警回执",
        "伤情鉴定报告",
        "录音 / 录像证据",
        "证人姓名和联系方式",
    ],
    "family_vulnerable_groups": [
        "结婚证 / 离婚协议草稿",
        "财产证明（房产证/银行存款证明）",
        "子女户口本",
        "家暴或侵害证据（报警记录/伤情照片/验伤报告，如有）",
    ],
    "traffic_personal_injury": [
        "交警事故认定书",
        "现场照片 / 行车记录仪视频",
        "伤情诊断证明 / 医疗费票据",
        "对方保险信息",
    ],
    # ── 以下 3 领域为通用兜底清单（待权威化，见法律智能体项目说明书.md）──
    "intellectual_property": [
        "权属证明（商标注册证/著作权登记/专利证书）",
        "侵权实物或页面截图并公证",
        "侵权时间证据（首次发现日期）",
        "损失证明或对方获利证明",
    ],
    "administrative_remedies": [
        "行政决定书 / 处罚决定书原件",
        "送达凭证或收到日期证据",
        "行政复议决定（如已申请）",
        "能证明违法或损害的材料",
    ],
    "medical_education_tax": [
        "完整病历（门诊/住院/手术记录）",
        "检查检验报告",
        "医疗费用票据",
        "封存的病历或实物证据（如有）",
    ],
    "cyber_data_fraud": [
        "转账记录 / 支付凭证",
        "对方账号信息（平台ID/手机号/收款账户）",
        "聊天记录或通话录音截图",
        "报警回执 / 平台投诉记录",
    ],
    "environment_pollution": [
        "污染现场照片或视频",
        "监测或检测报告（如有）",
        "向环保部门举报的回执",
        "损害证明（医疗记录/财产损失凭证）",
    ],
    "mediation_notary_arbitration": [
        "仲裁协议 / 调解协议 / 公证书",
        "已有裁决书或调解书",
        "基础争议的合同或凭证",
        "履行情况证据（付款记录/催告记录）",
    ],
}

# ── 通用兜底证据清单（other / 归类失败 / 模板留空时使用）──────────────
GENERIC_EVIDENCE = [
    "书面凭证（合同/协议/聊天记录）",
    "付款或损失凭证",
    "双方沟通记录",
    "报警或投诉回执（如有）",
]

# ── 9. 文书交付（方案 Word 版 + 已有官方模板引用，不再代填新文书） ─────────
# 原 DOC_GEN_PROMPT / DOC_AUDIT_PROMPT / DOC_TYPE_MAP / DOC_TEMPLATES 所支撑的
# “LLM 代填新文书”机制已整体删除（见 doc_generator.export_plan_word）：用户请求
# “生成文书”时导出已生成的维权行动方案为 Word，并引用已有官方空白模板，
# 绝不代用户填写新的起诉状/申请书等文书。DOC_TEMPLATES 手工模板字典已一并删除。


STRATEGY_SYNTHESIS_PROMPT = """你是法护通的案件策略中枢。请基于完整案件快照、争点识别、争点法律适用分析和最终检索法条，自主形成一份面向用户的行动策略数据。

这不是固定场景分类任务。你可以根据材料主动发现新的争点、矛盾、程序节点、证据缺口、对方可能主张和用户尚未意识到的维权机会。不要因为没有预设字段就放弃有价值的推理。

推理边界：
1. 完整案件快照中的事实、状态、冲突和证据状态是事实边界；不要把线索、猜测或控制语句写成已确认事实。
2. 可以使用一般法律知识组织事实关系和提出条件分支，但具体法条名称、条号和原文只能来自“最终检索法条包”。
3. 每个法律结论都要说明成立条件、当前依据和仍需核实的事实；不能把可能性写成已经确定的责任或结果。
4. 优先行动、后续程序和证据计划必须互相衔接，避免重复，也不要为了填满字段而编造渠道、期限或费用。
5. 输出 JSON，不输出 Markdown、思维过程或对本提示词的解释。
6. priority_actions、evidence_plan、risk_boundaries 和 conditions_that_change_result 必须能回溯到 issue_analyses 中的 legal_element_matrix 和 opponent_counterarguments：哪些要件已满足、哪些要件未知、哪些材料能改变判断、对方会如何反驳。禁止只写“报警、保存证据、联系平台”这类脱离争点要件的通用建议。
7. 若“最终检索法条包”中包含 issue_authorities，则每个争点的 priority_actions、evidence_plan 和 risk_boundaries 必须优先使用该争点对应的 authorities，并说明每个行动对应哪个要件。
8. 同一 JSON 中还必须输出 adversarial_execution_review，从办案机关、对方、平台、执行者四个角度做反方压力测试。

输出格式：
{{
  "strategy_plan": {{
    "headline_assessment": {{
      "position": "当前最重要的案件判断",
      "supporting_reason": "哪些事实、证据和法条分析支持该判断",
      "uncertainty": "哪些未确认事实会改变判断"
    }},
    "priority_actions": [
      {{"action": "现在做什么", "object": "对谁或向哪个对象", "purpose": "解决什么问题", "why_now": "为什么此刻优先", "risk": "不做或做错的风险"}}
    ],
    "procedure_path": [
      {{"order": 1, "step": "后续程序步骤", "trigger": "何种条件或前一步结果触发", "expected_change": "该步骤推进后会改变什么"}}
    ],
    "evidence_plan": [
      {{"item": "证据或材料", "status": "obtained|lead|unavailable|unknown", "proof_target": "要证明的事实", "action": "取得、固定或核验动作", "priority": "high|medium|low", "why": "与争点的关系"}}
    ],
    "opponent_arguments": ["对方、办案机关或其他主体可能提出的观点及应对"],
    "institution_focus": ["向办案机关、仲裁机构、法院或其他机构沟通时应抓住的重点"],
    "risk_boundaries": ["当前不能确定的责任、期限、金额或程序结论"],
    "conditions_that_change_result": ["会改变案件定性、程序路径或请求结果的具体条件"],
    "source_issue_ids": ["issue_1"],
    "source_law_refs": ["法律名称+条号"]
  }},
  "adversarial_execution_review": {{
    "adverse_points": [
      {{"point": "对用户不利的点", "source": "来自哪个事实/证据/要件", "impact": "影响什么判断", "countermeasure": "用户如何应对", "changes_conclusion": true}}
    ],
    "evidence_weaknesses": [
      {{"item": "证据或材料", "why": "为什么证明力不足", "impact": "影响什么争点", "remedy": "如何补强"}}
    ],
    "unmet_legal_elements": [
      {{"element": "未满足或未知的法律要件", "law": "法律名称+条号", "why": "当前为什么无法判断", "impact": "影响什么结论", "what_changes_it": "需要什么事实或证据"}}
    ],
    "procedure_risks": ["时效、管辖、前置程序或顺序风险"],
    "opponent_arguments": [
      {{"argument": "对方/平台/办案机关可能提出的抗辩", "response": "用户应如何回应", "evidence_needed": "需要补什么材料"}}
    ],
    "premise_risks": ["方案依赖但尚未证实的乐观前提"],
    "must_disclose": ["最终方案必须明确披露的不利或不确定结论"],
    "current_procedure_stage": "用户当前实际处于哪个程序阶段",
    "next_procedure_stage": "下一阶段应做什么",
    "next_stage_trigger": "进入下一阶段需要什么条件或材料",
    "conditional_paths": [
      {{"condition": "什么事实/证据成立", "path": "应走哪条路径", "if_false": "条件不成立时怎么办"}}
    ],
    "actionability_checks": [
      {{"action": "行动", "object": "对谁做", "purpose": "目的", "materials": ["所需材料"], "expected_result": "预期结果", "next_if_blocked": "受阻后下一步", "ok": true}}
    ],
    "duplicate_actions": ["重复或应合并的行动"]
  }}
}}

## 完整案件快照
{case_snapshot}

## 争点识别
{issue_map}

## 争点法律适用分析
{issue_analyses}

## 最终检索法条包
{legal_basis_packet}

## 证据状态摘要
{evidence_summary}
"""


ADVERSARIAL_EXECUTION_REVIEW_PROMPT = """你是案件对抗与执行推演审查员。请从办案机关、对方当事人、平台、执行者四个角度，对当前策略进行反方压力测试。

## 完整案件快照
{case_snapshot}

## AI识别的法律争点
{issue_map}

## 争点法律适用分析
{issue_analyses}

## 最终检索法条包
{legal_basis}

## AI策略中枢
{strategy_plan}

要求：
1. 所有判断必须来自输入；不得臆造事实、证据或法条。
2. 不利点要具体，不能只写“证据不足”“需进一步核实”；要写明什么材料或动作能改变该点。
3. 每个行动必须能回答：对谁做、具体做什么、需要什么材料、预期结果、被拒绝或受阻后下一步。
4. 条件分支必须说明什么条件成立走哪条路径，条件不成立会怎样。
5. 只输出 JSON，不输出 Markdown 或解释。

输出格式：
{{
  "review": {{
    "adverse_points": [
      {{"point": "对用户不利的点", "source": "来自哪个事实/证据/要件", "impact": "影响什么判断", "countermeasure": "用户如何应对", "changes_conclusion": true}}
    ],
    "evidence_weaknesses": [
      {{"item": "证据或材料", "why": "为什么证明力不足", "impact": "影响什么争点", "remedy": "如何补强"}}
    ],
    "unmet_legal_elements": [
      {{"element": "未满足或未知的法律要件", "law": "法律名称+条号", "why": "当前为什么无法判断", "impact": "影响什么结论", "what_changes_it": "需要什么事实或证据"}}
    ],
    "procedure_risks": ["时效、管辖、前置程序或顺序风险"],
    "opponent_arguments": [
      {{"argument": "对方/平台/办案机关可能提出的抗辩", "response": "用户应如何回应", "evidence_needed": "需要补什么材料"}}
    ],
    "premise_risks": ["方案依赖但尚未证实的乐观前提"],
    "must_disclose": ["最终方案必须明确披露的不利或不确定结论"],
    "current_procedure_stage": "用户当前实际处于哪个程序阶段",
    "next_procedure_stage": "下一阶段应做什么",
    "next_stage_trigger": "进入下一阶段需要什么条件或材料",
    "conditional_paths": [
      {{"condition": "什么事实/证据成立", "path": "应走哪条路径", "if_false": "条件不成立时怎么办"}}
    ],
    "actionability_checks": [
      {{"action": "行动", "object": "对谁做", "purpose": "目的", "materials": ["所需材料"], "expected_result": "预期结果", "next_if_blocked": "受阻后下一步", "ok": true}}
    ],
    "duplicate_actions": ["重复或应合并的行动"]
  }}
}}
"""


PLAN_CRITIQUE_PROMPT = """你是法律方案批判员。你的任务不是重写方案，而是找出当前初稿中必须修正的问题。

## 完整案件快照
{case_snapshot}

## AI识别的法律争点
{issue_map}

## 争点法律适用分析
{issue_analyses}

## 最终检索法条包
{legal_basis}

## 案件对抗与执行推演审查
{adversarial_execution_review}

## 待批判方案初稿
{draft}

要求：
1. 只根据输入材料挑错，不得臆造事实、证据或法条。
2. 重点检查：
   - 初稿是否回避 adversarial_execution_review 中的不利点
   - 法条是否只引用检索池内的条文
   - 每个争点是否有要件拆解、证据状态、条件分支
   - 每个行动是否有对象、目的、所需材料、受阻后下一步
   - 是否存在重复表述、泛泛建议、绝对化承诺或要求用户补充后再分析
   - 初稿是否缺失【法条要件核对】【反方压力测试】【条件分支】【证据作战图】【决策边界与条件】等必输栏目；缺失即 issue type=missing_section
   - 是否存在以下泛化模板句式，出现即 issue type=generic_boilerplate：
     “需要结合完整事实、证据和办案机关认定”
     “当前仅作阶段性分析”
     “现有信息可以支持继续采取低风险的证据保全和程序咨询行动”
     “如关键事实、证据或程序状态不同，法律评价可能随之变化”
     “具体以办案机关认定为准”
3. 如果问题不足以影响方案，verdict 必须是 acceptable，不要为了显得严格而强行要求修订。

只输出 JSON：
{{
  "verdict": "acceptable|revise",
  "issues": [
    {{
      "type": "legal_element_gap|unverified_claim|contradiction|generic_action|citation_issue|missing_procedure|redundancy",
      "target": "方案中哪个部分",
      "issue": "具体问题",
      "fix": "应如何修订"
    }}
  ]
}}
"""


PLAN_REVISION_PROMPT = """你是法律方案修订员。请根据批判结果修订初稿，生成用户最终看到的完整方案。

## 完整案件快照
{case_snapshot}

## AI识别的法律争点
{issue_map}

## 争点法律适用分析
{issue_analyses}

## 最终检索法条包
{legal_basis}

## 案件对抗与执行推演审查
{adversarial_execution_review}

## 待批判方案初稿
{draft}

## 批判结果
{critique}

要求：
1. 保留初稿已经正确、清楚、可执行的内容。
2. 只修改批判结果中列出的问题，不得为了改写而全篇重写。
3. 法条名称、条号和原文只能来自“最终检索法条包”。
4. 不得新增检索池外法条，不得把未知事实写成已确认事实，不得要求用户补充后再分析。
5. 不得输出【关键缺失信息清单】【强烈建议】等违禁栏目；信息不足用【优势与劣势】或【决策边界与条件】说明。
6. 同一事实、行动、证据和渠道不要重复两次以上。
7. 如果批判结果包含 generic_boilerplate，必须用本案具体事实、法律要件和证据状态重写对应段落；禁止保留“需要结合完整事实、证据和办案机关认定”“当前仅作阶段性分析”“现有信息可以支持继续采取低风险的证据保全和程序咨询行动”“如关键事实、证据或程序状态不同，法律评价可能随之变化”等模板句式。
8. 如果批判结果包含 missing_section，必须补回缺失栏目，并使用 legal_element_review 和 adversarial_review_block 中的内容；不得依赖系统自动补全。
9. 直接输出修订后的完整 Markdown 方案，不输出 JSON，不解释过程。
"""


DOMAIN_LABELS: dict[str, str] = {
    "labor_social_security":        "劳动/社保",
    "consumer_market":              "消费维权",
    "contracts_property_housing":   "合同/房产/租房",
    "criminal_public_security":     "刑事/治安",
    "family_vulnerable_groups":     "婚姻家庭/妇女儿童",
    "traffic_personal_injury":      "交通事故/人身损害",
    "medical_education_tax":        "医疗/教育/税务纠纷",
    "administrative_remedies":      "行政救济/行政复议",
    "intellectual_property":        "知识产权",
    "environment_pollution":        "环境保护",
    "cyber_data_fraud":             "网络/数据/诈骗",
    "mediation_notary_arbitration": "调解/公证/仲裁",
    "other":                        "其他",
}
