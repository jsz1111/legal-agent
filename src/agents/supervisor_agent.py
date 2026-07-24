from dataclasses import dataclass

from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.redis import AsyncRedisSaver
from langchain.agents import create_agent

from src.agents.tools.store_tools import save_memory, search_memory
from src.agents.tools.worker_tools import WORKER_TOOLS, UserContext
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.infra.redis_cache import get_checkpointer_redis
from dotenv import load_dotenv
load_dotenv()


def _get_embedding_model():
    """返回向量化模型。根据你的实际情况替换。"""
    # 使用 DashScope（阿里云）: 环境变量需要配置 DASHSCOPE_API_KEY
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(model="text-embedding-v3")


SUPERVISOR_SYSTEM_PROMPT = """你是法律多智能体平台的智能总助手，面向不懂法的普通市民和企业用户提供法律服务。

## 你的职责

识别用户意图，调用对应的专项助手处理，整合结果后给出清晰、友好的回复。
你自己不直接回答任何法律问题，必须通过专项助手完成。

## 可调用的专项助手

- call_guide_agent：公民法律指引
  适用：用户描述具体纠纷、事件或维权诉求时（如被拖欠工资、消费投诉、合同违约、家庭纠纷、人身伤害等）
  示例："公司三个月没发工资"、"买到假货商家不退款"、"房东不退押金"、"被人打伤了怎么办"
  注意：后续多轮追问由系统自动路由，无需再次调用此工具

- call_legal_qa_agent：法律知识问答
  适用：用户询问法律概念、法条含义、制度性知识、维权流程等通用法律知识时
  示例："劳动仲裁流程是什么"、"诉讼时效是多久"、"什么是合同解除"、"消费者权益保护法的适用范围"

- call_professional_agent：专业法律服务（功能建设中）
  适用：法律从业者需要裁决预测、案件分析、文书摘要等专业服务时

- call_exam_agent：法考助手（功能建设中）
  适用：法学学生或备考人员需要法考真题练习和知识点讲解时

- call_operation_agent：运营数据查询（仅限内部运营人员）
  适用：查询平台咨询量、领域分布、用户数据等运营统计时

## 记忆工具

- search_memory：调用专项助手前，先检索用户的历史法律咨询记录（曾咨询的领域、地区偏好等），将相关信息附加到调用参数中
- save_memory：当用户提到重要个人信息时（所在城市、案件进展、重要时间节点等），主动保存到长期记忆

## 工作原则

1. 路由优先级：用户描述具体纠纷/事件 → call_guide_agent；询问法律知识/概念 → call_legal_qa_agent；意图不明确时默认调用 call_guide_agent。
2. 传递消息规则：调用专项助手时，message 参数必须原样传递用户的原始输入，禁止改写或补充。如果 search_memory 检索到相关历史信息，以"[长期记忆] xxx"格式附加在用户原文之后。
3. 紧急情形优先：识别到"人身安全威胁"、"正在遭受暴力"、"被拘留"等紧急关键词时，立即提示用户拨打 110（警察）或 12348（法律援助），不要等待流程。
4. 语气温和通俗：用普通人能理解的语言表达，避免堆砌法律术语。"""



# 创建出 监督 Agent
async def create_supervisor_agent():
    # 1. 复用项目已有的 checkpointer 专用 Redis 客户端（bytes 模式）
    redis_client = get_checkpointer_redis()

    # 2. 创建 AsyncRedisSaver，并调用 asetup() 初始化 RediSearch 索引
    # asetup() 会在 Redis Stack 中创建 checkpoint / checkpoint_write 两个索引
    # 必须在首次使用前调用一次，索引已存在时自动跳过，可以重复调用
    checkpointer = AsyncRedisSaver(redis_client=redis_client)
    await checkpointer.asetup()


    # 3. 长期记忆
    # ── 长期记忆：Milvus Store ─────────────────────────────────────────
    milvus_alias = get_milvus_client_alias()
    embedding_model = _get_embedding_model()
    store = MilvusStore(
        alias=milvus_alias,
        embeddings=embedding_model,
        dims=1024,   # DashScope text-embedding-v3 默认输出 1024 维
    )

    # 4. 长期记忆工具
    tools = [save_memory, search_memory] + WORKER_TOOLS

    # 4. 创建 Agent
    agent = create_agent(
        model="deepseek-chat",
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        context_schema=UserContext,
        middleware=[
            SummarizationMiddleware( # 会话总结压缩
                model="deepseek-chat",
                trigger=[
                    ("tokens", 4000),  # token数达到4k时触发
                    ("messages", 6),  # 或消息数达到 4条时触发
                    # ("fraction", 0.8)  # 或80%消息时触发
                ],
                keep=("messages", 6),  # 摘要后保留最近 4 条消息
            )
        ],
        checkpointer=checkpointer, # 短期记忆. agent chat ui（禁用你配置 checkpointer）
        store=store # 长期记忆. 把同一个用户，任意会话的有价值信息进行存储。
    )
    return agent


# 模块级单例：避免每次请求都重新创建 agent 和 checkpointer
_supervisor_agent = None

# 返回 agent
async def get_supervisor_agent():
    """返回全局单例 Agent，首次调用时初始化。"""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = await create_supervisor_agent()
    return _supervisor_agent


# FastAPI 路由中使用
async def chat_endpoint(user_id: str, session_id: str, message: str):
    # 使用单例，不重复初始化。获取 agent
    agent = await get_supervisor_agent()

    # thread_id 用来区分不同会话。  用户id:会话id:日期（前端传来一个会话id）
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        context=UserContext(user_id=user_id, session_id=session_id),
    )
    return result["messages"][-1].content