# src/agents/tools/worker_tools.py

from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.workers.guide_agent import call_guide_agent_impl
from src.agents.workers.legal_qa_agent import get_legal_qa_agent
from src.agents.workers.professional_agent import handle_professional
from src.agents.workers.exam_agent import handle_exam
from src.agents.workers.operation_agent import get_operation_agent
from src.infra.redis_cache import get_checkpointer_redis
from src.core.config import get_settings

settings = get_settings()


@dataclass
class UserContext:
    user_id: str
    session_id: str


@tool
async def call_guide_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    启动公民法律指引流程。
    适用场景：用户描述具体法律纠纷或事件（拖欠工资、消费维权、合同违约、家庭纠纷等），
    需要法律依据、证据清单、维权路径和可操作步骤时。
    后续多轮追问由系统自动路由，无需再次调用此工具。

    Args:
        message: 用户描述的具体纠纷情况或维权诉求
    """
    session_id = runtime.context.session_id
    user_id = runtime.context.user_id
    print("🔧工具调用 call_guide_agent :", session_id, message)

    return await call_guide_agent_impl(
        message=message,
        user_id=user_id,
        session_id=session_id,
    )


@tool
async def call_legal_qa_agent(message: str) -> str:
    """
    调用法律知识问答Agent，回答法律知识类问题。
    适用场景：询问法律概念、法条含义、制度性知识、维权流程等通用法律知识时。
    示例："劳动仲裁的流程是什么"、"什么是诉讼时效"、"合同解除需要哪些条件"

    Args:
        message: 用户的法律知识问题
    """
    agent = get_legal_qa_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_professional_agent(message: str) -> str:
    """
    调用专业法律助手，面向法律从业者提供裁决预测、案件分析、文书摘要等服务。
    适用场景：律师或法务人员需要专业法律分析时。

    Args:
        message: 法律从业者的专业分析需求
    """
    return await handle_professional(message)


@tool
async def call_exam_agent(message: str) -> str:
    """
    调用法考助手，提供法考真题练习和知识点讲解。
    适用场景：法学学生或备考人员需要法考题目解析时。

    Args:
        message: 法考相关问题或练习需求
    """
    return await handle_exam(message)


@tool
async def call_operation_agent(message: str) -> str:
    """
    调用运营数据Agent，查询平台运营统计数据（仅限内部运营人员）。
    适用场景：运营人员查询用户量、咨询量、领域分布等运营数据时。

    Args:
        message: 运营人员的数据查询需求（自然语言）
    """
    agent = get_operation_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


# 所有 Worker 工具列表，供 Supervisor 使用
WORKER_TOOLS = [
    call_guide_agent,
    call_legal_qa_agent,
    call_professional_agent,
    call_exam_agent,
    call_operation_agent,
]
