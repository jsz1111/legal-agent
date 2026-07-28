# src/agents/workers/operation_agent.py
# 运营数据占位 Agent — 功能开发中

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from src.core.config import get_settings

settings = get_settings()

OPERATION_SYSTEM_PROMPT = """你是法律平台的运营数据助手。

## 你的职责
1. 查询平台运营统计数据（咨询量、领域分布等）
2. 注意：当前运营数据功能尚在开发中，暂无实际数据可查。

## 回复规则
- 直接告知用户"运营数据功能开发中，暂不可用。"
- 不要编造任何统计数据。"""


def create_operation_agent(db_session=None):
    llm = ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.2,
    )
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=OPERATION_SYSTEM_PROMPT,
        name="operation_agent",
    )


_operation_agent = None

def get_operation_agent():
    global _operation_agent
    if _operation_agent is None:
        _operation_agent = create_operation_agent()
    return _operation_agent
