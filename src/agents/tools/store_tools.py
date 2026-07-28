from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

def get_user_id(runtime: ToolRuntime):
    # 从 thread_id 中提取用户ID
    if runtime.config.get("configurable"):
        thread_id = runtime.config.get("configurable").get("thread_id")
        if thread_id:
            return thread_id.split(":")[0]
    return None

@tool
async def save_memory(
    content: str,
    runtime: ToolRuntime,
) -> str:
    """
    将重要信息保存到长期记忆中。
    当用户提到需要跨会话记住的内容时调用，例如：所在城市/地区、案件进展状态、
    重要时间节点（签合同日期、离职日期等）、已有证据情况、惯用维权渠道偏好等。

    Args:
        content: 要记住的内容，用一句话描述
    """

    # Skip print to avoid GBK encoding errors in Windows terminal
    user_id = get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    import time
    key = f"memory_{int(time.time())}"
    await runtime.store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    return f"已记住：{content}"


@tool
async def search_memory(
    query: str,
    runtime: ToolRuntime,
) -> str:
    """
    从长期记忆中检索与问题相关的历史信息。
    当需要回忆用户之前说过的内容、历史法律咨询、案件进展等时调用。

    Args:
        query: 检索关键词或问题
    """

    user_id = get_user_id(runtime)
    # Skip print to avoid GBK encoding errors in Windows terminal
    # print(f"[TOOL] search_memory: query={query}, user_id={user_id}")
    if not user_id:
        return "无法获取用户ID。"

    results = await runtime.store.asearch(
        ("users", user_id, "memories"),
        query=query,
        limit=5,
    )
    if not results:
        return "没有找到相关记忆。"

    memories = [f"- {item.value['content']}" for item in results]
    return "相关历史记忆：\n" + "\n".join(memories)