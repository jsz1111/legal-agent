"""
TruLens 会话配置 + LLM Provider + Dashboard 启动
"""
from urllib.parse import quote_plus

from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings

settings = get_settings()
DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"


def get_trulens_session() -> TruSession:
    """评估结果存入独立的 PostgreSQL 数据库"""
    user = quote_plus(settings.DB_USER)
    password = quote_plus(settings.DB_PASSWORD)
    return TruSession(
        database_url=f"postgresql://{user}:{password}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.TRULENS_DB_NAME}"
    )


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider（LLM-as-Judge）。
    注意：参数名是 model_engine，不是 model_name。
    前缀 openai/ 表示走 OpenAI 兼容协议。
    """
    return LiteLLM(
        model_engine=f"openai/{DEEPSEEK_FLASH_MODEL}",
        completion_kwargs={
            "api_key": settings.DASHSCOPE_API_KEY,
            "api_base": settings.BASE_URL_CHAT,
        },
    )


def launch_dashboard(port: int = 8501):
    """启动 TruLens Streamlit Dashboard"""
    from trulens.dashboard import run_dashboard
    session = get_trulens_session()
    run_dashboard(session=session, port=port)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    launch_dashboard(port)
