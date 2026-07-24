from urllib.parse import quote_plus

from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings

settings = get_settings()


def get_trulens_session() -> TruSession:
    """
    初始化 TruLens 会话。
    评估结果存入 PostgreSQL，与业务数据隔离。
    """
    user = quote_plus(settings.DB_USER)
    password = quote_plus(settings.DB_PASSWORD)
    database_url = (
        f"postgresql://{user}:{password}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.TRULENS_DB_NAME}"
    )
    session = TruSession(database_url=database_url)
    return session


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider。
    通过 LiteLLM 对接 DeepSeek (OpenAI 兼容 API)。
    """
    return LiteLLM(
        model_engine=f"openai/{settings.DEEPSEEK_MODEL}",
        completion_kwargs={
            "api_key": settings.DASHSCOPE_API_KEY,
            "api_base": settings.BASE_URL_CHAT,
        },
    )


def launch_dashboard(session: TruSession | None = None, port: int = 8501):
    """启动 TruLens Dashboard（Streamlit 界面）。"""
    from trulens.dashboard import run_dashboard
    s = session or get_trulens_session()
    run_dashboard(session=s, port=port)
