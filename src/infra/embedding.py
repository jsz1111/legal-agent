"""
Embedding 模型统一接口，支持 Ollama（本地）、DashScope（云端）和 Volcengine（火山方舟）三路径。
"""
from langchain_core.embeddings import Embeddings

from src.core.config import get_settings


def get_embedding_model() -> Embeddings:
    """
    获取 Embedding 模型，支持自动降级：
    - ollama: 本地模型（无网络依赖）
    - dashscope: 阿里云 DashScope API（需要网络和 API Key）
    - volcengine: 火山方舟 API（豆包 Doubao-embedding）

    配置方式：在 .env 文件中设置 EMBEDDING_PROVIDER
    """
    settings = get_settings()

    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    elif provider == "dashscope":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.DASHSCOPE_API_KEY,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            check_embedding_ctx_length=False,
        )

    elif provider == "volcengine":
        from src.infra.volcengine_embeddings import VolcengineMultimodalEmbeddings
        return VolcengineMultimodalEmbeddings(
            api_key=settings.VOLCENGINE_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimensions=1024,
        )

    else:
        raise ValueError(
            f"不支持的 embedding provider: {provider}。"
            f"请在 .env 中设置 EMBEDDING_PROVIDER 为 'ollama'、'dashscope' 或 'volcengine'"
        )
