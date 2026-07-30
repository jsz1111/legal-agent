# 读取环境变量
from functools import lru_cache
import math
from urllib.parse import quote_plus

from langchain_deepseek import ChatDeepSeek
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "legal-agent"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    # ── 法律指引业务配置 ──
    # 置信度分档阈值
    GUIDE_CONFIDENCE_HIGH: float = 0.65   # HIGH 档：高置信度，核心证据齐全即可达标
    GUIDE_CONFIDENCE_MID: float = 0.50    # MID 档：中等置信度
    # 澄清/追问轮次上限
    GUIDE_MAX_CLARIFY_ROUNDS: int = 2     # 最多澄清 2 轮
    GUIDE_MAX_ASK_ROUNDS: int = 6         # 事实+证据追问总上限
    GUIDE_SOFT_ASK_ROUNDS: int = 3        # 用户体验软上限，通常到此即按现有信息给方案
    GUIDE_MAX_FACT_ROUNDS: int = 3        # 最多追问关键事实 3 轮
    GUIDE_MAX_EVIDENCE_ROUNDS: int = 3    # 最多追问证据 3 轮
    GUIDE_MAX_LOW_INFO_ANSWERS: int = 2   # 连续“不知道/没有”后停止盘问
    GUIDE_MAX_COUNTER_QUESTIONS: int = 3  # 连续只反问 3 次后按现有信息收敛
    GUIDE_MAX_TOTAL_ROUNDS: int = 12      # 用户消息达到 12 轮时强制收敛
    GUIDE_SESSION_TTL: int = 86400         # 法律指引状态与文书保留 24 小时，支持离开页面后继续
    # 检索超时配置（秒）
    GUIDE_RETRIEVE_TIMEOUT_STATUTE: float = 8.0   # 法条检索超时
    GUIDE_RETRIEVE_TIMEOUT_CASE: float = 5.0      # 案例检索超时
    GUIDE_RETRIEVE_TIMEOUT_GRAPH: float = 3.0     # 图谱查询超时

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "legal"
    DB_PASSWORD: str = "legal123"
    DB_NAME: str = "legal_db"
    TRULENS_DB_NAME: str = "trulens_eval"
    LEGAL_STATISTICS_DB_NAME: str = "legal_statistics_db"
    LEGAL_STATISTICS_DB_USER: str = "legal_statistics_reader"
    LEGAL_STATISTICS_DB_PASSWORD: str = ""

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REDIS_SESSION_TTL: int = 3600  # Redis 会话状态 TTL（秒），默认1小时

    # 多模态支持（可选）
    ENABLE_MULTIMODAL: bool = False  # 是否启用多模态（图片理解）
    VL_MODEL: str = "qwen-vl-max"    # 阿里云视觉语言模型
    VL_API_KEY: str = ""             # 阿里云 API Key（为空则禁用多模态）
    MULTIMODAL_MAX_FILE_MB: int = 10
    MULTIMODAL_MAX_PIXELS: int = 24_000_000
    MULTIMODAL_TIMEOUT: int = 60
    MULTIMODAL_RETAIN_UPLOADS: bool = False

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "legal123"

    # 模型
    DASHSCOPE_API_KEY: str = ""
       # 聊天模型
    BASE_URL_CHAT: str = ""
    DEEPSEEK_API_KEY: str = ""
    CHAT_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"

    # Embedding 配置
    EMBEDDING_PROVIDER: str = "ollama"  # "ollama" | "dashscope" | "volcengine"
    EMBEDDING_MODEL: str = "bge-large"  # ollama: "bge-large" | dashscope: "text-embedding-v3" | volcengine: "Doubao-embedding"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # 火山方舟 API 配置
    VOLCENGINE_API_KEY: str = ""
    MINERU_API_URL: str = Field(
        default="",
        validation_alias=AliasChoices("MINERU_API_URL", "mineru_api_url"),
    )
    MINERU_BACKEND: str = Field(
        default="hybrid-auto-engine",
        validation_alias=AliasChoices("MINERU_BACKEND", "mineru_backend"),
    )
    MINERU_TIMEOUT: int = Field(
        default=50,
        validation_alias=AliasChoices("MINERU_TIMEOUT", "mineru_timeout"),
    )

    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    @field_validator("MINERU_API_URL", mode="before")
    @classmethod
    def normalize_mineru_api_url(cls, value):
        if not value:
            return ""
        return str(value).split("#", 1)[0].rstrip("/")

    @field_validator("MINERU_TIMEOUT", mode="after")
    @classmethod
    def normalize_mineru_timeout(cls, value: int) -> int:
        # Some env files store timeout in milliseconds; normalize to seconds.
        if value > 1000:
            return max(1, math.ceil(value / 1000))
        return max(1, value)

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def LEGAL_STATISTICS_DATABASE_URL(self) -> str:
        password = self.LEGAL_STATISTICS_DB_PASSWORD or self.DB_PASSWORD
        return (
            f"postgresql+asyncpg://{quote_plus(self.LEGAL_STATISTICS_DB_USER)}:"
            f"{quote_plus(password)}@{self.DB_HOST}:{self.DB_PORT}/"
            f"{quote_plus(self.LEGAL_STATISTICS_DB_NAME)}"
        )

    # 指定环境变量文件
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# 保存到内存缓存中。以后直接获取。
@lru_cache  # lru 把对象实例保存到内存中。这是一种单例的实现
def get_settings() -> Settings:
    return Settings()

def get_llm():
    return "deepseek-v4-flash"
