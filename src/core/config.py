# 读取环境变量
from functools import lru_cache
import math

from langchain_deepseek import ChatDeepSeek
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "legal-agent"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "legal"
    DB_PASSWORD: str = "legal123"
    DB_NAME: str = "legal_db"
    TRULENS_DB_NAME: str = "trulens_eval"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

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
    CHAT_MODEL: str = "deepseek-chat"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    VL_MODEL: str = "qwen-vl"
    DEEPSEEK_MODEL: str = "deepseek-chat"
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

    # 指定环境变量文件
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# 保存到内存缓存中。以后直接获取。
@lru_cache  # lru 把对象实例保存到内存中。这是一种单例的实现
def get_settings() -> Settings:
    return Settings()

def get_llm():
    return "deepseek-chat"
