import asyncio
from weakref import WeakKeyDictionary

import redis.asyncio as redis
from src.core.config import get_settings

settings = get_settings()

# 模块级别创建连接池（应用启动时初始化一次）
redis_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=True,
    encoding="utf-8",
    max_connections=50,  # 连接池最大连接数
    socket_keepalive=True,  # 启用 TCP keepalive
    socket_keepalive_options={
        1: 60,   # TCP_KEEPIDLE: 空闲 60s 后开始探测
        2: 10,   # TCP_KEEPINTVL: 探测间隔 10s
        3: 3,    # TCP_KEEPCNT: 探测 3 次失败后断开
    },
    health_check_interval=30,  # 每 30s 健康检查一次
)

# 模块级别创建 Redis 客户端实例（复用连接池）; _ 开头认为是 私有变量，不暴露出去
_redis_client = redis.Redis(connection_pool=redis_pool)

# 通用的 Redis 客户端获取函数
async def get_redis_client() -> redis.Redis:
    """
    FastAPI Depends 注入用。
    直接返回模块级别的 Redis 客户端实例，不需要每次创建新实例。
    连接池会自动管理连接的获取和归还。
    """
    return _redis_client


# redis.asyncio 的连接会绑定首次使用它的事件循环。按循环缓存可避免
# pytest、热重载或脚本多次 asyncio.run() 后复用已关闭循环中的连接。
_checkpointer_clients: WeakKeyDictionary[
    asyncio.AbstractEventLoop, redis.Redis
] = WeakKeyDictionary()


def _new_checkpointer_client() -> redis.Redis:
    pool = redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=False,  # RedisSaver 要求 bytes，不能 decode
        max_connections=30,
        socket_keepalive=True,
        socket_keepalive_options={
            1: 60,
            2: 10,
            3: 3,
        },
        health_check_interval=30,
    )
    return redis.Redis(connection_pool=pool)

def get_checkpointer_redis() -> redis.Redis:
    """
    返回供 RedisSaver（LangGraph checkpointer）专用的 Redis 客户端。
    decode_responses=False，以 bytes 模式运行，与业务 Redis 客户端隔离。
    """
    loop = asyncio.get_running_loop()
    client = _checkpointer_clients.get(loop)
    if client is None:
        client = _new_checkpointer_client()
        _checkpointer_clients[loop] = client
    return client


async def set_with_optional_ttl(
    client: redis.Redis,
    key: str,
    value,
    ttl: int | None,
):
    """Persist when ttl is zero/None; otherwise apply the configured lifetime."""
    if ttl and ttl > 0:
        return await client.set(key, value, ex=ttl)
    return await client.set(key, value)
