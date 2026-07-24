# src/api/routers/legal.py
# 法律知识库路由（占位，阶段三完善）

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/legal", tags=["legal"])


@router.get("/health")
async def legal_health():
    """法律模块就绪检查（占位）。"""
    return {"status": "ok", "module": "legal"}
