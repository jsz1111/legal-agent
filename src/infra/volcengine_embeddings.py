"""火山方舟多模态 Embedding 适配器"""
import asyncio
from typing import List
import httpx
from langchain_core.embeddings import Embeddings


class VolcengineMultimodalEmbeddings(Embeddings):
    """火山方舟多模态 Embedding 模型（支持纯文本）"""

    def __init__(
        self,
        api_key: str,
        model: str = "doubao-embedding-vision-251215",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        dimensions: int = 1024,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions
        self.client = httpx.AsyncClient(timeout=60.0)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量向量化文本"""
        url = f"{self.base_url}/embeddings/multimodal"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # 火山方舟的多模态 API 只支持单次请求单个 input，需要逐个调用
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "encoding_format": "float",
                "input": [{"type": "text", "text": text}],
                "dimensions": self.dimensions,
            }

            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # 提取 embedding（格式: {data: {embedding: [...]}}）
            embedding = None
            if isinstance(data, dict) and isinstance(data.get("data"), dict):
                embedding = data["data"].get("embedding")
            # 必须与入参一一对应：拿不到就抛错，不能静默跳过，
            # 否则返回列表变短，上游 zip(rows, embeddings) 会让向量与文本错位
            if not embedding:
                raise ValueError(f"火山方舟 embedding 响应缺少 embedding 字段: {str(data)[:200]}")
            embeddings.append(embedding)

        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """异步单条文本向量化"""
        result = await self.aembed_documents([text])
        return result[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步批量向量化（内部调用异步方法）"""
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """同步单条文本向量化（内部调用异步方法）"""
        return asyncio.run(self.aembed_query(text))
