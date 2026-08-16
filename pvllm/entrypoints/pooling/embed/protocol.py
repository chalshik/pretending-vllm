"""The OpenAI embeddings schema. R2.2, C5.

Upstream: vllm/entrypoints/pooling/embed/protocol.py
Tier: C

Shape only, as with every other protocol module: the field names, types and defaults
are the contract a client is written against, and the values behind them are
synthetic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pvllm.pooling_params import PoolingParams


class EmbeddingRequest(BaseModel):
    """`POST /v1/embeddings`."""

    model: str
    #: One string, one token-id list, or a batch of either. Unlike `/v1/completions`,
    #: batching *is* the normal shape here -- an embedding client sends a page of
    #: documents at a time -- so it is supported rather than refused.
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: str = "float"
    dimensions: int | None = None
    user: str | None = None
    priority: int = 0

    def to_pooling_params(self) -> PoolingParams:
        return PoolingParams(task="embed", dimensions=self.dimensions)


class EmbeddingResponseData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "list"
    created: int = 0
    model: str = ""
    data: list[EmbeddingResponseData] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


__all__ = [
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingResponseData",
    "UsageInfo",
]
