"""Serving /v1/models.

Upstream: vllm/entrypoints/openai/models/serving.py
Tier: C
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "pvllm"
    root: str | None = None
    max_model_len: int | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(default_factory=list)


class OpenAIServingModels:
    """Reports what this server serves."""

    def __init__(
        self, served_model_names: list[str], max_model_len: int, created: int
    ) -> None:
        self.served_model_names = served_model_names
        self.max_model_len = max_model_len
        self.created = created

    def show_available_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=name,
                    created=self.created,
                    root=name,
                    max_model_len=self.max_model_len,
                )
                for name in self.served_model_names
            ]
        )
