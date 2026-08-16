"""Serving /v1/models.

Upstream: vllm/entrypoints/openai/models/serving.py
Tier: C
"""

from __future__ import annotations

from typing import Any, Literal

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
        self,
        served_model_names: list[str],
        max_model_len: int,
        created: int,
        lora_modules: dict[str, Any] | None = None,
    ) -> None:
        self.served_model_names = served_model_names
        self.max_model_len = max_model_len
        self.created = created
        #: R16.1. Adapter name -> LoRARequest. An adapter is served under its own
        #: model name, so a client selects it the way it selects any model.
        self.lora_modules: dict[str, Any] = dict(lora_modules or {})

    def resolve(self, model: str) -> tuple[bool, Any]:
        """`(is_served, lora_request)` for a requested model name. R16.1."""
        if model in self.lora_modules:
            return True, self.lora_modules[model]
        return model in self.served_model_names, None

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
            + [
                # Adapters are listed with the base model as their root, which is
                # how a client discovers what it may ask for.
                ModelCard(
                    id=name,
                    created=self.created,
                    root=self.served_model_names[0],
                    max_model_len=self.max_model_len,
                )
                for name in self.lora_modules
            ]
        )


def build_lora_modules(lora_config: Any, specs: list[str] | None) -> dict[str, Any]:
    """`name=path` specs to `{name: LoRARequest}`. R16.1.

    Ids are assigned by position, so a given `--lora-modules` line always produces
    the same id -- which matters because the id partitions the prefix cache, and an
    id that shifted between restarts would silently invalidate every cached prefix
    for that adapter.
    """
    if not specs:
        return {}
    if lora_config is None:
        raise ValueError(
            "--lora-modules was given without --enable-lora. Serving an adapter "
            "changes both the memory budget and the admission constraint, so it is "
            "not inferred from the presence of a module."
        )

    from pvllm.lora.request import LoRARequest

    modules: dict[str, Any] = {}
    for index, spec in enumerate(specs, start=1):
        if "=" not in spec:
            raise ValueError(
                f"malformed --lora-modules entry {spec!r}; expected NAME=PATH"
            )
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(
                f"malformed --lora-modules entry {spec!r}; expected NAME=PATH"
            )
        if len(modules) >= lora_config.max_loras:
            raise ValueError(
                f"{len(specs)} adapters were given but max_loras is "
                f"{lora_config.max_loras}. Raise --max-loras, which also raises the "
                f"memory the adapters occupy."
            )
        modules[name] = LoRARequest(
            lora_name=name,
            lora_int_id=index,
            lora_path=path,
            rank=lora_config.max_lora_rank,
        )
    return modules
