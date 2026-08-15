"""Model configuration.

Upstream: vllm/config/model.py
Tier: C

Field names and defaults match upstream for every field pretending-vllm supports.
Upstream's `ModelConfig` is ~2,000 lines because it negotiates quantization, LoRA
targets, multimodal processors, and pooling. None of that applies here, so this keeps
the fields that reach the scheduler, the memory model, or the HTTP surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pvllm.logger import init_logger
from pvllm.sim.model_db import DTYPE_BYTES, ModelCard
from pvllm.transformers_utils.config import get_config

logger = init_logger(__name__)

#: Upstream's default model at the pin.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class ModelConfig:
    """Configuration for the model to serve."""

    model: str = DEFAULT_MODEL
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "auto"
    seed: int = 0
    revision: str | None = None
    tokenizer_revision: str | None = None
    max_model_len: int | None = None
    enforce_eager: bool = False
    max_logprobs: int = 20
    skip_tokenizer_init: bool = False
    served_model_name: str | None = None
    #: Explicit model card name or path, overriding the lookup by `model`.
    #: pretending-vllm-specific; the upstream counterpart is the HF hub.
    model_card: str | None = None

    hf_config: ModelCard = field(init=False)
    #: Resolved from `dtype`; `"auto"` takes the card's declared dtype.
    resolved_dtype: str = field(init=False)

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.served_model_name is None:
            self.served_model_name = self.model

        self.hf_config = get_config(self.model, self.model_card)

        if self.dtype == "auto":
            self.resolved_dtype = self.hf_config.dtype
        elif self.dtype in DTYPE_BYTES:
            self.resolved_dtype = self.dtype
        else:
            raise ValueError(
                f"unsupported dtype {self.dtype!r}; expected 'auto' or one of "
                f"{sorted(DTYPE_BYTES)}"
            )

        # R1.5: upstream clamps max_model_len to the model's positional limit and
        # errors when the user asks for more than the architecture supports.
        derived_max = self.hf_config.max_position_embeddings
        if self.max_model_len is None:
            self.max_model_len = derived_max
        elif self.max_model_len > derived_max:
            raise ValueError(
                f"max_model_len ({self.max_model_len}) is larger than the maximum "
                f"the model supports ({derived_max}). Lower max_model_len."
            )
        if self.max_model_len <= 0:
            raise ValueError(
                f"max_model_len must be positive, got {self.max_model_len}"
            )

        if self.tokenizer_mode not in ("auto", "mock", "slow"):
            raise ValueError(
                f"unsupported tokenizer_mode {self.tokenizer_mode!r}; expected "
                f"'auto', 'mock', or 'slow'"
            )

    # --- derived architecture, used by the memory and cost models ------------

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.resolved_dtype]

    def get_num_layers(self, tp_size: int = 1, pp_size: int = 1) -> int:
        """Layers resident on one device. Pipeline parallelism shards these."""
        return self.hf_config.num_hidden_layers // pp_size

    def get_num_kv_heads(self, tp_size: int = 1) -> int:
        """KV heads resident on one device. At least one, as upstream replicates
        rather than splitting a head when TP exceeds the KV head count."""
        return max(1, self.hf_config.num_key_value_heads // tp_size)

    def get_num_attention_heads(self, tp_size: int = 1) -> int:
        return max(1, self.hf_config.num_attention_heads // tp_size)

    def get_head_size(self) -> int:
        return self.hf_config.head_dim

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    @property
    def is_moe(self) -> bool:
        return self.hf_config.is_moe

    @property
    def architecture(self) -> str:
        return self.hf_config.architecture
