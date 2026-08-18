"""Model cards. Section 8.

Upstream: (none -- simulator)
Tier: D

No weights are ever read (NG1). What a model *is*, here, is its architecture: enough
dimensions to compute KV cache footprint exactly and FLOPs approximately.

**Parameter counts are derived, not quoted.** A card declares layer dimensions and
`num_parameters` is computed from them. Quoting a published figure alongside
hand-entered dimensions invites the two to disagree, and then `weight_bytes = P *
dtype_bytes` describes a different model than the one the FLOPs term describes. Deriving
makes the memory model and the cost model provably consistent with each other. A card
may still set `num_parameters_override` when it needs to match a published number, and
`parameter_count_discrepancy` reports the gap so the choice stays visible.

Cards are named by architecture (`dense-8b`) rather than by product, because these are
representative of a *class* of model, not measurements of a specific checkpoint.
`ALIASES` maps common Hugging Face ids onto them so a client that asks for
`meta-llama/Llama-3.1-8B-Instruct` gets a sensible card. An unknown id is an error, not
a guess: silently inventing an architecture would produce memory and throughput numbers
that are fiction *and* unlabeled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).parent / "models"

#: Bytes per element, by dtype name.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
    "int8": 1,
}

#: Common Hugging Face ids mapped onto the architecture card that represents them.
#: Approximate by construction -- see each card's `provenance`.
ALIASES: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "dense-0.6b",
    "Qwen/Qwen3-1.7B": "dense-0.6b",
    "meta-llama/Llama-3.1-8B": "dense-8b",
    "meta-llama/Llama-3.1-8B-Instruct": "dense-8b",
    "meta-llama/Llama-3-8B-Instruct": "dense-8b",
    "mistralai/Mistral-7B-Instruct-v0.3": "dense-8b",
    "Qwen/Qwen3-8B": "dense-8b",
    "meta-llama/Llama-3.1-70B": "dense-70b",
    "meta-llama/Llama-3.1-70B-Instruct": "dense-70b",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "moe-8x7b",
}


@dataclass(frozen=True)
class ModelCard:
    """A model's architecture. Everything needed for memory and cost, nothing else."""

    name: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    dtype: str = "bfloat16"
    architecture: str = "LlamaForCausalLM"
    tie_word_embeddings: bool = False
    #: MoE. `num_experts == 0` means dense. `num_experts_per_token` drives the
    #: *active* parameter count, which is what the compute term actually uses.
    num_experts: int = 0
    num_experts_per_token: int = 0
    #: R6.7. Hybrid attention. `sliding_window` is the window windowed layers use;
    #: `sliding_window_pattern` is the repeat length, with every `pattern`-th layer
    #: full attention and the rest windowed -- Gemma-3's 5:1 is `pattern = 6`. A card
    #: with a window and no pattern is *uniformly* windowed (Mistral-style); a card
    #: with neither is ordinary full attention.
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None
    #: R6.7. Multi-head latent attention (DeepSeek-V2/V3 class). Instead of storing a
    #: key and a value per KV head, MLA stores one *compressed latent* per token:
    #: `kv_lora_rank` dimensions of content plus `qk_rope_head_dim` of positional
    #: information. Both set means MLA; neither means ordinary attention.
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    #: R6.7. State-space (Mamba) layers. `hybrid_override_pattern` is the per-layer
    #: string these models publish -- 'M' a Mamba layer, '*' an attention layer, '-' an
    #: MLP layer -- and it is what makes a hybrid like Nemotron-H expressible: 24 Mamba
    #: layers, 4 attention, 24 MLP. The rest size one layer's recurrent state, which is
    #: constant per request however long the context gets.
    hybrid_override_pattern: str | None = None
    mamba_num_heads: int | None = None
    mamba_head_dim: int | None = None
    mamba_state_size: int | None = None
    mamba_n_groups: int = 1
    mamba_conv_kernel: int = 4
    #: Families differ here and it is a factor-of-two error to get wrong: Nemotron-H
    #: keeps the SSM state in float32 while Bamba and Falcon-H1 leave it at the model
    #: dtype. `None` means the model dtype.
    mamba_ssm_cache_dtype: str | None = None
    #: Set only when a card must match a published figure that the derivation misses.
    num_parameters_override: int | None = None
    provenance: str = "uncalibrated approximation"
    extra: dict[str, Any] = field(default_factory=dict)

    # --- derived architecture ------------------------------------------------

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def __post_init__(self) -> None:
        """Refuse a card that half-declares a feature.

        Cards are a documented extension point -- `load_model_card` takes an explicit
        path and its own error message invites you to add one -- so a typo in a field
        name is a reachable authoring mistake, and `from_dict` sweeps unknown keys
        into `extra` without complaint. The failure without this check is not a
        crash: it is a plausible capacity number that is wrong by 7.1x, produced from
        fields the architecture no longer uses.
        """
        if (self.kv_lora_rank is None) != (self.qk_rope_head_dim is None):
            declared, missing = (
                ("kv_lora_rank", "qk_rope_head_dim")
                if self.kv_lora_rank is not None
                else ("qk_rope_head_dim", "kv_lora_rank")
            )
            raise ValueError(
                f"model card {self.name!r} declares {declared} but not {missing}. "
                f"Multi-head latent attention needs both -- the cached latent is "
                f"kv_lora_rank + qk_rope_head_dim wide (R6.7). Set both for an MLA "
                f"model (DeepSeek-V2/V3 class) or neither for ordinary attention; "
                f"with only one the card would be sized as multi-head attention "
                f"from num_key_value_heads and head_dim, which an MLA model does "
                f"not use for KV."
            )

    @property
    def is_state_space(self) -> bool:
        """R6.7. Whether any layer holds a recurrent state instead of KV."""
        return self.num_mamba_layers > 0

    @property
    def num_mamba_layers(self) -> int:
        if not self.hybrid_override_pattern:
            return 0
        return self.hybrid_override_pattern.count("M")

    @property
    def num_attention_layers(self) -> int:
        """Layers that actually cache KV. Not `num_hidden_layers` for a hybrid."""
        if not self.hybrid_override_pattern:
            return self.num_hidden_layers
        return self.hybrid_override_pattern.count("*")

    def mamba_state_bytes_per_layer(self, tp_size: int = 1) -> int:
        """One Mamba layer's recurrent state for one request. R6.7, R10.2.

        Two tensors, both constant in context: a conv ring buffer
        `(conv_kernel - 1, conv_dim / tp)` and an SSM temporal state
        `(heads / tp, head_dim, state_size)`, where
        `conv_dim = heads * head_dim + 2 * n_groups * state_size`.

        The SSM half dominates -- 97% of the page on a Nemotron-H-class model -- and
        it is the half whose dtype differs by family, which is why the card carries
        `mamba_ssm_cache_dtype` separately.
        """
        assert self.mamba_num_heads is not None
        assert self.mamba_head_dim is not None
        assert self.mamba_state_size is not None

        conv_bytes = DTYPE_BYTES[self.dtype]
        ssm_bytes = DTYPE_BYTES[self.mamba_ssm_cache_dtype or self.dtype]

        d_inner = self.mamba_num_heads * self.mamba_head_dim
        conv_dim = d_inner + 2 * self.mamba_n_groups * self.mamba_state_size
        conv_elems = (self.mamba_conv_kernel - 1) * max(1, conv_dim // tp_size)
        ssm_elems = (
            max(1, self.mamba_num_heads // tp_size)
            * self.mamba_head_dim
            * self.mamba_state_size
        )
        return conv_elems * conv_bytes + ssm_elems * ssm_bytes

    def mamba_state_bytes(self, tp_size: int = 1) -> int:
        """Every Mamba layer's state for one request -- the constant per-request cost."""
        return self.num_mamba_layers * self.mamba_state_bytes_per_layer(tp_size)

    @property
    def use_mla(self) -> bool:
        """R6.7. Whether attention stores a compressed latent rather than K and V.

        `__post_init__` refuses a card that declares one half of the pair, so this
        really is the two-state property its field comment claims. Left unchecked,
        a card with `kv_lora_rank` and a misspelled `qk_rope_head_dim` took the
        ordinary-attention branch and was sized from `num_key_value_heads` and
        `head_dim` -- fields an MLA architecture still declares and no longer uses
        for KV -- so it reported 7.1x the KV per token with the rank sitting in a
        recognized field that was silently ignored.
        """
        return self.kv_lora_rank is not None and self.qk_rope_head_dim is not None

    @property
    def mla_head_size(self) -> int:
        """The latent's width per token per layer: content plus rope."""
        assert self.kv_lora_rank is not None and self.qk_rope_head_dim is not None
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def is_hybrid_attention(self) -> bool:
        """R6.7. Whether the layers do not all share one attention type."""
        return (
            self.sliding_window is not None and self.sliding_window_pattern is not None
        )

    def layer_is_full_attention(self, layer_index: int) -> bool:
        """Whether layer `layer_index` attends to the whole context.

        Gemma-3's convention, which is the one the published configs use: the
        *last* layer of each repeat is the full-attention one, so a pattern of 6 is
        five windowed layers followed by one full.
        """
        if not self.is_hybrid_attention:
            return self.sliding_window is None
        assert self.sliding_window_pattern is not None
        return (layer_index + 1) % self.sliding_window_pattern == 0

    @property
    def num_full_attention_layers(self) -> int:
        return sum(
            self.layer_is_full_attention(index)
            for index in range(self.num_hidden_layers)
        )

    @property
    def dtype_bytes(self) -> int:
        try:
            return DTYPE_BYTES[self.dtype]
        except KeyError:
            raise KeyError(
                f"model card {self.name!r} declares dtype {self.dtype!r}; known dtypes "
                f"are {sorted(DTYPE_BYTES)}"
            ) from None

    @property
    def embedding_parameters(self) -> int:
        params = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            params += self.vocab_size * self.hidden_size  # separate lm_head
        return params

    @property
    def attention_parameters_per_layer(self) -> int:
        q = self.hidden_size * self.num_attention_heads * self.head_dim
        kv = 2 * self.hidden_size * self.num_key_value_heads * self.head_dim
        o = self.num_attention_heads * self.head_dim * self.hidden_size
        return q + kv + o

    @property
    def mlp_parameters_per_layer(self) -> int:
        """Gated (SwiGLU) MLP: gate, up, and down projections."""
        one_expert = 3 * self.hidden_size * self.intermediate_size
        if not self.is_moe:
            return one_expert
        router = self.hidden_size * self.num_experts
        return self.num_experts * one_expert + router

    @property
    def router_parameters_per_layer(self) -> int:
        """The gate that scores experts. Zero for a dense model.

        Split out because expert parallelism divides the experts and *not* this: the
        router is a per-layer linear every rank computes in full, so folding it into
        the expert lump would shard something that is replicated.
        """
        return self.hidden_size * self.num_experts if self.is_moe else 0

    @property
    def expert_parameters_per_layer(self) -> int:
        """Every expert's weights for one layer, router excluded. R13.4."""
        return self.mlp_parameters_per_layer - self.router_parameters_per_layer

    @property
    def active_mlp_parameters_per_layer(self) -> int:
        """MLP parameters actually touched per token.

        For a dense model this equals `mlp_parameters_per_layer`. For MoE only the
        routed experts participate, which is why an MoE's compute cost is far below
        what its total parameter count suggests.
        """
        one_expert = 3 * self.hidden_size * self.intermediate_size
        if not self.is_moe:
            return one_expert
        return self.num_experts_per_token * one_expert

    @property
    def num_parameters(self) -> int:
        """Total parameters, derived from the declared dimensions."""
        if self.num_parameters_override is not None:
            return self.num_parameters_override
        return self._derived_parameters

    @property
    def _derived_parameters(self) -> int:
        # Two RMSNorms per layer plus a final norm. Tiny, but free to include.
        per_layer = (
            self.attention_parameters_per_layer
            + self.mlp_parameters_per_layer
            + 2 * self.hidden_size
        )
        return (
            self.embedding_parameters
            + self.num_hidden_layers * per_layer
            + self.hidden_size
        )

    @property
    def num_active_parameters(self) -> int:
        """Parameters touched per token -- the compute term's `P_active`."""
        per_layer = (
            self.attention_parameters_per_layer
            + self.active_mlp_parameters_per_layer
            + 2 * self.hidden_size
        )
        return (
            self.embedding_parameters
            + self.num_hidden_layers * per_layer
            + self.hidden_size
        )

    @property
    def parameter_count_discrepancy(self) -> float:
        """Relative gap between an overridden count and the derived one.

        Non-zero means the card's declared dimensions do not add up to the figure it
        reports. Surfaced so the disagreement is visible rather than silent.
        """
        if self.num_parameters_override is None:
            return 0.0
        derived = self._derived_parameters
        return (self.num_parameters_override - derived) / derived

    def kv_bytes_per_token(self, kv_dtype: str | None = None, tp_size: int = 1) -> int:
        """KV cache bytes for one token, on one device. R10.2.

        `2 *` covers key and value. KV heads are sharded across tensor-parallel ranks,
        which is why TP reduces per-device KV footprint as well as weight footprint.

        **R6.7. Neither holds under MLA**, and both differences move a capacity answer
        by a lot. There is one compressed latent rather than a key and a value, so the
        factor of two goes; and upstream's `get_num_kv_heads` returns 1 for MLA
        *before* it divides by `tensor_parallel_size`, so the latent is replicated on
        every rank rather than sharded. Scaling tensor parallelism buys weights and
        compute on a DeepSeek-class model and buys nothing at all on its KV cache --
        which is exactly the sort of thing a plan gets wrong from first principles.
        """
        dtype_bytes = DTYPE_BYTES[kv_dtype] if kv_dtype else self.dtype_bytes
        # R6.7. Attention layers only. A state-space hybrid's Mamba and MLP layers
        # cache no KV at all, so counting every hidden layer would charge Nemotron-H
        # for 52 layers of KV where it has 4.
        layers = self.num_attention_layers
        if self.use_mla:
            return self.mla_head_size * dtype_bytes * layers
        kv_heads_local = max(1, self.num_key_value_heads // tp_size)
        return 2 * kv_heads_local * self.head_dim * dtype_bytes * layers

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs, extra=extra)


@cache
def available_model_cards() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in MODELS_DIR.glob("*.json")))


@cache
def load_model_card(name: str) -> ModelCard:
    """Resolve a model name, Hugging Face id, or card path to a `ModelCard`."""
    candidate = Path(name)
    if candidate.suffix == ".json" and candidate.is_file():
        return ModelCard.from_dict(json.loads(candidate.read_text(encoding="utf-8")))

    resolved = ALIASES.get(name, name)
    path = MODELS_DIR / f"{resolved}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no model card for {name!r}. Bundled cards: "
            f"{list(available_model_cards())}. Known aliases: {sorted(ALIASES)}. "
            f"Add a card JSON under pvllm/sim/models/, or pass an explicit path via "
            f"the SimConfig `model_card` field. pretending-vllm will not invent an "
            f"architecture -- the memory and latency numbers would be fiction."
        )
    return ModelCard.from_dict(json.loads(path.read_text(encoding="utf-8")))
