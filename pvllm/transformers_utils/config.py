"""Model metadata loading.

Upstream: vllm/transformers_utils/config.py
Tier: C

Upstream fetches a `PretrainedConfig` from the Hugging Face hub or from disk. There is
no checkpoint here, so the same metadata comes from a model card (section 8).

`ModelCard`'s field names deliberately match the Hugging Face config attributes vLLM
actually reads -- `num_hidden_layers`, `hidden_size`, `num_attention_heads`,
`num_key_value_heads`, `head_dim`, `intermediate_size`, `vocab_size`,
`max_position_embeddings`, `tie_word_embeddings` -- so a card substitutes for a
`PretrainedConfig` at every call site that matters, and `ModelConfig.hf_config` reads
the same way it does upstream.

This module is the seam where "the model is simulated" is allowed to show. It sits
outside the boundary-enforced subtrees (`v1/core`, `v1/engine`, `entrypoints`) for
exactly the reason its upstream counterpart does: loading model metadata is a
prerequisite of building a config, not part of the control plane.
"""

from __future__ import annotations

from pvllm.sim.model_db import ModelCard, load_model_card


def get_config(model: str, model_card: str | None = None) -> ModelCard:
    """Resolve a model name to its architecture.

    Args:
        model: Model name or Hugging Face id, as a client would send it.
        model_card: Explicit card name or path, overriding the lookup. This is the
            escape hatch for modeling a checkpoint that has no bundled card.
    """
    return load_model_card(model_card or model)


def get_hf_text_config(config: ModelCard) -> ModelCard:
    """The text-model sub-config.

    Upstream unwraps multimodal wrappers here. Nothing is multimodal until M4, so the
    card is its own text config.
    """
    return config
