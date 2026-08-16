"""Multimodal inputs, as the engine sees them. R18.

Upstream: vllm/multimodal/inputs.py (the `MultiModalFeatureSpec` half)
Tier: C

A multimodal request is a token sequence with *placeholders*: a run of reserved token
ids standing in for where an image's embeddings will go. The engine never sees pixels.
It sees how many tokens the image occupies, where they sit in the prompt, and a hash
identifying the content -- which is exactly the information scheduling and caching
need, and exactly what a simulator can carry faithfully.

The hash is the load-bearing field. It decides two things:

* whether the encoder has to run at all, or the embeddings are already cached from
  another request (R18.1);
* whether two requests with the same image can share KV blocks, because it joins the
  prefix cache extra keys (R6.3). Two prompts that differ only in an image must not
  share, and two that share an image and a prefix must.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Token id reserved for placeholders. It sits *below* the mock tokenizer's byte
#: range (ids 256..511), in the block of special ids no text encodes to, so a
#: placeholder can never be confused for content. It is not derived from the
#: tokenizer: a real deployment's placeholder id comes from the model's config, and
#: this one is fixed so a trace is readable. Detokenizing it yields nothing --
#: `MockTokenizer.decode` skips ids below its byte offset.
PLACEHOLDER_TOKEN_ID = 4

#: The largest number of embeddings one multimodal item can produce here. Upstream
#: derives this per model from the processor and floors both encoder budgets with it
#: (`compute_mm_encoder_budget`); here every item is an image of fixed size, so the
#: bound is a constant. It is load-bearing rather than informational: an item larger
#: than the encoder cache can never be scheduled, and the scheduler's response to
#: "cannot be scheduled" is to try again next step, forever.
MAX_TOKENS_PER_MM_ITEM = 256


@dataclass(frozen=True)
class MultiModalFeatureSpec:
    """One multimodal item inside a request.

    Args:
        identifier: Content hash. Two items with the same identifier are the same
            image as far as every cache in the engine is concerned.
        modality: `image`, `audio`, or `video`. Carried so a cost model can charge
            different rates; only `image` is modeled.
        position: Index of the first placeholder token in the prompt.
        length: How many prompt tokens the item occupies.
        num_embeds: Encoder output embeddings, which is what the encoder cache
            budget is measured in. Usually equal to `length`.
    """

    identifier: str
    modality: str
    position: int
    length: int
    num_embeds: int

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(
                f"a multimodal item must occupy at least one token, got {self.length}"
            )
        if self.position < 0:
            raise ValueError(f"position must be non-negative, got {self.position}")


def content_hash(payload: str | bytes, modality: str = "image") -> str:
    """A stable identifier for multimodal content.

    sha256 of the bytes plus the modality, so the same bytes offered as an image and
    as a video frame are distinct entries -- the encoder produces different
    embeddings for them, and a cache that conflated the two would serve the wrong
    ones.

    Deterministic across processes, unlike `hash()`: the identifier reaches the
    prefix cache, and a per-process salt there would make cache behaviour
    irreproducible (B4).
    """
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    digest = hashlib.sha256(modality.encode("utf-8") + b"\x00" + raw)
    return digest.hexdigest()[:32]
