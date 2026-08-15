"""The simulated attention backend: metadata, no math.

Upstream: vllm/v1/attention/backend.py (counterpart, not a port)
Tier: D

R8.2: the metadata is **real**. `query_start_loc`, `seq_lens`, `slot_mapping`, and the
block table are computed exactly as a real backend computes them, because they are the
cost model's inputs -- which means a bug in them shows up as a wrong latency or a
failed slot-mapping assertion rather than being silently absorbed. That is the whole
reason to build them rather than fake them: metadata nobody reads can be wrong forever.

No attention is computed. `SimModel` produces tokens; this describes where their KV
would live.

Upstream's `AttentionBackend` is heavily torch-typed (`torch.dtype`, `torch.Tensor`,
CUDA graph support flags). This declares the same *shape* over numpy, which is why the
header says counterpart rather than port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class SimAttentionMetadata:
    """What a real attention kernel would need. R8.2.

    Field names match upstream's `CommonAttentionMetadata` so the mapping between the
    two is mechanical.
    """

    #: `[num_reqs + 1]`. Cumulative token counts; request i owns
    #: `[query_start_loc[i], query_start_loc[i + 1])`.
    query_start_loc: np.ndarray
    #: `[num_reqs]`. Total context per request, including tokens computed earlier.
    seq_lens: np.ndarray
    #: `[num_tokens]`. Where each token's KV is written (R8.3).
    slot_mapping: np.ndarray
    #: `[num_reqs, max_blocks_per_req]`.
    block_table: np.ndarray
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int

    #: Prefill and decode counts, split for the cost model. A request is decoding
    #: when it contributes exactly one token this step.
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    #: R5.9. Blocks shared by every request, which a real backend would use to run
    #: cascade attention over the shared prefix once instead of per request.
    num_common_prefix_blocks: int = 0

    @property
    def is_mixed_batch(self) -> bool:
        """Whether this step mixes prefill and decode.

        Relevant to R8.4: a mixed batch cannot use a captured graph, because the
        captured shapes are uniform-decode shapes.
        """
        return self.num_prefills > 0 and self.num_decodes > 0


class SimAttentionBackend:
    """The backend `SimPlatform.get_attn_backend_cls` resolves to."""

    #: Upstream advertises kernel constraints here; the simulator has no kernels, so
    #: it accepts whatever the model card declares.
    supported_dtypes: ClassVar[list[str]] = ["bfloat16", "float16", "float32"]
    supported_head_sizes: ClassVar[list[int]] = []

    @staticmethod
    def get_name() -> str:
        return "SIM_ATTN"

    @staticmethod
    def get_metadata_cls() -> type[SimAttentionMetadata]:
        return SimAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
    ) -> tuple[int, ...]:
        """The shape a real backend would allocate.

        Reported, never allocated: an 80 GiB KV cache must not need 80 GiB of host
        RAM. The memory ledger tracks the claim (R10.1); this describes its layout.
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return True

    @staticmethod
    def is_mla() -> bool:
        return False

    @staticmethod
    def supports_sliding_window() -> bool:
        # Sliding window needs a second KV cache group (R6.7), which lands in M4.
        return False
