"""One step's input, prepared.

Upstream: vllm/v1/worker/gpu/input_batch.py
Tier: B

F10, and the reason D6 chose the V2 runner: upstream computes all of this in numpy on
the CPU and only then mirrors it to device. That numpy half *is* the real logic, so
this is a near-verbatim port with the device copies removed -- not an approximation of
what the runner does, but the same computation.

The `_np` suffixes are upstream's and are kept: in upstream they distinguish the CPU
array from its device mirror, and preserving the names keeps a diff against upstream
mechanical even though there is no mirror here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InputBatch:
    """Everything one step needs, in batch order."""

    #: batch_idx -> req_id. The batch's ordering, which the output is indexed by.
    req_ids: list[str]
    num_reqs: int

    #: batch_idx -> slot index in `RequestState`.
    idx_mapping_np: np.ndarray

    #: `[num_reqs]`. Tokens scheduled per request this step.
    num_scheduled_tokens: np.ndarray
    num_tokens: int

    #: `[num_reqs + 1]`. Cumulative token offsets.
    query_start_loc_np: np.ndarray
    #: `[num_reqs]`. Context length per request after this step's tokens.
    seq_lens_np: np.ndarray

    #: `[num_tokens]`. Flattened token ids and their absolute positions.
    input_ids: np.ndarray
    positions: np.ndarray

    #: `[num_reqs]`. Prefill bookkeeping, kept separate from total length because a
    #: resumed request is re-fed output tokens as well as its prompt.
    prefill_len_np: np.ndarray
    num_computed_prefill_tokens_np: np.ndarray
    #: `[num_reqs]` bool: still prefilling, so no token is sampled for it this step.
    is_prefilling_np: np.ndarray

    #: Positions in the flattened batch that produce a sampled token -- the last
    #: token of each request that has finished prefilling.
    logits_indices: np.ndarray

    @property
    def num_sampled(self) -> int:
        return len(self.logits_indices)

    def __repr__(self) -> str:
        return (
            f"InputBatch(num_reqs={self.num_reqs}, num_tokens={self.num_tokens}, "
            f"num_sampled={self.num_sampled})"
        )


def sort_batch_req_ids(
    num_scheduled_tokens: dict[str, int], decode_query_len: int = 1
) -> list[str]:
    """Order the batch decodes-first.

    Upstream sorts so that uniform-length decodes are contiguous, which lets a
    captured graph cover them and lets attention split cleanly into a decode section
    and a prefill section. Ties break on request id so the order is total and the
    batch composition is reproducible -- without that, C1 would depend on dict
    iteration order.
    """
    return sorted(
        num_scheduled_tokens,
        key=lambda req_id: (num_scheduled_tokens[req_id] > decode_query_len, req_id),
    )
