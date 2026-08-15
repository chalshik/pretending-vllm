"""Persistent per-request state held by the worker.

Upstream: vllm/v1/worker/gpu/states.py
Tier: B

R7.3: the worker keeps this across steps and patches it from the scheduler's diff,
rather than rebuilding it each step. That is not an optimization detail worth
preserving for its own sake -- it is what makes the *cost* of state churn visible in
the design, and what the scheduler's `CachedRequestData` shape exists to serve.

A request occupies a **slot index** for its whole life, and the batch is addressed by
slot rather than by id. Slots are recycled through `free_indices`. Upstream backs
`all_token_ids` with a `max_num_reqs x max_model_len` tensor in unified memory,
because at 1024 requests and 128k context that array is gigabytes; here it is numpy,
allocated lazily per slot for the same reason -- a laptop must not need gigabytes to
simulate a device that does.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RequestState:
    """The worker's memory of every request it is currently serving."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        # Popped from the end, so a freed slot is reused immediately. Recycling
        # promptly keeps the live slot indices dense, which keeps the per-step
        # gather over them small.
        self.free_indices: list[int] = list(range(max_num_reqs))

        # Slot-indexed arrays. Upstream distinguishes prompt_len from prefill_len:
        # a request resumed after preemption is re-fed its prompt *and* the output
        # tokens it already produced, so prefill_len can exceed prompt_len. Features
        # that must treat prompt and output separately -- prompt logprobs, frequency
        # penalties -- depend on the distinction.
        self.prompt_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.prefill_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.total_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_prefill_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.max_seq_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_output_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        #: R18.1. request_id -> its multimodal items, for the encoder cost. Keyed by
        #: id rather than slot because it is read by request id from the scheduler
        #: output, which does not know slot indices.
        self.mm_features: dict[str, list[Any]] = {}

        # Allocated per slot on demand rather than as one dense 2D array: at
        # max_num_reqs=1024 and max_model_len=128k that array is 512 MiB of int32,
        # which a simulator has no reason to hold.
        self.all_token_ids: dict[int, list[int]] = {}

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> int:
        """Take a slot for a newly scheduled request. Returns its index."""
        if not self.free_indices:
            raise RuntimeError(
                f"no free request slots: the worker holds {self.num_reqs} requests "
                f"and max_num_seqs is {self.max_num_reqs}. The scheduler should not "
                f"have admitted another (R5.3)."
            )
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"request {req_id} has prefill_len {prefill_len} < prompt_len {prompt_len}"
        )

        self.prompt_len[req_idx] = prompt_len
        self.prefill_len[req_idx] = prefill_len
        self.total_len[req_idx] = prefill_len
        self.num_computed_tokens[req_idx] = num_computed_tokens
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.num_output_tokens[req_idx] = max(0, prefill_len - prompt_len)
        self.all_token_ids[req_idx] = list(all_token_ids)
        return req_idx

    def append_tokens(self, req_idx: int, token_ids: list[int]) -> None:
        """Record tokens the scheduler says arrived since the last step."""
        if not token_ids:
            return
        self.all_token_ids[req_idx].extend(token_ids)
        self.total_len[req_idx] += len(token_ids)
        self.num_output_tokens[req_idx] += len(token_ids)

    def set_num_computed_tokens(self, req_idx: int, num_computed: int) -> None:
        self.num_computed_tokens[req_idx] = num_computed
        self.num_computed_prefill_tokens[req_idx] = min(
            num_computed, int(self.prefill_len[req_idx])
        )

    def remove_request(self, req_id: str) -> int | None:
        """Release a request's slot. Returns the freed index, or None if unknown."""
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return None
        self.index_to_req_id.pop(req_idx, None)
        self.all_token_ids.pop(req_idx, None)
        self.free_indices.append(req_idx)
        return req_idx

    def is_prefilling(self, req_idx: int) -> bool:
        return bool(
            self.num_computed_prefill_tokens[req_idx] < self.prefill_len[req_idx]
        )

    def __repr__(self) -> str:
        return (
            f"RequestState(num_reqs={self.num_reqs}/{self.max_num_reqs}, "
            f"free_slots={len(self.free_indices)})"
        )
