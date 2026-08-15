"""Block tables and slot mapping.

Upstream: vllm/v1/worker/gpu/block_table.py
Tier: B

R7.4: block tables are materialized as integer arrays per request, sized
`max_model_len / block_size`, so their metadata memory is accounted for and the
indexing logic is real rather than implied.

**Slot mapping is where the KV manager gets checked (R8.3).** A slot is
`block_id * block_size + offset_within_block`; the mapping says where each token's KV
is written. Validating that every written slot lies inside a block the request
actually owns turns the simulator into a correctness oracle: a KV manager bug that
would silently corrupt another request's cache on real hardware -- and surface later
as garbled output nobody can trace -- raises here, at the step that caused it.

Upstream computes this in a Triton kernel. The arithmetic is identical; only the
execution differs.
"""

from __future__ import annotations

import numpy as np

from pvllm import envs

#: Marks a padded position with no KV slot. Upstream uses the same sentinel.
PAD_SLOT_ID = -1


class BlockTables:
    """Per-request block tables, one set per KV cache group."""

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        enable_caching: bool = False,
    ) -> None:
        self.enable_caching = enable_caching
        self.block_sizes = block_sizes
        self.num_kv_cache_groups = len(block_sizes)
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens

        # R7.4: sized for the worst case up front, exactly as upstream allocates it,
        # so the metadata cost of a large max_model_len is visible rather than
        # appearing only under load.
        self.max_blocks_per_req = [
            (max_model_len + block_size - 1) // block_size for block_size in block_sizes
        ]
        self.block_tables = [
            np.zeros((max_num_reqs, max_blocks), dtype=np.int32)
            for max_blocks in self.max_blocks_per_req
        ]
        #: How many entries of each row are valid.
        self.num_blocks = np.zeros(
            (self.num_kv_cache_groups, max_num_reqs), dtype=np.int32
        )

        self._debug_invariants = envs.PVLLM_DEBUG_INVARIANTS

    def set_block_ids(self, req_idx: int, block_ids: tuple[list[int], ...]) -> None:
        """Install a request's complete block table. Used when it is first scheduled."""
        for group_id, ids in enumerate(block_ids):
            if len(ids) > self.max_blocks_per_req[group_id]:
                raise ValueError(
                    f"request at slot {req_idx} was given {len(ids)} blocks in group "
                    f"{group_id}, but max_model_len allows only "
                    f"{self.max_blocks_per_req[group_id]}"
                )
            self.block_tables[group_id][req_idx, : len(ids)] = ids
            self.num_blocks[group_id, req_idx] = len(ids)

    def append_block_ids(self, req_idx: int, block_ids: tuple[list[int], ...]) -> None:
        """Extend a request's block table with newly allocated blocks.

        The incremental path (R7.3): a decode step usually adds nothing, and adds one
        block when the sequence crosses a block boundary.
        """
        for group_id, ids in enumerate(block_ids):
            if not ids:
                continue
            start = int(self.num_blocks[group_id, req_idx])
            end = start + len(ids)
            if end > self.max_blocks_per_req[group_id]:
                raise ValueError(
                    f"request at slot {req_idx} would hold {end} blocks in group "
                    f"{group_id}, exceeding the {self.max_blocks_per_req[group_id]} "
                    f"that max_model_len allows"
                )
            self.block_tables[group_id][req_idx, start:end] = ids
            self.num_blocks[group_id, req_idx] = end

    def clear(self, req_idx: int) -> None:
        """Forget a finished request's table."""
        self.num_blocks[:, req_idx] = 0

    def get_block_ids(self, req_idx: int, group_id: int = 0) -> np.ndarray:
        count = int(self.num_blocks[group_id, req_idx])
        return self.block_tables[group_id][req_idx, :count]

    def gather(self, req_indices: np.ndarray, group_id: int = 0) -> np.ndarray:
        """The block tables for a batch, in batch order.

        This is what the attention metadata carries, and what a real kernel would
        index to find each token's KV.
        """
        gathered: np.ndarray = self.block_tables[group_id][req_indices]
        return gathered

    def compute_slot_mapping(
        self,
        req_indices: np.ndarray,
        positions: np.ndarray,
        query_start_loc: np.ndarray,
        group_id: int = 0,
    ) -> np.ndarray:
        """Where each scheduled token's KV is written.

        `slot = block_id * block_size + (position % block_size)`.

        Args:
            req_indices: Slot index per request, in batch order.
            positions: Absolute position of every scheduled token, concatenated.
            query_start_loc: Where each request's tokens begin in `positions`.
        """
        block_size = self.block_sizes[group_id]
        table = self.block_tables[group_id]

        slot_mapping = np.full(len(positions), PAD_SLOT_ID, dtype=np.int64)
        for batch_idx, req_idx in enumerate(req_indices):
            start = int(query_start_loc[batch_idx])
            end = int(query_start_loc[batch_idx + 1])
            if start == end:
                continue
            token_positions = positions[start:end]
            block_offsets = token_positions // block_size

            # Checked before indexing, not after: numpy would otherwise raise a bare
            # IndexError that says nothing about which request over-ran its blocks.
            num_owned = int(self.num_blocks[group_id, req_idx])
            highest = int(block_offsets.max())
            if highest >= num_owned:
                raise AssertionError(
                    f"slot mapping runs past the block table: request at slot "
                    f"{req_idx} writes position {int(token_positions.max())}, which "
                    f"needs block index {highest}, but it owns only {num_owned} "
                    f"blocks. The scheduler scheduled a token the KV manager did not "
                    f"allocate for (R8.3)."
                )

            within_block = token_positions % block_size
            block_ids = table[req_idx, block_offsets]
            slot_mapping[start:end] = block_ids * block_size + within_block

        if self._debug_invariants:
            self._validate_slot_mapping(
                slot_mapping, req_indices, query_start_loc, group_id
            )
        return slot_mapping

    def _validate_slot_mapping(
        self,
        slot_mapping: np.ndarray,
        req_indices: np.ndarray,
        query_start_loc: np.ndarray,
        group_id: int,
    ) -> None:
        """R8.3, R21.1. Raises rather than warns, deliberately.

        On real hardware a slot landing in another request's block silently corrupts
        their KV cache, and the symptom -- one request emitting another's content,
        much later -- is close to untraceable. Here it fails at the step that caused
        it, which is the value of running a KV manager against a simulator at all.

        Two distinct checks, catching two distinct bugs:

        * **No slot written twice in one step.** Two requests scheduled together
          mapping to the same slot.
        * **No block held by two live requests.** The real corruption mode: the pool
          handed one block to two requests, which may not both be scheduled in the
          same step, so the per-step check alone would miss it until they collided.

        Note what is *not* checkable here: whether a block table matches the KV
        manager's own record. The table is the worker's copy of that record, so
        comparing it against itself would prove nothing. Cross-request uniqueness is
        the strongest statement the worker can make alone, and it happens to be the
        one that catches the bug that matters.
        """
        # Two requests writing the same slot in one step is a bug in every
        # configuration: even a shared prefix block is *read* by the second request,
        # never rewritten, so its positions are not scheduled.
        seen_slots: dict[int, int] = {}
        for batch_idx, req_idx in enumerate(req_indices):
            start = int(query_start_loc[batch_idx])
            end = int(query_start_loc[batch_idx + 1])
            for offset in range(start, end):
                slot = int(slot_mapping[offset])
                if slot == PAD_SLOT_ID:
                    continue
                if slot in seen_slots:
                    raise AssertionError(
                        f"slot {slot} written twice in one step, by requests at slots "
                        f"{seen_slots[slot]} and {int(req_idx)}. The block pool handed "
                        f"the same block to both (R21.1)."
                    )
                seen_slots[slot] = int(req_idx)

        # Cross-request block ownership is only exclusive when prefix caching is
        # off. With caching on, two requests sharing a prefix legitimately hold the
        # same blocks -- that is the entire mechanism -- and the worker cannot tell
        # legitimate sharing from double-allocation, because reference counts live
        # in the pool, not in the block table. The pool's own invariants
        # (BlockPool._check_invariants) cover that case instead.
        if not self.enable_caching:
            self.validate_block_ownership(group_id)

    def validate_block_ownership(self, group_id: int = 0) -> None:
        """No physical block may appear in two live requests' tables. R8.3.

        Scans every occupied row rather than only the scheduled ones, because the
        pool can hand a block to two requests that are never scheduled together --
        and the resulting corruption would then surface only much later, on the step
        they finally collide.

        **Only sound with prefix caching disabled.** Sharing a cached prefix means
        two requests holding the same blocks on purpose. Callers should not invoke
        this when caching is on; `compute_slot_mapping` does not.
        """
        owner_of_block: dict[int, int] = {}
        for req_idx in range(self.max_num_reqs):
            count = int(self.num_blocks[group_id, req_idx])
            if count == 0:
                continue
            for block_id in self.block_tables[group_id][req_idx, :count].tolist():
                previous = owner_of_block.get(int(block_id))
                if previous is not None and previous != req_idx:
                    raise AssertionError(
                        f"block {int(block_id)} is held by two live requests, at "
                        f"slots {previous} and {req_idx}. The block pool allocated it "
                        f"twice; on real hardware each would overwrite the other's KV "
                        f"cache (R8.3)."
                    )
                owner_of_block[int(block_id)] = req_idx

    def __repr__(self) -> str:
        return (
            f"BlockTables(groups={self.num_kv_cache_groups}, "
            f"block_sizes={self.block_sizes}, "
            f"max_blocks_per_req={self.max_blocks_per_req})"
        )
