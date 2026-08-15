"""The workloads the fidelity contract is asserted on. R21.3.

Upstream: (none -- pvllm addition)
Tier: B

Four workloads, each chosen because it is the smallest thing that makes one
conformance class *fail loudly* when the corresponding logic drifts. A workload that
happens to exercise a code path is not the same as one whose recording changes when
that path changes -- the second is what a regression suite needs.

They are deliberately tiny (`tiny-test` on `tiny-2gb`, single-digit requests). R21.5
budgets the whole suite at 30 seconds; a conformance run that took minutes would get
skipped locally and only fail in CI, which is where regressions become archaeology.

Every workload pins its config completely -- block size, budget, device card, seed --
because a golden recorded under a different config is not comparable and `compare`
refuses to pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Shared by the prefix-cache workload. Long enough to fill several blocks at the
#: workload's block size, because a shared prefix shorter than one block produces no
#: cache hit at all and the workload would silently assert nothing.
SHARED_PREFIX = (
    "You are a careful assistant. Answer using only the provided context. "
    "Cite the source of every claim. Do not speculate beyond the context. "
)


@dataclass(frozen=True)
class Workload:
    """One recorded scenario."""

    name: str
    #: What this workload would catch if it broke. Printed on failure, so a diff
    #: arrives with the reason the workload exists attached.
    pins: str
    prompts: list[str]
    max_tokens: int
    config: dict[str, Any] = field(default_factory=dict)

    def engine_kwargs(self) -> dict[str, Any]:
        """Engine args for this workload, with the invariant parts filled in."""
        return {
            "max_model_len": 256,
            "device_card": "tiny-2gb",
            "seed": 0,
            "disable_log_stats": True,
            **self.config,
        }


WORKLOADS: dict[str, Workload] = {
    "mixed-lengths": Workload(
        name="mixed-lengths",
        pins=(
            "C1. Prompts of different lengths arriving together, with more requests "
            "than max_num_seqs, so admission order, the token budget split, and the "
            "step count to drain are all exercised at once. A change to the "
            "three-phase schedule() -- or to when a request is admitted -- moves the "
            "step sequence here."
        ),
        prompts=[
            "short",
            "a somewhat longer prompt with more tokens in it",
            "tiny",
            "another prompt of middling length to vary the batch shape",
            "the longest prompt in this workload, written to span several blocks so "
            "that its prefill has to be split across more than one engine step",
            "brief",
        ],
        max_tokens=12,
        config={
            "block_size": 8,
            "max_num_batched_tokens": 24,
            "max_num_seqs": 3,
            # Roomy on purpose. Preemption has its own workload; if this one's step
            # sequence changes, the cause should be admission or the budget split,
            # not a victim choice bleeding in from elsewhere.
            "num_gpu_blocks_override": 96,
        },
    ),
    "shared-prefix": Workload(
        name="shared-prefix",
        pins=(
            "C3. Every request shares a multi-block prefix, so the hit rate is "
            "nonzero and the resident hash set is stable. Catches a change to the "
            "parent-chained hash, to the extra keys folded in, or to the "
            "one-token-recompute rule -- each of which moves the hit rate without "
            "necessarily breaking anything else."
        ),
        prompts=[SHARED_PREFIX + f"Question {i}: what happened?" for i in range(5)],
        max_tokens=8,
        config={
            "block_size": 8,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 4,
            "num_gpu_blocks_override": 64,
            "enable_prefix_caching": True,
        },
    ),
    "preemption": Workload(
        name="preemption",
        pins=(
            "C4. A block budget far too small for the concurrency, so the scheduler "
            "must preempt to make progress. Pins the victim choice (last in the "
            "running queue under FCFS), the preemption count, and -- because outputs "
            "are recorded too -- that a preempted request recomputes to the same "
            "tokens it would have produced uninterrupted (R21.1)."
        ),
        prompts=[
            "a prompt that needs several blocks of key value cache to hold",
            "a second prompt of similar length competing for the same blocks",
            "a third prompt, which is where the pressure starts to show",
            "a fourth prompt that cannot possibly fit alongside the others",
        ],
        max_tokens=24,
        config={
            "block_size": 8,
            "max_num_batched_tokens": 48,
            "max_num_seqs": 4,
            # Deliberately starved: each request needs ~13 blocks, so two fit and the
            # rest must wait for someone to be evicted. Starving it harder would make
            # the engine thrash, and a thrashing trace pins the thrash rather than
            # the victim policy.
            "num_gpu_blocks_override": 32,
            "enable_prefix_caching": False,
        },
    ),
    "chunked-prefill": Workload(
        name="chunked-prefill",
        pins=(
            "C1. One prompt far longer than the token budget, so its prefill is "
            "split across many steps while a short request decodes alongside it. "
            "Pins the chunk sizes and the prefill/decode interleaving -- the part of "
            "the budget arithmetic that a plausible-looking off-by-one leaves intact "
            "in every other workload."
        ),
        # ~100 tokens through the mock tokenizer against a 16-token budget, so the
        # long prompt takes seven prefill steps -- enough for the interleaving to be
        # a pattern rather than a coincidence.
        prompts=[
            " ".join(f"token{i}" for i in range(14)),
            "short companion",
        ],
        max_tokens=10,
        config={
            "block_size": 8,
            "max_num_batched_tokens": 16,
            "max_num_seqs": 4,
            "num_gpu_blocks_override": 64,
            "enable_chunked_prefill": True,
        },
    ),
}
