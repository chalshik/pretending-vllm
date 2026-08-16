"""What comes back across the simulation boundary.

Upstream: vllm/v1/outputs.py
Tier: B

`ModelRunnerOutput` is the return half of the one interface that crosses the boundary
(section 4). Upstream carries logprobs as torch tensors; here they are plain lists,
because the values are synthetic and only the schema and shape are contractual (NG3).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LogprobsLists:
    """Top-k logprobs for a batch of positions, as parallel lists.

    Values are synthetic. What is real is the shape: `len(logprob_token_ids) ==
    len(logprobs) == len(sampled_token_ranks)`, and each inner list has the requested
    k. A client that indexes into these gets what it would get from real vLLM.
    """

    logprob_token_ids: list[list[int]]
    logprobs: list[list[float]]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int) -> LogprobsLists:
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


@dataclass
class SamplerOutput:
    """The sampler's result for one step."""

    #: One list per request. More than one entry only under speculative decoding.
    sampled_token_ids: list[list[int]]
    logprobs_lists: LogprobsLists | None = None


@dataclass
class ModelRunnerOutput:
    """The return value of `execute_model`. The simulation boundary's output half."""

    #: Batch order. `req_id_to_index[req_ids[i]] == i`.
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    #: One list per request, indexed by `req_id_to_index`.
    sampled_token_ids: list[list[int]]
    logprobs: LogprobsLists | None = None
    prompt_logprobs_dict: dict[str, LogprobsLists | None] = field(default_factory=dict)
    #: Draft tokens proposed for the next step (R14).
    spec_token_ids: list[list[int]] | None = None
    #: KV connector transfer completions (R17).
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    #: R2.2. One vector per request that pooled this step, `None` for the rest.
    #: Indexed by `req_id_to_index`, like `sampled_token_ids`.
    pooler_output: list[list[float] | None] | None = None
    #: How long the modeled step took, in seconds. Feeds the clock and the metrics,
    #: and is labeled `modeled` wherever it surfaces (R9.5).
    modeled_duration: float = 0.0

    @classmethod
    def make_empty(cls) -> ModelRunnerOutput:
        return cls(req_ids=[], req_id_to_index={}, sampled_token_ids=[])


#: Returned when a step scheduled no work. A shared instance, matching upstream, so
#: the common empty-step path allocates nothing.
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput.make_empty()
