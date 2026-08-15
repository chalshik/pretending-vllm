"""Stop detection.

Upstream: vllm/v1/core/sched/utils.py
Tier: A

R11.5 requires stop-condition parity, and the *order* of these checks is the
specification. `min_tokens` is checked first and returns early, so a request below its
minimum cannot stop for any reason -- including EOS. Move that check below the EOS test
and `min_tokens` silently stops working exactly when the model emits EOS early, which
is the only time it matters.

`ignore_eos` is not tested here. The processor leaves `eos_token_id` unset when it is
requested, so the policy is decided once rather than at every stop check. Upstream does
the same.

Stop *strings* are not here: they need incremental detokenization and belong to the
output processor (R11.6), matching upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pvllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from pvllm.v1.request import Request


def check_stop(request: Request, max_model_len: int) -> bool:
    """Whether a request has finished, setting its status if so.

    Called after every appended token and before the output is built, so a stopped
    request never emits a token past its stop condition.
    """
    sampling_params = request.sampling_params
    assert sampling_params is not None

    # First, and returning early: below min_tokens, nothing stops the request.
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]

    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        # The OpenAI surface reports which token stopped it.
        request.stop_reason = last_token_id
        return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
