"""Seeded randomness. R19.2.

Upstream: (none -- simulator)
Tier: D

One global seed reproduces an entire run: arrival times, output lengths, sampled
tokens, and cost-model jitter.

The load-bearing property is that a request's RNG is derived from
``(seed, request_id)`` rather than drawn from a shared stream. A shared stream would
make every request's tokens depend on how requests happened to interleave, so adding
one request to a workload would change the output of all the others -- and any test
that depends on a specific request's output would be coupled to the whole schedule.
With per-request derivation, request ``abc`` produces the same tokens whether it ran
alone or seventeenth in a batch of two hundred.

Streams are derived with BLAKE2b rather than Python's ``hash()``, which is salted per
process and would make runs irreproducible across interpreter restarts.

This module and ``pvllm.sim.clock`` are the only places allowed to introduce
nondeterminism, and both are seeded. ``tests/unit/test_purity.py`` fails the build if
anything outside ``pvllm/sim/`` imports ``random`` or touches ``numpy.random``.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: Namespaces for engine-level streams that are not scoped to a request.
GLOBAL_STREAMS = ("jitter", "arrival", "workload")


def _derive_entropy(seed: int, namespace: str, key: str) -> int:
    """Derive a stable 128-bit seed from ``(seed, namespace, key)``.

    Stable across processes, platforms, and interpreter versions -- unlike
    ``hash()``, which is salted by ``PYTHONHASHSEED``.
    """
    payload = f"{seed}\x00{namespace}\x00{key}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


class RngFactory:
    """Derives independent, reproducible generators from one run seed.

    Args:
        seed: The run seed. Every generator this factory hands out is a pure
            function of this value plus the stream's name.
    """

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._request_cache: dict[str, np.random.Generator] = {}
        self._stream_cache: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        return self._seed

    def for_request(self, request_id: str) -> np.random.Generator:
        """The generator for one request's output length, tokens, and sampling.

        Cached, so repeated calls within a request continue the same stream rather
        than restarting it -- a request that draws 40 tokens must not draw the same
        token 40 times.
        """
        generator = self._request_cache.get(request_id)
        if generator is None:
            entropy = _derive_entropy(self._seed, "request", request_id)
            generator = np.random.default_rng(np.random.SeedSequence(entropy))
            self._request_cache[request_id] = generator
        return generator

    def for_position(
        self, request_id: str, position: int, seed: int | None = None
    ) -> np.random.Generator:
        """A generator for one request's token at one output position. R19.2.

        `seed` is the *request's* own `SamplingParams.seed` when it set one, and the
        engine seed otherwise. A client asking for a specific seed is asking for a
        reproducible completion, and honouring it here is the only place that can
        mean anything -- the branch in `ParentRequest._child_params` that offsets a
        seeded parent's children was, until this existed, an elaborate no-op.

        Derived from `(seed, request_id, position)` rather than drawn from the
        request's stream, so `for_position(r, 7)` gives the same answer however many
        times it is asked and whatever was asked before it. That idempotence is not a
        nicety:

        * **Speculative decoding needs it to be lossless.** Drafting samples
          positions ahead of where the request actually is. Off a shared stream those
          draws would consume the entropy the real tokens were going to use, so a
          speculated run would produce different text from an unspeculated one --
          and the whole claim of speculation is that it produces the same text
          faster.
        * **Preemption-recompute equivalence falls out for free** (R21.1). A
          recomputed request re-samples positions it already sampled; off a shared
          stream it would get different tokens the second time.

        Uncached, unlike `for_request`: the whole point is that the answer depends on
        the key rather than on history, so there is no stream state to keep.
        """
        # A client-supplied seed replaces the request id as well as the engine seed:
        # `seed` means "this completion", so two requests carrying the same seed and
        # the same parameters must produce the same text. Keeping the id in the key
        # would make the field mean "some completion, reproducibly", which is not
        # what any client sets it for. Without one, the id is what separates two
        # concurrent requests.
        entropy = (
            _derive_entropy(self._seed, "position", f"{request_id}:{position}")
            if seed is None
            else _derive_entropy(seed, "seeded-position", str(position))
        )
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def for_constraint(self, request_id: str) -> np.random.Generator:
        """The generator a constrained request's output is drawn from. R15.

        Derived from `(seed, request_id)` and *uncached*, so it is reproducible from
        the key alone. That is what lets the grammar backend prove a schema is
        satisfiable using the very generator generation will use: probe and
        generation produce the same string, so a compile that passes cannot be
        followed by a generation that fails.

        Off the request's own stream deliberately. That stream is stateful, so a
        probe run in the engine core and the generation run in the worker would
        draw from different positions -- and a schema whose conformance depends on
        the draw would compile on one and fail on the other, taking the engine down
        rather than the request.
        """
        entropy = _derive_entropy(self._seed, "constraint", request_id)
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def stream(self, name: str) -> np.random.Generator:
        """An engine-level stream that is not scoped to a request.

        Used for cost-model jitter and workload arrival processes. These are
        genuinely global -- jitter on step 900 depends on there having been 899 steps
        before it -- which is fine, because the step sequence is itself deterministic
        given the same workload and config.
        """
        generator = self._stream_cache.get(name)
        if generator is None:
            entropy = _derive_entropy(self._seed, "stream", name)
            generator = np.random.default_rng(np.random.SeedSequence(entropy))
            self._stream_cache[name] = generator
        return generator

    def forget_request(self, request_id: str) -> None:
        """Drop a finished request's generator.

        Called when a request leaves the engine, so a long-running server does not
        accumulate one generator per request ever seen. Safe because re-deriving
        yields an identical generator; only stream *position* is lost, and a finished
        request has no position left to lose.
        """
        self._request_cache.pop(request_id, None)

    def __repr__(self) -> str:
        return (
            f"RngFactory(seed={self._seed}, "
            f"live_requests={len(self._request_cache)}, "
            f"streams={sorted(self._stream_cache)})"
        )
