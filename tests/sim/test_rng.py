"""Seeded randomness. R19.2."""

from __future__ import annotations

from pvllm.sim.rng import RngFactory


def _draw(generator, n=6):
    return generator.integers(0, 1_000_000, n).tolist()


def test_same_seed_reproduces_the_run():
    assert _draw(RngFactory(7).for_request("r1")) == _draw(
        RngFactory(7).for_request("r1")
    )


def test_different_seeds_diverge():
    assert _draw(RngFactory(7).for_request("r1")) != _draw(
        RngFactory(8).for_request("r1")
    )


def test_request_streams_are_independent_of_interleaving():
    """The property the whole design turns on.

    A request must produce the same tokens whether it ran alone or seventeenth in a
    batch of two hundred. Drawing from a shared stream would couple every request's
    output to the arrival schedule.
    """
    solo = RngFactory(42)
    expected = _draw(solo.for_request("target"))

    interleaved = RngFactory(42)
    for other in ("a", "b", "c", "d"):
        _draw(interleaved.for_request(other))
    assert _draw(interleaved.for_request("target")) == expected


def test_repeated_access_continues_a_stream_rather_than_restarting_it():
    """A request drawing 40 tokens must not draw the same token 40 times."""
    factory = RngFactory(1)
    first = _draw(factory.for_request("r"), 3)
    second = _draw(factory.for_request("r"), 3)
    assert first != second


def test_named_streams_are_independent_of_each_other():
    factory = RngFactory(99)
    assert _draw(factory.stream("jitter")) != _draw(factory.stream("arrival"))


def test_streams_are_independent_of_request_generators():
    """Drawing jitter must not perturb any request's token stream."""
    quiet = RngFactory(5)
    expected = _draw(quiet.for_request("r"))

    noisy = RngFactory(5)
    _draw(noisy.stream("jitter"), 50)
    assert _draw(noisy.for_request("r")) == expected


def test_forgetting_a_request_is_safe_to_rederive():
    factory = RngFactory(3)
    expected = _draw(factory.for_request("r"))
    factory.forget_request("r")
    assert _draw(factory.for_request("r")) == expected


def test_derivation_is_stable_across_processes():
    """Guards against a regression to Python's salted `hash()`.

    These values are hardcoded on purpose: if the derivation changes, every recorded
    conformance trace becomes incomparable, so the change must be deliberate.
    """
    assert _draw(RngFactory(0).for_request("req-0"), 3) == [141042, 886445, 442175]
