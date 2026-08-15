"""pretending-vllm: a structurally faithful reimplementation of the vLLM V1 engine.

Upstream: vllm/__init__.py
Tier: C

Every layer above the device is real logic. Exactly one leaf is simulated: the thing
that turns scheduled token positions into logits while consuming time and memory.

    REAL  entrypoints -> processor -> EngineCoreClient -> EngineCore -> Scheduler
                                            |                          -> KVCacheManager
                                            v
    REAL                          Executor -> Worker
                                            |
    ============ SIMULATION BOUNDARY =======|===================================
                                            v
    FAKE                  SimModelRunner -> SimDevice (memory ledger + cost model)
                                            SimModel  (token generator)

Nothing outside ``pvllm.sim`` and ``pvllm.platforms.sim`` may invent a number, read a
clock, or draw randomness. That rule is enforced by ``tests/unit/test_purity.py``.

Upstream counterpart paths and fidelity tiers are declared in every module header and
checked against the vendored reference tree by ``tools/spec_sync.py``. See UPSTREAM.md.
"""

__version__ = "0.0.1"

#: The upstream vLLM version whose structure and behavior this package mirrors.
#: Recorded in every golden trace so a trace can be tied to a pin.
UPSTREAM_VERSION = "0.27.1"

__all__ = ["UPSTREAM_VERSION", "__version__"]
