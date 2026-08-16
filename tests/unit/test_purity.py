"""The simulation boundary, enforced as a test. B1, B3, R19.1.

Three rules, all of which are cheap to hold from the first commit and expensive to
restore once broken:

1. **Clock ownership (R19.1).** The engine core owns the clock. Nothing outside
   ``pvllm/sim/`` reads wall-clock time. Once a dozen call sites read the clock
   directly, determinism is gone and getting it back means auditing all of them --
   which is why D2 requires this from commit one.

2. **Randomness ownership (B3, R19.2).** One global seed reproduces an entire run.
   That only holds if ``pvllm/sim/rng.py`` is the sole source of randomness.

3. **No simulator awareness above the boundary (B1).** No ``if simulated:`` in
   ``v1/core``, ``v1/engine``, or ``entrypoints``, and no import of the ``sim``
   package from them. Selection happens through the platform abstraction.

Plus a structural check: no ``torch``, no CUDA, no ``transformers`` at import time
anywhere (NF1), and every module declares its upstream counterpart and tier (NF5).

These are AST checks rather than greps so that a mention inside a string or comment
does not trip them -- the docstrings in this repo discuss ``time.time`` constantly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "pvllm"

#: The only subtree allowed to read a clock or draw randomness.
SIM_PACKAGE = PACKAGE_ROOT / "sim"

#: Modules that must not know a simulator exists.
BOUNDARY_ENFORCED_SUBTREES = ("v1/core", "v1/engine", "entrypoints")

#: `now`/`utcnow` are here because `datetime.datetime.now()` is a wall clock and the
#: attribute check below already scopes itself to the `time` and `datetime` modules --
#: `time.now` does not exist, so nothing is caught spuriously. They were missing, and a
#: `strftime_now` added to the chat-template environment walked straight past the lint.
FORBIDDEN_TIME_ATTRS = {
    "time",
    "monotonic",
    "perf_counter",
    "process_time",
    "time_ns",
    "now",
    "utcnow",
}
FORBIDDEN_MODULES = {"torch", "transformers", "cupy", "triton"}
RANDOM_MODULES = {"random", "secrets"}


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py"))


def _relative(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT.parent))


def _is_sim(path: Path) -> bool:
    return SIM_PACKAGE in path.parents or path == SIM_PACKAGE


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _imported_module_roots(tree: ast.Module) -> set[str]:
    """Root module names imported by this file, from both import forms."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        # Relative imports have no module root to check.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _full_dotted_imports(tree: ast.Module) -> set[str]:
    """Fully dotted module paths imported, for subtree checks.

    `from a.b import c` records both `a.b` and `a.b.c`, because the name being
    imported may itself be a module. Recording only `a.b` let the ordinary
    module-import form -- `from pvllm.entrypoints.cli import openai` -- slip past both
    exemption guards below while the dotted form was caught, so each enforced its
    invariant against one of the two ways to write the same import.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


#: The one module outside `pvllm/sim/` allowed a wall clock, and the reason it is not
#: a hole in R19.1: `pvllm complete` / `pvllm chat` are *clients*. They run in a
#: different process from the engine, hold no engine state, and their `--stats` timing
#: measures this side of the socket -- which is why the output labels itself
#: `[client-measured]` and points at `/metrics` for the modeled numbers. Nothing in
#: the engine can read what it produces.
WALL_CLOCK_CLIENTS = (
    "pvllm/entrypoints/cli/openai.py",
    # R3.1. A model's chat template may call `strftime_now`, and Llama-3.x's default
    # one does in its opening lines. The date is read while *rendering a prompt* in
    # the frontend, before any request exists; it never reaches the engine, and a
    # fixed answer would stamp a stale date into every prompt. The guard below keeps
    # this from spreading the same way it does for the CLI client.
    "pvllm/tokenizers/hf.py",
)


def test_no_wall_clock_outside_sim() -> None:
    """R19.1: the engine core owns the clock; nothing else reads one."""
    violations: list[str] = []
    for path in _python_files():
        if _is_sim(path):
            continue
        if _relative(path) in WALL_CLOCK_CLIENTS:
            continue
        tree = _parse(path)

        # `import time` / `from time import ...`
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "time":
                        violations.append(
                            f"{_relative(path)}:{node.lineno}: imports `time`"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module == "time":
                for alias in node.names:
                    if alias.name in FORBIDDEN_TIME_ATTRS:
                        violations.append(
                            f"{_relative(path)}:{node.lineno}: imports time.{alias.name}"
                        )
            # `time.time()` style attribute access, even if `time` came in some
            # other way.
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TIME_ATTRS:
                value = node.value
                if isinstance(value, ast.Name) and value.id in {"time", "datetime"}:
                    violations.append(
                        f"{_relative(path)}:{node.lineno}: reads {value.id}.{node.attr}"
                    )

    assert not violations, (
        "Wall-clock access outside pvllm/sim/. The engine core owns the clock "
        "(R19.1); take the time from it instead.\n  " + "\n  ".join(violations)
    )


def test_no_randomness_outside_sim() -> None:
    """B3/R19.2: one seed reproduces a run, so randomness lives in sim/rng.py."""
    violations: list[str] = []
    for path in _python_files():
        if _is_sim(path):
            continue
        tree = _parse(path)

        for module in _imported_module_roots(tree) & RANDOM_MODULES:
            violations.append(f"{_relative(path)}: imports `{module}`")

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "random":
                value = node.value
                if isinstance(value, ast.Name) and value.id in {"np", "numpy"}:
                    violations.append(
                        f"{_relative(path)}:{node.lineno}: uses numpy.random"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module in {
                "numpy.random",
                "numpy",
            }:
                for alias in node.names:
                    if alias.name in {"random", "default_rng", "Generator"}:
                        violations.append(
                            f"{_relative(path)}:{node.lineno}: imports "
                            f"{node.module}.{alias.name}"
                        )

    assert not violations, (
        "Randomness outside pvllm/sim/. Derive a generator from RngFactory so the "
        "run stays reproducible from one seed (R19.2).\n  " + "\n  ".join(violations)
    )


def test_no_torch_or_transformers_anywhere() -> None:
    """NF1: no torch, no CUDA, no transformers at import time -- including in sim/."""
    violations: list[str] = []
    for path in _python_files():
        tree = _parse(path)
        for module in _imported_module_roots(tree) & FORBIDDEN_MODULES:
            violations.append(f"{_relative(path)}: imports `{module}`")

    assert not violations, (
        "A dependency this project exists to avoid was imported (NF1).\n  "
        + "\n  ".join(violations)
    )


def test_control_plane_does_not_import_sim() -> None:
    """B1: no code above the boundary knows it is talking to a simulator.

    The control plane reaches simulated classes only through `current_platform`,
    which hands back fully-qualified names it resolves at runtime.
    """
    violations: list[str] = []
    for path in _python_files():
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if not rel.startswith(BOUNDARY_ENFORCED_SUBTREES):
            continue
        for imported in _full_dotted_imports(_parse(path)):
            if imported == "pvllm.sim" or imported.startswith("pvllm.sim."):
                violations.append(f"{_relative(path)}: imports `{imported}`")

    assert not violations, (
        "Code above the simulation boundary imported the simulator (B1). Resolve it "
        "through `current_platform` instead.\n  " + "\n  ".join(violations)
    )


def test_no_simulator_branching_above_boundary() -> None:
    """B1: no `if simulated:` in v1/core, v1/engine, or entrypoints."""
    suspicious = {"simulated", "is_simulated", "is_sim", "sim_mode", "pretending"}
    violations: list[str] = []
    for path in _python_files():
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if not rel.startswith(BOUNDARY_ENFORCED_SUBTREES):
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.If):
                continue
            for sub in ast.walk(node.test):
                name = None
                if isinstance(sub, ast.Name):
                    name = sub.id
                elif isinstance(sub, ast.Attribute):
                    name = sub.attr
                if name in suspicious:
                    violations.append(
                        f"{_relative(path)}:{node.lineno}: branches on `{name}`"
                    )

    assert not violations, (
        "Code above the simulation boundary branched on being simulated (B1).\n  "
        + "\n  ".join(violations)
    )


def test_the_wall_clock_client_is_not_reachable_from_the_engine() -> None:
    """The exemption above is sound only while nothing in the engine can reach it.

    `pvllm complete` measures a wall clock, which R19.1 forbids everywhere the engine
    can see. That is fine for a client in another process and not fine the moment
    some engine module imports it for a helper -- so the exemption is one module wide
    and this is what keeps it there. Only the CLI entry point may reach it.
    """
    allowed = {
        "pvllm/entrypoints/cli/main.py",
        "pvllm/entrypoints/cli/openai.py",
        # The tokenizer registry is the only door to the chat-template renderer, and
        # `get_tokenizer` hands back a `TokenizerLike` -- so the engine reaches the
        # wall clock only through a protocol method that renders text.
        "pvllm/tokenizers/registry.py",
        "pvllm/tokenizers/hf.py",
    }
    violations: list[str] = []
    for path in _python_files():
        if _relative(path) in allowed:
            continue
        for imported in _full_dotted_imports(_parse(path)):
            if imported.startswith(
                ("pvllm.entrypoints.cli.openai", "pvllm.tokenizers.hf")
            ):
                violations.append(f"{_relative(path)}: imports `{imported}`")
    assert not violations, (
        "the wall-clock CLI client is only for the CLI entry point:\n"
        + "\n".join(violations)
    )


def test_only_the_debug_router_reads_simulator_state() -> None:
    """The one deliberate exception to B1, kept from spreading.

    `entrypoints/serve/dev/introspect.py` traverses through the worker into the
    simulator -- `device.recent_steps()`, the memory profile, the device card -- and
    it has to, because a cost-model breakdown *is* simulator state and showing it is
    the entire point of D9.

    That is sound only while two things hold. The introspector makes no decisions, so
    nothing the engine does can depend on it; and nothing else reaches for it, so the
    exemption stays one module wide. The first is a property of the code (every
    method returns a dict and mutates nothing). This test enforces the second.

    The import-based B1 check above cannot catch attribute traversal, so without this
    the next module to reach through `driver_worker.device` would pass silently.
    """
    allowed = "pvllm/entrypoints/serve/dev/"
    violations: list[str] = []
    for path in _python_files():
        rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        if rel.startswith(allowed):
            continue
        for imported in _full_dotted_imports(_parse(path)):
            if imported.startswith("pvllm.entrypoints.serve.dev"):
                violations.append(f"{_relative(path)}: imports `{imported}`")

    # api_server.py attaches the router, which is the sanctioned entry point.
    violations = [v for v in violations if "api_server.py" not in v]

    assert not violations, (
        "A module outside the debug router imported the engine introspector. It "
        "reads simulator state directly (the B1 exemption for D9) and must stay "
        "confined to entrypoints/serve/dev/.\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize("path", _python_files(), ids=_relative)
def test_module_declares_upstream_and_tier(path: Path) -> None:
    """NF5: every module names its upstream counterpart and fidelity tier.

    This is what `tools/spec_sync.py` reads to detect upstream drift, so a module
    without a header is invisible to that check.
    """
    docstring = ast.get_docstring(_parse(path))
    rel = _relative(path)

    if path.name == "__init__.py" and not (path.read_text().strip()):
        pytest.skip("empty package marker")

    assert docstring, f"{rel}: module has no docstring; NF5 requires an Upstream header"
    assert "Upstream:" in docstring, (
        f"{rel}: module docstring must name its upstream counterpart, e.g.\n"
        f"    Upstream: vllm/v1/core/sched/scheduler.py\n"
        f"or `Upstream: (none -- simulator)` for Tier D modules."
    )
    assert "Tier:" in docstring, (
        f"{rel}: module docstring must declare a fidelity tier (A/B/C/D). See UPSTREAM.md."
    )


def test_the_clock_advances_only_for_device_work() -> None:
    """R19.1, and the premise `--async-scheduling` is refused on.

    The engine charges modeled *device* time and nothing else: no CPU time, no
    scheduling time, no serialization time. That is what makes a virtual-clock run
    and a real-clock run report identical numbers, and it is also why upstream's
    async scheduler cannot be modeled here -- the thing it hides costs zero.

    Enforced rather than asserted in prose, because the day something starts
    charging CPU time is the day that refusal becomes wrong and nobody would notice.
    The allowed sites are the simulated device, the KV-transfer charges the engine
    core makes on its behalf, and the benchmark runner's arrival gaps.
    """
    allowed = {
        "pvllm/sim/device.py",
        "pvllm/sim/clock.py",
        "pvllm/v1/engine/core.py",
        "pvllm/benchmarks/lib/runner.py",
    }
    violations: list[str] = []
    for path in _python_files():
        rel = _relative(path)
        if rel in allowed:
            continue
        source = path.read_text()
        for lineno, line in enumerate(source.splitlines(), start=1):
            if ".advance(" in line or ".advance_async(" in line:
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    assert not violations, (
        "only the device (and the two sites that charge on its behalf) may advance "
        "the clock:\n" + "\n".join(violations)
    )
