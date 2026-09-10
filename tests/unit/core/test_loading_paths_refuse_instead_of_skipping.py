"""A missing key must refuse, never skip a load in silence.

Five strategies read `runtime_package.nbx_path` — a name that exists on no
package this engine builds — and, finding nothing, skipped the load without a
word. Measured 2026-09-09: the executor's package carries `root_path`, the
container carries `cache_path`, `nbx_path` neither. The methods were therefore
no-ops everywhere, and harmless only by the luck that the runtime loads the
weights first.

Same family as the vacuous gates: a name read, a step skipped, no trace.
"""
import ast
import pathlib

import pytest

from neurobrix.core.strategies.base import ExecutionStrategy

STRATEGIES = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "core" / "strategies"


class _Ctx:
    def __init__(self, package):
        self.runtime_package = package
        self._active_component = None


class _Strat(ExecutionStrategy):
    """Concrete enough to own the brick; nothing else."""
    def __init__(self, context):
        self.context = context

    def execute_component(self, *a, **k):  # pragma: no cover - unused
        raise NotImplementedError

    def prepare_inputs(self, *a, **k):  # pragma: no cover - unused
        raise NotImplementedError

    def handle_outputs(self, *a, **k):  # pragma: no cover - unused
        raise NotImplementedError


def _strat(package):
    return _Strat(_Ctx(package))


class _Pkg:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


# --------------------------------------------------------------- the resolver

def test_resolves_root_path_which_is_what_the_executor_package_carries():
    assert _strat(_Pkg(root_path="/art/model.nbx")).resolve_artifact_path("c") == "/art/model.nbx"


def test_resolves_cache_path_which_is_what_the_container_carries():
    assert _strat(_Pkg(cache_path="/cache/model.nbx")).resolve_artifact_path("c") == "/cache/model.nbx"


def test_nbx_path_still_wins_when_a_package_does_carry_it():
    pkg = _Pkg(nbx_path="/a.nbx", root_path="/b.nbx")
    assert _strat(pkg).resolve_artifact_path("c") == "/a.nbx"


def test_no_package_refuses_loudly_and_names_the_component():
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK.*'decoder'.*no runtime package"):
        _strat(None).resolve_artifact_path("decoder")


def test_package_without_any_path_refuses_and_says_what_it_tried():
    with pytest.raises(RuntimeError) as e:
        _strat(_Pkg(name="m")).resolve_artifact_path("decoder")
    msg = str(e.value)
    assert "ZERO FALLBACK" in msg
    for tried in ("nbx_path", "root_path", "cache_path", "path"):
        assert tried in msg, f"the refusal must name what it looked for: {tried}"


def test_an_empty_path_is_not_a_path():
    """`''` used to pass the old `if nbx_path:` guard as falsey and skip; it
    must now refuse, not be handed to a loader that would open the cwd."""
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        _strat(_Pkg(root_path="")).resolve_artifact_path("decoder")


# ------------------------------------------------------- the one legal no-op

def test_already_loaded_is_the_only_silence_left():
    class _Exec:
        _weights = {"w": 1}

        def load_weights(self, *a):  # pragma: no cover - must not be reached
            raise AssertionError("re-loaded weights that were already resident")

    strat = _strat(None)          # no package at all
    strat.context.component_executors = {"c": _Exec()}
    strat.load_weights("c")       # resident weights: doing nothing is correct


def test_not_loaded_and_no_package_no_longer_passes_in_silence():
    class _Exec:
        _weights = {}

        def load_weights(self, *a):  # pragma: no cover - must not be reached
            raise AssertionError("loaded from a path that does not exist")

    strat = _strat(None)
    strat.context.component_executors = {"c": _Exec()}
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        strat.load_weights("c")


# ------------------------------------------- no strategy keeps its own copy

def test_no_strategy_reads_the_absent_name_by_hand_any_more():
    offenders = []
    for f in STRATEGIES.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "'nbx_path'" in line and "getattr" in line:
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, (
        "a strategy resolves the artefact path by hand again; the brick is "
        "ExecutionStrategy.resolve_artifact_path:\n  " + "\n  ".join(offenders))


def test_every_strategy_that_loads_weights_routes_through_the_brick():
    """Any `executor.load_weights(p, ...)` must take `p` from the resolver."""
    bad = []
    for f in STRATEGIES.rglob("*.py"):
        whole = f.read_text()
        tree = ast.parse(whole)
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in funcs:
            calls = [n for n in ast.walk(fn)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "load_weights" and n.args]
            if not calls:
                continue
            src = ast.get_source_segment(whole, fn) or ""
            if "resolve_artifact_path" in src or "_nbx_path" in src:
                continue
            # a nested function takes the path from the scope that resolved it;
            # the enclosing functions must come from THIS tree, or `is` compares
            # nodes of two different parses and never matches
            outer = [o for o in funcs
                     if o is not fn and any(n is fn for n in ast.walk(o))]
            if any("resolve_artifact_path" in (ast.get_source_segment(whole, o) or "")
                   or "_nbx_path" in (ast.get_source_segment(whole, o) or "")
                   for o in outer):
                continue
            # the path may equally arrive as an argument or off the executor
            if any(isinstance(c.args[0], ast.Name) and c.args[0].id in
                   {a.arg for a in fn.args.args} for c in calls):
                continue
            bad.append(f"{f.name}:{fn.lineno}:{fn.name}")
    assert not bad, ("these load weights from a path of their own making:\n  "
                     + "\n  ".join(bad))
