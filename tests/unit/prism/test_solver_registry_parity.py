"""Every strategy the solver can choose must be one the registry can build.

`STRATEGY_REGISTRY`'s own comment states the rule — "Every strategy name that
Prism can emit MUST have a registry entry" — and nothing enforced it. On
2026-09-03 `cpu_streaming` was added to the solver's cascade and not to the
registry, so the LAST rung of the ladder, the one that exists so a model always
runs rather than being refused, crashed with

    ZERO FALLBACK: Unknown strategy 'cpu_streaming'.

the moment it was selected. A rule written in a comment is a wish; this file is
the rule.

The two names are read out of the code rather than listed here, so a strategy
added to either side is checked without anyone remembering to update a list.

TWO NAMESPACES (2026-10-12). `STRATEGY_REGISTRY` answers `get_strategy(plan.strategy)` —
the PLAN's strategy, one of the cascade's rungs. A component carries its own name in
`ComponentAllocation.strategy`, which `ExecutorFactory.STRATEGY_MAP` answers. Until
`expert_store` (3ee5ee57) every component name was also a rung, so one registry looked
like enough. `expert_store` is a component strategy only: Prism writes it into one
component of a `lazy_sequential` or `zero3` plan (`_place_component` Strategy 2.5,
`_zero3_resident_budgets`), and its runtime half is a ratchet state of Zero3Strategy. It
was registered as a plan strategy no rung emits — dead, and the force door accepted the
name and then refused it on every profile. Making it a rung would be a second name for
the lazy_sequential/zero3 plan it already is. So the component namespace is held here
by the same two directions: every name the solver writes into a component is buildable
by the factory, and a name only the factory holds is one the solver writes.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from neurobrix.core.prism.solver import PrismSolver
from neurobrix.core.strategies import STRATEGY_REGISTRY


def _solver_strategy_names() -> set[str]:
    """Names in the solver's cascade tuples: ("name", self._try_name)."""
    tree = ast.parse(inspect.getsource(PrismSolver))
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Tuple) and len(node.elts) == 2
                and isinstance(node.elts[0], ast.Constant)
                and isinstance(node.elts[0].value, str)
                and isinstance(node.elts[1], ast.Attribute)
                and node.elts[1].attr.startswith("_try_")):
            names.add(node.elts[0].value)
    return names


def test_the_solver_declares_strategies_at_all():
    """If the parse breaks, every other test here passes vacuously."""
    found = _solver_strategy_names()
    assert len(found) >= 8, f"parsed only {sorted(found)} — the extractor is broken"
    assert "single_gpu" in found and "cpu_streaming" in found


@pytest.mark.parametrize("name", sorted(_solver_strategy_names()))
def test_every_strategy_the_solver_can_choose_is_buildable(name):
    assert name in STRATEGY_REGISTRY, (
        f"the solver's cascade can select '{name}' but STRATEGY_REGISTRY "
        f"cannot build it, so choosing it raises ZERO FALLBACK at runtime.\n"
        f"Registered: {sorted(STRATEGY_REGISTRY)}"
    )


def test_the_registry_has_no_entry_the_solver_can_never_choose():
    """The other direction: a registered strategy nobody can select is dead
    code that looks like a capability."""
    unreachable = set(STRATEGY_REGISTRY) - _solver_strategy_names()
    assert not unreachable, (
        f"registered but unreachable from the solver's cascade: "
        f"{sorted(unreachable)}"
    )


def test_cpu_streaming_places_everything_on_the_host():
    """It shares CPUExecutionStrategy deliberately — same placement, and the
    plan's lazy loading_mode is what makes it a distinct rung. If it ever
    needs different PLACEMENT, it needs its own class."""
    from neurobrix.core.strategies.cpu_execution import CPUExecutionStrategy

    assert STRATEGY_REGISTRY["cpu_streaming"] is CPUExecutionStrategy
    assert STRATEGY_REGISTRY["cpu_execution"] is CPUExecutionStrategy


# ---------------------------------------------------------------------------
# The component namespace: ComponentAllocation.strategy -> ExecutorFactory.STRATEGY_MAP
# ---------------------------------------------------------------------------

def _strings_of(value: ast.AST) -> set[str]:
    """The strategy names an assigned expression can take: a string constant, either arm of a
    conditional, and — for `name if name in ("a", "b") else ...` — the names the guard admits."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return {value.value}
    if isinstance(value, ast.IfExp):
        out = _strings_of(value.body) | _strings_of(value.orelse)
        t = value.test
        if (isinstance(value.body, ast.Name) and isinstance(t, ast.Compare)
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.In)
                and isinstance(t.comparators[0], (ast.Tuple, ast.List, ast.Set))):
            out |= {e.value for e in t.comparators[0].elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        return out
    return set()


def _solver_component_strategy_names() -> set[str]:
    """Names the solver writes into `ComponentAllocation.strategy`: the dataclass default, the
    `strategy=` of every ComponentAllocation(...) it builds (a variable followed to its
    assignments in the same function), and every `<allocation>.strategy = ...` (a plan's own
    strategy is not a component's: targets on a `plan` are left to the cascade's test)."""
    from neurobrix.core.prism import solver as _solver
    tree = ast.parse(Path(inspect.getsourcefile(_solver)).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ComponentAllocation":
            for st in node.body:
                if (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                        and st.target.id == "strategy" and st.value is not None):
                    names |= _strings_of(st.value)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned: dict[str, set[str]] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        assigned.setdefault(tgt.id, set()).update(_strings_of(node.value))
                    elif (isinstance(tgt, ast.Attribute) and tgt.attr == "strategy"
                          and not (isinstance(tgt.value, ast.Name) and "plan" in tgt.value.id)):
                        names |= _strings_of(node.value)
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "ComponentAllocation"):
                for kw in node.keywords:
                    if kw.arg != "strategy":
                        continue
                    names |= _strings_of(kw.value)
                    if isinstance(kw.value, ast.Name):
                        names |= assigned.get(kw.value.id, set())
    return names


def _component_namespace() -> dict:
    from neurobrix.core.runtime.factory import ExecutorFactory
    return ExecutorFactory.STRATEGY_MAP


def test_the_solver_writes_component_strategies_at_all():
    """If the extractor breaks, the two component cells below pass vacuously."""
    found = _solver_component_strategy_names()
    assert {"single_gpu", "zero3", "expert_store", "block_scatter", "pipeline_parallel",
            "weight_sharding"} <= found, f"parsed only {sorted(found)} — the extractor is broken"


@pytest.mark.parametrize("name", sorted(_solver_component_strategy_names()))
def test_every_component_strategy_the_solver_writes_is_buildable(name):
    assert name in _component_namespace(), (
        f"the solver writes '{name}' into a ComponentAllocation but ExecutorFactory.STRATEGY_MAP "
        f"cannot build it, so the component's executor raises ZERO FALLBACK at load.\n"
        f"Component namespace: {sorted(_component_namespace())}")


def test_a_component_only_name_is_one_the_solver_writes():
    """The other direction: a name only the component namespace holds (no rung emits it) must be
    one the solver writes into a component — else it is dead code that looks like a capability."""
    only_component = set(_component_namespace()) - set(STRATEGY_REGISTRY)
    unreachable = only_component - _solver_component_strategy_names()
    assert not unreachable, (
        f"component-only strategy names the solver never writes into a ComponentAllocation: "
        f"{sorted(unreachable)}")


def test_expert_store_is_a_component_strategy_not_a_rung():
    """Where `expert_store` lives, stated: written into a component by the solver, built by the
    factory, served by Zero3Strategy through the lazy_sequential and zero3 rungs — and in neither
    plan registry, which `get_strategy(plan.strategy)` alone reads."""
    from neurobrix.core.strategies.triton import TRITON_REGISTRY
    assert "expert_store" in _solver_component_strategy_names()
    assert "expert_store" in _component_namespace()
    assert "expert_store" not in _solver_strategy_names()
    assert "expert_store" not in STRATEGY_REGISTRY
    assert "expert_store" not in TRITON_REGISTRY
    assert set(TRITON_REGISTRY) <= set(STRATEGY_REGISTRY), (
        f"the triton registry names plan strategies the plan registry does not: "
        f"{sorted(set(TRITON_REGISTRY) - set(STRATEGY_REGISTRY))}")
