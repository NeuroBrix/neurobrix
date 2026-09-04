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
