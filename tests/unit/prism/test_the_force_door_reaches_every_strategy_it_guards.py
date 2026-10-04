"""`NBX_FORCE_STRATEGY` accepts every strategy the engine can emit.

The door exists for "deterministic single-strategy selection for matrix validation and
debugging" — its own words. Its valid set was a HARDCODED LITERAL, and it had drifted from
the cascade it guards:

    NBX_FORCE_STRATEGY='layer_streaming' is invalid. Valid values: ['block_scatter',
    'component_placement', 'component_placement_lazy', 'cpu_execution', 'lazy_sequential',
    'pipeline_parallel', 'single_gpu', 'single_gpu_lifecycle', 'weight_sharding', 'zero3']

`layer_streaming`, `op_level_tiling` and `cpu_streaming` are all rungs of the cascade and all
three were rejected. So the instrument for validating strategies could not reach three of the
strategies it validates — and the last of those, `cpu_streaming`, is the rung that exists so
a model always runs rather than being refused.

Found 2026-09-22 while trying to exercise the streamed path, which is unreachable by SCORE on
this rack: under a tight-host V100 fixture Prism prints "lazy_sequential scored 260 ahead of
layer_streaming". layer_streaming is viable there and simply loses; the door is the only way
to run it, and the door refused the name.

The correct source was already three lines below in the same block — the OTHER error message
reads `sorted(n for n, _ in strategies)`. One branch read the cascade and the other held a
copy, and the copy was the one gating entry. The same shape as
test_solver_registry_parity.py, which exists because a strategy was added to the cascade and
not to the registry.

TWO QUESTIONS, KEPT APART. "Is this a real name?" is the REGISTRY's question — its comment
says every name Prism can emit must have an entry. "Is it available on this profile?" is the
cascade's, checked separately, and keeping them apart preserves the better message for a
multi-GPU strategy asked for on a single-GPU profile.
"""
from __future__ import annotations

import ast
import inspect

from neurobrix.core.prism import solver as _solver
from neurobrix.core.strategies import STRATEGY_REGISTRY


def _the_door() -> tuple[str, ast.FunctionDef, str]:
    """The PrismSolver method that reads NBX_FORCE_STRATEGY, found BY NAME in the file as it is
    on disk now — wherever the door lives, and never by a line number.

    `inspect.getsource(method)` slices the file at the line the method had WHEN IMPORTED; an edit
    of solver.py during a 42-minute suite shifted that slice 26 lines up and this gate read the
    middle of `solve()` instead of `_solve_at_rung` (suite_final_0abfe804, 2026-10-04: the door had
    not moved; the file under the running suite had). Parsing the file and locating the door by
    what it reads cannot be shifted that way."""
    path = inspect.getsourcefile(_solver)
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrismSolver")
    doors = []
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        reads = [c for c in ast.walk(fn)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                 and c.func.attr == "get" and c.args
                 and isinstance(c.args[0], ast.Constant) and c.args[0].value == "NBX_FORCE_STRATEGY"]
        if reads:
            doors.append(fn)
    assert len(doors) == 1, (
        f"NBX_FORCE_STRATEGY is read by {[d.name for d in doors]} in PrismSolver — the door is "
        "ONE place; none means it left the solver, two means a second copy of it")
    door = doors[0]
    return door.name, door, ast.get_source_segment(text, door)


def test_the_door_does_not_carry_its_own_copy_of_the_strategy_names():
    """A door with a hand-written list of what it guards drifts from what it guards."""
    name, fn, src = _the_door()
    i = src.find("NBX_FORCE_STRATEGY")
    window = src[i:i + 2500]
    assert "STRATEGY_REGISTRY" in window or "for n, _ in strategies" in window, (
        f"{name}: the valid set is not derived from the registry or the cascade — it is a copy, "
        "and a copy is what let layer_streaming, op_level_tiling and cpu_streaming be rejected")
    # Any literal of strategy NAMES inside the door's block is a copy, whatever its formatting:
    # a set, list or tuple whose elements are all strings and at least two of them registry
    # names. The block is the window above, in file lines (the method's other literals — a
    # membership test of the two host rungs further down — are not the door).
    first = fn.lineno + src[:i].count("\n")
    last = first + window.count("\n")
    names = set(STRATEGY_REGISTRY.keys())
    for node in ast.walk(fn):
        if not first <= getattr(node, "lineno", 0) <= last:
            continue
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)) and node.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            held = {e.value for e in node.elts} & names
            assert len(held) < 2, (
                f"{name} line {node.lineno}: a hand-written literal of strategy names "
                f"{sorted(held)} — the hardcoded copy is back")


def test_every_registry_strategy_is_an_accepted_name():
    """The registry is the set of names Prism can emit; the door must accept all of them."""
    names = sorted(STRATEGY_REGISTRY.keys())
    assert names, "the registry is empty — this cell would pass vacuously"
    for missing in ("layer_streaming", "op_level_tiling", "cpu_streaming"):
        assert missing in names, (
            f"{missing} is not in the registry; if the cascade emits it, "
            f"test_solver_registry_parity.py should already be red")


def test_the_three_that_were_rejected_are_the_newer_rungs():
    """Pins WHICH names had drifted, so the entry is about a real gap and not a tidy-up.
    These three are absent from the literal that was there before."""
    old_literal = {
        "single_gpu", "single_gpu_lifecycle", "component_placement", "pipeline_parallel",
        "block_scatter", "weight_sharding", "component_placement_lazy", "lazy_sequential",
        "zero3", "cpu_execution",
    }
    drifted = set(STRATEGY_REGISTRY.keys()) - old_literal
    assert drifted == {"layer_streaming", "op_level_tiling", "cpu_streaming"}, drifted
