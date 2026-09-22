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

import inspect

from neurobrix.core.prism import solver as _solver
from neurobrix.core.strategies import STRATEGY_REGISTRY


def test_the_door_does_not_carry_its_own_copy_of_the_strategy_names():
    """A door with a hand-written list of what it guards drifts from what it guards."""
    src = inspect.getsource(_solver.PrismSolver.solve)
    i = src.find("NBX_FORCE_STRATEGY")
    assert i > -1, "the force door moved — re-read this gate"
    window = src[i:i + 2500]
    assert "STRATEGY_REGISTRY" in window or "for n, _ in strategies" in window, (
        "the valid set is not derived from the registry or the cascade — it is a copy, and "
        "a copy is what let layer_streaming, op_level_tiling and cpu_streaming be rejected")
    assert '"single_gpu", "single_gpu_lifecycle",' not in window, (
        "the hardcoded literal is back")


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
