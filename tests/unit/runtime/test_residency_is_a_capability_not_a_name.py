"""The runtime asks a strategy whether it manages its own weight residency.

It used to ask whether the strategy was called `zero3`. A name cannot be
extended: the second strategy to manage its own residency — layer streaming,
which holds one segment of a component at a time — had no way to say so, and
the runtime would have loaded the whole component while the plan promised one
segment. That is the same shape of defect as a vendor prefix hard-coded into a
device test, and this milestone spent a night removing those.

The constraint on the change is that **zero3 behaves exactly as before**, and
these tests are what proves it rather than asserting it: every input the old
predicate answered True for, the new one answers True for, and nothing else
has changed its answer.
"""

from __future__ import annotations

import pytest

from neurobrix.core.strategies import (
    STRATEGY_REGISTRY, strategy_manages_weight_residency)


def _old_predicate(strategy_name: str, component_sub_strategy: str) -> bool:
    """The predicate this replaced, transcribed exactly.

    Kept here as the reference the new behaviour is compared against, so the
    equivalence is a test and not a claim in a commit message.
    """
    if strategy_name == "zero3":
        return True
    return component_sub_strategy == "zero3"


class _Alloc:
    def __init__(self, strategy):
        self.strategy = strategy


class _Plan:
    def __init__(self, components):
        self.components = components


class _Runtime:
    """The two lines of RuntimeExecutor this predicate depends on."""
    def __init__(self, strategy, plan):
        self.strategy = strategy
        self.plan = plan

    # the method under test, bound from the real class
    from neurobrix.core.runtime.executor import RuntimeExecutor as _RE
    _component_manages_own_residency = _RE._component_manages_own_residency


class _Strategy:
    def __init__(self, manages):
        self.manages_weight_residency = manages


def test_zero3_answers_exactly_as_it_did():
    """Plan-level and per-component, both directions."""
    z3 = _Strategy(True)
    other = _Strategy(False)

    cases = [
        # (plan strategy name, live strategy object, component sub-strategy)
        ("zero3", z3, ""),
        ("zero3", z3, "zero3"),
        ("zero3", z3, "single_gpu"),
        ("lazy_sequential", other, "zero3"),
        ("lazy_sequential", other, "single_gpu"),
        ("lazy_sequential", other, ""),
        ("single_gpu", other, ""),
    ]
    for name, strategy, sub in cases:
        plan = _Plan({"model": _Alloc(sub)} if sub else {"model": _Alloc("")})
        rt = _Runtime(strategy, plan)
        new = rt._component_manages_own_residency("model")
        old = _old_predicate(name, sub)
        assert new == old, (
            f"plan={name!r} sub={sub!r}: the old predicate said {old}, the "
            f"new one says {new}. zero3 must behave exactly as before.")


def test_zero3_declares_the_capability_and_the_registry_agrees():
    assert strategy_manages_weight_residency("zero3") is True
    # The registry is a lazy mapping: it hands back the CLASS, not its name.
    assert STRATEGY_REGISTRY["zero3"].__name__ == "Zero3Strategy"


def test_only_strategies_that_manage_residency_say_so():
    """The whole registry, so a new strategy cannot claim it by accident."""
    claimed = {n for n in STRATEGY_REGISTRY
               if strategy_manages_weight_residency(n)}
    assert claimed == {"zero3", "layer_streaming"}, (
        f"strategies claiming to manage their own weight residency: "
        f"{sorted(claimed)}. Adding one is a deliberate act — it changes how "
        f"the runtime drives that strategy's components.")


def test_an_unknown_sub_strategy_is_not_an_error():
    """The runtime asks this of every component, including ones with no
    sub-strategy at all."""
    assert strategy_manages_weight_residency("") is False
    assert strategy_manages_weight_residency("no such strategy") is False


def test_a_component_with_no_allocation_answers_false():
    rt = _Runtime(_Strategy(False), _Plan({}))
    assert rt._component_manages_own_residency("absent") is False
