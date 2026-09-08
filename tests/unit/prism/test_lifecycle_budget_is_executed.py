"""`single_gpu_lifecycle` must release what its budget says it releases.

Prism accepts that strategy on `sum(persistent weights) + max(one transient)`.
The classification was computed in the solver and then thrown away: the plan
carried no trace of it, `single_gpu_lifecycle` is registered to
`SingleGPUStrategy`, and that class treated eagerness as global — `if
self._eager: return` in both unload paths, so nothing was ever released.

The plan was therefore accepted on a budget that assumed a swap the executor
never performed, and the strategy over-planned by exactly the transients it
had promised to release. These pin the contract in both directions.

Nothing here involves a device string: the fix is vendor-agnostic by
construction, so hip:0 and mps:0 get it without a line more.
"""

from __future__ import annotations

import pytest

from neurobrix.core.strategies.base import StrategyContext
from neurobrix.core.strategies.single_gpu import SingleGPUStrategy


class _Executor:
    """Stands in for a GraphExecutor.

    `_weights` is not decoration: ExecutionStrategy.unload_weights checks
    `hasattr(executor, '_weights') and bool(executor._weights)` before it
    releases anything, so a fake without it is never unloaded and the test
    passes for the wrong reason.
    """

    def __init__(self):
        self.unloaded = 0
        self._weights = {"w": object()}

    def unload_weights(self):
        self.unloaded += 1
        self._weights = {}


def _ctx(loading_mode, transient=frozenset(), comps=("transformer", "vae", "text_encoder")):
    return StrategyContext(
        strategy_name="single_gpu_lifecycle",
        allocations={c: ("cuda:0", {}) for c in comps},
        component_executors={c: _Executor() for c in comps},
        loading_mode=loading_mode,
        transient_components=transient,
    )


def test_eager_without_a_classification_keeps_everything():
    """The plain single_gpu contract is unchanged: nothing is released."""
    s = SingleGPUStrategy(_ctx("eager"))
    for c in ("transformer", "vae", "text_encoder"):
        assert s._keeps(c), f"{c} must stay resident under a plain eager plan"


def test_eager_with_a_classification_keeps_only_the_persistent_ones():
    s = SingleGPUStrategy(_ctx("eager", frozenset({"vae", "text_encoder"})))
    assert s._keeps("transformer"), "the denoising loop's component is persistent"
    assert not s._keeps("vae"), "a transient must be released under an eager plan"
    assert not s._keeps("text_encoder"), "a transient must be released under an eager plan"


def test_lazy_keeps_nothing_whatever_the_classification():
    s = SingleGPUStrategy(_ctx("lazy", frozenset({"vae"})))
    for c in ("transformer", "vae", "text_encoder"):
        assert not s._keeps(c), "a lazy plan releases every component"


def test_unload_inactive_releases_transients_under_an_eager_plan():
    ctx = _ctx("eager", frozenset({"vae", "text_encoder"}))
    s = SingleGPUStrategy(ctx)
    s.unload_inactive_components(keep_component="transformer")
    assert ctx.component_executors["vae"].unloaded == 1
    assert ctx.component_executors["text_encoder"].unloaded == 1
    assert ctx.component_executors["transformer"].unloaded == 0, (
        "the kept component must never be unloaded"
    )


def test_unload_inactive_is_a_no_op_under_a_plain_eager_plan():
    ctx = _ctx("eager")
    s = SingleGPUStrategy(ctx)
    s.unload_inactive_components(keep_component="transformer")
    assert all(e.unloaded == 0 for e in ctx.component_executors.values()), (
        "plain single_gpu must not start releasing components"
    )


def test_a_released_transient_is_reloaded_on_its_next_use():
    """Releasing without clearing the loaded-set would skip the reload."""
    ctx = _ctx("eager", frozenset({"vae"}))
    s = SingleGPUStrategy(ctx)
    s._loaded_components.add("vae")
    s.unload_weights("vae")
    assert "vae" not in s._loaded_components, (
        "a released component must not still count as loaded, or its next use "
        "runs against weights that are gone"
    )


def test_the_plan_carries_the_classification_to_the_executor():
    """The solver's classification has to survive as far as the strategy."""
    import inspect
    from neurobrix.core.runtime import executor as ex

    src = inspect.getsource(ex)
    assert "transient_components=frozenset(" in src, (
        "RuntimeExecutor no longer passes the classification into "
        "StrategyContext; the lifecycle budget is unenforced again"
    )
