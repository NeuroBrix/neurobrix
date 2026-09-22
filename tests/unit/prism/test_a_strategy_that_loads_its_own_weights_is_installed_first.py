"""A strategy that loads its own weights is installed BEFORE the load, and the
whole-component load does not happen.

`_ensure_weights_loaded` did this, in this order:

    executor.load_weights(nbx_path, component)              # the WHOLE component
    ...
    if self._component_manages_own_residency(comp_name):    # asked ten lines too late
        install_fn(comp_name, executor)

`_component_manages_own_residency`'s own docstring names the outcome as the thing it exists
to prevent — "the runtime would have loaded its whole component while the plan promised one
segment" — and it gated the INSTALL, not the LOAD.

MEASURED on all four class-1 MoE models, 2026-09-22, `--hardware default-ff6008b7` pinned to
the 16 GB card they ran on:

    Strategy: layer_streaming
    live_tracked=0MB -> GPU malloc failed for 32 332 025 856 bytes  (30.1 GiB)
    driver_total=16151MB
    ... and not one LAYERDIAG line: `segmented_run` was never reached.

After: the segmentation installs, and

    [layer_streaming] 'model': 3 segments, one resident at a time
    malloc failed for 14 645 594 112 bytes  (13.6 GiB)   live_tracked=2160MB

30.1 GiB -> 13.6 GiB, and the segmented path is entered.

TWO CAPABILITIES, DELIBERATELY SEPARATE
---------------------------------------
`manages_weight_residency` is not enough to gate the load, because zero3 declares it too and
zero3 manages residency THROUGH the loader — `load_component_weights` partitions its blocks
onto pinned host, so zero3 NEEDS the whole-component load. layer_streaming loads per segment
inside `segmented_run`, so for it that load is fatal. Conflating the two would have broken
zero3 to fix layer streaming.

WHAT THIS DOES NOT FIX, stated rather than implied: the run still OOMs, now by ~286 MB
(13 966 MB asked, 13 680 MB free with 2 160 MB already live). That is a segment-BUDGET
question — the plan cut three segments against a figure that does not match what is free at
execution — and it is a different defect from the ordering one this file gates.
"""
from __future__ import annotations

from neurobrix.core.strategies.base import ExecutionStrategy
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy
from neurobrix.core.strategies.zero3 import Zero3Strategy


def test_the_two_capabilities_are_distinct():
    assert ExecutionStrategy.manages_weight_residency is False
    assert ExecutionStrategy.loads_own_weights is False
    # zero3: manages residency, but THROUGH the loader — it must keep getting the load.
    assert Zero3Strategy.manages_weight_residency is True
    assert Zero3Strategy.loads_own_weights is False
    # layer_streaming: both.
    assert LayerStreamingStrategy.manages_weight_residency is True
    assert LayerStreamingStrategy.loads_own_weights is True


def test_zero3_is_byte_identical_on_this_path():
    """The whole point of the narrower flag: zero3's route through
    `_ensure_weights_loaded` is unchanged, because it does not declare the new capability."""
    assert Zero3Strategy.loads_own_weights is False, (
        "zero3 now claims to load its own weights — its blocks are partitioned INSIDE "
        "load_component_weights, so skipping that load would leave it with nothing resident")


def test_the_load_is_gated_on_the_narrower_flag_not_the_wider_one():
    """A door, because the numbers above cannot be reproduced without a 16 GB card and a
    MoE container. What can be asserted anywhere is that the skip reads `loads_own_weights`
    and not `manages_weight_residency` — reading the wider one is what would break zero3."""
    import inspect
    from neurobrix.core.runtime import executor as _ex
    src = inspect.getsource(_ex.RuntimeExecutor._ensure_weights_loaded)
    i_skip = src.find("loads_own_weights")
    i_load = src.find("executor.load_weights(")
    assert i_skip > -1, "the load is no longer gated on loads_own_weights"
    assert i_skip < i_load, (
        "the capability is consulted AFTER the load — which is the original defect, "
        "where the install was gated and the load was not")


def test_install_happens_before_the_skip_returns():
    """If the strategy is not installed, nothing replaces `run` and the component executes
    with no weights at all — worse than the OOM this replaces."""
    import inspect
    from neurobrix.core.runtime import executor as _ex
    src = inspect.getsource(_ex.RuntimeExecutor._ensure_weights_loaded)
    block = src[src.find("loads_own_weights"):]
    i_install = block.find("install_fn(comp_name, executor)")
    i_return = block.find("return")
    assert i_install > -1 and i_return > i_install, (
        "the early return happens before install_for_executor — the component would run "
        "with no weights and no segmentation")
