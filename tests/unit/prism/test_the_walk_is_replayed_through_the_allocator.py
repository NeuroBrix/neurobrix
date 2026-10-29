"""Prism's walk of a component replayed as the Triton allocator serves it (M3, docs/nbxtensor/COST.md section 2): its
allocations and frees, the hot loop's deferred queue holding each dead graph tensor until the queue reaches its cliff
or its count, through the allocator's pool (`nbxtensor.memory.pool.simulate`) — `ActivationProfile.held_peak_bytes`,
what `--explain-plan` states beside each component. Measured 2026-10-29: Kokoro's decoder walks a 34 MB live peak and
the allocator held 1 213 MB — the queue, not a factor; replayed, 1 407 MB.

What each cell would do if the code were wrong: a queue that frees at once (the held peak would be the live peak: red),
a queue that never drains (the held peak would be every byte allocated: red), no replay under the compiled engine's
rules (a figure where none is: red)."""
from neurobrix.core.prism.profiler import ActivationProfiler
from tests.unit.prism.test_layer_partition import _chain

MB = 1024 * 1024


def _walk(drain_cliff, drain_count=512):
    return ActivationProfiler(_chain(8, weight_mb=1, act_mb=4)).estimate_peak_memory(
        dtype_bytes=2, pool={"parked_cap": 0, "drain_cliff": drain_cliff, "drain_count": drain_count})


def test_without_a_queue_the_allocator_holds_the_live_peak():
    ap = _walk(0)
    assert ap.held_peak_bytes == ap.peak_bytes, (ap.held_peak_bytes, ap.peak_bytes)


def test_the_queue_holds_the_dead_until_its_cliff():
    live = _walk(0).peak_bytes                       # the chain: one 4 MB output beside its 4 MB input
    held = _walk(12 * MB).held_peak_bytes            # three dead outputs wait before the drain
    never = _walk(1 << 40).held_peak_bytes           # a cliff never reached: every output until the end
    assert live < held < never, (live, held, never)
    assert never == 8 * 4 * MB, never                # the eight outputs, none freed mid-run (the input is the caller's)


def test_the_count_drains_the_queue_too():
    assert _walk(1 << 40, drain_count=1).held_peak_bytes == _walk(0).held_peak_bytes


def test_no_rules_no_replay():
    ap = ActivationProfiler(_chain(4, weight_mb=1)).estimate_peak_memory(dtype_bytes=2)
    assert ap.held_peak_bytes is None
