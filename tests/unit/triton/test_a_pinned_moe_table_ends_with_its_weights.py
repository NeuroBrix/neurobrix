"""A pinned MoE pointer table ends with the weights it pins.

On Metal a pointer table PINS: it holds its expert tensors and one whole-allocation wrap per arena
(`PtrTables.pins`, `triton_ext_driver.pinned_addresses`), because the addresses in the table are valid exactly
while those are held. Two of its lifetimes had no end:

* a CACHED table (resident weights) lived until LRU eviction, 256 tables later. So unloading a component left
  its weights pinned: the tensors could not be freed and the wrap kept the arena mapped. Under
  `layer_streaming` every unloaded segment stayed in memory beside the next one. Measured on the Mac
  (deepseek-moe-16b-chat, gate cell 2026-10-04 07:00, 4 segments "one resident at a time"): while the second
  segment loaded, swap went 3.4 -> 10.2 GB in 30 s and the guard killed the run;
* a PER-CALL table (weights promoted for one call) was dropped with `del tables`: its pins never exited, so the
  driver kept the wrap of every promoted allocation, and a stale pin count on addresses the allocator reissues.

Now the per-call table releases its pins when the call returns, and unloading weights releases every cached
table that pins (`MemoryManager.unload_weights`, the unload boundary every strategy goes through). On CUDA a
table pins nothing, so nothing changes there.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor  # noqa: E402

E, K_, H, F, T = 8, 2, 64, 32, 5


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("a table pins on the Metal driver only; on CUDA it holds integers")
        from neurobrix.triton import triton_ext_driver as drv
        return drv
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


def _host(arr):
    t = NBXTensor.empty_cpu(arr.shape, NBXDtype.float16)
    ctypes.memmove(t.data_ptr(), np.ascontiguousarray(arr).ctypes.data, arr.nbytes)
    return t


@pytest.fixture
def unload(monkeypatch):
    """`MemoryManager.unload_weights`, with its device sync bound to the NBX allocator. `device_sync` asks which
    engine the PROCESS loaded and takes torch's when torch is there: a --triton run never loads torch, but a test
    session does as soon as one other test imports it, and "cuda:0" then reaches `torch.cuda` on a Mac. What is
    under test is the pin release at the unload boundary, not that dispatch."""
    from neurobrix.core.memory import MemoryManager, manager
    monkeypatch.setattr(manager, "device_sync", lambda dev: DeviceAllocator.sync_device())
    monkeypatch.setattr(manager, "device_empty_cache", lambda dev: None)
    return MemoryManager.unload_weights


def _live_tensors_in(spans):
    """How many live NBXTensors point inside these (base, nbytes) allocations. The allocator's own tables cannot
    answer "was it freed": it reissues a freed address to the next allocation of that size."""
    import gc
    n = 0
    for o in gc.get_objects():
        if isinstance(o, NBXTensor):
            try:
                p = o.data_ptr()
            except Exception:                          # noqa: BLE001
                continue
            n += any(b <= p < b + size for b, size in spans)
    return n


def _fused_call(where, seed):
    """One `execute_moe_fused` on stacked slabs, resident on the device or on the host (promoted per call).
    Returns the slabs' dict, as an executor's weights dict would hold them, and the output."""
    from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
    from neurobrix.triton.moe import execute_moe_fused
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((T, H)) * 0.5).astype(np.float16)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    w_in = (rng.standard_normal((E, H, 2 * F)) * 0.1).astype(np.float16)
    w_out = (rng.standard_normal((E, F, H)) * 0.1).astype(np.float16)
    hs, gs = NBXTensor.from_numpy(h), NBXTensor.from_numpy(probs)
    make = NBXTensor.from_numpy if where == "resident" else _host
    weights = {"in": make(w_in), "out": make(w_out)}
    attrs = {"num_experts": E,
             "stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out",
                                 "ffn_dim": F, "input_linear_in_axis": 0,
                                 "gate_offset": 0, "output_linear_in_axis": 0},
             "expert_gate_weight_ids": [], "expert_up_weight_ids": [], "expert_down_weight_ids": []}
    gate, up, down = expert_weight_lists(attrs, weights.get)
    out = execute_moe_fused(gs, hs, gate, up, down, top_k=K_, num_experts=E,
                            cache_key=f"pinned-lifetime-{where}-{seed}")
    got = np.asarray(out.numpy(), dtype=np.float64)
    assert np.isfinite(got).all()
    return weights, got


def test_a_per_call_table_releases_its_pins_when_the_call_returns():
    drv = _metal_or_skip()
    wraps, counts = set(drv._RESIDENT_WRAPS), dict(drv._PIN_COUNTS)
    weights, _ = _fused_call("host", seed=41)
    left = set(drv._RESIDENT_WRAPS) - wraps
    assert not left, (
        f"the call returned and the driver still keeps the wrap of {len(left)} allocation(s) it promoted for "
        f"that call: a per-call table's pins never exited")
    assert dict(drv._PIN_COUNTS) == counts, "the call returned and left addresses pinned"
    del weights


def test_unloading_the_weights_releases_the_cached_tables_that_pin_them(unload):
    drv = _metal_or_skip()
    from neurobrix.triton import moe as M
    weights, _ = _fused_call("resident", seed=43)
    spans = [(t.data_ptr(), DeviceAllocator._cuda_ptr_size[t.data_ptr()]) for t in weights.values()]
    assert all(b in drv._RESIDENT_WRAPS for b, _ in spans), "a resident table that pins nothing is an accident"
    assert _live_tensors_in(spans) > 2, "the cached table's pins hold the expert views: the instrument sees them"

    unload(weights)                                    # what every strategy's unload goes through

    assert not any(t.pins for t in M._ptr_cache.values()), "a cached table still pins after the unload"
    assert not drv._RESIDENT_WRAPS, (
        f"the weights were unloaded and the driver still keeps {len(drv._RESIDENT_WRAPS)} whole-allocation "
        f"wrap(s): the arena stays mapped beside whatever loads next")
    assert not drv._PIN_COUNTS and not drv._PINNED_WRAPS, "the unload left addresses pinned"
    live = _live_tensors_in(spans)
    assert live == 0, f"{live} tensor(s) of the unloaded slabs are still alive: a pin still holds them"


def test_a_table_rebuilt_after_an_unload_reads_the_new_weights(unload):
    """The cache is keyed by the weights' addresses, and an allocator reissues addresses: a table that survived
    an unload would be found again by a later load at the same addresses, with pins on the tensors that died.
    A guard on the fixed behaviour (the same values after a reload), not the red test: before the fix the dead
    tensors were never freed, so their addresses were never reissued."""
    _metal_or_skip()
    weights, first = _fused_call("resident", seed=47)
    unload(weights)
    weights, second = _fused_call("resident", seed=47)  # the same sizes: the allocator may hand back the addresses
    unload(weights)
    assert np.array_equal(first, second)
