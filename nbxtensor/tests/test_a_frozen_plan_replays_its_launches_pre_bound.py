"""A frozen plan's direct replay runs each launch as the driver resolved it once (`prepare_launch`, a BOUND action):
on Metal the pipeline, each pointer's (buffer, offset) and the packed scalars; on CUDA its own argument cells and one
cuLaunchKernel. A plan's addresses never move, so nothing is looked up at replay.

What this test does if the code were wrong:
- the Metal cell fails when a bound replay computes on the recorded CONTENTS instead of the live ones (its inputs are
  rewritten in place between the recording and the replay: a token, a refilled expert slot), when it gives other
  bytes than the live ops, or when it still looks an address up;
- the assert / print cells fail when a kernel with a trailing assert or print block is pre-bound (one a launch);
- the CUDA cells (a stand-in cuLaunchKernel: no card here) fail when the bound launch does not hand the recorded values
  in their cells, or when a launch `launch` would refuse is pre-bound instead of left live.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from nbxtensor.backend import frozen_plan as FP


def _metal():
    from nbxtensor.backend.runtime import gpu_backend
    if gpu_backend() != "metal":
        pytest.skip(f"the Metal bound replay; this backend is {gpu_backend()!r}")
    from nbxtensor.backend import launcher
    launcher.install()
    FP.install_seams()


def _write(t, values) -> None:
    """Rewrite a device tensor's contents in place (its address unchanged), as a new token or a refilled slot is."""
    from nbxtensor.memory.allocator import DeviceAllocator
    host = np.ascontiguousarray(values, dtype=np.float32)
    DeviceAllocator.memcpy(t.data_ptr(), host.ctypes.data, host.nbytes, 1)


def _record(run):
    from nbxtensor.memory.slab import SlabAllocator
    slab = SlabAllocator(64 << 20, 0)
    records, failed, broken = FP.record(run, slab, {})
    assert failed is None and broken is None, (failed, broken)
    return records, slab


def test_a_bound_replay_computes_on_the_live_contents_and_looks_nothing_up(monkeypatch):
    _metal()
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.backend.metal_device import runtime
    rng = np.random.default_rng(0)
    a0, b0 = rng.standard_normal(4096).astype(np.float32), rng.standard_normal(4096).astype(np.float32)
    a, b = NBXTensor.from_numpy(a0).to("cuda:0"), NBXTensor.from_numpy(b0).to("cuda:0")
    out = {}
    records, _slab = _record(lambda: out.setdefault("d", EW.mul(EW.add(a, b), a)))
    actions = [FP._bound(t, r) for t, r in records]
    assert [t for t, _r in actions].count(FP.BOUND) == [t for t, _r in records].count(FP.KERNEL) >= 2
    a1, b1 = rng.standard_normal(4096).astype(np.float32), rng.standard_normal(4096).astype(np.float32)
    _write(a, a1), _write(b, b1)
    looked = []
    rt = runtime()
    real = rt.buffer_for_pointer
    monkeypatch.setattr(rt, "buffer_for_pointer", lambda p: looked.append(p) or real(p))
    for tag, rec in actions:
        FP._run_action(tag, rec)
    got = out["d"].to_cpu().numpy()
    monkeypatch.undo()
    live = EW.mul(EW.add(a, b), a).to_cpu().numpy()
    assert got.tobytes() == live.tobytes()                       # the replay computed on the rewritten inputs
    assert not np.array_equal(got, (a0 + b0) * a0)
    assert looked == []                                          # nothing resolved at replay


@pytest.mark.parametrize("layout", ["assert_layout", "print_layout"])
def test_a_kernel_with_an_assert_or_print_block_stays_live(layout):
    """A kernel whose emitter appended an assert or print block (one a launch, read after it) is never pre-bound: the
    Metal driver's prepare_launch leaves it to the live launch."""
    _metal()
    from nbxtensor.backend.launcher import active_driver
    trailing = SimpleNamespace(print_layout=None, assert_layout=None, exposes_addresses=False, reads_addresses=False,
                               cross_tg_barrier=False)
    setattr(trailing, layout, object())
    assert active_driver().prepare_launch(object(), (1, 1, 1), (32, 1, 1), 0, [("ptr", 1)], types=["*fp32"],
                                          trailing=trailing) is None


def _cuda_driver(counts, holds=True):
    from nbxtensor.backend import launcher as L
    calls = []

    def launch(function, gx, gy, gz, bx, by, bz, shared, stream, arr, extra):
        cells = [ctypes.cast(arr[i], ctypes.POINTER(ctypes.c_uint64)).contents.value for i in range(3)]
        calls.append(((gx, gy, gz, bx, by, bz), shared, stream, cells))
        return 0

    driver = L.CudaDriver.__new__(L.CudaDriver)
    driver.lib = SimpleNamespace(cuLaunchKernel=launch)
    driver._param_counts = counts
    driver._check = lambda rc, what: (_ for _ in ()).throw(RuntimeError(f"{what} {rc}"))
    return driver, calls


def test_a_cuda_bound_launch_is_one_call_with_its_recorded_cells(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "holds", staticmethod(lambda p: True))
    fn = ctypes.c_void_p(0x1234)
    driver, calls = _cuda_driver({0x1234: 3})
    params = [("ptr", 0xA000), ("u64", 7), ("ptr", 0xB000)]
    bound = driver.prepare_launch(fn, (4, 2), (128,), 0, params)
    bound.run(0)
    bound.run(0)
    assert calls == [((4, 2, 1, 128, 1, 1), 0, 0, [0xA000, 7, 0xB000])] * 2


@pytest.mark.parametrize("case", ["count", "foreign", "trailing", "empty"])
def test_a_cuda_launch_launch_would_refuse_stays_live(monkeypatch, case):
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "holds", staticmethod(lambda p: case != "foreign"))
    fn = ctypes.c_void_p(0x1234)
    driver, _calls = _cuda_driver({0x1234: 2 if case == "count" else 3})
    params = [("ptr", 0xA000), ("u64", 7), ("ptr", 0xB000)]
    grid = (0, 1) if case == "empty" else (4, 2)
    trailing = object() if case == "trailing" else None
    assert driver.prepare_launch(fn, grid, (128,), 0, params, trailing=trailing) is None
