"""Step 2 of the state engine on Metal (docs/design/state-engine.md §3.3 point 3, §4.3): a released page returns to the
free list only after the work that read it completed.

Triton branch: the Apple vendor files declare `memory.async_release: false` — a page comes back AT ONCE — and what makes
that true is the order of every write that can reuse it, not a fence: nbxtensor runs every kernel and blit on one
serial stream (`backend/metal_dispatch.py`), so a later one runs after the kernels that read the page, and the host
writes through a copy that flushes that stream first. (Until 2026-11-12 it was true because every launch waited for
the device; that wait is gone.) The test that holds the declaration rewrites from the host, at once, a page a kernel
long enough to be caught running is still reading: the kernel must read the old contents, or the declaration is false
and the backend's fence (nbxtensor.memory.pages.MetalStateFence, nbxtensor/docs/BACKENDS.md) becomes mandatory.

Compiled branch: through MPS events (core/state/mps_fence.py). A backend without a fence is refused by name.
"""
import ctypes
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from nbxtensor.memory.pages import PagePool, PageTable
import nbxtensor.backend.runtime as _nbx_runtime  # the module the moved names live in (M2 cut of tensor.py)



def _has_metal():
    nbx_tensor = pytest.importorskip("nbxtensor.tensor")
    try:
        return _nbx_runtime.gpu_backend() == "metal"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="no Apple GPU here")

SPIN = """
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float x = i;
  for (int k = 0; k < 4000000; k++) { x = x * 1.0000001f + 0.5f; }
  out[i] = x;
}
"""


def _wait(fence, pool, deadline_s=30.0):
    t = time.time()
    while not fence.drain(pool):
        assert time.time() - t < deadline_s, "the fence never completed"
        time.sleep(0.001)


_APPLE = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors" / "apple"
_N, _BLOCK = 4096, 256


def test_the_apple_vendor_files_release_a_page_at_once():
    for f in sorted(_APPLE.glob("*.yml")):
        memory = (yaml.safe_load(f.read_text()) or {}).get("memory") or {}
        if "async_release" in memory:
            assert memory["async_release"] is False, f"{f.name}: a page is released at once on Metal (one serial stream)"
    pool = PagePool(num_pages=2, tokens_per_page=16, async_release=False)
    t = PageTable(pool)
    t.reserve(32)
    t.free()
    assert pool.free_pages == 2 and pool.pending_pages == 0


def test_a_page_a_running_kernel_reads_is_rewritten_only_after_it():
    """The page a kernel is still reading, released and rewritten at once from the host through the engine's copy
    path (`DeviceAllocator.memcpy`, what a page refill takes): the kernel reads the OLD contents. Every write that can
    reuse a page is ordered after the kernels before it: a kernel or a blit by the one serial stream they share, the
    host because a copy flushes that stream first."""
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from nbxtensor.backend.launcher import launch
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.tensor import NBXTensor

    @triton.jit
    def _slow_copy(src_ptr, out_ptr, steps, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = off.to(tl.float32) * 0.0
        for _ in range(steps):
            x = x * 0.5 + 1.0                        # 2.0 after enough steps: the time a kernel takes to run
        tl.store(out_ptr + off, tl.load(src_ptr + off) + x - 2.0)

    page = NBXTensor.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    out = NBXTensor.from_numpy(np.zeros(_N, dtype=np.float32))
    launch(_slow_copy, (_N // _BLOCK,), page, out, 8, BLOCK=_BLOCK)                    # compiled
    launch(_slow_copy, (_N // _BLOCK,), page, out, 200000, BLOCK=_BLOCK)               # still running on return
    refill = np.full(_N, -1.0, dtype=np.float32)
    DeviceAllocator.memcpy(page.data_ptr(), refill.ctypes.data, refill.nbytes, kind=1)  # the page, rewritten at once
    got = out.to_cpu().numpy()
    assert int((got == 7.0).sum()) == _N, (
        f"{int((got == 7.0).sum())} of {_N} values read the page's old contents: the host rewrote a page a kernel was "
        f"still reading, so memory.async_release: false on Apple is false — release pages past the backend's fence "
        f"(nbxtensor.memory.pages.MetalStateFence)")


def test_the_compiled_branch_frees_a_page_past_its_mps_event():
    import torch
    from neurobrix.core.state.mps_fence import MpsStateFence
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    fence = MpsStateFence(torch)
    pool = PagePool(num_pages=1, tokens_per_page=16, async_release=True)
    t = PageTable(pool)
    t.reserve(16)
    a = torch.randn(4096, 4096, device="mps")
    for _ in range(20):
        a = a @ a * 1e-3                                           # queued work that "reads" the page
    t.free(fence=fence.record())
    assert fence.drain(pool) == 0, "the matmuls are still queued: the page stays in the release queue"
    _wait(fence, pool)
    assert pool.free_pages == 1
