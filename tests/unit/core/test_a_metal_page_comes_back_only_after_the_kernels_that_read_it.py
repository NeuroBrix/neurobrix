"""Step 2 of the state engine on Metal (docs/design/state-engine.md §3.3 point 3, §4.3): a released page returns to the
free list only after the work that read it completed.

Triton branch: the Apple vendor files declare `memory.async_release: false` — a page comes back AT ONCE — and what makes
that true is the launch, not a fence: nbxtensor's launcher drains NeuroBrix's own queue before every kernel and waits
for the kernel after it (`_nbx_queue_drain()` / `_native().synchronize()`, nbxtensor/backend/triton_ext_driver.py), so
no kernel is running when the host or that queue touches a page. The test that holds the declaration launches a kernel
long enough to be caught running and reads its output from the host at the instant the launch returns: complete, or
the declaration is false. Seen red with the post-launch synchronize() made a no-op (0 of 4096 values written). The day
the launch stops waiting, the backend's fence (triton/state_fence.py, docs/nbxtensor/BACKENDS.md) becomes mandatory.

Compiled branch: through MPS events (core/state/mps_fence.py). A backend without a fence is refused by name.
"""
import ctypes
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from neurobrix.core.state.pages import PagePool, PageTable
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
            assert memory["async_release"] is False, f"{f.name}: a page is released at once on Metal (the launch waits)"
    pool = PagePool(num_pages=2, tokens_per_page=16, async_release=False)
    t = PageTable(pool)
    t.reserve(32)
    t.free()
    assert pool.free_pages == 2 and pool.pending_pages == 0


def test_a_triton_launch_on_metal_returns_only_once_its_kernel_completed():
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from nbxtensor.backend.launcher import launch
    from nbxtensor.tensor import NBXTensor

    @triton.jit
    def _spin(out_ptr, steps, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = off.to(tl.float32) * 0.0
        for _ in range(steps):
            x = x * 0.5 + 1.0
        tl.store(out_ptr + off, x)

    launch(_spin, (_N // _BLOCK,), NBXTensor.from_numpy(np.zeros(_N, dtype=np.float32)), 8, BLOCK=_BLOCK)  # compiled
    out = NBXTensor.from_numpy(np.zeros(_N, dtype=np.float32))
    launch(_spin, (_N // _BLOCK,), out, 200000, BLOCK=_BLOCK)
    # The host's view of the buffer, read with no synchronize of its own (numpy() would copy through a path that may wait).
    got = np.ctypeslib.as_array((ctypes.c_float * _N).from_address(out.data_ptr())).copy()
    assert int((got == 2.0).sum()) == _N, (
        f"{int((got == 2.0).sum())} of {_N} values written when the launch returned: the kernel was still running, so "
        f"memory.async_release: false on Apple is false — keep the post-launch wait, or release pages past the "
        f"backend's fence (triton/state_fence.py)")


def test_a_backend_without_a_fence_is_refused_by_name():
    from neurobrix.triton.state_fence import MetalStateFence
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: the Apple backend has no fence"):
        MetalStateFence(object())


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
