"""Step 2 of the state engine on Metal (docs/design/state-engine.md §3.3 point 3, §4.3): a released page returns to the
free list only after the work that read it completed — on the Triton branch through the Apple backend's fence, on the
compiled branch through MPS events.

What this test does if the code is wrong: with a kernel still running on the page, the page must stay in the release
queue (granting it would hand the next writer memory the GPU is still reading, which on MPS gives a wrong answer and
no error); once the device reports completion the drain must give it back. A backend without a fence is refused by name
(no silent synchronous fallback).
"""
import time

import pytest

from neurobrix.core.state.pages import PagePool, PageTable
import nbxtensor.backend.runtime as _nbx_runtime  # the module the moved names live in (M2 cut of tensor.py)



def _has_metal():
    nbx_tensor = pytest.importorskip("nbxtensor.tensor")
    try:
        return _nbx_runtime._detect_gpu_backend() == "metal"
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


def test_the_triton_branch_frees_a_page_past_the_backends_fence():
    from triton_apple_backend import metal_native as m
    from neurobrix.triton.state_fence import MetalStateFence
    fence = MetalStateFence(m)
    pool = PagePool(num_pages=2, tokens_per_page=16, async_release=True)
    t = PageTable(pool)
    t.reserve(32)
    kernel = m.load_metallib(m.compile_source(SPIN, "safe")).get_function("spin")
    buf = m.alloc(4 * 1024)
    kernel(buf, threads=[1024, 1, 1], group_size=[256, 1, 1])     # the "read" of the pages
    t.free(fence=fence.record())
    assert fence.drain(pool) == 0 and pool.free_pages == 0, "the kernel is still running: no page may come back"
    _wait(fence, pool)
    assert pool.free_pages == 2 and pool.pending_pages == 0


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
