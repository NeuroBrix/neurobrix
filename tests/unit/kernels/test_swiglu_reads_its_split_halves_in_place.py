"""The fused SwiGLU reads the gate and up halves of a fused projection where they lie: (B, S, N) views of a (B, S, 2N)
buffer, row stride 2N, never copied. Copied, mochi-1-preview's (2, 7 950, 16 384) fp32 projection cost 2 x 497 MB the
graph never priced and its transformer overran the 8 GB rung by them (2026-10-28).

What this test would do if the code were wrong: a copy of either half (three allocations instead of one — red); the
second half read from the first's offset (values differ from silu(gate) * up — red)."""
import numpy as np
import pytest

from nbxtensor.tensor import DeviceAllocator, NBXTensor



def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


def test_the_halves_of_a_split_are_read_in_place_and_give_silu_gate_times_up(monkeypatch):
    from nbxtensor.ops import wrappers as W
    rng = np.random.default_rng(0)
    host = rng.standard_normal((2, 7, 64)).astype(np.float32)
    fused = NBXTensor.from_numpy(host).to_cuda(0)
    gate, up = fused[:, :, :32], fused[:, :, 32:]
    sizes = []
    orig = DeviceAllocator.malloc_cuda
    monkeypatch.setattr(DeviceAllocator, "malloc_cuda",
                        staticmethod(lambda n, *a, **k: (sizes.append(n), orig(n, *a, **k))[1]))
    out = W.swiglu_fused_wrapper(gate, up)
    assert sizes == [2 * 7 * 32 * 4], f"allocations beside the output: {sizes}"
    g, u = host[:, :, :32], host[:, :, 32:]
    ref = g / (1.0 + np.exp(-g)) * u
    assert tuple(out.shape) == (2, 7, 32)
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)
