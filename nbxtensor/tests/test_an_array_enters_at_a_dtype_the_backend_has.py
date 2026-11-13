"""`NBXTensor.from_numpy_as(arr, dtype, device)`: a float64 array (a positional embedding from its formula) enters as
a tensor of `dtype` with the values the formula gave, on every backend; `new_tensor` places values at a tensor's own
dtype and card.

Why: on 2026-09-28 the Sana 4K container failed at the transformer's first add in both Triton modes on Metal at every
off-build size (3072x4096, 2048x2560). The embedding was recomputed at the runtime resolution in float64, placed on the
device as float64 and cast to bf16 by a kernel; Apple GPUs have no float64 and the Metal lowering narrows it silently:
1 139 of 143 360 bits right, 3 255 non-finite. The ATen branch casts through torch on the host and renders.

Measured before this door: casting the real Sana 4K grids float64 -> float32 -> bf16 on the host gives the same bits as
torch's single float64 -> bf16 rounding, 0 of 41 287 680 elements differing at 3072x4096, 2048x2560 and 1024x1024.
Seen red on Metal with the device-side float64 cast (values garbage, non-finite) before the host cast. The rule lived
in the engine (`computed_array_to_container`) until 2026-11-12; the library holds it now."""
import numpy as np
import pytest

from nbxtensor.backend.runtime import gpu_backend
from nbxtensor.dtype.convert import f32_to_bf16_bits
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor

pytestmark = pytest.mark.skipif(gpu_backend() is None, reason="no device")


def _grid():
    pos = np.arange(64, dtype=np.float64)[:, None] / 64.0
    freq = 1.0 / (10000 ** (np.arange(1120, dtype=np.float64) / 1120))
    return np.concatenate([np.sin(pos * freq), np.cos(pos * freq)], axis=1)[None]     # (1, 64, 2240) float64


@pytest.mark.parametrize("target", ["bfloat16", "float16", "float32"])
def test_the_values_are_the_formulas_whatever_the_backend(target):
    g = _grid()
    out = NBXTensor.from_numpy_as(g, target, device=0)
    assert out.nbx_dtype.name == target
    got = out.to(NBXDtype.float32).numpy().astype(np.float32); DeviceAllocator.device_synchronize()
    assert np.isfinite(got).all(), f"{int((~np.isfinite(got)).sum())} non-finite values in the placed embedding"
    if target == "bfloat16":
        want_bits = f32_to_bf16_bits(g.astype(np.float32))
        assert np.array_equal(f32_to_bf16_bits(got), want_bits), "the placed bf16 bits are not the formula's"
    else:
        assert np.allclose(got, g.astype(np.float32), rtol=0, atol=(2e-3 if target == "float16" else 0))


def test_new_tensor_takes_the_tensor_s_dtype_and_card_and_leaves_the_current_device():
    like = NBXTensor.zeros((2, 3), dtype=NBXDtype.float16, device="cuda:0")
    before = DeviceAllocator.get_device()
    t = like.new_tensor(np.arange(6, dtype=np.float32).reshape(2, 3) / 4)
    assert t.nbx_dtype == NBXDtype.float16 and t.device_idx == like.device_idx
    assert DeviceAllocator.get_device() == before
    assert np.array_equal(t.to(NBXDtype.float32).numpy(), np.arange(6, dtype=np.float32).reshape(2, 3) / 4)
