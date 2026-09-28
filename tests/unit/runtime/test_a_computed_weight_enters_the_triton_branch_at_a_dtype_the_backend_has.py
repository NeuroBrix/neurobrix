"""A computed-at-runtime weight (the sincos 2D positional embedding, float64 from its formula) enters the Triton
branch's container with the values the formula gave, on every backend.

Why: on 2026-09-28 the Sana 4K container failed at the transformer's first add in both Triton modes on Metal at every
off-trace size (3072x4096, 2048x2560). The embedding was recomputed at the runtime resolution in float64, placed on the
device as float64 and cast to bf16 by a kernel; Apple GPUs have no float64 and the Metal lowering narrows it silently:
1 139 of 143 360 bits right, 3 255 non-finite. The ATen branch casts through torch on the host and renders.

Measured before this door: casting the real Sana 4K grids float64 -> float32 -> bf16 on the host gives the same bits as
torch's single float64 -> bf16 rounding, 0 of 41 287 680 elements differing at 3072x4096, 2048x2560 and 1024x1024.
Seen red on Metal with the device-side float64 cast (values garbage, non-finite) before the host cast."""
import numpy as np
import pytest
from neurobrix.core.runtime.graph_executor import computed_array_to_container
from neurobrix.kernels.autotune_certify import f32_to_bf16_bits
from neurobrix.kernels.nbx_tensor import NBXDtype, DeviceAllocator


def _grid():
    pos = np.arange(64, dtype=np.float64)[:, None] / 64.0
    freq = 1.0 / (10000 ** (np.arange(1120, dtype=np.float64) / 1120))
    return np.concatenate([np.sin(pos * freq), np.cos(pos * freq)], axis=1)[None]     # (1, 64, 2240) float64


@pytest.mark.parametrize("target", ["bfloat16", "float16", "float32"])
def test_the_values_are_the_formulas_whatever_the_backend(target):
    from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
    if _detect_gpu_backend() is None:
        pytest.skip("no device")
    g = _grid()
    out = computed_array_to_container(g, "cuda:0", target)
    assert out.nbx_dtype.name == target
    got = out.to(NBXDtype.float32).numpy().astype(np.float32); DeviceAllocator.device_synchronize()
    assert np.isfinite(got).all(), f"{int((~np.isfinite(got)).sum())} non-finite values in the placed embedding"
    if target == "bfloat16":
        want_bits = f32_to_bf16_bits(g.astype(np.float32))
        assert np.array_equal(f32_to_bf16_bits(got), want_bits), "the placed bf16 bits are not the formula's"
    else:
        assert np.allclose(got, g.astype(np.float32), rtol=0, atol=(2e-3 if target == "float16" else 0))
