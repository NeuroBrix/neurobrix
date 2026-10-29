"""The one device conversion of bfloat16 patterns to float16 (`ops.wrappers.bf16_to_fp16_gpu`) gives, in place and
out of place, the bits of the one host conversion (`nbxtensor.dtype.convert.bf16_bits_to_f16`) — the weight loader
narrows its raw bf16 buffers with it in place (`triton/weight_loader._half_in_place`)."""
import numpy as np
import pytest

from nbxtensor.dtype.convert import bf16_bits_to_f16


def _device():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _device(), reason="needs a device")


def _patterns():
    bits = np.arange(2 ** 16, dtype=np.uint32).astype(np.uint16)
    keep = ~np.isnan((bits.astype(np.uint32) << np.uint32(16)).view(np.float32))   # NaN payloads aside
    return bits[keep]


@pytest.mark.parametrize("in_place", [False, True])
def test_every_finite_pattern_is_the_host_rule(in_place):
    from nbxtensor.ops.wrappers import bf16_to_fp16_gpu
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    bits = _patterns()
    src = NBXTensor.from_numpy(bits.view(np.int16).copy())
    out = None
    if in_place:
        out = NBXTensor(src.data_ptr(), tuple(src.shape), tuple(src.stride()), NBXDtype.float16, "cuda",
                        owns_data=False, device_idx=src._device_idx, base=src)
    got = bf16_to_fp16_gpu(src, out=out).numpy().view(np.uint16)
    want = bf16_bits_to_f16(bits).view(np.uint16)
    assert np.array_equal(got, want)
