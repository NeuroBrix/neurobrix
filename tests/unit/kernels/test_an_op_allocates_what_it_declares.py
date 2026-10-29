"""The cost an op declares (nbxtensor.ops._cost) is what its wrapper allocates on the device: each family run once on
the device with the allocator's pool emptied, the allocator's own peak above its level at entry against the
declaration's `peak_bytes` (R-NBXT 4: Prism will price from the declaration, so a declaration that is not the
execution is a defect). Shapes whose every buffer is a whole number of device pages, so no rounding blurs a byte."""
import numpy as np
import pytest

from nbxtensor.ops._cost import (Spec, elementwise_binary, elementwise_unary, native_layer_norm, reduction,
                                 rms_norm, softmax)


def _device():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _device(), reason="the allocator's peak needs a device")


@pytest.fixture(autouse=True)
def _this_machine(monkeypatch):
    """The kernels read their launch facts from the run's hardware profile (`set_hardware_profile`): this host's."""
    from neurobrix.core.prism.autodetect import load_default_profile
    from nbxtensor.ops import wrappers as W
    previous = W.get_hardware_profile()
    W.set_hardware_profile(load_default_profile())
    yield
    W.set_hardware_profile(previous)


@pytest.fixture(autouse=True)
def _no_pool(monkeypatch):
    """The measurement with the allocator's pool off: every buffer the op takes reaches the driver counter. With it
    on, a block an earlier test parked serves the op's output, and pool blocks count as live: the delta read 0 inside
    the GPU suite while each cell was green alone (the Dell, V100, 2026-10-29)."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA._maybe_init_pool()
    DA.device_synchronize(0)
    DA.empty_cache_pool()
    monkeypatch.setattr(DA, "_pool_enabled", False)


def _measure(fn):
    """The allocator's peak above its level during ONE call, after a first call (which may tune the kernel's
    configuration, the tuning's own buffers being no part of the op)."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    fn()
    DA.device_synchronize(0)
    base = DA._cuda_live_bytes.get(0, 0)
    DA._cuda_peak_bytes[0] = base
    out = fn()
    DA.device_synchronize(0)
    peak = DA._cuda_peak_bytes.get(0, 0) - base
    del out
    return peak


def _x(shape, dtype, transpose=False):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    a = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    t = NBXTensor.from_numpy(a)
    if dtype != "float32":
        t = t.to(getattr(NBXDtype, dtype))
    return t.transpose(0, 1) if transpose else t


def _spec(t):
    return Spec(tuple(t.shape), tuple(t.stride()), t.nbx_dtype.name)


R, C = 256, 1024


@pytest.mark.parametrize("transpose", [False, True])
def test_unary(transpose):
    from nbxtensor.ops import wrappers as W
    x = _x((C, R) if transpose else (R, C), "float16", transpose)
    assert _measure(lambda: W.exp(x)) == elementwise_unary(_spec(x)).peak_bytes


def test_binary_with_a_cast():
    from nbxtensor.ops import wrappers as W
    a, b = _x((R, C), "float16"), _x((R, C), "float32")
    want = elementwise_binary(_spec(a), _spec(b), "float32", (R, C)).peak_bytes
    assert _measure(lambda: W.add(a, b)) == want


@pytest.mark.parametrize("dim", [-1, 0])
def test_softmax(dim):
    from nbxtensor.ops import wrappers as W
    x = _x((R, C), "float16")
    assert _measure(lambda: W.softmax(x, dim)) == softmax(_spec(x), dim).peak_bytes


@pytest.mark.parametrize("transpose", [False, True])
def test_rms_norm(transpose):
    from nbxtensor.ops import wrappers as W
    x = _x((C, R) if transpose else (R, C), "float16", transpose)
    w = _x((x.shape[-1],), "float16")
    assert _measure(lambda: W.rms_norm(x, w)) == rms_norm(_spec(x), True).peak_bytes


def test_layer_norm():
    from nbxtensor.ops import wrappers as W
    x, w, b = _x((R, C), "float16"), _x((C,), "float16"), _x((C,), "float16")
    want = native_layer_norm(_spec(x), (C,), True, True).peak_bytes
    assert _measure(lambda: W.native_layer_norm(x, (C,), w, b)) == want


@pytest.mark.parametrize("dim", [-1, 0])
def test_sum(dim):
    from nbxtensor.ops import wrappers as W
    x = _x((R, C), "float16")
    assert _measure(lambda: W.sum_wrapper(x, dim)) == reduction(_spec(x), dim, False, "float16").peak_bytes
