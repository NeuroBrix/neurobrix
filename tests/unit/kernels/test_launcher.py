"""The NeuroBrix launcher (docs/internal/metal_launcher_contract.md): Triton
compiles, the dispatch layer launches through a vendor driver of its own.

Two gates: (1) byte identity with upstream's `kernel[grid](...)` on a house
kernel; (2) the launch path imports no torch — measured in a subprocess, the
same instrument as the end-of-run proof (tools/r33_sys_modules_probe.py)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[3] / "src"


def _cuda():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda(), reason="CUDA device required")


def _nbx(arr):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _host(t, np_dtype):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = np.empty(t._nbytes, dtype=np.uint8)
    DeviceAllocator.memcpy(buf.ctypes.data, t.data_ptr(), t._nbytes, 2)
    return buf.view(np_dtype).reshape(t.shape)


def test_launch_is_byte_identical_to_upstream_on_a_house_kernel():
    from neurobrix.kernels.launcher import launch
    from neurobrix.kernels.ops.fft_op import scale_kernel
    rng = np.random.default_rng(7)
    n = 3001                                              # a masked tail
    r0, i0 = rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    a_r, a_i = _nbx(r0), _nbx(i0)
    b_r, b_i = _nbx(r0), _nbx(i0)
    grid = ((n + 1023) // 1024,)
    scale_kernel[grid](a_r, a_i, n, 0.37, BLOCK_SIZE=1024)                 # upstream launcher
    launch(scale_kernel, grid, b_r, b_i, n, 0.37, BLOCK_SIZE=1024)         # NeuroBrix launcher
    assert _host(a_r, np.float32).tobytes() == _host(b_r, np.float32).tobytes()
    assert _host(a_i, np.float32).tobytes() == _host(b_i, np.float32).tobytes()
    assert not np.array_equal(_host(b_r, np.float32), r0)                  # it did run


def test_the_launch_path_imports_no_torch():
    code = r"""
import sys
import numpy as np
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels.launcher import launch
from neurobrix.kernels.ops.fft_op import scale_kernel
DeviceAllocator.set_device(0)
n = 2048
r = NBXTensor.from_numpy(np.arange(n, dtype=np.float32)); i = NBXTensor.from_numpy(np.ones(n, dtype=np.float32))
launch(scale_kernel, (2,), r, i, n, 2.0, BLOCK_SIZE=1024)
buf = np.empty(n * 4, dtype=np.uint8); DeviceAllocator.memcpy(buf.ctypes.data, r.data_ptr(), n * 4, 2)
assert buf.view(np.float32)[5] == 10.0, buf.view(np.float32)[:8]
print("torch" in sys.modules)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                         env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0",
                              "HOME": str(Path.home())})
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == "False", f"the launch path pulled torch:\n{out.stderr[-800:]}"


def test_our_specialisation_matches_triton_s_binder():
    """The parity proof of the Python binder: for tensors (aligned, unaligned),
    ints (1, 16, 17, 0, 2**31, -5, 2**63), floats, bools, None and a tl.dtype,
    our (type, attr) equals what Triton's C++ specialiser returns."""
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import BaseBackend
    import triton.language as tl
    from neurobrix.kernels.launcher import specialize_arg
    t16 = _nbx(np.zeros(64, dtype=np.float16))
    ti = _nbx(np.zeros(64, dtype=np.int64))
    cases = [t16, ti, 1, 16, 17, 0, 2 ** 31, -5, 2 ** 40 * 16, 2 ** 63, 0.37, True, None]
    for arg in cases:
        for specialize, align in ((True, True), (False, True), (True, False)):
            theirs = tuple(native_specialize_impl(BaseBackend, arg, False, specialize, align))
            ours = specialize_arg(arg, specialize, align)
            assert ours == theirs, (arg, specialize, align, ours, theirs)
    assert specialize_arg(tl.float16) == ("constexpr", tl.float16)


def test_our_binder_returns_triton_s_triple_on_a_real_kernel():
    from triton.compiler import make_backend
    from triton.runtime.jit import create_function_from_signature
    from neurobrix.kernels.launcher import nbx_binder, target
    from neurobrix.kernels.ops.fused_moe import fused_moe_kernel
    import triton.language as tl
    backend = make_backend(target())
    theirs_fn = create_function_from_signature(fused_moe_kernel.signature, fused_moe_kernel.params, backend)
    a = _nbx(np.zeros((16, 64), dtype=np.float16)); p = _nbx(np.zeros(4, dtype=np.int64))
    o = _nbx(np.zeros((32, 64), dtype=np.float16)); w = _nbx(np.zeros(32, dtype=np.float32))
    s = _nbx(np.zeros(64, dtype=np.int32)); e = _nbx(np.zeros(4, dtype=np.int32)); n = _nbx(np.zeros(1, dtype=np.int32))
    args = (a, p, o, w, s, e, n, 64, 64, 64, 32, 64, 1, 64, 1, 64, 1)
    kw = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, MUL_ROUTED_WEIGHT=False,
              top_k=2, compute_type=tl.float16, TOPK_DIVIDE=True, num_warps=4, num_stages=2)
    b1, s1, o1 = theirs_fn(*args, **kw)
    b2, s2, o2 = nbx_binder(fused_moe_kernel, args, kw)
    assert list(b1.keys()) == list(b2.keys()) and all(b1[k] is b2[k] or b1[k] == b2[k] for k in b1)
    assert [tuple(x) for x in s1] == [tuple(x) for x in s2], (s1, s2)
    assert o1 == o2
