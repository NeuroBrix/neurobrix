"""The fused SwiGLU is one element-wise launch at the element-wise class's certified tile (`elementwise_launch.ew_meta`).
Its former fixed 16 x 1024 tile (128 elements a thread) spilled on an M4 Pro: 84 us a launch at a decode step's
(1, 5632), 22 launches a token, 1.8 ms of TinyLlama's 15.6 (the per-kernel GPU profile, 2026-11-13). Each element is
computed alone, so no tile can move a value: the bytes are the former tile's.

What this test would do if the code were wrong: the wrapper launching a tile of its own instead of the certified
class's (red on the configuration it asks for); any output byte different from the former tile's, at a decode row, a
prefill block, a ragged width, strided halves and both dtypes (red)."""
import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


import triton
import triton.language as tl


@triton.jit
def _former_kernel(gate_ptr, up_ptr, output_ptr, M, N, stride_gm, stride_gn, stride_um, stride_un, stride_om, stride_on,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """The SwiGLU kernel as it was (2-D tile), kept here as the bytes' witness."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    gate = tl.load(gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un, mask=mask, other=0.0).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, result, mask=mask)


# the launcher routes a kernel by the module it was written in (`launcher._is_ours`): the witness is launched as the
# library's former kernel was, through the library's launcher
_former_kernel.fn.__module__ = "nbxtensor.ops.fused_moe"


def _former_tile(gate2d, up2d):
    """The 16 x 1024 tile, 4 warps: the bytes every certified run was made with."""
    from nbxtensor.ops import elementwise_wrappers as EW
    M, N = gate2d.shape
    out = NBXTensor.empty((M, N), dtype=gate2d._dtype, device=f"cuda:{gate2d._device_idx}")
    bn = min(triton.next_power_of_2(N), 1024)
    EW.set_device(gate2d)
    _former_kernel[(triton.cdiv(M, 16), triton.cdiv(N, bn))](
        gate2d, up2d, out, M, N, gate2d.stride(0), gate2d.stride(1), up2d.stride(0), up2d.stride(1),
        out.stride(0), out.stride(1), BLOCK_M=16, BLOCK_N=bn, num_warps=4)
    return out


@pytest.mark.parametrize("m,n,split", [(1, 5632, False), (23, 5632, False), (3, 1000, False), (40, 384, False),
                                       (7, 32, True)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_element_wise_launch_writes_the_former_tiles_bytes(m, n, split, dtype):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops import elementwise_wrappers as EW
    rng = np.random.default_rng(m * 7 + n)
    dt = getattr(NBXDtype, dtype)
    if split:                                   # the halves of a fused projection: row stride 2N
        fused = NBXTensor.from_numpy(rng.standard_normal((m, 2 * n)).astype(np.float32)).to_cuda(0).to(dt)
        gate, up = fused[:, :n], fused[:, n:]
    else:
        gate = NBXTensor.from_numpy(rng.standard_normal((m, n)).astype(np.float32)).to_cuda(0).to(dt)
        up = NBXTensor.from_numpy(rng.standard_normal((m, n)).astype(np.float32)).to_cuda(0).to(dt)
    new = EW.swiglu_fused_wrapper(gate, up).to(NBXDtype.float32).numpy()
    old = _former_tile(gate, up).to(NBXDtype.float32).numpy()
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32))


def test_the_wrapper_launches_at_the_element_wise_class_certificate(monkeypatch):
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.ops.elementwise_launch import ew_meta
    seen = []

    class Spy:
        def __getitem__(self, grid):
            seen.append(grid)
            return lambda *a, **k: seen.append(k)

    monkeypatch.setattr(EW, "silu_mul_split_kernel", Spy())
    gate = NBXTensor.from_numpy(np.ones((1, 5632), np.float32)).to_cuda(0)
    EW.swiglu_fused_wrapper(gate, gate)
    meta = ew_meta(5632)
    assert seen == [(triton.cdiv(5632, meta["BLOCK_SIZE"]),), meta]
