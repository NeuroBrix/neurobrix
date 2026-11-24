"""The fused SwiGLU is one element-wise launch at the element-wise class's certified tile (`elementwise_launch.ew_meta`).
Its former fixed 16 x 1024 tile (128 elements a thread) spilled on an M4 Pro: 84 us a launch at a decode step's
(1, 5632), 22 launches a token, 1.8 ms of TinyLlama's 15.6 (the per-kernel GPU profile, 2026-11-13). Each element is
computed alone, so no tile can move a value.

The bytes are the CHAIN's (library f8ebc61, 2026-11-23, "a fusion computes what its chain computes"): silu(gate)
rounded to the stored dtype, then the product rounded at its store, no fused multiply-add. The former 16 x 1024 tile kept
silu in fp32 and differed from the chain in bf16 (11,203 of 41,472 elements), so it is no longer the witness: the
library's own two launches, `silu` then `mul`, are. The grid is handed as a function of the tile (library 82dcd22,
`elementwise_launch.ew_grid_of`): the launch evaluates it at the tile it is given.

What this test would do if the code were wrong: the wrapper launching a tile of its own instead of the certified
class's, or a grid that is not the class tile's (red on the configuration it asks for and the grid it evaluates to);
any output byte different from the chain's, at a decode row, a prefill block, a ragged width, strided halves and both
dtypes (red)."""
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


def _chain(gate, up):
    """The chain of ops the fusion replaces: silu(gate), rounded at its store, then mul(., up) — the library's own two
    element-wise launches."""
    from nbxtensor.ops import elementwise_wrappers as EW
    return EW.mul(EW.silu(gate), up)


@pytest.mark.parametrize("m,n,split", [(1, 5632, False), (23, 5632, False), (3, 1000, False), (40, 384, False),
                                       (7, 32, True)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_element_wise_launch_writes_the_chains_bytes(m, n, split, dtype):
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
    fused = EW.swiglu_fused_wrapper(gate, up)
    chain = _chain(gate, up)
    assert fused.nbx_dtype == chain.nbx_dtype
    new, old = fused.to(NBXDtype.float32).numpy(), chain.to(NBXDtype.float32).numpy()
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32)), int((new != old).sum())


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
    grid, kwargs = seen
    assert callable(grid), "the grid is a function of the launch's tile (ew_grid_of)"
    assert grid(kwargs) == (triton.cdiv(5632, meta["BLOCK_SIZE"]),)
    assert {k: kwargs[k] for k in meta} == meta and kwargs.get("enable_fp_fusion") is False
