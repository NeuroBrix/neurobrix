"""triton-ext's launch ABI, exercised through NeuroBrix's launcher.

The defect this pins, measured 2026-09-16 on M4 Pro (campaign ledger,
`abi_scalars.py`): NeuroBrix launched triton-ext-compiled kernels through the
FORK's driver. The fork binds each scalar to its own Metal argument slot;
triton-ext packs every scalar into ONE buffer and binds it after the pointers.
Bound the fork's way, the first scalar landed and the rest arrived as zero —

    kernel saw HW,C,K = [4096, 0, 0]   expected [4096, 180, 180]

— so every mask went false, every `tl.load` took its `other`, the `tl.store` was
fully masked, and the output buffer kept the zeros it was allocated with. No
exception anywhere. The same bare kernel passed on torch/mps tensors, which is
what flipped the attribution from triton-ext to us.

The second silent-zero, found while building the adapter: triton-ext dispatches
on its OWN command queue, not the one the allocator orders its copies against.
The kernel computed correctly and the host still read the initial zeros. Both
are correctness, not performance, so both are asserted here.

Skipped wherever triton-ext is not installed, which includes the Dell.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
pytest.importorskip("triton_apple_backend")

from neurobrix.triton.metal_backend import selected_metal_backend  # noqa: E402


@pytest.fixture(autouse=True)
def _force_ext(monkeypatch):
    monkeypatch.setenv("NEUROBRIX_METAL_BACKEND", "triton_ext")
    if selected_metal_backend() != "triton_ext":
        pytest.skip("triton-ext is not the selectable Metal backend here")


def _kernel():
    import triton.language as tl

    @triton.jit
    def axpy(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs,
                 tl.load(x_ptr + offs, mask=m) * alpha
                 + tl.load(y_ptr + offs, mask=m), mask=m)
    return axpy


def test_scalars_arrive_whole_and_the_host_sees_the_writes():
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.launcher import active_driver
    from neurobrix.triton.triton_ext_driver import TritonExtDriver

    assert isinstance(active_driver(), TritonExtDriver)

    n, BLOCK, alpha = 4096, 256, 2.5
    rng = np.random.default_rng(0)
    xh = rng.standard_normal(n).astype(np.float32)
    yh = rng.standard_normal(n).astype(np.float32)
    x, y = NBXTensor.from_numpy(xh), NBXTensor.from_numpy(yh)
    o = NBXTensor.from_numpy(np.zeros(n, dtype=np.float32))

    _kernel()[(triton.cdiv(n, BLOCK),)](x, y, o, n, alpha, BLOCK=BLOCK)

    got, ref = o.numpy(), xh * alpha + yh
    assert not (got == 0).all(), (
        "every element is exactly zero: either the scalars were bound the "
        "fork's way (masks all false, stores all masked) or the dispatch was "
        "never ordered against the host read")
    err = np.abs(got - ref).max() / max(1e-30, float(np.abs(ref).max()))
    assert err < 1e-5, f"max relative error {err:.3e}"


def test_a_parameter_the_allocator_does_not_own_is_refused_not_guessed():
    """A pointer has to be wrapped with its LENGTH, which only the allocator
    knows. An unknown pointer must be named, never bound at a guessed size."""
    from neurobrix.triton.triton_ext_driver import _buffer_for

    with pytest.raises(RuntimeError, match="does not record it"):
        _buffer_for(0xDEADB000, "*fp32")


def test_launching_without_parameter_types_is_refused():
    """The scalars share one packed buffer whose field offsets come from the
    Triton types. Without them the packing would be silently misaligned."""
    from neurobrix.triton.triton_ext_driver import TritonExtDriver

    with pytest.raises(RuntimeError, match="needs the Triton type"):
        TritonExtDriver.instance().launch(
            object(), (1, 1, 1), (32, 1, 1), 0, 0, [("i32", 1)], names=["n"])
