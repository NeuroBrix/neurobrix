"""Every op in INPLACE_SAFE_UNARY gives the same answer writing into its input.

`OpLevelTilingEngine.INPLACE_SAFE_UNARY` claims each of these kernels is strictly
element-wise with matched offsets -- one `tl.load(input_ptr + offset)`, one
`tl.store(output_ptr + offset)`, no cross-lane read -- so one pointer passed twice
is correct. That claim is a COST-AND-EQUIVALENCE sentence of exactly the kind this
project has learned not to leave unpinned: nothing re-checks it when a kernel is
rewritten to read a neighbour, and the failure is silent wrong numbers rather than
a crash.

So each op runs both ways on the same input and the results are compared BITWISE.
Not `allclose`: these are the same kernel over the same values in the same order,
so any difference at all is the in-place path reading something it has already
overwritten, and a tolerance would hide exactly that.

NO RIG DOOR. `tests/unit/kernels/_rig.py` guards cells that MEASURE, because a
timing taken beside another process is a different measurement. This cell times
nothing; a numerical identity is the same on a busy card as an idle one.

SHAPES: `n = 3 * 1024 + 7` spans four BLOCK_SIZE=1024 blocks and ends mid-block, so
the tail mask is exercised -- at a block-aligned length every lane is valid and a
masking bug is invisible. The input spans negatives, positives and exact zero
because `leaky_relu`, `relu` and `elu` all branch on the sign, and a strictly
positive probe would run one side of every one of them.

SEEN RED: with the wrapper's `output = out if out is not None else ...` reverted
to an unconditional `NBXTensor.empty_like(x)`, all 14 fail.

WHAT THIS GATE CANNOT SEE, measured rather than assumed. Shifting `leaky_relu`'s
kernel to store at `output_ptr + offset + 1` leaves all 14 GREEN, because both
arms run the same kernel and an error on both sides is invisible to a differential
-- the contaminated-oracle shape. An earlier draft of this docstring claimed that
injection turned the gate red; it was written before the injection was run, and it
was wrong. Kernel correctness is not this file's question, but a purely
self-referential comparison is worth little on its own, so
`test_the_values_match_an_oracle_outside_the_engine` checks the three ops with an
unambiguous closed form against NumPy.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_an_elementwise_op_can_write_into_its_input.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.module.tiling_engine import OpLevelTilingEngine

pytestmark = pytest.mark.parametrize(
    "op_type,fn_name", sorted(OpLevelTilingEngine.INPLACE_SAFE_UNARY.items()))

N = 3 * 1024 + 7          # four blocks, last one partial: the tail mask runs


def _probe():
    rng = np.random.default_rng(20260918)
    x = rng.uniform(-4.0, 4.0, N).astype(np.float32)
    x[0], x[1], x[2] = 0.0, -0.0, -3.5      # the sign branches, explicitly
    return x


# The guard covers THE MACHINE, not the names. A catch-all around the import
# would turn a wrapper that lost its `out` parameter into the same skip as a
# rig-less checkout, and a skip is invisible in a count -- register 67, where
# five reds became five skips exactly this way. So the import is at module
# scope and unguarded; only the device is optional.
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers

_TOTAL = DeviceAllocator.device_count()
needs_a_card = pytest.mark.skipif(_TOTAL == 0, reason="needs a CUDA device")


@needs_a_card
def test_in_place_equals_out_of_place_bitwise(op_type, fn_name):
    fn = getattr(wrappers, fn_name)
    x = _probe()

    fresh = fn(NBXTensor.from_numpy(x.copy()))
    DeviceAllocator.sync_device()
    expected = fresh.numpy().copy()

    target = NBXTensor.from_numpy(x.copy())
    returned = fn(target, out=target)
    DeviceAllocator.sync_device()
    got = returned.numpy().copy()

    assert returned.data_ptr() == target.data_ptr(), (
        f"{fn_name}(out=target) returned a different buffer, so nothing was "
        f"saved and the caller's liveness proof bought nothing")
    np.testing.assert_array_equal(
        got, expected,
        err_msg=f"{op_type}: writing into the input changed the result")


@needs_a_card
def test_the_input_buffer_actually_carries_the_result(op_type, fn_name):
    """The point is the SAVING: `target` itself must hold the answer.

    A wrapper that quietly allocated and returned a new tensor would pass an
    equality check against the out-of-place path while saving nothing -- the
    allocation this whole mechanism exists to avoid would still happen.
    """
    fn = getattr(wrappers, fn_name)
    x = _probe()
    target = NBXTensor.from_numpy(x.copy())
    before = target.data_ptr()
    fn(target, out=target)
    DeviceAllocator.sync_device()
    after = target.numpy()

    assert target.data_ptr() == before
    # and it is no longer the input: an activation that left the buffer
    # untouched would pass every check above.
    assert not np.array_equal(after, x), (
        f"{op_type}: the input buffer is unchanged, so the kernel wrote "
        f"somewhere else")


# Ops whose definition admits exactly one reading, so an oracle can be written
# without re-deriving the kernel's own choices. `gelu`, `mish` and `silu` are
# excluded deliberately: their approximations differ between implementations and
# an oracle for them would be a second opinion, not a reference.
_ORACLES = {
    "relu": lambda v: np.maximum(v, 0.0),
    "leaky_relu": lambda v: np.where(v >= 0, v, 0.01 * v),
    "hardswish": lambda v: v * np.clip(v + 3.0, 0.0, 6.0) / 6.0,
}


@needs_a_card
def test_the_values_match_an_oracle_outside_the_engine(op_type, fn_name):
    """The differential above is blind to an error both arms share.

    Comparing the in-place path against the out-of-place path proves they agree;
    it cannot prove either is right, because one kernel serves both. So the ops
    with an unambiguous closed form are checked against NumPy, which knows
    nothing about this engine.
    """
    oracle = _ORACLES.get(fn_name)
    if oracle is None:
        pytest.skip(f"{fn_name} has no implementation-independent closed form")
    fn = getattr(wrappers, fn_name)
    x = _probe()
    target = NBXTensor.from_numpy(x.copy())
    fn(target, out=target)
    DeviceAllocator.sync_device()
    np.testing.assert_allclose(target.numpy(), oracle(x.astype(np.float64)),
                               rtol=1e-6, atol=1e-6)
