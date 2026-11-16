"""A session's fixed-size state (nbxtensor.memory.fixed_state) on the machine's accelerator (CUDA or Metal): allocated
at zero, written in place, reordered exactly as the library's own index gather (`index_select_wrapper`) and as numpy's
`take`, copied, rewritten, restored to the copied bytes, forgotten with its bytes given back to the allocator. On a host
with no device the suite's one mechanism (tests/_no_device.py) skips it by name. Torch-free.

What this test would do if the code were wrong: a buffer not zeroed on the card, a copy enqueued after the writes it must
precede, a reorder that reads a row it already overwrote, a buffer that moves, bytes not freed by `forget` -> red."""
import gc

import numpy as np
import pytest

from nbxtensor import device
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory import fixed_state as F
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.ops.index_wrappers import index_select_wrapper
from nbxtensor.tensor import NBXTensor

KINDS = {"recurrent": (["batch", 4, 128, 128], NBXDtype.float32, np.float32),
         "conv": (["batch", 640, 3], NBXDtype.bfloat16, np.uint16)}
OPS = ("l0", "l1", "l2")
BATCH = 4


@pytest.fixture
def card():
    if device.count() < 1:
        pytest.skip("needs a GPU device: none visible to this process")
    DA.set_device(0)
    return 0


def _host(t, npd):
    DA.device_synchronize(t.device_idx)
    out = np.empty(tuple(t.shape), dtype=npd)
    DA.memcpy(out.ctypes.data, t.data_ptr(), t.nbytes(), 2)
    return out


def _random(rng, shape, npd):
    if npd == np.uint16:     # bf16 bits of finite values: a float32's upper half
        return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return rng.standard_normal(shape).astype(npd)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_verb_keeps_the_declared_bytes_on_the_card(card, kind):
    shape, dtype, npd = KINDS[kind]
    d = F.StateDeclaration(kind, shape, dtype, OPS)
    live0 = DA.memory_allocated(card) - DA.pool_cached_bytes(card)
    s = F.FixedState(d, BATCH, card)
    assert s.held_bytes() == d.nbytes(BATCH)
    assert all(not _host(s.tensor(op), npd).any() for op in OPS)
    addrs = [s.tensor(op).data_ptr() for op in OPS]
    rng = np.random.default_rng(11)
    src = {op: _random(rng, d.shape(BATCH), npd) for op in OPS}
    for op in OPS:
        s.write(op, NBXTensor.from_numpy(src[op], dtype=dtype, device=card))
    assert all(np.array_equal(_host(s.tensor(op), npd), src[op]) for op in OPS)

    index = [3, 3, 0, 1]
    witness = {op: _host(index_select_wrapper(s.tensor(op), 0, NBXTensor.from_numpy(
        np.array(index, dtype=np.int64), device=card)), npd) for op in OPS}
    s.copy()
    s.reorder(index)
    for op in OPS:
        got = _host(s.tensor(op), npd)
        assert np.array_equal(got, witness[op]) and np.array_equal(got, np.take(src[op], index, axis=0))

    s.zero()
    assert all(not _host(s.tensor(op), npd).any() for op in OPS)
    s.restore()
    assert all(np.array_equal(_host(s.tensor(op), npd), src[op]) for op in OPS)
    assert [s.tensor(op).data_ptr() for op in OPS] == addrs and not s.has_copy

    s.forget()
    del witness
    gc.collect()
    DA.device_synchronize(card)
    live = DA.memory_allocated(card) - DA.pool_cached_bytes(card)     # parked blocks are the pool's, not held
    assert live - live0 < d.nbytes(BATCH)                              # the state's bytes went back to the allocator
