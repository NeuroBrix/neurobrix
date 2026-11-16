"""A session's fixed-size state (nbxtensor.memory.fixed_state), on each kind ('recurrent': float32 always; 'conv': the
last taps-1 input columns at the op's dtype): allocated at zero from a declaration, its bytes exactly the declaration's,
written in place, zeroed, reordered along the batch as an index gather, the one copy of a turn taken and restored,
forgotten; every refusal by name. Card-less: the allocator's direct operations are host memory here (the device run of
the same verbs is test_a_fixed_state_on_a_card_keeps_its_bytes_through_every_verb.py). Torch-free.

What this test would do if the code were wrong: a buffer not zeroed (the fake memory starts at 0xA5), a buffer that moves
on a write, a reorder that reads a row it already overwrote, a restore that gives back anything but the copied bytes, a
verb accepted on a forgotten state, a declaration of the wrong dtype or shape accepted -> red."""
import ctypes
import gc

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory import fixed_state as F
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.tensor import NBXTensor

KINDS = {   # kind -> (shape, dtype, the numpy container of its bits)
    "recurrent": (["batch", 2, 4, 3], NBXDtype.float32, np.float32),
    "conv": (["batch", 5, 3], NBXDtype.bfloat16, np.uint16),
}
OPS = ("layers.0.linear_attn", "layers.1.linear_attn", "layers.3.linear_attn")
BATCH = 3


@pytest.fixture
def host(monkeypatch):
    """The allocator's direct operations on host memory: fresh blocks hold 0xA5, the current device is recorded."""
    blocks, cur = {}, {"dev": 0}

    def malloc(nbytes, dev_idx=None, pooled=True):
        buf = ctypes.create_string_buffer(int(nbytes))
        ctypes.memset(buf, 0xA5, int(nbytes))
        blocks[ctypes.addressof(buf)] = (buf, cur["dev"] if dev_idx is None else dev_idx)
        return ctypes.addressof(buf)

    monkeypatch.setattr(DA, "malloc_direct", staticmethod(malloc))
    monkeypatch.setattr(DA, "free_direct", staticmethod(lambda ptr: blocks.pop(ptr, None)))
    monkeypatch.setattr(DA, "memcpy_direct", staticmethod(lambda d, s, n, k=3: ctypes.memmove(d, s, n)))
    monkeypatch.setattr(DA, "memset_direct", staticmethod(lambda p, v, n: ctypes.memset(p, v, n)))
    monkeypatch.setattr(DA, "sync_direct", staticmethod(lambda: None))
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: cur["dev"]))
    monkeypatch.setattr(DA, "set_device", staticmethod(lambda d: cur.__setitem__("dev", int(d))))
    monkeypatch.setattr(DA, "_interceptor", None)
    yield blocks
    gc.collect()


def _bits(t, np_dtype):
    return np.frombuffer(ctypes.string_at(t.data_ptr(), t.nbytes()), dtype=np_dtype).reshape(tuple(t.shape))


def _random(rng, shape, np_dtype):
    if np_dtype == np.uint16:
        return rng.integers(0, 1 << 16, size=shape, dtype=np.uint16)
    return rng.standard_normal(shape).astype(np_dtype)


def _state(kind, fence=None, devices=0):
    shape, dtype, _ = KINDS[kind]
    return F.FixedState(F.StateDeclaration(kind, shape, dtype, OPS), BATCH, devices, fence=fence)


def _fill(state, kind, rng):
    shape, dtype, npd = KINDS[kind]
    src = {}
    for op in OPS:
        src[op] = _random(rng, state.declaration.shape(BATCH), npd)
        state.write(op, NBXTensor.from_numpy(src[op], dtype=dtype))
    return src


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_state_is_allocated_at_zero_and_holds_the_declarations_bytes_exactly(host, kind):
    shape, dtype, npd = KINDS[kind]
    s = _state(kind)
    per_op = int(np.prod(shape[1:])) * BATCH * np.dtype(npd).itemsize
    assert s.declaration.nbytes(BATCH) == len(OPS) * per_op == s.held_bytes()
    assert sum(len(b) for b, _ in host.values()) == s.held_bytes()          # no byte allocated beyond the declaration
    for op in OPS:
        t = s.tensor(op)
        assert tuple(t.shape) == (BATCH, *shape[1:]) and t.nbx_dtype == dtype
        assert not _bits(t, npd).any()


def test_the_declared_bytes_of_the_first_hybrid_model_are_the_design_figures():
    rec = F.StateDeclaration("recurrent", ["batch", 48, 128, 128], "float32", [f"r{i}" for i in range(48)])
    conv = F.StateDeclaration("conv", ["batch", 10240, 3], "bfloat16", [f"c{i}" for i in range(48)])
    assert rec.nbytes(1) == 150_994_944 and conv.nbytes(1) == 48 * 10240 * 3 * 2
    assert rec.nbytes(4) == 4 * rec.nbytes(1)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_write_lands_in_place_in_its_own_buffer(host, kind):
    s = _state(kind)
    addrs = [s.tensor(op).data_ptr() for op in OPS]
    src = _fill(s, kind, np.random.default_rng(1))
    assert [s.tensor(op).data_ptr() for op in OPS] == addrs
    for op in OPS:
        assert np.array_equal(_bits(s.tensor(op), KINDS[kind][2]), src[op])


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_zero_clears_every_buffer_in_place(host, kind):
    s = _state(kind)
    addrs = [s.tensor(op).data_ptr() for op in OPS]
    _fill(s, kind, np.random.default_rng(2))
    s.zero()
    assert [s.tensor(op).data_ptr() for op in OPS] == addrs
    assert not any(_bits(s.tensor(op), KINDS[kind][2]).any() for op in OPS)


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("index", [[2, 0, 1], [1, 1, 0], [0, 0, 0], [0, 1, 2]])
def test_a_reorder_is_the_index_gather_along_the_batch_in_place(host, kind, index):
    s = _state(kind)
    src = _fill(s, kind, np.random.default_rng(3))
    addrs = [s.tensor(op).data_ptr() for op in OPS]
    before = sum(len(b) for b, _ in host.values())
    s.reorder(index)
    gc.collect()
    assert [s.tensor(op).data_ptr() for op in OPS] == addrs
    assert sum(len(b) for b, _ in host.values()) == before                  # the gather's twin is freed
    for op in OPS:
        assert np.array_equal(_bits(s.tensor(op), KINDS[kind][2]), np.take(src[op], index, axis=0))


@pytest.mark.parametrize("index", [[0, 1], [0, 1, 2, 0], [0, 1, 3], [0, -1, 1], [0, 1.0, 2], [True, 0, 1]])
def test_a_reorder_index_that_is_not_one_row_per_sequence_is_refused_by_name(host, index):
    with pytest.raises(ValueError, match="reorder index"):
        _state("recurrent").reorder(index)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_copy_then_writes_then_a_restore_gives_back_the_copied_bytes(host, kind):
    s = _state(kind)
    rng = np.random.default_rng(4)
    kept = _fill(s, kind, rng)
    addrs = [s.tensor(op).data_ptr() for op in OPS]
    s.copy()
    assert s.has_copy and s.copy_bytes() == s.held_bytes() == s.declaration.nbytes(BATCH)
    _fill(s, kind, rng)                                                       # the turn writes the state
    s.reorder([2, 2, 0])
    assert not np.array_equal(_bits(s.tensor(OPS[0]), KINDS[kind][2]), kept[OPS[0]])
    s.restore()
    assert [s.tensor(op).data_ptr() for op in OPS] == addrs
    for op in OPS:
        assert np.array_equal(_bits(s.tensor(op), KINDS[kind][2]), kept[op])
    assert not s.has_copy and s.copy_bytes() == 0                             # the turn is over: the copy is freed


def test_one_copy_is_kept_at_a_time_and_a_missing_copy_is_refused_by_name(host):
    s = _state("conv")
    with pytest.raises(RuntimeError, match="no copy is kept"):
        s.restore()
    with pytest.raises(RuntimeError, match="no copy is kept"):
        s.release_copy()
    s.copy()
    with pytest.raises(RuntimeError, match="a copy is already kept"):
        s.copy()
    s.release_copy()
    gc.collect()
    assert not s.has_copy and sum(len(b) for b, _ in host.values()) == s.held_bytes()


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_forgotten_state_frees_its_buffers_and_refuses_every_verb_by_name(host, kind):
    s = _state(kind)
    s.copy()
    src = NBXTensor.from_numpy(_random(np.random.default_rng(5), s.declaration.shape(BATCH), KINDS[kind][2]),
                               dtype=KINDS[kind][1])
    s.forget()
    gc.collect()
    assert s.forgotten and s.held_bytes() == 0 and s.copy_bytes() == 0
    assert sum(len(b) for b, _ in host.values()) == src.nbytes()               # only the source tensor is left
    for verb in (lambda: s.tensor(OPS[0]), lambda: s.write(OPS[0], src), s.zero, lambda: s.reorder([0, 1, 2]),
                 s.copy, s.restore, s.release_copy, s.forget):
        with pytest.raises(RuntimeError, match="is forgotten"):
            verb()


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_write_that_is_not_the_declarations_is_refused_by_name(host, kind):
    shape, dtype, npd = KINDS[kind]
    s = _state(kind)
    rng = np.random.default_rng(6)
    other = NBXDtype.float16 if dtype != NBXDtype.float16 else NBXDtype.float32
    with pytest.raises(ValueError, match="declared"):
        s.write(OPS[0], NBXTensor.from_numpy(_random(rng, (BATCH + 1, *shape[1:]), npd), dtype=dtype))
    with pytest.raises(ValueError, match="declared"):
        s.write(OPS[0], NBXTensor.zeros((BATCH, *shape[1:]), dtype=other, device="cuda:0"))
    with pytest.raises(ValueError, match="on its own card"):
        s.write(OPS[0], NBXTensor.zeros((BATCH, *shape[1:]), dtype=dtype, device="cuda:1"))
    with pytest.raises(KeyError, match="not one it serves"):
        s.tensor("layers.2.self_attn")


def test_each_op_has_its_buffer_on_the_card_it_is_placed_on(host):
    s = _state("recurrent", devices=[0, 1, 1])
    assert [s.tensor(op).device_idx for op in OPS] == [0, 1, 1]
    assert [host[s.tensor(op).data_ptr()][1] for op in OPS] == [0, 1, 1]
    for devices in ([0, 1], [0, 1, -1], "cuda:0", None):
        with pytest.raises(ValueError, match="devices"):
            _state("recurrent", devices=devices)


@pytest.mark.parametrize("kind,shape,dtype,ops,match", [
    ("kv", ["batch", 4, 64], "float16", OPS, "paged KV cache"),
    ("delta", ["batch", 2, 4, 3], "float32", OPS, "unknown kind"),
    ("recurrent", ["batch", 2, 4, 3], "bfloat16", OPS, "float32 always"),
    ("recurrent", ["batch", 2, 4, 3], "float16", OPS, "float32 always"),
    ("conv", ["batch", 5, 3], "int32", OPS, "not a floating dtype"),
    ("recurrent", ["batch", 2, 4], "float32", OPS, "is not"),
    ("conv", ["batch", 5, 3, 1], "float16", OPS, "is not"),
    ("conv", [1, 5, 3], "float16", OPS, "is not"),
    ("conv", ["batch", "channels", 3], "float16", OPS, "only the first axis is symbolic"),
    ("recurrent", ["batch", 2, 0, 3], "float32", OPS, "only the first axis is symbolic"),
    ("conv", ["batch", 5, 3], "float16", (), "non-empty"),
    ("conv", ["batch", 5, 3], "float16", ("a", "a"), "listed twice"),
])
def test_a_declaration_that_is_not_its_kinds_is_refused_by_name(kind, shape, dtype, ops, match):
    with pytest.raises(ValueError, match=match):
        F.StateDeclaration(kind, shape, dtype, ops)


def test_a_batch_that_is_not_a_count_of_sequences_is_refused_by_name(host):
    d = F.StateDeclaration("conv", ["batch", 5, 3], "float16", OPS)
    for batch in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="positive count of sequences"):
            d.shape(batch)


class Fence:
    def __init__(self):
        self.calls = []

    def drain(self, owner):
        self.calls.append(("drain", owner))

    def record(self):
        self.calls.append(("record",))


def test_every_host_issued_rewrite_is_inside_the_states_fence(host):
    fence = Fence()
    s = _state("conv", fence=fence)
    _fill(s, "conv", np.random.default_rng(7))
    s.zero()
    s.reorder([1, 0, 2])
    s.reorder([0, 1, 2])                                     # the identity moves nothing
    s.copy()
    s.restore()
    assert fence.calls == [("drain", s), ("record",)] * (len(OPS) + 4)
