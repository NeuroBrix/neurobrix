"""A fixed-size state kept for a session: the state of the layers whose memory does not grow with the tokens.

Two kinds sit beside the KV cache (whose device pools are memory/paged_kv.py and whose session bookkeeping is
memory/kv_session.py), in the same form: one buffer per op it serves, on the card that runs the op, at a fixed address
for the state's lifetime (a recorded replay bakes it), rewritten in place by the host only inside the state's fence.

- 'recurrent': the delta rule's state, one `[batch, heads, k_dim, v_dim]` per op, float32 always (never narrowed).
- 'conv': a causal convolution's last taps-1 input columns, one `[batch, channels, taps-1]` per op, at the op's dtype.

A declaration names the kind, the shape (its first axis the symbol "batch", every other axis a size), the dtype and the
ordered op ids it serves; its bytes are exact, a function of the batch alone (no sequence axis). The verbs are the
library's: allocate at zero, the buffer a kernel writes in place, write, zero, reorder along the batch, the one copy a
serving turn keeps and its restore, forget. The POLICY (what it costs, when a copy is taken, what a failed turn does) is
the caller's. Whatever is not supported is refused by name. Zero torch (R33): NBXTensor and the allocator only.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size, parse_dtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor

STATE_KINDS = ("recurrent", "conv")
BATCH = "batch"
_AXES = {"recurrent": ("heads", "k_dim", "v_dim"), "conv": ("channels", "taps-1")}
_FLOATS = (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32, NBXDtype.float64)
_D2D = 3


class StateDeclaration:
    """What a fixed state holds: `kind` (one of STATE_KINDS), `shape` (["batch", sizes...]), `dtype` (an NBXDtype or a
    name) and `ops` (the ordered ids of the ops it serves, one buffer each). Refused by name when any is not the kind's."""

    def __init__(self, kind: str, shape: Sequence, dtype, ops: Sequence[str]):
        self.kind = _kind(kind)
        self.sizes = _sizes(self.kind, shape)
        self.dtype = _dtype(self.kind, dtype)
        self.ops = _ops(ops)

    def shape(self, batch: int) -> Tuple[int, ...]:
        """One op's buffer for `batch` sequences."""
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError(f"fixed state '{self.kind}': batch {batch!r} is not a positive count of sequences")
        return (batch, *self.sizes)

    def nbytes(self, batch: int) -> int:
        """The bytes the state holds for `batch` sequences over every op it serves: what it costs, exactly."""
        return len(self.ops) * math.prod(self.shape(batch)) * dtype_size(self.dtype)

    def __repr__(self) -> str:
        return f"StateDeclaration({self.kind!r}, [{BATCH!r}, {', '.join(map(str, self.sizes))}], " \
               f"{self.dtype.name}, {len(self.ops)} ops)"


def _kind(kind) -> str:
    if kind == "kv":
        raise ValueError("fixed state: kind 'kv' grows with the tokens; it is the paged KV cache's "
                         "(nbxtensor.memory.paged_kv), never a fixed state")
    if kind not in STATE_KINDS:
        raise ValueError(f"fixed state: unknown kind {kind!r}; the kinds are {', '.join(STATE_KINDS)}")
    return kind


def _sizes(kind: str, shape) -> Tuple[int, ...]:
    axes, shape = _AXES[kind], list(shape)
    if len(shape) != len(axes) + 1 or shape[0] != BATCH:
        raise ValueError(f"fixed state '{kind}': shape {shape} is not [{BATCH!r}, {', '.join(axes)}]")
    for name, size in zip(axes, shape[1:]):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"fixed state '{kind}': axis {name} is {size!r} in {shape}; only the first axis is "
                             f"symbolic ({BATCH!r}), every other is a positive size")
    return tuple(shape[1:])


def _dtype(kind: str, dtype) -> NBXDtype:
    dt = dtype if isinstance(dtype, NBXDtype) else parse_dtype(dtype)
    if kind == "recurrent" and dt != NBXDtype.float32:
        raise ValueError(f"fixed state 'recurrent': dtype {dt.name}; the recurrent state is float32 always, never "
                         f"narrowed")
    if dt not in _FLOATS:
        raise ValueError(f"fixed state '{kind}': dtype {dt.name} is not a floating dtype")
    return dt


def _ops(ops) -> Tuple[str, ...]:
    ops = tuple(ops)
    if not ops or not all(isinstance(o, str) and o for o in ops):
        raise ValueError(f"fixed state: the op ids {list(ops)} are not a non-empty list of names")
    if len(set(ops)) != len(ops):
        raise ValueError(f"fixed state: an op id is listed twice in {list(ops)}; each op has one buffer")
    return ops


class FixedState:
    """The buffers of `declaration` for `batch` sequences, allocated at zero: op i's on `devices` (one index for every
    op, or one per op in the declaration's order: the placement is the caller's). `fence` (optional, `record()` /
    `drain(owner)`): taken around every host-issued rewrite, as the KV cache's, on a device whose queue may still read
    the buffers. Every buffer keeps its address until `forget`."""

    def __init__(self, declaration: StateDeclaration, batch: int, devices: Union[int, Sequence[int]], fence=None):
        self.declaration = declaration
        self.batch = int(batch)
        shape = declaration.shape(batch)
        self.fence = fence
        self._held: Optional[Dict[str, NBXTensor]] = {
            op: _zeros(shape, declaration.dtype, dev) for op, dev in zip(declaration.ops, _devices(declaration, devices))}
        self._copy: Optional[Dict[str, NBXTensor]] = None

    # -- what it holds ------------------------------------------------------------------------------------------------
    @property
    def forgotten(self) -> bool:
        return self._held is None

    def tensor(self, op_id: str) -> NBXTensor:
        """Op `op_id`'s buffer: the one its kernel reads and writes in place."""
        return self._live()[self._op(op_id)]

    def held_bytes(self) -> int:
        """Bytes the buffers hold (the declaration's bytes for this batch; 0 once forgotten). The copy is apart."""
        return 0 if self._held is None else sum(t.nbytes() for t in self._held.values())

    def copy_bytes(self) -> int:
        """Bytes the kept copy holds, 0 when none is kept."""
        return 0 if self._copy is None else sum(t.nbytes() for t in self._copy.values())

    @property
    def has_copy(self) -> bool:
        return self._copy is not None

    # -- the verbs ----------------------------------------------------------------------------------------------------
    def write(self, op_id: str, src: NBXTensor) -> None:
        """`src` written into op `op_id`'s buffer in place: its shape, dtype and device are the buffer's."""
        dst = self.tensor(op_id)
        if tuple(src.shape) != tuple(dst.shape) or src.nbx_dtype != dst.nbx_dtype:
            raise ValueError(f"fixed state '{self.declaration.kind}': writing {list(src.shape)} {src.nbx_dtype.name} "
                             f"into op {op_id!r}, declared {list(dst.shape)} {dst.nbx_dtype.name}")
        if src.device_idx != dst.device_idx:
            raise ValueError(f"fixed state '{self.declaration.kind}': writing from device {src.device_idx} into op "
                             f"{op_id!r} on device {dst.device_idx}; the state is written on its own card")
        with _Fenced(self):
            DeviceAllocator.set_device(dst.device_idx)
            dst.copy_(src)

    def zero(self) -> None:
        """Every buffer back to zero in place (a new conversation in the same state)."""
        with _Fenced(self):
            for t in self._live().values():
                DeviceAllocator.set_device(t.device_idx)
                DeviceAllocator.memset_cuda(t.data_ptr(), 0, t.nbytes())

    def reorder(self, index: Sequence[int]) -> None:
        """Every buffer's batch rows gathered by `index` in place: row i becomes what row index[i] held (a row may be
        repeated, as when several continuations come from one sequence). `index` has one entry per sequence."""
        live, rows = self._live(), self._rows(index)
        if rows == list(range(self.batch)):
            return
        with _Fenced(self):
            for t in live.values():
                _gather_rows(t, rows)

    def copy(self) -> None:
        """The one copy a turn keeps of every buffer (taken at its start): refused while one is kept."""
        live = self._live()
        if self._copy is not None:
            raise RuntimeError(f"fixed state '{self.declaration.kind}': a copy is already kept; it is restored or "
                               f"released before another is taken (one copy at a time)")
        kept = {op: _twin(t) for op, t in live.items()}
        with _Fenced(self):
            for op, t in live.items():
                _bytes(kept[op], t)
        self._copy = kept

    def restore(self) -> None:
        """Every buffer given back the kept copy's bytes in place, then the copy released."""
        live, kept = self._live(), self._kept()
        with _Fenced(self):
            for op, t in live.items():
                _bytes(t, kept[op])
        self._copy = None

    def release_copy(self) -> None:
        """The kept copy freed (its turn succeeded): refused when none is kept."""
        self._live()
        self._kept()
        self._copy = None

    def forget(self) -> None:
        """Every buffer and the copy freed; any verb after this is refused by name."""
        self._live()
        self._held, self._copy = None, None

    # -- checks -------------------------------------------------------------------------------------------------------
    def _live(self) -> Dict[str, NBXTensor]:
        if self._held is None:
            raise RuntimeError(f"fixed state '{self.declaration.kind}' of ops {list(self.declaration.ops)} is "
                               f"forgotten: its buffers are freed")
        return self._held

    def _kept(self) -> Dict[str, NBXTensor]:
        if self._copy is None:
            raise RuntimeError(f"fixed state '{self.declaration.kind}': no copy is kept")
        return self._copy

    def _op(self, op_id: str) -> str:
        if op_id not in self.declaration.ops:
            raise KeyError(f"fixed state '{self.declaration.kind}': op {op_id!r} is not one it serves "
                           f"({len(self.declaration.ops)} ops)")
        return op_id

    def _rows(self, index: Sequence[int]):
        rows = list(index)
        bad = [r for r in rows if isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < self.batch]
        if len(rows) != self.batch or bad:
            raise ValueError(f"fixed state '{self.declaration.kind}': reorder index {rows} is not {self.batch} rows "
                             f"in [0, {self.batch})")
        return rows


class _Fenced:
    """The state's fence around a host-issued rewrite: drained before, recorded after (none: nothing)."""

    def __init__(self, state: FixedState):
        self.state = state

    def __enter__(self):
        if self.state.fence is not None:
            self.state.fence.drain(self.state)

    def __exit__(self, *exc):
        if self.state.fence is not None and exc[0] is None:
            self.state.fence.record()
        return False


def _devices(declaration: StateDeclaration, devices) -> Tuple[int, ...]:
    one = isinstance(devices, int) and not isinstance(devices, bool)
    out = (devices,) * len(declaration.ops) if one else tuple(devices) if isinstance(devices, (list, tuple)) else ()
    if len(out) != len(declaration.ops) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                                                    for d in out):
        raise ValueError(f"fixed state '{declaration.kind}': devices {devices!r} are not one index, or one per op "
                         f"({len(declaration.ops)})")
    return out


def _zeros(shape, dtype: NBXDtype, device_idx: int) -> NBXTensor:
    DeviceAllocator.set_device(device_idx)
    return NBXTensor.zeros(shape, dtype=dtype, device=f"cuda:{device_idx}")


def _twin(t: NBXTensor) -> NBXTensor:
    """An uninitialised buffer of `t`'s shape and dtype on `t`'s card (every byte of it is written next)."""
    DeviceAllocator.set_device(t.device_idx)
    return NBXTensor.empty(tuple(t.shape), dtype=t.nbx_dtype, device=f"cuda:{t.device_idx}")


def _bytes(dst: NBXTensor, src: NBXTensor) -> None:
    DeviceAllocator.set_device(dst.device_idx)
    DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), dst.nbytes(), _D2D)


def _gather_rows(t: NBXTensor, rows) -> None:
    """`t`'s rows gathered by `rows` in place, through a twin of `t` the moved rows are read from (freed after)."""
    src = _twin(t)
    _bytes(src, t)
    row = t.nbytes() // int(t.shape[0])
    for i, r in enumerate(rows):
        if r != i:
            DeviceAllocator.memcpy(t.data_ptr() + i * row, src.data_ptr() + r * row, row, _D2D)
