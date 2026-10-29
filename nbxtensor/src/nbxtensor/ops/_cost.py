"""The cost an op declares: what it allocates, for how long, what it moves and computes (R-NBXT 4) — prototype.

A planner (NeuroBrix's Prism) prices a model from these declarations instead of re-deriving what the allocator and the
kernels do. Everything here is plain data (R-NBXT 11): an operand is a `Spec(shape, strides, dtype)`, a cost is a
`Cost` of `Buffer`s with integer lifetimes, byte counts and flop counts. The declarations read the same layout rules
the ops execute (`nbxtensor.shape`) and the same widths (`nbxtensor.dtype`).

Prototype scope (docs/nbxtensor/COST.md): the element-wise, reduction and normalisation families, mirroring today's
wrappers (NeuroBrix `kernels/wrappers.py`) line by line, each declaration citing the allocation it mirrors. Where a
wrapper takes a decision the library does not hold yet (the binary ops' result dtype, the DtypeEngine's output
dtype), the caller passes it in; when the wrappers move into the library (M2/M3), `cost` reads the same function the
kernel launch reads, so the declaration and the execution cannot drift.

Lifetimes are steps of the op's own sequence: a buffer lives over [first, last]; today every transient lives to the
op's return (the wrappers keep it in a local name); the operands are the caller's and are never counted. `peak_bytes` is the most the op holds at once beyond its operands, outputs included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from nbxtensor.dtype.itemsize import itemsize
from nbxtensor.shape import contiguous_strides, is_contiguous, numel

__all__ = ["Spec", "Buffer", "Cost", "elementwise_unary", "elementwise_binary", "softmax", "rms_norm",
           "native_layer_norm", "reduction"]


@dataclass(frozen=True)
class Spec:
    """An operand: its shape, its strides in elements, its dtype name."""
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str

    @staticmethod
    def contiguous(shape: Sequence[int], dtype: str) -> "Spec":
        return Spec(tuple(shape), contiguous_strides(tuple(shape)), dtype)

    @property
    def nbytes(self) -> int:
        return numel(self.shape) * itemsize(self.dtype)


@dataclass(frozen=True)
class Buffer:
    """A buffer the op allocates: its bytes, what it is, and the steps it lives over (`first` to `last`, inclusive).
    `returned` buffers are the op's outputs: they outlive the op (their `last` is the op's last step)."""
    name: str
    nbytes: int
    first: int
    last: int
    returned: bool = False


@dataclass(frozen=True)
class Cost:
    buffers: Tuple[Buffer, ...]
    bytes_moved: int        # every byte read and written, copies included
    flops: int

    @property
    def outputs(self) -> Tuple[Buffer, ...]:
        return tuple(b for b in self.buffers if b.returned)

    @property
    def workspace(self) -> Tuple[Buffer, ...]:
        return tuple(b for b in self.buffers if not b.returned)

    @property
    def peak_bytes(self) -> int:
        steps = {s for b in self.buffers for s in (b.first, b.last)}
        return max((sum(b.nbytes for b in self.buffers if b.first <= s <= b.last) for s in steps), default=0)


class _Steps:
    """The op's sequence: each allocation takes the next step. Every buffer lives until the op returns: the wrappers
    hold their transients in local names (the cast, the contiguous copy, the accumulator), which Python releases at
    the return, never earlier."""

    def __init__(self):
        self.buffers = []
        self.moved = 0
        self.step = 0

    def alloc(self, name: str, nbytes: int, returned: bool = False) -> int:
        self.step += 1
        self.buffers.append([name, int(nbytes), self.step, None, returned])
        return len(self.buffers) - 1

    def free(self, index: int) -> None:
        """The buffer dies here (a transient the wrapper drops before it returns: a reassigned local, a callee's
        return): it lives over [its step, this step]."""
        self.step += 1
        self.buffers[index][3] = self.step

    def done(self, flops: int) -> Cost:
        end = self.step
        return Cost(tuple(Buffer(n, b, f, end if l is None else l, r) for n, b, f, l, r in self.buffers),
                    self.moved, int(flops))

    def nest(self, cost: "Cost", prefix: str) -> list:
        """A callee's cost inside this op's sequence: its buffers allocated in its order, its transients freed when it
        returns; its returned buffers are the caller's to free (`free` on the indices returned)."""
        base = self.step
        kept = []
        for b in cost.buffers:
            self.buffers.append([f"{prefix}: {b.name}", b.nbytes, base + b.first,
                                 None if b.returned else base + b.last, False])
            if b.returned:
                kept.append(len(self.buffers) - 1)
        self.step = base + max((b.last for b in cost.buffers), default=0)
        self.moved += cost.bytes_moved
        return kept


def _contiguous_copy(st: _Steps, x: Spec, name: str) -> Optional[int]:
    """`x.contiguous()`: no buffer when `x` already lies row-major (`shape.is_contiguous`), else one copy of its
    elements, read strided and written dense."""
    if is_contiguous(x.shape, x.strides):
        return None
    st.moved += 2 * x.nbytes
    return st.alloc(name, x.nbytes)


def elementwise_unary(x: Spec, out_dtype: Optional[str] = None, flops_per_element: int = 1) -> Cost:
    """A unary element-wise op (`wrappers._prepare_unary`: `x.contiguous()`, then `empty_like` the output; `out_dtype`
    when the op writes another dtype, e.g. a comparison's bool)."""
    st = _Steps()
    _contiguous_copy(st, x, "contiguous input")
    out = Spec.contiguous(x.shape, out_dtype or x.dtype)
    st.alloc("output", out.nbytes, returned=True)
    st.moved += x.nbytes + out.nbytes
    return st.done(flops_per_element * numel(x.shape))


def elementwise_binary(a: Spec, b: Spec, common_dtype: str, out_shape: Sequence[int],
                       out_dtype: Optional[str] = None, flops_per_element: int = 1) -> Cost:
    """A binary element-wise op of two tensors (`wrappers._prepare_binary_strided`): each operand not already at
    `common_dtype` is cast — a COPY at that dtype (the RoPE lesson: an in-kernel widen moves the compiler's
    contraction) —, then both are read by strides as broadcast VIEWS (no expand copy), and the output is dense at
    the broadcast shape. `common_dtype` and `out_shape` are the wrapper's decisions, passed in until the wrapper's
    promotion lives here."""
    st = _Steps()
    for name, x in (("a cast", a), ("b cast", b)):
        if x.dtype != common_dtype:
            cast = Spec(x.shape, x.strides, common_dtype)
            st.moved += x.nbytes + cast.nbytes
            st.alloc(name, cast.nbytes)
    out = Spec.contiguous(out_shape, out_dtype or common_dtype)
    st.alloc("output", out.nbytes, returned=True)
    st.moved += numel(a.shape) * itemsize(common_dtype) + numel(b.shape) * itemsize(common_dtype) + out.nbytes
    return st.done(flops_per_element * numel(out_shape))


def softmax(x: Spec, dim: int) -> Cost:
    """`wrappers.softmax`: the reduced dim moved last (`movedim(dim, -1).contiguous()`: a copy unless that view is
    already dense), the output `empty_like` the 2-D view, returned moved back (a view). Fused and looped kernels
    alike hold nothing else. Arithmetic per element: the row max, the shift, the exponential, the row sum, the
    division."""
    dim = dim % len(x.shape)
    order = [d for d in range(len(x.shape)) if d != dim] + [dim]
    moved = Spec(tuple(x.shape[d] for d in order), tuple(x.strides[d] for d in order), x.dtype)
    st = _Steps()
    _contiguous_copy(st, moved, "rows made contiguous")
    st.alloc("output", x.nbytes, returned=True)
    st.moved += x.nbytes + x.nbytes
    return st.done(5 * numel(x.shape))


def rms_norm(x: Spec, has_weight: bool, out_dtype: Optional[str] = None) -> Cost:
    """`wrappers.rms_norm`: `x.contiguous()`; when that made a fresh copy and the output dtype is the input's, the
    output is written INTO the copy (no second buffer); otherwise a new output. Arithmetic per element: the square,
    the row sum, the scale by the reciprocal root, and the weight's product when there is one."""
    out_dt = out_dtype or x.dtype
    st = _Steps()
    copy = _contiguous_copy(st, x, "contiguous input")
    out_bytes = numel(x.shape) * itemsize(out_dt)
    if copy is not None and out_dt == x.dtype:
        st.buffers[copy][4] = True                                 # the copy IS the output
        st.buffers[copy][0] = "output (in the contiguous copy)"
    else:
        st.alloc("output", out_bytes, returned=True)
    st.moved += x.nbytes + out_bytes + (x.shape[-1] * itemsize(x.dtype) if has_weight else 0)
    return st.done((4 if has_weight else 3) * numel(x.shape))


def native_layer_norm(x: Spec, normalized_shape: Sequence[int], has_weight: bool, has_bias: bool) -> Cost:
    """`wrappers.native_layer_norm`: `x.contiguous()`, the output `empty_like` it, and the row statistics (mean and
    1/std, float32, one per row) — returned, as aten::native_layer_norm returns (output, mean, rstd). Arithmetic per
    element: the mean's sum, the centring, the square, the variance's sum, the scale; the weight's product and the
    bias's sum when present."""
    feat = numel(normalized_shape)
    rows = numel(x.shape) // feat
    st = _Steps()
    _contiguous_copy(st, x, "contiguous input")
    st.alloc("output", x.nbytes, returned=True)
    st.alloc("mean", rows * itemsize("float32"), returned=True)
    st.alloc("rstd", rows * itemsize("float32"), returned=True)
    affine = feat * itemsize(x.dtype) * (int(has_weight) + int(has_weight and has_bias))
    st.moved += 2 * x.nbytes + 2 * rows * itemsize("float32") + affine
    return st.done((5 + int(has_weight) + int(has_weight and has_bias)) * numel(x.shape))


def reduction(x: Spec, dim: Optional[int], keepdim: bool, out_dtype: str, cast_to: Optional[str] = None) -> Cost:
    """`wrappers.sum_wrapper` / `mean_wrapper` over one dim or all: the input cast to the op's `dtype` keyword when
    given (`cast_to`, a copy), the reduced dim moved last and made contiguous (a copy unless already dense), one
    float32 accumulator per output element, then cast to the result dtype (`out_dtype`: the input's for a float
    input, int64 for sum of an integer one) — a second buffer unless that dtype is float32 (`.to` returns the tensor
    itself then). One addition per input element."""
    st = _Steps()
    src = x
    if cast_to is not None and cast_to != x.dtype:
        src = Spec(x.shape, x.strides, cast_to)
        st.moved += x.nbytes + src.nbytes
        st.alloc("dtype keyword cast", src.nbytes)
    if dim is None:
        rows = 1
        _contiguous_copy(st, src, "flattened input")
    else:
        d = dim % len(src.shape)
        order = [i for i in range(len(src.shape)) if i != d] + [d]
        moved = Spec(tuple(src.shape[i] for i in order), tuple(src.strides[i] for i in order), src.dtype)
        rows = numel(src.shape) // src.shape[d]
        _contiguous_copy(st, moved, "reduced dim made contiguous")
    st.alloc("float32 accumulator", rows * itemsize("float32"), returned=out_dtype == "float32")
    st.moved += src.nbytes + rows * itemsize("float32")
    if out_dtype != "float32":
        st.alloc("result", rows * itemsize(out_dtype), returned=True)
        st.moved += rows * itemsize("float32") + rows * itemsize(out_dtype)
    return st.done(numel(x.shape))
