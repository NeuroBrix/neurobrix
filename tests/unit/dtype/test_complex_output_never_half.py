"""An op whose TRACED OUTPUT is complex never receives half inputs.

Measured on 2026-09-10 (torch 2.5.1+cu121), the promotion that motivates
this file::

    fp16 tensor * complex64 tensor  -> complex64   (safe: the wider complex wins)
    fp16 tensor * 1j                -> complex32   (the trap)
    torch.complex(fp16, fp16)       -> complex32   (the trap)
    exp / angle / irfft(complex32)  -> RAISE ("not implemented for 'ComplexHalf'")

The trap is NOT "a half tensor meets a complex tensor" — that promotes
correctly. It is a half tensor meeting a complex value the graph does not
carry as an input: Kokoro's decoder traces

    aten::mul   input_dtypes=['float32']  output_dtypes=['complex64']

where the second operand is the Python scalar ``1j``. Nothing in the op's
INPUTS says this op produces a complex, so no runtime inspection of the
arguments can decide it needs protection. The container already answered the
question, per op, in ``output_dtypes`` — that is the signal the engine reads.

This is why the hardcoded ``{"polar", "view_as_complex"}`` entries in
AMP_FP32_OPS cannot be completed by adding op names: the survey of the
cached containers (5 components, 490 complex-touching ops) shows complex
produced by ``aten::mul``, ``aten::complex``, ``aten::stft`` and
``aten::_fft_r2c`` as well — and a generic ``mul`` can produce a complex in
any model ever traced.

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_complex_output_never_half.py
"""
from __future__ import annotations

import torch

from neurobrix.core.dtype.engine import DtypeEngine


def _engine(compute_dtype=torch.float16) -> DtypeEngine:
    return DtypeEngine(compute_dtype, graph_dtype=torch.float16, amp_enabled=True)


def test_mul_by_complex_scalar_keeps_complex64_under_fp16_compute():
    """Kokoro decoder: phase * 1j. The traced output is complex64; a fp16
    left operand would silently make it complex32, which the very next op
    (aten::exp) has no kernel for."""
    eng = _engine()
    op = eng.compile_op("aten::mul", torch.mul,
                        {"input_dtypes": ["float32"],
                         "output_dtypes": ["complex64"]},
                        op_uid="mul::218")
    out = op(torch.ones(4, dtype=torch.float16), 1j)
    assert out.dtype == torch.complex64, (
        f"traced output is complex64, engine produced {out.dtype}")


def test_aten_complex_from_half_parts_keeps_complex64():
    """MiniCPM-o hift / chatterbox s3gen: aten::complex(real, imag) with two
    fp32 parts traced to complex64. torch.complex(fp16, fp16) is complex32
    and the istft that consumes it has no ComplexHalf kernel."""
    eng = _engine()
    op = eng.compile_op("aten::complex", torch.complex,
                        {"input_dtypes": ["float32", "float32"],
                         "output_dtypes": ["complex64"]},
                        op_uid="complex::7")
    out = op(torch.ones(4, dtype=torch.float16), torch.ones(4, dtype=torch.float16))
    assert out.dtype == torch.complex64, (
        f"traced output is complex64, engine produced {out.dtype}")


def test_complex128_output_does_not_downcast_float64_inputs():
    """The guard is a FLOOR (half -> fp32), not a leveller (everything ->
    fp32). Wan2.1's RoPE traces view_as_complex(float64) -> complex128 on
    80 blocks; forcing fp32 there would change the traced dtype contract."""
    eng = _engine()
    op = eng.compile_op("aten::view_as_complex", torch.view_as_complex,
                        {"input_dtypes": ["float64"],
                         "output_dtypes": ["complex128"]},
                        op_uid="view_as_complex::3")
    out = op(torch.ones(4, 2, dtype=torch.float64))
    assert out.dtype == torch.complex128, (
        f"traced output is complex128, engine produced {out.dtype}")


def test_real_output_op_is_untouched_by_the_complex_guard():
    """A control: the guard must not change any op whose traced output is
    real. aten::mul on two fp16 reals stays fp16."""
    eng = _engine()
    op = eng.compile_op("aten::mul", torch.mul,
                        {"input_dtypes": ["float16", "float16"],
                         "output_dtypes": ["float16"]},
                        op_uid="mul::4")
    out = op(torch.ones(4, dtype=torch.float16), torch.ones(4, dtype=torch.float16))
    assert out.dtype == torch.float16


# ─────────────────────────────────────────────────────────────────────
# R30 — the same invariant in the sequential and triton dispatch paths
# ─────────────────────────────────────────────────────────────────────

def test_sequential_dispatch_upcasts_half_at_a_complex_output_op():
    """--sequential goes through amp_cast_inputs, not compile_op. It reads
    the same container fact or the oracle diverges from compiled (R30)."""
    eng = _engine()
    args = eng.amp_cast_inputs("aten::mul",
                               [torch.ones(4, dtype=torch.float16), 1j],
                               op_uid="mul::218",
                               op_record={"output_dtypes": ["complex64"]})
    assert args[0].dtype == torch.float32
    assert torch.mul(*args).dtype == torch.complex64


def test_sequential_dispatch_keeps_float64_at_a_complex128_op():
    """Floor, not leveller — in the sequential path too."""
    eng = _engine()
    args = eng.amp_cast_inputs("aten::view_as_complex",
                               [torch.ones(4, 2, dtype=torch.float64)],
                               op_uid="view_as_complex::3",
                               op_record={"output_dtypes": ["complex128"]})
    assert args[0].dtype == torch.float64


def test_vendor_fp32_pin_does_not_retype_a_complex_op():
    """A keep-in-fp32 pin is a precision choice among REAL dtypes; it may not
    narrow a traced complex128 contract to complex64 by levelling its float64
    input. The complex floor wins over the pin."""
    eng = _engine()
    eng.set_precision_contract(True, fp32_op_uids={"view_as_complex::3"},
                               narrow_op_uids=frozenset())
    op = eng.compile_op("aten::view_as_complex", torch.view_as_complex,
                        {"output_dtypes": ["complex128"]},
                        op_uid="view_as_complex::3")
    assert op(torch.ones(4, 2, dtype=torch.float64)).dtype == torch.complex128


def test_triton_wrap_op_upcasts_half_at_a_complex_output_op():
    """The Triton branch carries complex too (NBXDtype.complex64/complex128 —
    Wan's RoPE freqs are complex128), so the mirror engine needs the same
    floor. Dtype crosses that boundary as a string: no torch here (R33)."""
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine

    class _T:
        """Stand-in for an NBXTensor: a CUDA-less unit test cannot allocate
        one, and the policy under test only reads dtype and casts."""
        def __init__(self, dt): self.nbx_dtype = dt
        def is_floating_point(self):
            return self.nbx_dtype in (NBXDtype.float16, NBXDtype.bfloat16,
                                      NBXDtype.float32, NBXDtype.float64)
        def is_contiguous(self): return True
        def contiguous(self): return self
        def to(self, dt): return _T(dt)

    eng = TritonDtypeEngine(NBXDtype.float16)
    seen = {}
    def _kernel(a, b):
        seen["a"] = a.nbx_dtype
        return a
    wrapped = eng.wrap_op("mul", _kernel, op_uid="mul::218",
                          op_record={"output_dtypes": ["complex64"]})
    wrapped(_T(NBXDtype.float16), _T(NBXDtype.complex64))
    assert seen["a"] == NBXDtype.float32


def test_triton_wrap_op_keeps_float64_at_a_complex128_op():
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine

    class _T:
        def __init__(self, dt): self.nbx_dtype = dt
        def is_floating_point(self):
            return self.nbx_dtype in (NBXDtype.float16, NBXDtype.bfloat16,
                                      NBXDtype.float32, NBXDtype.float64)
        def is_contiguous(self): return True
        def contiguous(self): return self
        def to(self, dt): return _T(dt)

    eng = TritonDtypeEngine(NBXDtype.float16)
    seen = {}
    def _kernel(a):
        seen["a"] = a.nbx_dtype
        return a
    wrapped = eng.wrap_op("view_as_complex", _kernel, op_uid="vac::3",
                          op_record={"output_dtypes": ["complex128"]})
    wrapped(_T(NBXDtype.float64))
    assert seen["a"] == NBXDtype.float64


# ─────────────────────────────────────────────────────────────────────
# The guard is only alive if every dispatch site still hands it the fact
# ─────────────────────────────────────────────────────────────────────

def test_every_dispatch_site_passes_the_op_record():
    """A guard keyed on the container's `output_dtypes` is dead the moment a
    call site stops passing it — and dead silently, since a missing `attrs`
    reads as "not complex" and every test above still passes.

    That is the replaced-call-site failure family (the R33 peel blinding the
    decode replay; the fusion proxy blinding the per-op recorder): the watcher
    survives, its seam does not. So the seam itself is pinned here — every
    `wrap_op` / `amp_cast_inputs` call under src/ names `attrs`.
    """
    import ast
    import pathlib

    watched = {"wrap_op", "amp_cast_inputs"}
    offenders = []
    scanned = calls = 0
    # From __file__, never the CWD: resolved relatively, this gate read ZERO
    # files from any other directory — a frozen worktree included — and
    # reported green. A gate that cannot say what it read has not run.
    src = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
    assert src.is_dir(), f"engine source not found at {src}"
    for path in src.rglob("*.py"):
        scanned += 1
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if name not in watched:
                continue
            calls += 1
            if not any(kw.arg == "op_record" for kw in node.keywords):
                offenders.append(f"{path}:{node.lineno} {name}() without op_record=")
    assert not offenders, (
        "a complex-output op reaching these sites would be narrowed to "
        "complex32 with no error:\n  " + "\n  ".join(offenders))
    # What the gate actually read — without these it passes on an empty sweep.
    assert scanned > 100, f"only {scanned} source files scanned"
    assert calls >= 5, f"only {calls} watched call sites found"


# ─────────────────────────────────────────────────────────────────────
# The predicate is duplicated across the two branches (R33: triton/ may
# not import the torch-carrying core module). Pin the copies together.
# ─────────────────────────────────────────────────────────────────────

def test_both_branches_read_the_container_identically():
    """core/dtype/engine.py and triton/dtype.py each carry their own
    `traced_output_is_complex` — the Triton branch cannot import the core one
    without pulling torch in at import time (R33). Duplication is the
    doctrine's answer there (AMP_FP32_OPS is duplicated for the same reason),
    but two copies can drift. They read a STRING, so the whole surface they
    could drift on is representation — pinned here.
    """
    from neurobrix.core.dtype import engine as core_mod
    from neurobrix.core.dtype.engine import traced_output_is_complex as core
    from neurobrix.triton import dtype as triton_mod
    from neurobrix.triton.dtype import traced_output_is_complex as triton

    assert core_mod._COMPLEX_DTYPE_NAMES == triton_mod._COMPLEX_DTYPE_NAMES, (
        "the two branches recognise different complex dtype names")

    cases = [
        ({"output_dtypes": ["complex64"]}, True),
        ({"output_dtypes": ["complex128"]}, True),
        ({"output_dtypes": ["complex32"]}, True),
        ({"output_dtypes": ["torch.complex64"]}, True),      # a torch.dtype
        ({"output_dtypes": ["NBXDtype.complex64"]}, True),   # an NBXDtype
        ({"output_dtypes": ["float32", "complex64"]}, True),  # multi-output
        ({"output_dtypes": ["float32"]}, False),
        ({"output_dtypes": ["float64"]}, False),
        ({"output_dtypes": []}, False),
        ({"input_dtypes": ["complex64"]}, False),   # INPUTS are not the signal
        ({}, False),
        (None, False),
        ("not-a-dict", False),
    ]
    for attrs, expected in cases:
        assert core(attrs) is expected, f"core disagrees on {attrs!r}"
        assert triton(attrs) is expected, f"triton disagrees on {attrs!r}"


def test_predicate_fires_on_the_real_containers_if_present():
    """Integration pin: the key the engines read is the key the tracer
    writes. Skipped where the cache is not present."""
    import json
    import os
    import pytest

    from neurobrix.core.dtype.engine import traced_output_is_complex as core
    from neurobrix.triton.dtype import traced_output_is_complex as triton

    g = os.path.expanduser(
        "~/.neurobrix/cache/Kokoro-82M/components/decoder/graph.json")
    if not os.path.exists(g):
        pytest.skip("Kokoro-82M container not extracted on this machine")

    ops = json.load(open(g)).get("ops") or {}
    flagged = {uid for uid, op in ops.items() if core(op)}
    # aten.mul::218 is `phase * 1j`: input_dtypes=['float32'] and
    # output_dtypes=['complex64'] — the op no name list and no argument
    # inspection can reach.
    assert "aten.mul::218" in flagged
    assert all(core(op) is triton(op) for op in ops.values())


# ─────────────────────────────────────────────────────────────────────
# The compiled seam — the only mode where the ComplexHalf crash is real
# ─────────────────────────────────────────────────────────────────────

def test_compiled_resolver_seam_applies_the_floor():
    """`compile_op` takes its dict POSITIONALLY, so the keyword gate above
    cannot see the compiled path. Pin it by behaviour, at the seam the
    compiled sequence actually calls."""
    from neurobrix.core.runtime.graph.compiled_ops import CompiledOpResolver

    r = CompiledOpResolver(device=torch.device("cpu"), dtype=torch.float16)
    complex_op = r.get_op_func("mul", {"output_dtypes": ["complex64"]},
                               op_uid="mul::218")
    assert complex_op(torch.ones(4, dtype=torch.float16), 1j).dtype == torch.complex64
    real_op = r.get_op_func("mul", {"output_dtypes": ["float16"]}, op_uid="mul::4")
    assert real_op(torch.ones(4, dtype=torch.float16),
                   torch.ones(4, dtype=torch.float16)).dtype == torch.float16


def test_compiled_sequence_hands_the_resolver_the_output_dtypes():
    """The fact the compiled rule reads is merged into `attrs` in a DIFFERENT
    file from its consumer: compiled_sequence.py copies `op_data`'s top-level
    `output_dtypes` into the dict it passes to the resolver. Delete that copy
    and every behavioural test in this file still passes while compiled mode
    silently loses the floor — the seam has no other witness, so it is pinned
    here."""
    import ast
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
            / "core" / "runtime" / "graph" / "compiled_sequence.py")
    assert path.is_file(), f"compiled sequence not found at {path}"
    tree = ast.parse(path.read_text(), filename=str(path))
    merged = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for tgt in node.targets
        if isinstance(tgt, ast.Subscript)
        and isinstance(tgt.value, ast.Name) and tgt.value.id == "attrs"
        and isinstance(tgt.slice, ast.Constant) and tgt.slice.value == "output_dtypes"
    ]
    assert merged, (
        f"{path.name} no longer copies op_data['output_dtypes'] into the attrs "
        "it hands the op resolver — compiled mode cannot see that an op "
        "produces a complex, and narrows its inputs to half with no error")


def test_triton_pin_does_not_retype_a_complex_op():
    """R30 mirror of test_vendor_fp32_pin_does_not_retype_a_complex_op: the
    complex floor sits ahead of the calibration islands in the Triton engine
    too, so a pinned op keeps its float64 rather than being levelled."""
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine

    class _T:
        def __init__(self, dt): self.nbx_dtype = dt
        def is_floating_point(self):
            return self.nbx_dtype in (NBXDtype.float16, NBXDtype.bfloat16,
                                      NBXDtype.float32, NBXDtype.float64)
        def is_contiguous(self): return True
        def contiguous(self): return self
        def to(self, dt): return _T(dt)

    eng = TritonDtypeEngine(NBXDtype.float16)
    eng.set_precision_contract(True, fp32_op_uids={"vac::3"}, narrow_op_uids=())
    seen = {}
    wrapped = eng.wrap_op("view_as_complex", lambda a: seen.setdefault("a", a.nbx_dtype),
                          op_uid="vac::3", op_record={"output_dtypes": ["complex128"]})
    wrapped(_T(NBXDtype.float64))
    assert seen["a"] == NBXDtype.float64
