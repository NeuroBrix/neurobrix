"""The operand copies Prism prices (`runtime_widths.operand_casts`) are the ones each engine's dtype wrapper makes.

Prism cannot call a wrapper (the ATen one imports torch; a Triton one needs a device), so it MIRRORS their choice of
copies. This file is the drift door: each engine's own wrapper — `DtypeEngine.compile_op` (core/dtype/engine.py) and
`TritonDtypeEngine.wrap_op` (triton/dtype.py) — is handed real tensors and a spy kernel; every operand the spy
receives as ANOTHER object than the one passed is a copy, at the dtype it arrives in. Prism's list for the same op,
over the same operand dtypes, must be that list. The self-managed Triton wrappers (mm, bmm, addmm, conv2d) align their
operands inside the kernel wrapper by `launch_keys` functions that Prism calls itself: not mirrored, not tested here.

What it would do if the mirror drifted: a class whose wrapper changes (an op moved between AMP sets, a promote
threshold changed, the squaring guard dropped) gives a list that differs — red, naming the op.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.runtime_widths import conservative_contract, runtime_dtypes

torch = pytest.importorskip("torch")
NONE = conservative_contract("test: no contract")

# (op_type, operand dtypes, the same tensor twice)
CASES = [
    ("aten::convolution", ["float32", "float16"], False),          # lower precision: x to fp16
    ("aten::mm", ["float16", "float16"], False),                   # fp16 needing fp32: both up
    ("aten::bmm", ["float32", "float16"], False),
    ("aten::native_group_norm", ["float16", "float16", "float16"], False),   # AMP_FP32: all up
    ("aten::layer_norm", ["float32", "float16"], False),
    ("aten::add", ["float32", "float16"], False),                  # promote: the narrower widened
    ("aten::cat", ["float16", "float16"], False),                  # promote, nothing wider than 2 bytes
    ("aten::mul", ["float16", "float16"], True),                   # the fp16 squaring guard
    ("aten::mul", ["float16", "float32"], False),
    ("aten::gelu", ["float32"], False),                            # no class: no copy
]


def _prism(op_type, dtypes, same, engine):
    ins = ["input::a", "input::a"] if same else [f"input::x{i}" for i in range(len(dtypes))]
    T = {t: {"shape": [4, 4], "dtype": d, "is_input": True, "seam_alias_of": "test"} for t, d in zip(ins, dtypes)}
    T["out"] = {"shape": [4, 4], "dtype": "float16"}
    op = {"op_uid": "op::0", "op_type": op_type, "input_tensor_ids": ins, "output_tensor_ids": ["out"],
          "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}
    dag = {"tensors": T, "ops": {"op::0": op}, "execution_order": ["op::0"], "input_tensor_ids": sorted(set(ins)),
           "output_tensor_ids": ["out"], "torch_dtype": "float16"}
    casts = {}
    runtime_dtypes(dag, "float16", engine, has_native_bf16=False, attention_operands="narrowest", contract=NONE,
                   stores_fp64=False, casts=casts)
    return sorted((ins.index(t) if not same else 0, d) for t, d in casts.get("op::0", []))


def _spied(wrap, args):
    seen = []

    def spy(*a, **k):
        seen.extend(a)
        return a[0]
    wrap(spy)(*args)
    return sorted((0 if args[0] is args[-1] and len(args) == 2 and args[0] is args[1] else i, _dtype(x))
                  for i, (x, orig) in enumerate(zip(seen, args)) if x is not orig)


def _dtype(x):
    d = getattr(x, "dtype", None)
    name = str(getattr(d, "name", None) or d).replace("torch.", "")
    return {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}.get(name, name)     # NBXDtype's short names


@pytest.mark.parametrize("op_type,dtypes,same", CASES, ids=[f"{c[0]}:{'/'.join(c[1])}{':same' if c[2] else ''}"
                                                          for c in CASES])
def test_the_compiled_engine_s_copies_are_priced(op_type, dtypes, same):
    from neurobrix.core.dtype.engine import DtypeEngine
    eng = DtypeEngine(torch.float16, graph_dtype=torch.float16)
    if same:
        t = torch.zeros(4, 4, dtype=getattr(torch, dtypes[0]))
        args = (t, t)
    else:
        args = tuple(torch.zeros(4, 4, dtype=getattr(torch, d)) for d in dtypes)
    got = _spied(lambda f: eng.compile_op(op_type, f, {}, "op::0"), args)
    assert _prism(op_type, dtypes, same, "compiled") == got, f"{op_type} {dtypes}: Prism vs the engine's {got}"


def _cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda(), reason="the Triton wrappers' copies are NBXTensors on a device")
@pytest.mark.parametrize("engine", ["triton", "triton_sequential"])
@pytest.mark.parametrize("op_type,dtypes,same", [c for c in CASES if c[0] not in ("aten::mm", "aten::bmm")],
                         ids=[f"{c[0]}:{'/'.join(c[1])}{':same' if c[2] else ''}" for c in CASES
                              if c[0] not in ("aten::mm", "aten::bmm")])
def test_the_triton_engine_s_copies_are_priced(op_type, dtypes, same, engine):
    import numpy as np
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine
    eng = TritonDtypeEngine(NBXDtype.float16, has_native_bf16=False, graph_dtype="float16", stores_fp64=False)

    def nbx(d):
        return NBXTensor.from_numpy(np.zeros((4, 4), dtype=np.float32)).to_cuda().to(NBXDtype[d])
    if same:
        t = nbx(dtypes[0])
        args = (t, t)
    else:
        args = tuple(nbx(d) for d in dtypes)
    name = op_type.split("::")[-1]
    got = _spied(lambda f: eng.wrap_op(name, f, "op::0", {}), args)
    assert _prism(op_type, dtypes, same, engine) == got, f"{op_type} {dtypes}: Prism vs the engine's {got}"
