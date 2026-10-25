"""In a half-precision graph, an op the trace computed entirely in float32 is the vendor's own fp32 island; both
engines' dtype policies keep it fp32 on ANY hardware (core/dtype/traced_island.is_traced_fp32_island, one predicate).
Sana_1600M_1024px_MultiLing on the M4 Pro, 2026-10-25: the compiled branch input-cast the Gemma rotary GEMM
(`inv_freq @ position_ids`, traced fp32 in a float16 graph) to bf16 — "bf16 range = fp32 range" — and the angles of
506 positions lost ~2 rad; the image was a different composition from the Triton branch's (19.8 dB).

What this test would do if the code were wrong: under bf16 compute the compiled policy picks the lower-precision
(input-cast) wrapper for the traced-fp32 GEMM -> red (seen red before the fix, 2026-10-25). fp16 hardware keeps its
answer (the GEMM was already fp32-protected there), and a GEMM traced at the graph's half width is still cast."""
from __future__ import annotations

import pytest
import torch

from neurobrix.core.dtype.engine import DtypeEngine
from neurobrix.core.dtype.traced_island import is_traced_fp32_island

ISLAND = {"input_dtypes": ["float32", "float32"], "output_dtypes": ["float32"]}
HALF = {"input_dtypes": ["float16", "float16"], "output_dtypes": ["float16"]}
CAST_OUT = {"input_dtypes": ["float32"], "output_dtypes": ["float16"]}


def test_the_predicate():
    assert is_traced_fp32_island(ISLAND, "float16") and is_traced_fp32_island(ISLAND, "bfloat16")
    assert not is_traced_fp32_island(ISLAND, "float32")          # an fp32 graph has no island: it IS fp32
    assert not is_traced_fp32_island(HALF, "float16")
    assert not is_traced_fp32_island(CAST_OUT, "float16")        # the cast out of the island
    assert not is_traced_fp32_island({"input_dtypes": ["int64"], "output_dtypes": ["float32"]}, "float16")


def _picked(compute, hardware, record, monkeypatch):
    eng = DtypeEngine(compute, graph_dtype=torch.float16, hardware=hardware, device="cpu")
    monkeypatch.setattr(eng, "_make_lower_precision_wrapper", lambda f: "lower")
    monkeypatch.setattr(eng, "_make_fp32_wrapper", lambda f: "fp32")
    return eng.compile_op("aten::bmm", torch.ops.aten.bmm.default, record, op_uid="aten.bmm::0")


def test_the_compiled_policy_keeps_a_traced_fp32_gemm_fp32_under_bf16(monkeypatch):
    assert _picked(torch.bfloat16, ("apple", "apple_m4_pro"), ISLAND, monkeypatch) == "fp32"


def test_a_half_traced_gemm_is_still_cast_under_bf16(monkeypatch):
    assert _picked(torch.bfloat16, ("apple", "apple_m4_pro"), HALF, monkeypatch) == "lower"


def test_fp16_hardware_keeps_its_answer(monkeypatch):
    assert _picked(torch.float16, ("nvidia", "volta"), ISLAND, monkeypatch) == "fp32"


@pytest.mark.parametrize("record,expected", [(ISLAND, "fp32"), (HALF, "lower")])
def test_the_triton_policy_reads_the_same_predicate(record, expected, monkeypatch):
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.dtype import TritonDtypeEngine
    eng = TritonDtypeEngine(NBXDtype.bfloat16, graph_dtype="float16", stores_fp64=False)
    monkeypatch.setattr(eng, "_wrap_lower_precision", lambda f: "lower")
    monkeypatch.setattr(eng, "_wrap_fp32", lambda f: "fp32")
    assert eng.wrap_op("baddbmm", lambda *a: None, op_uid="aten.baddbmm::0", op_record=record) == expected
