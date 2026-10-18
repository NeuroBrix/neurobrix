"""What the compiled engine's implementation of an op holds beside its input and output while it runs is the vendor file's
per-op table (`memory.compiled_op_workspace`), priced as the conv3d workspace is — never a default.

MPS decomposes `aten::linalg_vector_norm` into full-size intermediates before the fp32 copy the precision wrapper makes:
VACE's 52-frame encoder chunk held 9.95 GiB at that op where Prism said 3.2 GB live (the Mac, 2026-10-18). ATen's CUDA
reductions run in one kernel pass. A vendor that knows an op takes workspace it has not measured writes `null`, and a
plan holding that op is refused by name on that device until the value is in.

What each test would do if the code were wrong: a table ignored prices the norm at zero (red at the +bytes assertion);
a missing key priced at zero passes the refusal test's `raises` no more (red); a `null` priced at zero likewise (red);
an op the table does not name charged anyway (red at the equality); the conv3d path double-charged (red)."""
from __future__ import annotations

import types

import pytest

from neurobrix.core.prism.op_workspace import op_workspace_fraction, op_workspace_fractions
from neurobrix.core.prism.profiler import ActivationProfiler


def test_the_table_is_read_and_refused_by_name():
    assert op_workspace_fractions({"compiled_op_workspace": {}}) == {}
    assert op_workspace_fractions({"compiled_op_workspace": {"aten::linalg_vector_norm": 2.5}}) == {"aten::linalg_vector_norm": 2.5}
    assert op_workspace_fractions({"compiled_op_workspace": {"aten::linalg_vector_norm": None}}) == {"aten::linalg_vector_norm": None}
    with pytest.raises(KeyError, match="compiled_op_workspace"):
        op_workspace_fractions({"compiled_conv3d_workspace_input_fraction": 0.0})
    with pytest.raises(ValueError, match="not an op type"):
        op_workspace_fractions({"compiled_op_workspace": {"linalg_vector_norm": 1.0}})
    with pytest.raises(ValueError, match=">= 0"):
        op_workspace_fractions({"compiled_op_workspace": {"aten::mean": -1}})
    assert op_workspace_fraction({"aten::mean": 0.5}, "aten::mean") == 0.5
    assert op_workspace_fraction({"aten::mean": 0.5}, "aten::add") == 0.0
    with pytest.raises(RuntimeError, match="aten::linalg_vector_norm.*not measured"):
        op_workspace_fraction({"aten::linalg_vector_norm": None}, "aten::linalg_vector_norm", "apple/apple_m4_pro")


def _dag():
    T = {"input::x": {"shape": [1, 8, 4, 4], "dtype": "float16", "is_input": True},
         "n": {"shape": [1, 1, 4, 4], "dtype": "float16"}, "y": {"shape": [1, 8, 4, 4], "dtype": "float16"}}
    O = {"norm::0": {"op_type": "aten::linalg_vector_norm", "input_tensor_ids": ["input::x"], "output_tensor_ids": ["n"],
                     "attributes": {"args": [{"type": "tensor", "tensor_id": "input::x"}]}},
         "div::0": {"op_type": "aten::div", "input_tensor_ids": ["input::x", "n"], "output_tensor_ids": ["y"],
                    "attributes": {"args": [{"type": "tensor", "tensor_id": "input::x"}, {"type": "tensor", "tensor_id": "n"}]}}}
    return {"tensors": T, "ops": O, "execution_order": ["norm::0", "div::0"], "input_tensor_ids": ["input::x"],
            "output_tensor_ids": ["y"], "torch_dtype": "float16"}


def test_the_profiler_adds_the_fraction_of_the_input_at_the_op_and_nothing_elsewhere():
    g = _dag()
    base = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2)
    def table(op_type, x_shape, x_width, out_width):
        n = 1
        for d in x_shape:
            n *= d
        return int(n * x_width * 3.0) if op_type == "aten::linalg_vector_norm" else 0
    priced = ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, op_workspace=table)
    assert priced.peak_op_uid == "norm::0"
    # x is the caller's (not handed): the norm's in-op figure is its output n (32 B) + 3 x the input's 256 B of workspace;
    # without the table the peak is the div's: n (32 B) + y (256 B)
    assert priced.peak_bytes == 32 + 3 * 256
    assert base.peak_bytes == 32 + 256


def test_the_solver_prices_the_compiled_engine_from_the_vendor_table_and_the_triton_engines_from_their_kernels(monkeypatch):
    from neurobrix.core.prism import PrismSolver
    from neurobrix.core.prism import attention_scratch as A
    S = PrismSolver()
    prof = types.SimpleNamespace(devices=[types.SimpleNamespace(memory_mb=16384, architecture="volta", brand="nvidia", index=0)])
    monkeypatch.setattr(A, "vendor_memory", lambda d: {"compiled_op_workspace": {"aten::linalg_vector_norm": 2.0}})
    S._mode = "compiled"
    hook = S._op_workspace(prof)
    assert hook("aten::linalg_vector_norm", [1, 8, 4, 4], 2, 2) == 128 * 2 * 2 and hook("aten::add", [1, 8, 4, 4], 2, 2) == 0
    S._mode = "triton"
    assert S._op_workspace(prof) is None
    assert S._op_workspace(None) is None


@pytest.mark.parametrize("vendor,arch,expect", [("nvidia", "volta", {}), ("apple", "apple_m4_pro", {"aten::linalg_vector_norm": None})])
def test_every_shipped_vendor_file_states_its_table(vendor, arch, expect):
    from neurobrix.core.prism.attention_scratch import vendor_memory_of
    assert op_workspace_fractions(vendor_memory_of(vendor, arch)) == expect
