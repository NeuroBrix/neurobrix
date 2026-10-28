"""The nbxtensor-vs-torch benchmark's judgement, card-less (benchmarks/nbxtensor_vs_torch): a faster wrong result is a
loss, never a win; the bound is the narrowest dtype the op reads, writes or is recorded writing; two ops differing
only in which tensors they read are one instance; the inputs are deterministic and an index stays in its extent."""
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[3] / "benchmarks" / "nbxtensor_vs_torch"
sys.path.insert(0, str(BENCH))

import instances as I  # noqa: E402
import summary as S  # noqa: E402

K = pytest.importorskip("kernel_bench")


def _row(key, contender, status, median, note=""):
    return {"key": key, "op_type": "aten::mm", "shapes": "[[2, 2], [2, 2]]", "dtypes": "float16,float16", "sites": 1,
            "contender": contender, "status": status, "median_ms": median, "note": note}


def test_a_faster_wrong_result_is_a_loss():
    rows = [_row("a", "torch_eager", "ok", "1.0"), _row("a", "torch_compile[default]", "ok", "0.8"),
            _row("a", "nbxtensor", "wrong", "0.1")]
    lost, won, total = S.kernel_losses(rows)
    assert total == 1 and won == 0 and lost[0][3] == "wrong"


def test_slower_than_the_best_right_torch_is_a_loss_and_a_wrong_torch_is_no_benchmark():
    rows = [_row("a", "torch_eager", "ok", "1.0"), _row("a", "torch_compile[default]", "wrong", "0.1"),
            _row("a", "nbxtensor", "ok", "1.2"),
            _row("b", "torch_eager", "ok", "1.0"), _row("b", "nbxtensor", "ok", "0.9")]
    lost, won, total = S.kernel_losses(rows)
    assert total == 2 and won == 1
    assert len(lost) == 1 and lost[0][1]["key"] == "a" and lost[0][0] == pytest.approx(1.2)


def test_the_bound_is_the_narrowest_dtype_read_written_or_recorded():
    assert K._bound(["float32", "float32", "float32"]) == K.TOLERANCE["float32"]
    assert K._bound(["float32", "float32", "bfloat16"]) == K.TOLERANCE["bfloat16"]
    assert K._bound(["int64", "bool"]) is None


def test_two_ops_reading_other_tensors_are_one_instance():
    a = [{"type": "tensor", "tensor_id": "x::0"}, {"type": "scalar", "value": 2}]
    b = [{"type": "tensor", "tensor_id": "y::7"}, {"type": "scalar", "value": 2}]
    c = [{"type": "tensor", "tensor_id": "y::7"}, {"type": "scalar", "value": 3}]
    assert I._literal(a) == I._literal(b) != I._literal(c)


def test_the_inputs_are_deterministic_and_an_index_stays_in_its_extent():
    inst = {"shapes": [[8, 16], [4], []], "dtypes": ["float16", "int64", "float32"]}
    x, y = K.host_inputs(inst), K.host_inputs(inst)
    assert all(np.array_equal(p, q) for p, q in zip(x, y))
    assert x[1].dtype == np.int64 and x[1].max() < 16 and x[2].shape == ()
