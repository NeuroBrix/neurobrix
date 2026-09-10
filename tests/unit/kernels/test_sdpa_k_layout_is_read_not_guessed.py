"""K's layout comes from the graph, never from its shape.

PyTorch's SDPA math decomposition hands the traced graph a K already
transposed to (batch, heads, head_dim, seq). Every engine has to put it
back, and each used to recognise the situation by comparing shapes:

    k.shape[-2] == q.shape[-1] and k.shape[-1] != q.shape[-1]

That test is undecidable exactly when the sequence length equals the head
dimension. TinyLlama's head_dim is 64, so at a 64-token prompt the
correction silently did not fire: measured 2026-09-07 on Apple, the
last-position logits gave argmax 29892 at 6.41 where the float64 oracle
says 3864 at 22.36 — |delta| 23.27 across the vocabulary — while 63 and 65
tokens were exact to 0.02. Both branches carried it, because both sniffed
the same ambiguous shapes.
"""
import pytest

from neurobrix.core.runtime.graph_executor import GraphExecutor
from neurobrix.core.runtime.graph.compiled_ops import _k_is_pre_transposed


def _dag(*, transposed: bool, seq: int, head_dim: int = 64, heads: int = 32):
    """A minimal q/k/v -> SDPA graph, with or without the decomposition's
    transpose on K's chain."""
    q_shape = [1, heads, seq, head_dim]
    k_traced = [1, heads, head_dim, seq] if transposed else [1, heads, seq, head_dim]
    ops = {
        "q_src::0": {"op_type": "aten::clone", "input_tensor_ids": [],
                     "output_tensor_ids": ["q::0"], "attributes": {}},
        "v_src::0": {"op_type": "aten::clone", "input_tensor_ids": [],
                     "output_tensor_ids": ["v::0"], "attributes": {}},
        "k_src::0": {"op_type": "aten::clone", "input_tensor_ids": [],
                     "output_tensor_ids": ["k_base::0"], "attributes": {}},
        "sdpa::0": {"op_type": "aten::scaled_dot_product_attention",
                    "input_tensor_ids": ["q::0", "k::0", "v::0"],
                    "output_tensor_ids": ["out::0"], "attributes": {}},
    }
    tensors = {
        "q::0": {"shape": q_shape},
        "v::0": {"shape": q_shape},
        "k_base::0": {"shape": [1, heads, seq, head_dim]},
        "k::0": {"shape": k_traced},
        "out::0": {"shape": q_shape},
    }
    if transposed:
        ops["t::0"] = {"op_type": "aten::transpose",
                       "input_tensor_ids": ["k_base::0"],
                       "output_tensor_ids": ["k::0"],
                       "attributes": {"args": [
                           {"type": "tensor", "tensor_id": "k_base::0"},
                           {"type": "scalar", "value": -2},
                           {"type": "scalar", "value": -1}]}}
    else:
        ops["t::0"] = {"op_type": "aten::clone",
                       "input_tensor_ids": ["k_base::0"],
                       "output_tensor_ids": ["k::0"], "attributes": {}}
    return {"ops": ops, "tensors": tensors}


def _mark(dag):
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._dag = dag
    ex._mark_sdpa_k_layout()
    return dag["ops"]["sdpa::0"]["attributes"]


@pytest.mark.parametrize("seq", [23, 63, 64, 65])
def test_a_pre_transposed_k_is_recognised_at_every_length(seq):
    """Including the one where the shapes cannot say: seq == head_dim."""
    attrs = _mark(_dag(transposed=True, seq=seq))
    assert attrs["nbx_k_pre_transposed"] is True
    assert attrs["nbx_v_pre_transposed"] is False


@pytest.mark.parametrize("seq", [23, 63, 64, 65])
def test_a_plain_k_is_left_alone_at_every_length(seq):
    attrs = _mark(_dag(transposed=False, seq=seq))
    assert attrs["nbx_k_pre_transposed"] is False


def test_the_square_case_is_where_the_shape_test_was_blind():
    """At seq == head_dim the two layouts have the SAME shape, which is why
    the recorded answer had to exist."""
    square = _dag(transposed=True, seq=64, head_dim=64)
    assert (square["tensors"]["k::0"]["shape"]
            == square["tensors"]["k_base::0"]["shape"])


class _T:
    """The smallest thing that answers `ndim` and `shape`."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)


def test_the_runtime_reader_prefers_the_recorded_answer():
    q = _T([1, 32, 64, 64])
    k = _T([1, 32, 64, 64])
    assert _k_is_pre_transposed(q, k, {"nbx_k_pre_transposed": True}, "sdpa") is True
    assert _k_is_pre_transposed(q, k, {"nbx_k_pre_transposed": False}, "sdpa") is False


def test_the_runtime_reader_refuses_the_square_case_rather_than_guess():
    q = _T([1, 32, 64, 64])
    k = _T([1, 32, 64, 64])
    with pytest.raises(RuntimeError, match="square"):
        _k_is_pre_transposed(q, k, {}, "sdpa")


def test_the_runtime_reader_still_reads_an_unambiguous_shape():
    q = _T([1, 32, 63, 64])
    assert _k_is_pre_transposed(q, _T([1, 32, 64, 63]), {}, "sdpa") is True
    assert _k_is_pre_transposed(q, _T([1, 32, 63, 64]), {}, "sdpa") is False
