"""The work one run of a component does (core/prism/work.py): two flops per multiply-add of a product, a convolution
and an attention, read from the resolved shapes; a weight's bytes counted once whichever op meets it first (a view's
transpose included); a view moves no activation bytes.

What these tests would do if the code were wrong: a contraction read on the wrong operand (addmm's bias) gives the
wrong flops — the first case; a weight read through `aten::t` counted as an activation, or not at all, fails the
third; a view's copy counted fails it too.
"""
from neurobrix.core.prism import work as W


def test_products_convolutions_and_attention_count_their_multiply_adds():
    assert W.op_flops("aten::mm", [[64, 32], [32, 16]], [[64, 16]], {}) == 2 * 64 * 16 * 32
    assert W.op_flops("aten::addmm", [[16], [64, 32], [32, 16]], [[64, 16]], {}) == 2 * 64 * 16 * 32
    assert W.op_flops("aten::convolution", [[1, 8, 10, 10], [4, 8, 3, 3]], [[1, 4, 8, 8]], {}) == 2 * 256 * 72
    assert W.op_flops("aten::scaled_dot_product_attention", [[2, 4, 100, 64], [2, 4, 50, 64], [2, 4, 50, 64]],
                      [[2, 4, 100, 64]], {}) == 4 * 8 * 100 * 50 * 64
    assert W.op_flops("aten::add", [[3, 5], [3, 5]], [[3, 5]], {}) == 15


class _Profiler:
    def __init__(self, dag):
        self.tensors, self.ops, self.execution_order = dag["tensors"], dag["ops"], dag["execution_order"]

    def _resolve_shape(self, meta, symbol_map):
        return [symbol_map.get(d, d) if isinstance(d, str) else d for d in meta["shape"]]

    def _compute_size(self, shape, meta, default):
        n = 1
        for d in shape:
            n *= d
        return n * {"float16": 2, "float32": 4}.get(meta.get("dtype"), default)


def test_a_weight_read_through_a_view_is_counted_once_and_a_view_moves_nothing():
    dag = {"tensors": {"x": {"shape": ["s", 32], "dtype": "float16"},
                       "w": {"shape": [16, 32], "dtype": "float16", "is_parameter": True},
                       "wt": {"shape": [32, 16], "dtype": "float16"},
                       "y": {"shape": ["s", 16], "dtype": "float16"}},
           "ops": {"t": {"op_type": "aten::t", "input_tensor_ids": ["w"], "output_tensor_ids": ["wt"]},
                   "mm": {"op_type": "aten::mm", "input_tensor_ids": ["x", "wt"], "output_tensor_ids": ["y"]}},
           "execution_order": ["t", "mm"]}
    w = W.component_work(_Profiler(dag), {"s": 10})
    assert w.flops == 2 * 10 * 16 * 32 + 16 * 32                  # the product, plus the view's one per element
    assert w.weight_bytes == 16 * 32 * 2
    assert w.activation_bytes == 10 * 32 * 2 + 10 * 16 * 2       # x read, y written; the weight's view is the weight
