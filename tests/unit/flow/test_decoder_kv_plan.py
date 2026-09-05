"""The decoder self-attention plan is derived from the graph's dataflow: a
cross-attention's keys/values do not derive from the token inputs."""
from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan


def _dag():
    return {"input_tensor_ids": ["input::input_ids", "input::encoder_hidden_states"], "ops": {
        "aten.embedding::0": {"op_type": "aten::embedding", "input_tensor_ids": ["param::tok", "input::input_ids"], "output_tensor_ids": ["h0"]},
        "aten.arange::0": {"op_type": "aten::arange", "input_tensor_ids": [], "output_tensor_ids": ["pos"]},
        "aten.mm::0": {"op_type": "aten::mm", "input_tensor_ids": ["h0", "wq"], "output_tensor_ids": ["q"], "input_shapes": [[1, 7, 1280], [1280, 1280]]},
        "aten.mm::1": {"op_type": "aten::mm", "input_tensor_ids": ["h0", "wk"], "output_tensor_ids": ["k"]},
        "aten.mm::2": {"op_type": "aten::mm", "input_tensor_ids": ["h0", "wv"], "output_tensor_ids": ["v"]},
        "aten._scaled_dot_product_efficient_attention::0": {"op_type": "aten::_scaled_dot_product_efficient_attention", "input_tensor_ids": ["q", "k", "v"], "output_tensor_ids": ["a0"], "input_shapes": [[1, 20, 7, 64], [1, 20, 7, 64], [1, 20, 7, 64]]},
        "aten.add::0": {"op_type": "aten::add", "input_tensor_ids": ["h0", "a0"], "output_tensor_ids": ["h1"]},
        "aten.mm::3": {"op_type": "aten::mm", "input_tensor_ids": ["h1", "wq2"], "output_tensor_ids": ["q2"]},
        "aten.mm::4": {"op_type": "aten::mm", "input_tensor_ids": ["input::encoder_hidden_states", "wk2"], "output_tensor_ids": ["k2"]},
        "aten.mm::5": {"op_type": "aten::mm", "input_tensor_ids": ["input::encoder_hidden_states", "wv2"], "output_tensor_ids": ["v2"]},
        "aten._scaled_dot_product_efficient_attention::1": {"op_type": "aten::_scaled_dot_product_efficient_attention", "input_tensor_ids": ["q2", "k2", "v2"], "output_tensor_ids": ["a1"], "input_shapes": [[1, 20, 7, 64], [1, 20, 1500, 64], [1, 20, 1500, 64]]},
        "aten.add::1": {"op_type": "aten::add", "input_tensor_ids": ["h1", "a1"], "output_tensor_ids": ["h2"]},
        # a second self-attention AFTER the cross-attention: its K/V derive from tokens AND encoder — still self
        "aten.mm::6": {"op_type": "aten::mm", "input_tensor_ids": ["h2", "wk3"], "output_tensor_ids": ["k3"]},
        "aten.mm::7": {"op_type": "aten::mm", "input_tensor_ids": ["h2", "wv3"], "output_tensor_ids": ["v3"]},
        "aten._scaled_dot_product_efficient_attention::2": {"op_type": "aten::_scaled_dot_product_efficient_attention", "input_tensor_ids": ["q2", "k3", "v3"], "output_tensor_ids": ["a2"], "input_shapes": [[1, 20, 7, 64], [1, 20, 7, 64], [1, 20, 7, 64]]},
    }}


def test_self_and_cross_attentions_are_told_apart_by_dataflow():
    p = decoder_self_attention_plan(_dag())
    assert p["num_layers"] == 2 and p["num_heads"] == 20 and p["head_dim"] == 64
    assert p["self_attn_uids"] == ["aten._scaled_dot_product_efficient_attention::0",
                                   "aten._scaled_dot_product_efficient_attention::2"]
    assert p["cross_attn_uids"] == ["aten._scaled_dot_product_efficient_attention::1"]
    assert p["arange_uids"] == ["aten.arange::0"]


def test_graph_without_attention_yields_no_plan():
    assert decoder_self_attention_plan({"input_tensor_ids": [], "ops": {}}) is None
