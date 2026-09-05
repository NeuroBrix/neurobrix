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


def _dag_with_positional_table_slice():
    """A whisper-class decoder traced with the positions as rows [0, seq_len)
    of a parameter table (no arange anywhere)."""
    d = _dag()
    del d["ops"]["aten.arange::0"]
    d["symbolic_context"] = {"symbols": {
        "s0": {"name": "batch", "source": "input::input_ids::dim_0", "trace_value": 1},
        "s1": {"name": "seq_len", "source": "input::input_ids::dim_1", "trace_value": 7},
        "s3": {"name": "seq_len", "source": "input::encoder_hidden_states::dim_1", "trace_value": 1500}}}
    d["ops"]["aten.slice::0"] = {
        "op_type": "aten::slice", "input_tensor_ids": ["param::embed_positions.weight"],
        "output_tensor_ids": ["pos"],
        "attributes": {"args": [{"type": "tensor", "tensor_id": "param::embed_positions.weight"},
                                {"type": "scalar", "value": 0}, {"type": "scalar", "value": 0},
                                {"type": "symbol", "id": "s1", "trace": 7}]}}
    # a slice of the ENCODER states by the encoder length is not a positional table
    d["ops"]["aten.slice::1"] = {
        "op_type": "aten::slice", "input_tensor_ids": ["param::other"], "output_tensor_ids": ["x"],
        "attributes": {"args": [{"type": "tensor", "tensor_id": "param::other"},
                                {"type": "scalar", "value": 0}, {"type": "scalar", "value": 0},
                                {"type": "symbol", "id": "s3", "trace": 1500}]}}
    return d


def test_a_positional_table_slice_is_the_second_positional_mechanism():
    p = decoder_self_attention_plan(_dag_with_positional_table_slice())
    assert p["arange_uids"] == []
    assert p["position_slice_uids"] == ["aten.slice::0"]


def test_the_arange_form_reports_no_positional_slice():
    p = decoder_self_attention_plan(_dag())
    assert p["arange_uids"] == ["aten.arange::0"] and p["position_slice_uids"] == []


def test_position_slice_interceptor_shifts_the_window_by_the_cache_length():
    import torch
    from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper
    w = KVCacheAttentionWrapper.__new__(KVCacheAttentionWrapper)
    table = torch.arange(10, dtype=torch.float32).unsqueeze(1)   # row i holds value i
    w._is_prefill = True; w._position_offset = 0
    assert w.intercept_position_slice(table, 0, 0, 7, 1).squeeze(1).tolist() == list(range(7))
    w._is_prefill = False; w._position_offset = 7
    assert w.intercept_position_slice(table, 0, 0, 1, 1).squeeze(1).tolist() == [7.0]
    w._position_offset = 8
    assert w.intercept_position_slice(table, 0, 0, 1, 1).squeeze(1).tolist() == [8.0]
