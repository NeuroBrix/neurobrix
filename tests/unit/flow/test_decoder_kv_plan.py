"""The decoder self-attention plan is derived from the graph's dataflow: a
cross-attention's keys/values do not derive from the token inputs."""
from conftest import _dag, _dag_with_positional_table_slice  # noqa: F401
from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan


def test_self_and_cross_attentions_are_told_apart_by_dataflow():
    p = decoder_self_attention_plan(_dag())
    assert p["num_layers"] == 2 and p["num_heads"] == 20 and p["head_dim"] == 64
    assert p["self_attn_uids"] == ["aten._scaled_dot_product_efficient_attention::0",
                                   "aten._scaled_dot_product_efficient_attention::2"]
    assert p["cross_attn_uids"] == ["aten._scaled_dot_product_efficient_attention::1"]
    assert p["arange_uids"] == ["aten.arange::0"]


def test_graph_without_attention_yields_no_plan():
    assert decoder_self_attention_plan({"input_tensor_ids": [], "ops": {}}) is None


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
