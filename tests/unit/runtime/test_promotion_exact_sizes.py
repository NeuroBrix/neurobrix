"""Seq-length scalar promotion: size entries match EXACTLY, slice ends keep +1
(2026-09-02, Allegro-TI2V 4-mode).

The runtime promotes concrete scalars equal to a seq_len symbol's trace
value (offset 0..1) to symbolic references, in both engines. The +1 fuzz
exists for the `seq_len + 1` pattern of causal-mask / BOS SLICING. Applied
to the SIZE list of an expand, the head count 24 (= text_len 23 + 1) was
rewritten as `text_len + 1` and the cross-attention mask expanded to
[B, 78, 1, 77] at runtime. Pins, on the triton mirror and the compiled
original: an expand / view size entry equal to seq_len + 1 stays literal;
the seq_len itself still promotes; a slice end of seq_len + 1 still
promotes with offset 1.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/runtime/test_promotion_exact_sizes.py -v
"""
from __future__ import annotations

import copy


def _dag():
    sym = {"type": "symbol", "id": "s6", "trace": 23}
    return {
        "symbolic_context": {"symbols": {
            "s6": {"name": "seq_len", "trace_value": 23, "source": "input::encoder_hidden_states::dim_1"},
        }},
        "tensors": {
            "input::encoder_hidden_states": {"shape": [3, 23, 64], "symbolic_shape": {"dims": [3, sym, 64]}},
            "param::w": {"shape": [64, 64], "weight_name": "w"},
        },
        "ops": {
            "aten.expand::0": {"op_type": "aten::expand", "attributes": {"args": [
                {"type": "tensor", "tensor_id": "input::encoder_hidden_states"},
                {"type": "list", "value": [3, 24, 1, 23]}]}},
            "aten.view::0": {"op_type": "aten::view", "attributes": {"args": [
                {"type": "tensor", "tensor_id": "input::encoder_hidden_states"},
                {"type": "list", "value": [-1, 24, 96]}]}},
            "aten.slice::0": {"op_type": "aten::slice", "attributes": {"args": [
                {"type": "tensor", "tensor_id": "input::encoder_hidden_states"},
                {"type": "scalar", "value": 1}, {"type": "scalar", "value": 0},
                {"type": "scalar", "value": 24}, {"type": "scalar", "value": 1}]}},
        },
        "execution_order": ["aten.expand::0", "aten.view::0", "aten.slice::0"],
    }


def _check(ops):
    ex = ops["aten.expand::0"]["attributes"]["args"][1]["value"]
    assert ex[1] == 24, f"the head count was promoted: {ex[1]}"
    assert isinstance(ex[3], dict) and ex[3].get("symbol_id") == "s6" and ex[3].get("offset") == 0, ex
    vw = ops["aten.view::0"]["attributes"]["args"][1]["value"]
    assert vw[1] == 24, f"the view's head count was promoted: {vw[1]}"
    sl = ops["aten.slice::0"]["attributes"]["args"][3]
    assert isinstance(sl, dict) and sl.get("symbol_id") == "s6" and sl.get("offset") == 1, sl


def test_triton_promotion_sizes_exact_slice_ends_keep_plus_one():
    from neurobrix.triton.promotion import promote_seq_len_scalars
    dag = _dag()
    promote_seq_len_scalars(dag, dag["tensors"], dag["ops"])
    _check(dag["ops"])


def test_compiled_promotion_sizes_exact_slice_ends_keep_plus_one():
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    dag = _dag()
    seq = CompiledSequence.__new__(CompiledSequence)
    seq.dag = dag
    seq._promote_seq_len_scalars_to_symbolic(dag["tensors"], dag["ops"])
    _check(dag["ops"])


def test_cross_branch_injection_protects_architectural_config_constants():
    """A literal equal to a profile.json config int (the head count) is never
    rewritten by the cross-branch expression injection, in an expand
    broadcast slot or a view slot, even when an expression with that trace
    value exists in the graph (text_len + 1 = 24 = num_attention_heads)."""
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    s10 = {"type": "symbol", "id": "s10", "trace": 23}
    plus1 = {"type": "add", "left": s10, "right": 1, "trace": 24}
    dag = {
        "symbolic_context": {"symbols": {"s10": {"name": "seq_len", "trace_value": 23, "source": "input::mask::dim_1"}}},
        "tensors": {
            "input::mask": {"shape": [3, 1, 1, 23], "symbolic_shape": {"dims": [3, 1, 1, s10]}},
            "aten.pad::0::out_0": {"shape": [3, 24], "symbolic_shape": {"dims": [3, plus1]}},
            "input::q": {"shape": [3, 385, 2304], "symbolic_shape": {"dims": [3, 385, 2304]}},
        },
        "ops": {
            "aten.expand::0": {"op_type": "aten::expand", "input_tensor_ids": ["input::mask"],
                               "input_shapes": [[3, 1, 1, 23]],
                               "attributes": {"args": [{"type": "tensor", "tensor_id": "input::mask"},
                                                       {"type": "list", "value": [3, 24, 1, 23]}]}},
            "aten.view::0": {"op_type": "aten::view", "input_tensor_ids": ["input::q"],
                             "input_shapes": [[3, 385, 2304]],
                             "attributes": {"args": [{"type": "tensor", "tensor_id": "input::q"},
                                                     {"type": "list", "value": [3, -1, 24, 96]}],
                                            "shape": [3, -1, 24, 96]}},
        },
        "execution_order": ["aten.expand::0", "aten.view::0"],
    }
    import copy

    def run(constants):
        d = copy.deepcopy(dag)
        seq = CompiledSequence.__new__(CompiledSequence)
        seq.dag = d
        seq._config_constants = set(constants)
        seq._propagate_cross_branch_expressions(d["tensors"], d["ops"])
        ex = d["ops"]["aten.expand::0"]["attributes"]["args"][1]["value"]
        vw = d["ops"]["aten.view::0"]["attributes"].get("shape")
        return ex[1], vw[2]

    assert run({24, 96, 2}) == (24, 24)          # protected: the head count stays literal
    injected = run(set())
    assert injected[0] == plus1                 # the unguarded pass DID rewrite the broadcast slot (the 2026-09-02 defect)


def test_both_mirrors_skip_a_seq_len_whose_trace_value_is_a_config_constant():
    """R30: a seq_len symbol whose trace value equals a profile.json config
    int is a collision on BOTH mirrors — the triton promotion (which also
    serves the PyTorch-sequential oracle) takes the same set."""
    from neurobrix.triton.promotion import promote_seq_len_scalars
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    dag = _dag()
    dag["symbolic_context"]["symbols"]["s6"]["trace_value"] = 24  # collides with a head count of 24
    dag["tensors"]["input::encoder_hidden_states"]["symbolic_shape"]["dims"][1]["trace"] = 24
    for op in dag["ops"].values():
        for a in op["attributes"]["args"]:
            if isinstance(a, dict) and a.get("type") == "list":
                a["value"] = [25 if v == 24 else v for v in a["value"]]
    dag["ops"]["aten.slice::0"]["attributes"]["args"][3]["value"] = 24
    import copy
    d1 = copy.deepcopy(dag); promote_seq_len_scalars(d1, d1["tensors"], d1["ops"], config_constants={24})
    assert d1["ops"]["aten.slice::0"]["attributes"]["args"][3] == {"type": "scalar", "value": 24}
    d2 = copy.deepcopy(dag); seq = CompiledSequence.__new__(CompiledSequence); seq.dag = d2; seq._config_constants = {24}
    seq._promote_seq_len_scalars_to_symbolic(d2["tensors"], d2["ops"])
    assert d2["ops"]["aten.slice::0"]["attributes"]["args"][3] == {"type": "scalar", "value": 24}
    d3 = copy.deepcopy(dag); promote_seq_len_scalars(d3, d3["tensors"], d3["ops"])
    assert d3["ops"]["aten.slice::0"]["attributes"]["args"][3].get("symbol_id") == "s6"  # unguarded: promoted
