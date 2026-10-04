"""The compiled sequence re-keys no literal, and a graph not written at the fixed point is refused.

The compiled sequence's cross-branch compensation re-keyed an expand / view / reshape literal
equal to the trace value of the only expression of that value in the graph. On graphs the
single write leaves at the symbolic fixed point it could only corrupt literals (2026-10-04):

* Ming-Lite-Omni-1.5 `model.model`: the re-propagated MoE token counts (`s0*s1*6 - 1106`,
  trace 4) became the only expressions of trace 4, and the GQA broadcast `expand [s0,4,4,s1,128]`
  and its merge `_unsafe_view [s0,16,s1,128]` were rewritten from them: 2176 heads at a
  275-token prompt, "size of tensor a (16) must match the size of tensor b (2176)".
* Sana_1600M_4Kpx_BF16 `vae view::31`: the literal 4096 was merged into an expression of
  trace 1024 and evaluated at 4K to 1536.

The pass is deleted (owner, 2026-10-04: no compensation, fix at the source, zero fallback) and
the loader refuses, by name, a container whose graphs do not declare the fixed point
(`symbolic_fixed_point` in extracted_values, written by the single write).

What would this file do if the code were wrong? With the pass still in `compile()`, the two
literal cells fail on the exact expressions above (measured on the base tree, 1a6a7544). With
the loader's door removed, the refusal cell fails (the load runs past the topology); with the
door keyed on anything but each graph's own declaration, the partial-stamp cell or the
non-graph cell fails.
"""
from __future__ import annotations

import copy
import json

import pytest
import torch

from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.nbx.fixed_point import FIXED_POINT_FLAG, unstamped_components
from neurobrix.nbx.neurotax import NEUROTAX_VERSION


def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


def _t(dims, concrete):
    return {"shape": concrete, "dtype": "float32",
            "symbolic_shape": {"dims": dims, "concrete": concrete}}


def _op(op_type, src, dst, in_shape, dims, key):
    return {"op_type": op_type, "input_tensor_ids": [src], "output_tensor_ids": [dst],
            "input_shapes": [in_shape],
            "attributes": {"args": [{"type": "tensor", "tensor_id": src},
                                    {"type": "list", "value": dims}], key: dims}}


def _ming_dag():
    """Ming's first GQA chain beside one re-propagated MoE count of trace value 4."""
    s0, s1 = _sym("s0", 5), _sym("s1", 37)
    rows = {"type": "mul", "left": {"type": "mul", "left": s0, "right": s1, "trace": 185},
            "right": 6, "trace": 1110}
    moe = {"type": "sub", "left": rows, "right": 1106, "trace": 4}
    return {
        "symbolic_context": {"symbols": {
            "s0": {"trace_value": 5, "source": "input::k::dim_0"},
            "s1": {"trace_value": 37, "source": "input::k::dim_3"}}},
        "tensors": {
            "input::k": _t([s0, 4, 1, s1, 128], [5, 4, 1, 37, 128]),
            "k_exp": _t([s0, 4, 4, s1, 128], [5, 4, 4, 37, 128]),
            "k_view": _t([s0, 16, s1, 128], [5, 16, 37, 128]),
            "moe_rows": _t([moe, 2048], [4, 2048]),
        },
        "ops": {
            "aten.expand::3": _op("aten::expand", "input::k", "k_exp", [5, 4, 1, 37, 128],
                                  [s0, 4, 4, s1, 128], "size"),
            "aten._unsafe_view::3": _op("aten::_unsafe_view", "k_exp", "k_view",
                                        [5, 4, 4, 37, 128], [s0, 16, s1, 128], "shape"),
        },
        "execution_order": ["aten.expand::3", "aten._unsafe_view::3"],
        "input_tensor_ids": ["input::k"], "output_tensor_ids": ["k_view"],
    }


def _sana_vae_dag():
    """Sana's vae `view::31` [s0,1024,4,s1,s2] -> [s0,4096,s1,s2] beside an expression of
    trace value 1024."""
    s0, s1, s2 = _sym("s0", 1), _sym("s1", 128), _sym("s2", 128)
    e1024 = {"type": "mul", "left": {"type": "mul", "left": s0, "right": 32, "trace": 32},
             "right": {"type": "floordiv", "left": s1, "right": 4, "trace": 32}, "trace": 1024}
    return {
        "symbolic_context": {"symbols": {
            "s0": {"trace_value": 1, "source": "input::x::dim_0"},
            "s1": {"trace_value": 128, "source": "input::x::dim_3"},
            "s2": {"trace_value": 128, "source": "input::x::dim_4"}}},
        "tensors": {
            "input::x": _t([s0, 1024, 4, s1, s2], [1, 1024, 4, 128, 128]),
            "y": _t([s0, 4096, s1, s2], [1, 4096, 128, 128]),
            "attn": _t([e1024, 32], [1024, 32]),
        },
        "ops": {"aten.view::31": _op("aten::view", "input::x", "y", [1, 1024, 4, 128, 128],
                                     [s0, 4096, s1, s2], "shape")},
        "execution_order": ["aten.view::31"],
        "input_tensor_ids": ["input::x"], "output_tensor_ids": ["y"],
    }


def _compiled_ops(dag):
    seq = CompiledSequence(copy.deepcopy(dag), torch.device("cpu"), torch.float32)
    seq.compile()
    return seq.dag["ops"]


def test_ming_s_gqa_heads_stay_literal_through_compile():
    ops = _compiled_ops(_ming_dag())
    assert ops["aten.expand::3"]["attributes"]["args"][1]["value"][2] == 4
    assert ops["aten._unsafe_view::3"]["attributes"]["args"][1]["value"][1] == 16
    assert ops["aten._unsafe_view::3"]["attributes"]["shape"][1] == 16


def test_sana_s_vae_channels_stay_literal_through_compile():
    ops = _compiled_ops(_sana_vae_dag())
    assert ops["aten.view::31"]["attributes"]["args"][1]["value"][1] == 4096
    assert ops["aten.view::31"]["attributes"]["shape"][1] == 4096


def _container(tmp_path, graphs, others=(), stamped=()):
    d = tmp_path / "m"
    (d / "runtime").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"model_name": "m", "neurotax_version": NEUROTAX_VERSION}))
    for name in graphs:
        (d / "components" / name).mkdir(parents=True)
        (d / "components" / name / "graph.json").write_text("{}")
    for name in others:
        (d / "components" / name).mkdir(parents=True)
    values = {name: {FIXED_POINT_FLAG: "671cafd"} for name in stamped}
    values["_global"] = {}
    (d / "topology.json").write_text(json.dumps({"extracted_values": values}))
    for rel in ("runtime/variables.json", "runtime/defaults.json"):
        (d / rel).write_text("{}")
    return d


def test_a_graph_without_the_declaration_is_refused_by_name(tmp_path):
    d = _container(tmp_path, graphs=("model", "resampler"), stamped=("model",))
    with pytest.raises(RuntimeError, match="SYMBOLIC FIXED POINT") as e:
        NBXRuntimeLoader().load(str(d))
    assert "['resampler']" in str(e.value) and str(d) in str(e.value)


def test_each_graph_answers_for_itself_and_a_component_without_a_graph_needs_nothing(tmp_path):
    d = _container(tmp_path, graphs=("model", "lm_head"), others=("tokenizer",),
                   stamped=("model", "lm_head"))
    assert unstamped_components(d, json.loads((d / "topology.json").read_text())) == []
    try:
        NBXRuntimeLoader().load(str(d))
    except RuntimeError as e:
        assert "SYMBOLIC FIXED POINT" not in str(e)
    except Exception:
        pass  # the empty fixture fails further on; only the door is under test
