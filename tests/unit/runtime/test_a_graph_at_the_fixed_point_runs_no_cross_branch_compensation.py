"""A graph at Forge's re-propagation fixed point runs with no cross-branch compensation.

`CompiledSequence._propagate_cross_branch_expressions` is a compensation layer for builds
whose tracer left a cross-branch dimension literal: it collects every expression the graph's
tensors carry, keyed by TRACE VALUE, and rewrites a literal in an expand / view / reshape that
happens to equal one. Its own docstring states that a correctly traced graph must behave
identically with it off, and the Triton branch never had it (R30).

Measured 2026-10-04 on two graphs Forge's single write (ec71599) leaves at the fixed point:

* Ming-Lite-Omni-1.5 `model.model`: the re-propagation symbolised the MoE per-expert token
  counts (`s0*s1*6 - 1106`, trace 4; `- 1107`, trace 3) and turned the M-RoPE plane count into
  the literal 3, so those expressions became the ONLY ones of trace value 3 and 4. The pass
  then rewrote the GQA broadcast `expand [s0, 4, 4, s1, 128]` and its merge
  `_unsafe_view [s0, 16, s1, 128]` into `(s0*s1*6 - 1106) * 4`: at a 275-token prompt
  (s0=1, s1=275) the K view became 2176 heads and the first attention failed
  ("size of tensor a (16) must match the size of tensor b (2176)"). 168 rewrites in the staged
  graph, 0 in the installed one.
* Sana_1600M_4Kpx_BF16 `vae`: the literal 4096 of `view::31` (1024 channels x 4) was merged
  into an expression whose trace value is 1024 and evaluated at 4K to 1536.

The single write records the fixed point on every component it writes (`symbolic_fixed_point`
in the container's `extracted_values`, the value is the writer's revision), and a component that
carries it never reaches the pass.

What would this file do if the code were wrong? Without the door the first two cells fail:
the pass rewrites the literal 4 / 16 of the Ming fixture and the literal 4096 of the Sana
fixture. With the door keyed on anything but the component's own declaration, the third cell
fails (a container without the flag — an old build — still needs the compensation). With the
executor not reading the flag, the fourth fails.
"""
from __future__ import annotations

import copy

import pytest

import neurobrix.core.runtime.registry_flags as rf
from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
from neurobrix.nbx import component_flags


def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


def _moe_count(trace):
    # Forge's re-propagated per-expert count: s0*s1*6 - (1110 - trace)
    total = {"type": "mul", "left": {"type": "mul", "left": _sym("s0", 5), "right": _sym("s1", 37),
                                     "trace": 185}, "right": 6, "trace": 1110}
    return {"type": "sub", "left": total, "right": 1110 - trace, "trace": trace}


def _t(dims, concrete):
    return {"symbolic_shape": {"dims": dims, "concrete": concrete}}


def _ming_dag():
    """The GQA chain of Ming's first attention + one MoE count carrying trace value 4."""
    s0, s1 = _sym("s0", 5), _sym("s1", 37)
    return {
        "tensors": {
            "k": _t([s0, 4, 1, s1, 128], [5, 4, 1, 37, 128]),
            "k_exp": _t([s0, 4, 4, s1, 128], [5, 4, 4, 37, 128]),
            "k_view": _t([s0, 16, s1, 128], [5, 16, 37, 128]),
            "moe_rows": _t([_moe_count(4), 2048], [4, 2048]),
        },
        "ops": {
            "aten.expand::3": {
                "op_type": "aten::expand", "input_tensor_ids": ["k"], "output_tensor_ids": ["k_exp"],
                "input_shapes": [[5, 4, 1, 37, 128]],
                "attributes": {"args": [{"type": "tensor", "tensor_id": "k"},
                                        {"type": "list", "value": [s0, 4, 4, s1, 128]}],
                               "size": [s0, 4, 4, s1, 128]}},
            "aten._unsafe_view::3": {
                "op_type": "aten::_unsafe_view", "input_tensor_ids": ["k_exp"],
                "output_tensor_ids": ["k_view"], "input_shapes": [[5, 4, 4, 37, 128]],
                "attributes": {"args": [{"type": "tensor", "tensor_id": "k_exp"},
                                        {"type": "list", "value": [s0, 16, s1, 128]}],
                               "shape": [s0, 16, s1, 128]}},
        },
    }


def _sana_vae_dag():
    """Sana's vae `view::31`: [s0, 1024, 4, s1, s2] -> [s0, 4096, s1, s2], next to a tensor
    carrying an expression of trace value 1024 (the merge the pass synthesises x 4)."""
    s0, s1, s2 = _sym("s0", 1), _sym("s1", 128), _sym("s2", 128)
    e1024 = {"type": "mul", "left": {"type": "mul", "left": s0, "right": 32, "trace": 32},
             "right": {"type": "floordiv", "left": s1, "right": 4, "trace": 32}, "trace": 1024}
    return {
        "tensors": {
            "x": _t([s0, 1024, 4, s1, s2], [1, 1024, 4, 128, 128]),
            "y": _t([s0, 4096, s1, s2], [1, 4096, 128, 128]),
            "attn": _t([e1024, 32], [1024, 32]),
        },
        "ops": {
            "aten.view::31": {
                "op_type": "aten::view", "input_tensor_ids": ["x"], "output_tensor_ids": ["y"],
                "input_shapes": [[1, 1024, 4, 128, 128]],
                "attributes": {"args": [{"type": "tensor", "tensor_id": "x"},
                                        {"type": "list", "value": [s0, 4096, s1, s2]}],
                               "shape": [s0, 4096, s1, s2]}},
        },
    }


def _run_pass(dag, at_fixed_point):
    d = copy.deepcopy(dag)
    seq = CompiledSequence.__new__(CompiledSequence)
    seq.dag = d
    seq._config_constants = set()
    seq._graph_at_fixed_point = at_fixed_point
    seq._propagate_cross_branch_expressions(d["tensors"], d["ops"])
    return d["ops"]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("NBX_DISABLE_CROSS_BRANCH", raising=False)
    monkeypatch.setattr(rf, "_REGISTRY_CACHE", None)
    monkeypatch.setattr(rf, "_find_registry_yaml", lambda: None)
    component_flags.clear()
    yield
    component_flags.clear()
    rf._REGISTRY_CACHE = None


def test_ming_s_gqa_heads_stay_literal_at_the_fixed_point():
    ops = _run_pass(_ming_dag(), at_fixed_point=True)
    assert ops["aten.expand::3"]["attributes"]["args"][1]["value"][2] == 4
    assert ops["aten._unsafe_view::3"]["attributes"]["args"][1]["value"][1] == 16
    assert ops["aten._unsafe_view::3"]["attributes"]["shape"][1] == 16


def test_sana_s_vae_channels_stay_literal_at_the_fixed_point():
    ops = _run_pass(_sana_vae_dag(), at_fixed_point=True)
    assert ops["aten.view::31"]["attributes"]["shape"][1] == 4096
    assert ops["aten.view::31"]["attributes"]["args"][1]["value"][1] == 4096


def test_an_old_build_without_the_declaration_keeps_its_compensation():
    """The door is the component's own declaration, nothing else: the same graphs without it are
    rewritten exactly as the 2026-10-04 runs measured (this is the defect the door closes)."""
    ming = _run_pass(_ming_dag(), at_fixed_point=False)
    assert isinstance(ming["aten._unsafe_view::3"]["attributes"]["shape"][1], dict)
    sana = _run_pass(_sana_vae_dag(), at_fixed_point=False)
    assert isinstance(sana["aten.view::31"]["attributes"]["shape"][1], dict)


def test_the_executor_reads_the_declaration_the_container_carries(tmp_path):
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    (tmp_path / "manifest.json").write_text('{"model_name": "Ming-Lite-Omni-1.5"}')
    component_flags.register("Ming-Lite-Omni-1.5", {
        "model.model": {"symbolic_fixed_point": "ec71599"},
        "vision": {"hidden_size": 2048},
    })
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._cache_path = str(tmp_path)
    ex._component_name = "model.model"
    assert ex._graph_at_fixed_point() is True
    ex._component_name = "vision"
    assert ex._graph_at_fixed_point() is False
