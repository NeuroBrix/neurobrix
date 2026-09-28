"""An activation dim contracted against a WEIGHT's dim is that weight's extent — a literal, never a symbol.

2026-09-27: PixArt-XL-2-1024-MS's timestep sinusoid is 256 wide (Timesteps(num_channels=256), typed
inside diffusers) and the tracer bound its half, 128, to the latent height (128 at the 1024 trace);
off the trace aten.addmm::0 received 2x160 against the 256x1152 time projection. The parameter
itself carried no symbol, so the parameter half of tools/weights_are_not_symbolic.py was blind to
it; this is the contraction half. Before it the function does not exist: this fails.
"""
import importlib.util
import sys
from pathlib import Path


def _tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "weights_are_not_symbolic.py"
    spec = importlib.util.spec_from_file_location("weights_are_not_symbolic_c", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _graph(act_dim):
    s4 = {"type": "symbol", "id": "s4", "trace": 128}
    return {
        "tensors": {
            "param::proj.weight": {"is_parameter": True, "symbolic_shape": {"dims": [1152, 256]}},
            "param::proj.bias": {"is_parameter": True, "symbolic_shape": {"dims": [1152]}},
            "aten.t::0::out_0": {"producer_op_uid": "aten.t::0", "symbolic_shape": {"dims": [256, 1152]}},
            "aten.cat::1::out_0": {"producer_op_uid": "aten.cat::1", "symbolic_shape": {"dims": [2, act_dim]}},
            "input::h": {"symbolic_shape": {"dims": [2, s4, s4]}},
            "aten.mm::9::out_0": {"producer_op_uid": "aten.mm::9", "symbolic_shape": {"dims": [2, s4, s4]}},
        },
        "ops": {
            "aten.t::0": {"op_type": "aten::t", "input_tensor_ids": ["param::proj.weight"]},
            "aten.addmm::0": {"op_type": "aten::addmm",
                              "input_tensor_ids": ["param::proj.bias", "aten.cat::1::out_0", "aten.t::0::out_0"]},
            # two activations: a symbolic contraction here is the request's own, never flagged
            "aten.bmm::0": {"op_type": "aten::bmm", "input_tensor_ids": ["input::h", "aten.mm::9::out_0"]},
        },
    }


def test_a_symbol_contracted_against_a_weight_is_named():
    T = _tool()
    bad = T.offending_contractions(_graph({"type": "add", "left": 128, "right": {"type": "symbol", "id": "s4"}, "trace": 256}))
    assert [(u, t, ax) for u, t, ax, _ in bad] == [("aten.addmm::0", "aten.cat::1::out_0", 1)]


def test_a_literal_contraction_and_an_activation_pair_pass():
    assert _tool().offending_contractions(_graph(256)) == []
