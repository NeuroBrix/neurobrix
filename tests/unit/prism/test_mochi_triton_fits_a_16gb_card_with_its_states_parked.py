"""A temporal stream's states parked on the host between chunks (strategy D, Hocine 2026-10-27 11:01).

mochi-1-preview triton on ONE V100-16GB: its VAE in chunks of one frame peaked at 11 281 MB (Prism) / 16 103 MiB
(measured, f53ca7e2) — the 38 causal states (5 043 MB) live through the whole chunk, the ones it makes (graph
outputs) from their op to the chunk's end and the ones it reads from before its first op. Parked, each state the
chunk makes leaves for the host at its op (`temporal_stream_runner.park_state`) and each one it reads comes back right
before its first reader (`ParkedTensor`, the sequences' `defer_inputs`); Prism prices exactly that
(`estimate_peak_memory` deferred_inputs / parked_outputs). The states stay on the card when the chunk fits so.

What each test would do if the code were wrong: a deferred input counted from the first op, or a parked output kept
to the end — the synthetic figures equal the unparked ones (red); the planner never parking — mochi triton 16g is
refused (red on 1fe9dcec); the planner parking a stream that fits — the 32g plan carries park_states (red)."""
from __future__ import annotations

import copy

from neurobrix.core.prism.profiler import ActivationProfiler
from tests.unit.prism._pinned_machine import pin_host


def _t(n):
    return {"shape": [1, n], "dtype": "float16"}


def _op(uid, ins, outs):
    return {"op_uid": uid, "op_type": "aten::mul", "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def _chunk():
    """op0 makes the state `so` from the chunk; op2 is the peak (a, b); op3 reads the state `s` it was handed."""
    T = {"input::x": _t(1000), "input::s": _t(500), "so": _t(300), "a": _t(2000), "b": _t(2000), "c": _t(100)}
    O = [_op("op0", ["input::x"], ["so"]), _op("op1", ["input::x"], ["a"]), _op("op2", ["a"], ["b"]),
         _op("op3", ["b", "input::s"], ["c"])]
    return {"tensors": T, "ops": {o["op_uid"]: o for o in O}, "execution_order": [o["op_uid"] for o in O],
            "input_tensor_ids": ["input::x", "input::s"], "output_tensor_ids": ["c", "so"], "torch_dtype": "float16"}


def test_a_parked_state_is_priced_only_while_it_is_on_the_card():
    g = _chunk()

    def peak(**kw):
        return ActivationProfiler(g).estimate_peak_memory(dtype_bytes=2, handed_inputs=True, **kw).peak_bytes
    assert peak() == 2 * (500 + 300 + 2000 + 2000)
    assert peak(deferred_inputs=frozenset({"input::s"})) == 2 * (300 + 2000 + 2000)
    assert peak(parked_outputs=frozenset({"so"})) == 2 * (500 + 2000 + 2000)
    assert peak(deferred_inputs=frozenset({"input::s"}), parked_outputs=frozenset({"so"})) == 2 * (2000 + 2000)


def _solve(monkeypatch, memory_mb, total_mb):
    from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, profile
    from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config, no_door
    from neurobrix.core.prism import PrismSolver
    prof = copy.deepcopy(V100_16GB)
    prof["devices"][0].update(memory_mb=memory_mb)
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, total_mb, 306, "an idle V100")
    c, ic = _input_config("mochi-1-preview", "triton")
    return PrismSolver().solve_smart(c, profile(prof), ic, mode="triton")


def test_mochi_triton_plans_on_one_16gb_card_with_its_states_parked(monkeypatch):
    p = _solve(monkeypatch, 16384, 16151)
    stream = (p.component_stream or {}).get("vae")
    assert stream and stream["park_states"], (p.strategy, stream, getattr(p, "stream_declined", None))
    assert not p.execution_door, p.execution_door


def test_a_stream_that_fits_keeps_its_states_on_the_card(monkeypatch):
    p = _solve(monkeypatch, 32768, 32501)
    stream = (p.component_stream or {}).get("vae")
    assert stream and not stream["park_states"], stream
