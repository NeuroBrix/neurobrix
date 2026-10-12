"""A component larger than the rung streams INSIDE ITSELF — never a refusal while streaming can run it.

Projecting the census table onto the Mac's profile (Apple M4 Pro, 18 186 MB unified, 24 576 MB host;
tools/derived_census.py table, every rung, both Triton modes), Prism refused five models at the top
rung with "the last rung needs only the largest single component to fit" (the Mac, 2026-10-04 02:16,
results/derived_table_2026_10_04/table_30.log). Reproduced here on merge-queue-17 (e5f03258), the
machine pinned, the rung imposed through the door. Three causes, each a rule of Prism:

1. THE PARTITIONER KEPT DEAD TENSORS ALIVE. `LayerPartitioner.live_activation_curve` freed a tensor
   at its last CONSUMER only; an output no op reads (layer-norm statistics, attention log-sum-exps,
   RoPE `copy_` results) stayed alive to the last op. Wan2.2-I2V-A14B's transformer at 480x832x81:
   108 800 MB of "activations alone" — the profiler's own liveness (`dag_last_uses`, which frees a
   dead output at its producer) prices it 7 737. One liveness for both walks now.
2. ANOTHER PHASE'S WEIGHTS WERE RESERVED BESIDE THE SEGMENTS, and a bigger rung refused what a
   smaller one planned: CogVideoX-5b-I2V at 16 384 kept its 11 003 MB text encoder whole and reserved
   it beside the transformer's segments (3 088 MB left); at 8 192 the encoder was itself streamed and
   the plan held. A layer_streaming plan is lazy, its serving session never persistent, and both
   engines' iterative handlers unload a pre_loop component after it runs: another phase holds
   nothing. The transformer then fits ONE piece of the partitioner's figure while the whole rungs
   refused it by the arena's figure — it is cut in two instead of declined.
3. A NO-PHASE FLOW'S WHOLE COMPONENTS FILLED THE RUNG: Ming-Lite-Omni-1.5's 24 115 MB of components
   that each fit whole, reserved beside its 62 GB language model against 15 073 MB usable. The
   component whose reserve weighs most is streamed too, until the segments fit; the KV check then
   prices the cut's own window, not every streamed peak summed.

RETRACED (2026-10-05): Wan2.1-I2V-14B-480P-Diffusers and Allegro-TI2V were excluded by name: they
refused because their `vae_encoder` graphs FROZE the time axis — Wan2.1's output is
`[1, 3, 16, h', w']` (batch and latent time concrete; Wan2.2's is `[s0, f(s1), 16, h', w']`) and
Allegro's `[1, (s0*s1 ...), 4, h', w']` (batch fused into time, the known batch freeze). No tile maps
the time symbol, and the graph cannot run the request's frames anyway: a Forge re-trace (principle 1),
not a Prism rule. With that encoder's activations brought under the rung, both plan layer_streaming
(probe, 2026-10-04) — it is their only cause. Forge re-traced both encoders (2026-10-04); they are
now in STREAMED, and the door that named them asserts their time axis stays one symbol.

SEEN RED on e5f03258 (2026-10-04): every plan cell refused (the five texts above), the dead-output
cell kept 1 000 000 bytes alive, the reserve cell counted another phase's weights, the lazy cell
read `eager` on a door-imposed card. Injections on the fix (7bbad2bd), each in its own worktree:
liveness back to consumers only -> Wan2.2 (both Triton modes) and the dead-output cell RED; promotion
off -> Ming RED; the one-piece cut off -> CogVideoX RED; the other-phase reserve back to weights -> the
reserve cell RED while CogVideoX still PLANS — promotion streams its text encoder beside the
transformer instead, so the plan holds and only the reserve cell guards that rule.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. python -m pytest -q \\
     tests/unit/prism/test_a_component_over_the_rung_streams_inside_itself.py
"""
from __future__ import annotations


import sys
from pathlib import Path

import pytest

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.layer_partition import LayerPartitioner
from neurobrix.core.prism.solver import ComponentMemory
from neurobrix.nbx import NBXContainer
from tests.unit.prism._graph import as_graph
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung, no_door,
                                              pin_host, profile)

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import trace_request as TR  # noqa: E402

MB = 1024 * 1024
MODES = ["triton", "triton_sequential", "compiled"]
#: Wan2.1-I2V-14B-480P and Allegro-TI2V joined on 2026-10-05: Forge re-traced their `vae_encoder`
#: with a symbolic time axis (topologies of 2026-10-04 23:31 and 23:18), the frozen-encoder door
#: below turned red as it was written to, and both now plan layer_streaming at the top rung.
STREAMED = ["CogVideoX-5b-I2V", "Ming-Lite-Omni-1.5", "Wan2.2-I2V-A14B-Diffusers",
            "Wan2.1-I2V-14B-480P-Diffusers", "Allegro-TI2V"]
#: The encoders the door named, now re-traced: their time axis is ONE symbol (asserted below).
RETRACED_ENCODER = ["Wan2.1-I2V-14B-480P-Diffusers", "Allegro-TI2V"]


def _input_config(model: str, mode: str):
    """The model's DERIVED request (tools/trace_request.derived_request), resolved by the run's own
    `request_input_config` — the request the census plans."""
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    flag = {"triton": "--triton", "triton_sequential": "--triton-sequential", "compiled": "--compiled"}[mode]
    args = create_parser().parse_args(["run", "--model", model, *TR.derived_request(model), flag])
    c = NBXContainer.load(str(container_root(model)))
    man = c.get_manifest() or {}
    return c, request_input_config(args, man, man.get("family"), c.cache_path)


def _plan(model, mode):
    c, ic = _input_config(model, mode)
    s = PrismSolver()
    return s.solve_smart(c, profile(APPLE_M4_PRO), ic, mode=mode), s


# ───────────────── the five, on the Mac's profile ─────────────────

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", STREAMED)
def test_at_the_top_rung_the_component_streams_inside_itself(monkeypatch, model, mode):
    pin_host(monkeypatch, 24576, 18186, "the Mac's idle reading, its profile's capacity")
    impose_rung(monkeypatch, 16384)
    p, s = _plan(model, mode)
    usable = s._usable_mb(s._prepare_devices(profile(APPLE_M4_PRO))[0])
    if mode == "compiled" and p.strategy != "layer_streaming":
        # The compiled engine holds a whole component without the Triton arena's factor
        # (`_live_activation_mb`): CogVideoX's transformer fits whole there, and a whole rung wins.
        assert p.strategy not in ("cpu_execution", "cpu_streaming"), p.strategy
        return
    if p.cfg_split_components and p.strategy != "layer_streaming":
        # One guidance branch per pass holds every component resident — a class above streaming
        # (`PrismSolver._split_guidance_batch`): Allegro-TI2V's transformer, whose FFN at the guidance
        # batch is what made it stream. Resident is the stronger answer to "never a refusal".
        assert "holds every component resident" in p.selection_reason, p.selection_reason
        assert all(not d.startswith(("cpu", "zero3")) for a in p.components.values() for d in a.devices), (
            {n: a.devices for n, a in p.components.items()})
        return
    assert p.strategy == "layer_streaming", f"{model} [{mode}] planned {p.strategy!r}"
    assert p.loading_mode == "lazy"
    assert p.device_window_mb is not None and p.device_window_mb <= usable + 1e-6, (
        f"{model} [{mode}]: window {p.device_window_mb} MB over the usable {usable:.0f} MB")


#: OWED by the token-axis brick (this branch, prism-an-oversized-op-is-split-at-the-source): Ming's
#: vision tower adds its mask to [heads, N, N] attention scores — 4 420 MB in, 4 420 MB out, alive
#: together: 9 878 MB while that op runs (`LayerPartitioner.op_peak_curve`), against 1 946 MB of
#: segment budget on the busy Mac. Priced after the op (5 458 MB), the plan streamed it and
#: under-priced the run by the score tensor. The slice that serves it runs the queries in slices with
#: the keys whole, which `chunked_region` does not yet do ("S on both operands' free dimensions").
#: Strict: the day the slice lands this cell turns red and the mark is removed.
OWED_BUSY = {"Ming-Lite-Omni-1.5": "attention scores sliced on the query axis (chunked_region)"}


@pytest.mark.parametrize("model", [pytest.param(m, marks=pytest.mark.xfail(strict=True, reason=OWED_BUSY[m]))
                                   if m in OWED_BUSY else m for m in STREAMED])
def test_on_the_busy_mac_the_plan_streams_at_the_rung_its_reading_gives(monkeypatch, model):
    """~12 000 MB free: no door, the unified descent from the reading's own rung."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 12000, "the Mac with ~12 GB free")
    p, _ = _plan(model, "triton")
    assert p.strategy == "layer_streaming", f"{model} planned {p.strategy!r}"


@pytest.mark.parametrize("model", RETRACED_ENCODER)
def test_a_retraced_encoder_maps_time_through_one_symbol(model):
    """The door that replaced `a_frozen_encoder_time_axis_is_named_not_planned_around`: those
    encoders froze their time axis (Wan2.1: `[1, 3, 16, h', w']`; Allegro: batch folded into time).
    Re-traced, the latent time dim of their output is an expression of the frame symbol alone —
    the property the tile planner reads — and the top-rung plan above streams them."""
    import json
    g = json.loads((container_root(model) / "components" / "vae_encoder" / "graph.json").read_text())
    out = g["tensors"][g["output_tensor_ids"][0]]["symbolic_shape"]["dims"]

    def symbols(d):
        if isinstance(d, dict):
            if d.get("type") == "symbol":
                return {d["id"]}
            return set().union(*(symbols(v) for v in d.values() if isinstance(v, (dict, list))))
        if isinstance(d, list):
            return set().union(*(symbols(v) for v in d)) if d else set()
        return set()
    time_dims = [d for d in out[1:3] if symbols(d)]
    assert time_dims, f"{model}: the encoder's latent time is concrete again: {out[1:3]}"
    assert all(len(symbols(d)) == 1 for d in time_dims), (
        f"{model}: the latent time folds several symbols (a batch freeze): {time_dims}")


# ───────────────── the rules, on graphs and figures built here ─────────────────

def test_an_output_no_op_reads_is_dead_at_its_producer():
    """Op `a` makes `x` (read by `b`) and `stat` (read by nothing, 1 000 000 bytes). After `a` the
    curve holds `x` only. Consumers-only liveness kept `stat` alive to the end."""
    t = lambda n: {"shape": [n], "dtype": "uint8"}
    g = {"tensors": {"in": t(10), "x": t(100), "stat": t(1_000_000), "y": t(100)},
         "ops": {"a": {"op_type": "aten::native_layer_norm", "input_tensor_ids": ["in"],
                       "output_tensor_ids": ["x", "stat"]},
                 "b": {"op_type": "aten::relu", "input_tensor_ids": ["x"], "output_tensor_ids": ["y"]}},
         "execution_order": ["a", "b"], "output_tensor_ids": ["y"]}
    assert LayerPartitioner(as_graph(g)).live_activation_curve() == [100, 100]


def test_another_phase_reserves_nothing_beside_the_segments(monkeypatch):
    s = PrismSolver()
    m = {n: ComponentMemory(n, w * MB, a * MB, int((w + a) * MB * 0.05))
         for n, w, a in (("text_encoder", 9000, 100), ("transformer", 1000, 400),
                         ("transformer_2", 800, 300), ("vae", 100, 5000))}
    comps = list(m.items())
    monkeypatch.setattr(s, "_flow_topology", lambda c: {"flow": {
        "type": "iterative_process", "pre_loop": ["text_encoder"],
        "loop": {"components": ["transformer", "transformer_2"]}, "post_loop": ["vae"]}})
    assert s._resident_beside_streamed(None, comps, {"text_encoder"}) == 0
    assert s._resident_beside_streamed(None, comps, {"transformer"}) == m["transformer_2"].total_bytes
    # a whole component NO phase names is concurrent: nothing says it is unloaded
    extra = ComponentMemory("extra", 700 * MB, 70 * MB, 0)
    assert (s._resident_beside_streamed(None, comps + [("extra", extra)], {"text_encoder"})
            == extra.total_bytes)


def test_a_layer_streaming_plan_is_lazy_even_where_the_door_leaves_the_card_roomy():
    """`layer_streaming` is no AllocationStrategy member; its loading mode came from the summed CUDA
    capacity and read `eager` whenever the plan's totals sat under 90 % of a card the door had
    imposed a smaller rung on — a serving session then keeps every weight of every phase resident
    (serving/engine.py), which the reserve above prices at nothing."""
    from neurobrix.core.prism.solver import DeviceState
    from neurobrix.core.prism.structure import DeviceBrand, DeviceSpec
    from tests.unit.prism._pinned_machine import V100_16GB
    s = PrismSolver()
    spec = DeviceSpec(index=0, name="dev", memory_mb=16384, compute_capability="7.0",
                      supports_dtypes=["float16"], architecture="volta", brand=DeviceBrand.NVIDIA)
    dev = DeviceState(device_string="cuda:0", capacity_mb=15565.0, spec=spec, recommended_mb=15565.0)
    mem = {"m": ComponentMemory("m", 100 * MB, 100 * MB, 0)}
    plan = s._build_plan({"m": ("cuda:0", {})}, mem, [dev], {"m": "float16"}, profile(V100_16GB),
                         "layer_streaming")
    assert plan.loading_mode == "lazy"
