"""The placement map binds the request's own extents as asked; the trace floor is for a guess.

`build_symbol_map(placement_floor=True)` floored EVERY named symbol at its trace value, one axis
at a time. The floor was written for a guessed binding — Qwen3-Omni's audio axis named `seq_len`
taking the global text length (~128) instead of its 441-frame trace, 2026-08-10 — and it also
caught the request's frames, height and width, which are not guesses. Wan2.1-VACE traced at
480x832 and asked at 832x480 (the vendor's portrait size, the same area) priced its width at the
trace's 104 latent columns where it asked for 60: a 104x104 grid, the transformer 18.04 GiB
instead of 10.41 GiB, and the plan fell from lazy_sequential to cpu_streaming (2026-09-29).

Injection, seen RED: the floor back on time/height/width -> the swap and the fewer-frames cases.
The `seq_len` case is the floor's own reason and stays floored.
"""
from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig

GRAPH = {
    "execution_order": [], "ops": {}, "tensors": {},
    "symbolic_context": {"symbols": {
        "s0": {"name": "batch", "trace_value": 1, "source": "input::hidden_states::dim_0"},
        "s1": {"name": "time", "trace_value": 21, "source": "input::hidden_states::dim_2"},
        "s2": {"name": "height", "trace_value": 60, "source": "input::hidden_states::dim_3"},
        "s3": {"name": "width", "trace_value": 104, "source": "input::hidden_states::dim_4"},
        "s4": {"name": "seq_len", "trace_value": 441, "source": "input::input_features::dim_1"},
    }, "expressions": {}},
}


def _map(height, width, frames, seq_len=None):
    req = InputConfig(batch_size=1, height=height, width=width, num_frames=frames,
                      temporal_compression=4, vae_scale=8, seq_len=seq_len)
    return ActivationProfiler(GRAPH).build_symbol_map(req, placement_floor=True, flow=False)


def test_the_swapped_aspect_is_priced_at_its_own_grid():
    m = _map(832, 480, 81)
    assert (m["s2"], m["s3"]) == (104, 60)
    assert m["s2"] * m["s3"] == 60 * 104, "the same area costs the same"


def test_fewer_frames_are_priced_at_their_own_count():
    assert _map(480, 832, 33)["s1"] == 9          # (33 - 1) // 4 + 1


def test_a_guessed_sequence_length_keeps_the_floor():
    assert _map(480, 832, 81, seq_len=128)["s4"] == 441


class _Flow:
    """The request's flow, as `FlowBindings.overrides` hands it to the map: the CFG batch."""
    def overrides(self, dag, input_config):
        return {"s0": 2}


def test_the_flows_batch_is_priced_at_the_flows_batch():
    """Wan2.2-I2V's transformer traced at batch 3 (a collision-free stimulus, R39); the flow runs
    the CFG pair, 2. Floored, the plan priced 3 (12.26 GiB for 8.2) and fell to cpu_streaming."""
    g = dict(GRAPH, symbolic_context={"symbols": dict(GRAPH["symbolic_context"]["symbols"], s0={
        "name": "batch", "trace_value": 3, "source": "input::hidden_states::dim_0"}), "expressions": {}})
    req = InputConfig(batch_size=2, height=480, width=832, num_frames=81, temporal_compression=4,
                      vae_scale=8, flow=_Flow())
    assert ActivationProfiler(g).build_symbol_map(req, placement_floor=True)["s0"] == 2


def test_an_image_encoder_is_priced_at_the_view_the_run_feeds_it():
    """Wan2.1-I2V's image encoder declares `height`/`width` symbols (trace 224) on `pixel_values`; the
    run feeds it the CLIP view of the build's own processor (224x224), whatever the request. Bound by
    name, the unfloored map priced the request's latent grid (60x104 at 480x832) — a guess the old
    per-axis floor only covered by coincidence (review 2026-09-29). The flow binds the view."""
    from neurobrix.core.prism.flow_bindings import FlowBindings
    from tests.unit.prism._pinned_machine import container_root
    import json
    root = container_root("Wan2.1-I2V-14B-480P-Diffusers")
    topo = json.loads((root / "topology.json").read_text())
    dag = json.loads((root / "components" / "image_encoder" / "graph.json").read_text())
    req = InputConfig(batch_size=2, height=480, width=832, num_frames=81, temporal_compression=4,
                      vae_scale=8, flow=FlowBindings(topo, root))
    m = ActivationProfiler(dag).build_symbol_map(req, placement_floor=True)
    table = dag["symbolic_context"]["symbols"]
    got = {table[s]["name"]: v for s, v in m.items() if s in table and table[s]["name"] in ("height", "width")}
    assert got == {"height": 224, "width": 224}, got
