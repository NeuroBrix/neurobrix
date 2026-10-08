"""A model that fits whole is planned whole — Prism reads what the FLOW keeps loaded, and prices no
buffer the engine never allocates.

The Mac measured (2026-10-04 00:5x, 04:20): orpheus-3b streamed at ~10-12 GB free although its
7 241 MB of weights and a 152 MB KV cache fit (196 s for 16 tokens, every token re-reading the
language model's segments), and Janus-Pro-7B streamed on the idle machine (2 281 s, its harness cell
timed out). Reproduced on the Mac's profile, the machine pinned. Four causes, none of them a rung too
small:

1. THE TRANSPOSE OF A WEIGHT WAS PRICED AS AN ALLOCATION. The Triton sequence removes `aten::t` on a
   parameter at bind time (`_eliminate_weight_transpose_ops`), triton_sequential runs it as a stride
   view, and `mm`/`addmm` read the view in place. The profiler counted its output: orpheus's
   `lm_head` (156 940 x 3 072) planned 927 MB of "activations" for 7 MB of logits, and at 12 000 MB
   free the plan's host side overshot the reading by that much — rung 8 192, then streamed. Free on
   the TRITON engines only, and only where every reader is `mm`/`addmm`: the compiled engine keeps
   the bytes (on fp16 hardware its fp32 wrapper copies the weight per call — a real buffer, not yet
   measured), and so does a transpose `matmul`'s batched route expands.
2. THE OUTPUT BOUNDARY PRICED THE LANGUAGE MODEL'S LOGITS AS THE SAVED IMAGE. Janus's language model
   returns `[s0, s1, 102 400]`; at the placement floor that is 360 448 000 elements, 3 437 MB of
   host at the image family's save cost, for a 384x384 picture the DECODER returns.
3. A PLAN THAT LOADS ON DEMAND WAS HELD TO EVERY COMPONENT AT ONCE on a unified device: its host
   side added the plan's total. The triton image decode releases the language model before the
   decoder loads (`triton/flow/autoregressive.py execute`), on the first request of a process.
4. THE KV CHECK SUMMED EVERY COMPONENT for the same plan.

The lifecycle is the flow's own (`core/flow/base.py resident_together(topology, engine, served)`):
the two handlers of the image decode release at different points, so the phases are per engine; the
triton handler leaves the decoder loaded when the request ends, so a SERVED plan has one phase. The
KV cache is in no phase — `session.cleanup()` keeps its buffers — and is held against whichever
phase is dearest. A text decode declares none — its head and its codec stage run beside the loaded
model in both engines — and keeps the SUM.

SEEN RED on 13f517b8 (2026-10-04): orpheus at 12 000 MB free and Janus on the idle Mac planned
layer_streaming; the lm_head cell read 927 MB; the output cell 3 437 MB; the phase, KV and host cells
had no function to call.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. python -m pytest -q \\
     tests/unit/prism/test_a_model_that_fits_whole_is_planned_whole.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.core.flow.base import resident_together, saved_output_components
from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.layer_partition import LayerPartitioner
from neurobrix.core.prism.profiler import ActivationProfiler, weight_transposes_read_in_place
from neurobrix.core.prism.solver import unified_device_bytes
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, no_door,
                                              pin_host, profile)

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import trace_request as TR  # noqa: E402

MB = 1024 * 1024
TRITON_MODES = ["triton", "triton_sequential"]
#: The strategies that hold every component they run WHOLE on the device.
WHOLE = {"single_gpu", "single_gpu_lifecycle", "lazy_sequential"}


def _plan(model: str, mode: str):
    """The model's DERIVED request (tools/trace_request.derived_request) planned on the Mac's profile."""
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    flag = {"triton": "--triton", "triton_sequential": "--triton-sequential", "compiled": "--compiled"}[mode]
    args = create_parser().parse_args(["run", "--model", model, *TR.derived_request(model), flag])
    c = NBXContainer.load(str(container_root(model)))
    man = c.get_manifest() or {}
    s = PrismSolver()
    return s.solve_smart(c, profile(APPLE_M4_PRO), request_input_config(args, man, man.get("family"), c.cache_path),
                         mode=mode), s


def _topology(model: str) -> dict:
    return json.loads((container_root(model) / "topology.json").read_text())


# ───────────────────── the two models, on the Mac's profile ─────────────────────

@pytest.mark.parametrize("mode", TRITON_MODES)
def test_orpheus_is_planned_whole_at_12_gb_free(monkeypatch, mode):
    """12 000 MB free, no door: 7 241 MB of weights, a 152 MB cache. It streamed."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 12000, "the Mac with ~12 GB free")
    p, _ = _plan("orpheus-3b-0.1-ft", mode)
    assert p.strategy in WHOLE, f"orpheus [{mode}] planned {p.strategy!r}"
    assert p.device_window_mb is None
    fp = p.host_footprint
    need_mb = (fp["total_bytes"] - fp["resident_bytes"]) / MB
    assert need_mb <= 12000, f"the plan's host side asks {need_mb:.0f} MB of 12 000 free"


def test_janus_is_planned_whole_on_the_idle_mac(monkeypatch):
    """18 186 MB free, no door: the language model whole at the 16 384 rung. It streamed at 12 288.

    The triton mode only. triton_sequential prices the same language model's activations at
    3 410 MB (2 035 under triton); whole, at the arena's factor, that is 16 813 MB against 15 073
    usable, so that mode streams at the 16 384 rung — by its own figure, not by a lifecycle."""
    mode = "triton"
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p, _ = _plan("Janus-Pro-7B", mode)
    assert p.strategy in WHOLE, f"Janus [{mode}] planned {p.strategy!r}"
    assert p.device_window_mb is None
    fp = p.host_footprint
    assert (fp["total_bytes"] - fp["resident_bytes"]) / MB <= 18186
    # the host side holds the decode phase, not the decoder beside it
    total = sum(int(m.total_bytes) for m in p.component_memory.values())
    decoder = int(p.component_memory["gen_vision_model"].total_bytes)
    assert fp["device_bytes"] == total - decoder, (fp["device_bytes"] / MB, total / MB, decoder / MB)


def test_a_language_model_over_the_rung_is_still_streamed(monkeypatch):
    """The lifecycle frees nothing the decode needs: Janus's 12 380 MB language model at the Mac's
    measured 15 725 MB free (rung 12 288) cannot be held whole, and is not."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 15725, "the Mac at its 04:20 reading")
    p, _ = _plan("Janus-Pro-7B", "triton")
    assert p.strategy == "layer_streaming", p.strategy


# ───────────────────── the transpose of a weight ─────────────────────

def _linear(reader="aten::mm", weight="param::weight", returned=False, second_reader=None):
    """`t(weight) -> reader`, 1 000 x 100 float32: the transposed view is 400 000 bytes."""
    dag = {
        "tensors": {
            weight: {"shape": [1000, 100], "dtype": "float32", "is_parameter": True},
            "input::x": {"shape": [1, 100], "dtype": "float32"},
            "aten.t::0::out_0": {"shape": [100, 1000], "dtype": "float32"},
            "reader::0::out_0": {"shape": [1, 1000], "dtype": "float32"},
        },
        "ops": {
            "aten.t::0": {"op_type": "aten::t", "input_tensor_ids": [weight],
                          "output_tensor_ids": ["aten.t::0::out_0"]},
            "reader::0": {"op_type": reader, "input_tensor_ids": ["input::x", "aten.t::0::out_0"],
                          "output_tensor_ids": ["reader::0::out_0"]},
        },
        "execution_order": ["aten.t::0", "reader::0"],
        "output_tensor_ids": ["reader::0::out_0"] + (["aten.t::0::out_0"] if returned else []),
    }
    if second_reader:
        dag["tensors"]["other::0::out_0"] = {"shape": [100, 1000], "dtype": "float32"}
        dag["ops"]["other::0"] = {"op_type": second_reader, "input_tensor_ids": ["aten.t::0::out_0"],
                                  "output_tensor_ids": ["other::0::out_0"]}
        dag["execution_order"].append("other::0")
    return dag


def test_a_weight_transpose_read_in_place_is_no_buffer():
    assert weight_transposes_read_in_place(_linear("aten::mm")) == {"aten.t::0"}
    assert weight_transposes_read_in_place(_linear("aten::addmm")) == {"aten.t::0"}
    assert weight_transposes_read_in_place(_linear("aten::mm", weight="buffer::table")) == {"aten.t::0"}
    # every other shape keeps its bytes: a reader that copies (matmul's batched route expands the
    # weight — Ming's image_vae), a second reader that is not a contraction, a transpose the graph
    # returns, the transpose of an ACTIVATION (no pass removes it)
    assert weight_transposes_read_in_place(_linear("aten::expand")) == set()
    assert weight_transposes_read_in_place(_linear("aten::matmul")) == set()
    assert weight_transposes_read_in_place(_linear("aten::mm", second_reader="aten::clone")) == set()
    assert weight_transposes_read_in_place(_linear("aten::mm", returned=True)) == set()
    assert weight_transposes_read_in_place(_linear("aten::mm", weight="aten.add::0::out_0")) == set()


def test_the_estimate_drops_it_only_for_an_engine_that_reads_in_place():
    """The caller says which engine it prices: the compiled estimate keeps the transposed bytes."""
    profiler = ActivationProfiler(_linear("aten::mm"))
    kept = profiler.estimate_peak_memory().peak_bytes
    free = profiler.estimate_peak_memory(in_place_weight_reads=True).peak_bytes
    assert kept - free == 400_000, (kept, free)
    copied = ActivationProfiler(_linear("aten::expand"))
    assert (copied.estimate_peak_memory(in_place_weight_reads=True).peak_bytes
            == copied.estimate_peak_memory().peak_bytes)


def test_the_partitioner_still_prices_a_cut_between_a_transpose_and_its_contraction():
    """On the partitioner's curve the figure is the cost of a CUT: a seam between `t(weight)` and
    the `mm` that reads it would carry the weight out of the piece that loaded it. Priced at zero
    there, that seam looked free (the pinned boundaries of DeepSeek-Coder-V2-Lite moved onto an
    `aten.t`, 2026-10-04)."""
    assert LayerPartitioner(_linear("aten::mm"), {}).live_activation_curve() == [400_000, 4_000]


def test_orpheus_lm_head_is_priced_at_its_logits_on_triton(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p, _ = _plan("orpheus-3b-0.1-ft", "triton")
    head = p.component_memory["lm_head"]
    assert head.weight_bytes > 900 * MB
    assert head.activation_bytes < 64 * MB, f"lm_head activations {head.activation_bytes / MB:.0f} MB"


def test_the_compiled_estimate_keeps_the_transposed_weight(monkeypatch):
    """Not symmetry for its own sake: on fp16 hardware the compiled engine's fp32 wrapper copies the
    weight for a contraction outside an fp16-safe contract (core/dtype/engine.py
    `_make_fp32_wrapper`), and the transpose's bytes are the only price near that copy. The day the
    copy is measured and priced as itself, this cell is re-read."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p, _ = _plan("orpheus-3b-0.1-ft", "compiled")
    head = p.component_memory["lm_head"]
    assert head.activation_bytes > 900 * MB, f"lm_head activations {head.activation_bytes / MB:.0f} MB"


# ───────────────────── the flow's lifecycle ─────────────────────

def test_an_image_decode_releases_its_language_model_before_its_decoder_on_triton():
    topo = _topology("Janus-Pro-7B")
    gen = topo["flow"]["generation"]
    lm, decoder, head = gen["lm_component"], gen["decoder_component"], gen["head_component"]
    triton = resident_together(topo, "triton")
    assert not any(lm in ph and decoder in ph for ph in triton), triton
    assert any(lm in ph and head in ph for ph in triton) and any(decoder in ph and head in ph for ph in triton)
    # the compiled handler decodes with the language model loaded; an unnamed engine is held to it;
    # and so is a SERVED triton plan — that handler never unloads the decoder, so from the second
    # request of a session it is loaded during the decode
    for phases in (resident_together(topo, "compiled"), resident_together(topo, None),
                   resident_together(topo, "triton", served=True)):
        assert any(lm in ph and decoder in ph for ph in phases), phases
    assert saved_output_components(topo) == {decoder}


def test_the_phases_are_the_release_order_the_handlers_have():
    """The door that turns red when a handler changes what it releases, and when: `RESIDENT_PHASES`
    is a reading of two `execute` bodies, and nothing else ties it to them."""
    import inspect
    from neurobrix.core.flow.autoregressive import AutoregressiveHandler
    from neurobrix.triton.flow.autoregressive import TritonAutoregressiveHandler
    triton = inspect.getsource(TritonAutoregressiveHandler.execute)
    image = triton.index("if is_image:")
    released, decoded = triton.index("session.cleanup()", image), triton.index("strategy.process_output(")
    assert image < released < decoded, "triton: the language model is no longer released before the decoder"
    assert "_unload_non_lm_weights(" not in triton, (
        "triton: the handler now unloads the decoder when the request ends — a served plan may take "
        "the two phases too (core/flow/base.py _autoregressive_phases)")
    compiled = inspect.getsource(AutoregressiveHandler.execute)
    assert (compiled.index("strategy.process_output(") < compiled.index("session.cleanup()")
            < compiled.index("self._unload_non_lm_weights(")), (
        "compiled: the decoder no longer runs beside the loaded language model — one phase is no "
        "longer what this handler does")
    # a text decode: the codec stage precedes the session's release in both
    assert triton.index("self._run_snac_codec_decoder(") < triton.rindex("session.cleanup()")
    assert compiled.index("self._run_snac_codec_decoder(") < compiled.index("session.cleanup()")


def test_a_text_decode_declares_no_phases():
    """orpheus: the head and the codec stage run beside the loaded model in both engines."""
    topo = _topology("orpheus-3b-0.1-ft")
    assert resident_together(topo, "triton") is None and resident_together(topo, "compiled") is None
    assert saved_output_components(topo) is None


def test_the_iterative_phases_are_the_same_on_both_engines():
    flow = {"flow": {"type": "iterative_process", "pre_loop": ["enc"], "loop": {"components": ["dit"]},
                     "post_loop": ["vae"]}}
    assert (resident_together(flow, "triton") == resident_together(flow, "compiled")
            == resident_together(flow) == [{"enc"}, {"dit"}, {"vae"}])


# ───────────────────── the arithmetic, on a built flow ─────────────────────

def _mem(total_mb, act_mb=0.0, over_mb=0.0):
    return SimpleNamespace(total_bytes=int(total_mb * MB), activation_bytes=int(act_mb * MB),
                           overhead_bytes=int(over_mb * MB))


#: an image decode: a language model, its head, its decoder, and a tower the generation never names
COMPS = {"lm": _mem(10_000, act_mb=900), "head": _mem(200), "dec": _mem(700), "tower": _mem(600)}
ON_DEVICE = {n: ("mps:0", {}) for n in COMPS}
KV = 500 * MB


def _built(tmp_path, mode, generation_type="autoregressive_image"):
    (tmp_path / "topology.json").write_text(json.dumps({"flow": {
        "type": "autoregressive_generation",
        "generation": {"type": generation_type, "lm_component": "lm", "head_component": "head",
                       "decoder_component": "dec"}}}))
    s = PrismSolver()
    s._lm_component_name = "lm"
    s._mode = mode
    return s, SimpleNamespace(cache_path=tmp_path)


def _kv(s, strategy, container):
    return s._resident_bytes_for_kv_check(strategy, ON_DEVICE, COMPS, profile(APPLE_M4_PRO), KV,
                                          container=container) / MB


def test_the_kv_check_holds_the_cache_against_the_dearest_phase(tmp_path):
    s, c = _built(tmp_path, "triton")
    # on demand, triton, one run: {lm, head} is the dearest phase, with the unnamed tower — the
    # decoder is never loaded beside the model
    assert _kv(s, "lazy_sequential", c) == 10_000 - 500 + 200 + 600
    assert _kv(s, "cpu_streaming", c) == 10_000 - 500 + 200 + 600
    # an eager strategy releases nothing it loads
    assert _kv(s, "component_placement", c) == 10_000 - 500 + 200 + 700 + 600
    # the compiled handler decodes beside the loaded model
    s, c = _built(tmp_path, "compiled")
    assert _kv(s, "lazy_sequential", c) == 10_000 - 500 + 200 + 700 + 600
    # a text decode declares no phases: the SUM
    s, c = _built(tmp_path, "triton", generation_type="autoregressive_text")
    assert _kv(s, "lazy_sequential", c) == 10_000 - 500 + 200 + 700 + 600


def test_a_decoder_dearer_than_the_model_is_what_the_cache_is_held_against(tmp_path, monkeypatch):
    """The cache's buffers outlive the language model: when the decoder's phase is the dearer one,
    it is that phase the cache must fit beside — not the model's."""
    s, c = _built(tmp_path, "triton")
    monkeypatch.setitem(COMPS, "dec", _mem(12_000))
    assert _kv(s, "lazy_sequential", c) == 200 + 12_000 + 600


def test_a_served_plan_keeps_what_the_handler_never_unloads(tmp_path):
    """A session's second request finds the decoder loaded (the triton handler leaves it): the SUM,
    whether the session is hot or degraded to cold."""
    s, c = _built(tmp_path, "triton")
    s._serve_mode = s._serve_requested = True
    assert _kv(s, "lazy_sequential", c) == 10_000 - 500 + 200 + 700 + 600
    s._serve_cold_fallback = True
    assert _kv(s, "component_placement", c) == 10_000 - 500 + 200 + 700 + 600


def test_a_plan_that_loads_on_demand_holds_its_dearest_phase_on_a_unified_device(tmp_path, monkeypatch):
    s, c = _built(tmp_path, "triton")
    plan = SimpleNamespace(component_memory=COMPS, device_window_mb=None, loading_mode="lazy",
                           total_memory_mb=11_500.0, kv_cache_plan=SimpleNamespace(memory_bytes=KV))
    peak = s._peak_loaded_bytes(c, plan)
    assert peak == (10_000 + 200 + 600) * MB          # {lm, head} + the tower; not {head, dec}
    apple, v100 = profile(APPLE_M4_PRO), profile(V100_16GB)
    assert unified_device_bytes(plan, apple, peak) == peak
    assert unified_device_bytes(plan, v100, peak) == 0                       # a discrete card: another pool
    plan.loading_mode = "eager"
    assert unified_device_bytes(plan, apple, peak) == 11_500 * MB            # eager keeps them all
    plan.loading_mode, plan.device_window_mb = "lazy", 4_000.0
    assert unified_device_bytes(plan, apple, 0) == 4_000 * MB                # a streamed plan: at least its window
    plan.layer_stream_plan = {"lm": object()}                                 # lm streamed: its phases hold the rest whole
    s._layer_stream_cost = {n: int(m.total_bytes) for n, m in COMPS.items()}  # what that rung prices, none tiled
    streamed_peak = s._peak_loaded_bytes(c, plan)
    assert streamed_peak == (200 + 700 + 500 + 600) * MB                      # {head, dec} + the cache + the tower
    assert unified_device_bytes(plan, apple, 9_000 * MB) == 9_000 * MB        # and a dearer whole phase (2026-10-08)
    del plan.layer_stream_plan
    # the decoder's phase carries the cache the model left behind
    monkeypatch.setitem(COMPS, "dec", _mem(9_800))
    assert s._peak_loaded_bytes(c, plan) == (200 + 9_800 + 500 + 600) * MB
    # no phases declared: the plan's total
    s, c = _built(tmp_path, "triton", generation_type="autoregressive_text")
    assert s._peak_loaded_bytes(c, plan) == (10_000 + 200 + 9_800 + 600) * MB


def test_a_streamed_component_in_two_phases_reserves_the_dearest(tmp_path):
    """The head runs beside the language model, then beside the decoder: streamed, it reserves the
    dearer of the two beside its segments, with the tower no phase names."""
    s, c = _built(tmp_path, "triton")
    comps = sorted(COMPS.items(), key=lambda kv: -kv[1].total_bytes)
    assert s._resident_beside_streamed(c, comps, {"head"}) == (10_000 + 600) * MB
    assert s._resident_beside_streamed(c, comps, {"lm"}) == (200 + 600) * MB
    assert s._resident_beside_streamed(c, comps, {"dec"}) == (200 + 600) * MB


@pytest.mark.parametrize("minimum", [1, 2], ids=["run", "serve-two-turns"])
def test_segments_cut_in_another_phase_still_reserve_the_cache(tmp_path, minimum):
    """A whole language model carries the cache's estimate in its total, and that total is reserved
    beside the segments only where the model IS beside them. A streamed decoder on triton runs after
    the model is released and the cache's buffers are still allocated: the whole need is reserved."""
    def reserve(mode, streamed):
        s, c = _built(tmp_path, mode)
        s._target_dtype_str = "bfloat16"
        s._estimate_kv_cache_bytes = lambda *_: KV
        s._kv_min_bytes = lambda *_: minimum * KV
        comps = sorted(COMPS.items(), key=lambda kv: -kv[1].total_bytes)
        return s._kv_reserve_beside(c, comps, streamed, s._resident_beside_streamed(c, comps, streamed))
    need = minimum * KV
    assert reserve("triton", {"dec"}) == (need, need)             # the model is in another phase
    assert reserve("triton", {"head"}) == (need - KV, need)       # the model, whole, is beside the head
    assert reserve("compiled", {"dec"}) == (need - KV, need)      # one phase: the model is beside
    assert reserve("triton", {"lm"}) == (need, need)              # the model itself streamed


def test_the_output_boundary_prices_what_the_flow_saves(monkeypatch):
    """Janus: the decoder's 1 x 3 x 384 x 384 picture, not the language model's logits."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p, s = _plan("Janus-Pro-7B", "triton")
    assert s._output_elements["gen_vision_model"] == 3 * 384 * 384
    assert s._output_elements["language_model"] > 100 * s._output_elements["gen_vision_model"]
    assert 0 < p.host_footprint["output_bytes"] < 16 * MB, p.host_footprint["output_bytes"] / MB
