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
STREAMED = ["CogVideoX-5b-I2V", "Wan2.2-I2V-A14B-Diffusers",
            "Wan2.1-I2V-14B-480P-Diffusers", "Allegro-TI2V"]
#: Ming-Lite-Omni-1.5 left STREAMED on 2026-10-14: its vision tower was priced at its TRACE's 17 020 patches
#: (9.65 GiB of activations) because no flow rule bound it to the request's image; priced at the image
#: (1 024 patches, 0.05 GiB — test_a_vision_tower_is_priced_at_the_requests_image.py) its components fit and
#: the plan holds them on the device, never streamed, never refused, never on the host.
WHOLE_AT_THE_IMAGE = ["Ming-Lite-Omni-1.5"]
#: The encoders the door named, now re-traced: their time axis is ONE symbol (asserted below).
RETRACED_ENCODER = ["Wan2.1-I2V-14B-480P-Diffusers", "Allegro-TI2V"]


def _input_config(model: str, mode: str):
    """The model's DERIVED request (tools/trace_request.derived_request), resolved by the run's own
    `request_input_config` — the request the census plans."""
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    flag = {"triton": "--triton", "triton_sequential": "--triton-sequential", "compiled": "--compiled"}[mode]
    c = NBXContainer.load(str(container_root(model)))
    args = create_parser().parse_args(["run", "--model", model, *TR.derived_request(model), flag])
    man = c.get_manifest() or {}
    return c, request_input_config(args, man, man.get("family"), c.cache_path)


def _plan(model, mode):
    c, ic = _input_config(model, mode)
    s = PrismSolver()
    return s.solve_smart(c, profile(APPLE_M4_PRO), ic, mode=mode), s


# ───────────────── the five, on the Mac's profile ─────────────────

def _rung_below_the_largest_component(monkeypatch, model, mode) -> int:
    """The highest ladder rung under the model's largest whole component (its cost at the top rung): the rung at
    which that component streams inside itself and nothing smaller is asked of the cut."""
    from neurobrix.core.prism.memory_budget import memory_ladder_mb
    no_door(monkeypatch)
    p, s = _plan(model, mode)
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    c = NBXContainer.load(str(container_root(model)))
    whole = max(s._whole_component_mb(c, n, m, dev) for n, m in p.component_memory.items())
    below = [r for r in memory_ladder_mb() if r < whole]
    assert below, (model, whole)
    return below[-1]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", STREAMED)
def test_under_a_rung_below_its_largest_component_the_component_streams_inside_itself(monkeypatch, model, mode):
    """Under the budget law of 2026-10-25 (the margin is the allocator's reserve, no longer a 12 % hedge) these
    transformers fit the 16 384 rung whole per phase and plan lazy_sequential there, rightly; the rule this cell
    holds — a component over the rung streams inside itself, never a refusal — is proven at the ladder rung just
    under the model's largest whole component (CogVideoX-5b 13 455 MB -> 12 288; Allegro-TI2V 11 586 -> 11 264)."""
    pin_host(monkeypatch, 24576, 18186, "the Mac's idle reading, its profile's capacity")
    impose_rung(monkeypatch, _rung_below_the_largest_component(monkeypatch, model, mode))
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


#: Ming's busy-Mac cell was marked owed (strict xfail) for a query-axis slice of its vision tower's
#: [heads, N, N] attention scores — 4 420 MB each, at the TRACE's N = 17 020 patches. At the request's image
#: N is 1 024 and the scores are 16 MB: the debt was the mispricing (2026-10-14), not a missing slice. The
#: cell moved to WHOLE_AT_THE_IMAGE below.
OWED_BUSY = {}


@pytest.mark.parametrize("model", [pytest.param(m, marks=pytest.mark.xfail(strict=True, reason=OWED_BUSY[m]))
                                   if m in OWED_BUSY else m for m in STREAMED])
def test_on_the_busy_mac_the_idle_plan_runs_if_it_fits_else_the_machine_is_replanned(monkeypatch, model):
    """~12 000 MB free, no door. The plan is sized by the pool's facts (the owner, 2026-10-25 15:16): when its host
    side fits the memory free now, the busy machine runs the idle machine's plan; when it does not, the machine is
    RE-PLANNED on what is free and said in clear (the owner, 2026-10-26 19:21), never refused while a plan fits.
    Before 19:21 the door refused it by name. What this test would do if the code were wrong: a plan that changed with
    room to spare, or a door left shut (or a silent re-plan) on a plan that does not fit — RED."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    idle, _ = _plan(model, "triton")
    pin_host(monkeypatch, 24576, 12000, "the Mac with ~12 GB free")
    busy, _ = _plan(model, "triton")
    assert not busy.execution_door, (model, busy.execution_door)
    if busy.busy_replan is None:
        assert busy.strategy == idle.strategy, (model, busy.strategy, idle.strategy)
    else:
        assert busy.busy_replan["from"] == idle.strategy, (model, busy.busy_replan, idle.strategy)
        assert "machine busy" in busy.selection_reason and "12000 MB free now" in busy.selection_reason, (
            busy.selection_reason)
