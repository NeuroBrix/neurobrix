"""On unified memory each phase of a lazy plan is held to the host with ITS OWN load, and a component
whose whole cost and load overflow that room is streamed — never the next phase's component.

Wan2.1-T2V-1.3B-Diffusers, triton, 480x832, 4 steps, on the Mac's `default-9f169c79` at 14 687 MB free
(the Mac, to_dell 2026-10-05 13:40): Prism walked 12 288 -> 11 264 -> 8 192 -> 6 144 and streamed the
2.7 GB transformer in eight segments, re-read on every step (~10 min a step). Every rung fit the
device; each was refused by the host side:

* 12 288 (lazy_sequential): 46 698 MB — the VAE priced untiled (37 853 MB) though the plan tiles it;
* 11 264 / 8 192 (layer_streaming): 17 211 / 16 377 MB — the umt5 encoder's 4 006 MB float32
  embedding, loaded twice (8 012 MB), added to a window that belongs to another phase.

The machine is pinned as the Mac read it: 24 576 MB, 14 687 MB available, the planning process holding
238 MB. No card.
"""
import json
import sys

import pytest

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism import host_footprint as HF
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, no_door, pin_host, profile

MODEL = "Wan2.1-T2V-1.3B-Diffusers"
FREE_MB = 14687
HELD_NOW_MB = 238


@pytest.fixture(scope="module")
def _request():
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    root = container_root(MODEL)
    man = json.loads((root / "manifest.json").read_text())
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "a red apple rolling slowly",
                                       "--steps", "4", "--height", "480", "--width", "832", "--triton"])
    return root, request_input_config(args, man, man.get("family"), root)


def _plan(monkeypatch, request):
    root, ic = request
    no_door(monkeypatch)
    monkeypatch.delenv("NBX_CENSUS", raising=False)
    monkeypatch.delenv("NBX_CENSUS_DEVICES", raising=False)
    pin_host(monkeypatch, 24576, FREE_MB, "the Mac, 2026-10-05 13:40")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(HF, "process_footprint_now", lambda: HELD_NOW_MB << 20)
    s = PrismSolver()
    return s, s.solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO), ic, mode="triton")


def test_the_transformer_stays_whole_at_the_rung_the_reading_gives(monkeypatch, _request):
    s, p = _plan(monkeypatch, _request)
    streamed = set(p.layer_stream_plan or {})
    need = (p.host_footprint["total_bytes"] - p.host_footprint["resident_bytes"]) >> 20
    assert "transformer" not in streamed, (p.strategy, streamed, p.unified_rungs_tried)
    assert [r for r, _s, _n in p.unified_rungs_tried] == [12288], p.unified_rungs_tried
    assert need <= FREE_MB, (need, p.unified_rungs_tried)
    assert "vae" in (p.component_tiling or {}), p.component_tiling


def test_a_tiled_component_is_held_at_its_tile(monkeypatch, _request):
    """The phases the host side pairs loads with price a tiled component at what it holds."""
    s, p = _plan(monkeypatch, _request)
    phases = s._unified_device_phases(NBXContainer.load(str(_request[0])), p, s._prepare_devices(
        profile(APPLE_M4_PRO)))
    vae = [held for held, names in phases if names == {"vae"}]
    assert vae and vae[0] < p.component_memory["vae"].total_bytes // 4, (vae, p.component_memory["vae"])
    # The transformer's phase holds it whole — more than the streaming window, which counts only the
    # phases a streamed component runs in — and the host side is paired phase by phase from these.
    dit = [held for held, names in phases if names == {"transformer"}]
    assert dit and dit[0] >= p.component_memory["transformer"].total_bytes > p.device_window_mb * 2**20, dit
    assert p.host_footprint["phased"], p.host_footprint


def test_each_phase_pairs_its_own_load():
    """Pure arithmetic: the load of phase A is never added to the device bytes of phase B."""
    from types import SimpleNamespace as NS
    plan = NS(loading_mode="lazy", components={"a": NS(dtype="bfloat16", device="mps:0", shard_map={}),
                                                "b": NS(dtype="bfloat16", device="mps:0", shard_map={})},
              component_memory={})
    keys = {"a": {"emb": 4000 << 20}, "b": {"w": 50 << 20}}
    fp = HF.host_footprint(plan, keys, {}, "triton", None, {"bfloat16": 2}, lambda k: False,
                           device_bytes=8000 << 20, device_phases=[(6000 << 20, {"a"}), (8000 << 20, {"b"})])
    assert fp["device_and_loading_bytes"] == max(6000 + 8000, 8000 + 100) << 20
    unpaired = HF.host_footprint(plan, keys, {}, "triton", None, {"bfloat16": 2}, lambda k: False,
                                 device_bytes=8000 << 20)
    assert unpaired["device_and_loading_bytes"] == (8000 + 8000) << 20
