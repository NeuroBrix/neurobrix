"""A streamed plan's device price on a unified device covers EVERY phase of its flow, not only the
phase its segments stream in.

Rebuilt 2026-10-08 on the M4 Pro from the request of CogVideoX-2b's native run
(`native_2026_10_08/measure/CogVideoX-2b.native.log`, 480x720, 49 frames, 4 steps, 16 981 MB free):
layer_streaming streams only `text_encoder`; the unified device term was its window, 4 610 MB,
while the flow's next phases hold the transformer whole (4 622 MB) and then the VAE at its tiled
cost (4 436 MB). The window is the right figure for cutting segments — a component in another
phase is beside nothing (`_resident_beside_streamed`) — and the wrong one for what the run holds
at its worst moment, which the host price adds on a unified device.

The oracle reads the plan's own components: each phase without a streamed component holds at least
its components' totals, a tiled one at least its tiled activations.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_a_streamed_plan_prices_every_phase_on_a_unified_device.py
"""
from __future__ import annotations

from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, pin_host, profile

MODEL = "CogVideoX-2b"
REQUEST = ["--prompt", "a red apple rolling slowly across a wooden table", "--seed", "42",
           "--steps", "4", "--height", "480", "--width", "720"]


def test_every_phase_of_a_streamed_plan_is_inside_its_unified_device_price(monkeypatch):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism import PrismSolver
    from neurobrix.nbx import NBXContainer
    pin_host(monkeypatch, 24576, 16981, "the CogVideoX-2b native run's reading, 2026-10-08")
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    container = NBXContainer.load(str(container_root(MODEL)))
    manifest = container.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", MODEL, *REQUEST])
    request = request_input_config(args, manifest, manifest.get("family"), container._cache_path)
    s = PrismSolver()
    p = s.solve_smart(container, profile(APPLE_M4_PRO), request, mode="compiled")
    assert p.strategy == "layer_streaming" and p.device_window_mb, f"precondition: streamed ({p.strategy!r})"
    streamed = set(p.layer_stream_plan or {})
    tiling = p.component_tiling or {}

    def held(name):
        if name in tiling:
            return int(tiling[name]["tiled_activation_bytes"])
        return int(p.component_memory[name].total_bytes)

    phases = [ph for ph in s._flow_phases(container) if not ph & streamed]
    assert phases, "precondition: the flow has a phase without a streamed component"
    dearest = max(sum(held(n) for n in ph if n in p.component_memory) for ph in phases)
    device = p.host_footprint["device_bytes"]
    assert device >= dearest, (f"unified device priced {device / 2**20:.0f} MB (window "
                               f"{p.device_window_mb:.0f} MB); a phase without a streamed "
                               f"component holds {dearest / 2**20:.0f} MB")
