"""A streamed component's segments are cut against what is live BESIDE them, not every peak.

`_try_layer_streaming` reserved `resident_beside` — every whole component's TOTAL, activations
included — beside the streamed component's segments. The flow runs components one after another:
the iterative handlers of both engines unload each pre_loop component after it runs and the
loop's before post_loop (`core/flow/base.py resident_together`, from 38751c12). A whole component
of ANOTHER phase keeps at most its weights beside the segments; its activation peak is never live
at the same time. Once Prism priced the fp32 activations the engines execute
(runtime_widths.py), PixArt-XL-2-1024-MS's VAE decode at 1024x2048 grew to 5 120 MB and, reserved
beside the TEXT ENCODER's segments, refused a plan that streams on the Mac (18 prism cells,
2026-09-28).

SEEN RED on 86aa0d87 (the width commit, before the co-residency reserve), 2026-09-28:
  * test_another_phase_s_activation_peak_does_not_shrink_the_segments — PixArt refused
    (layer_streaming declined: the text encoder's segments had no room for aten.embedding::0);
  * test_the_reserve_counts_weights_of_other_phases_and_totals_of_its_own — AttributeError,
    the reserve did not exist as a rule.
And with the rule's `n in phase` test replaced by True (every component concurrent again) on the
fix, both are red.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. python -m pytest -q \
     tests/unit/prism/test_a_streamed_component_reserves_only_what_runs_beside_it.py
"""
from __future__ import annotations

from types import SimpleNamespace

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung, pin_host,
                                              profile)

MB = 1024 * 1024


def _mem(name, w, a):
    return ComponentMemory(name, w * MB, a * MB, int((w + a) * MB * 0.05))


def test_the_reserve_counts_weights_of_other_phases_and_totals_of_its_own(monkeypatch):
    s = PrismSolver()
    comps = [("text_encoder", _mem("text_encoder", 9000, 100)),
             ("transformer", _mem("transformer", 1000, 400)),
             ("transformer_2", _mem("transformer_2", 800, 300)),
             ("vae", _mem("vae", 100, 5000))]
    flow = {"flow": {"type": "iterative_process", "pre_loop": ["text_encoder"],
                     "loop": {"components": ["transformer", "transformer_2"]},
                     "post_loop": ["vae"]}}
    monkeypatch.setattr(s, "_flow_topology", lambda c: flow)
    m = dict(comps)

    def weights(n):
        mm = m[n]
        return int(mm.weight_bytes + mm.overhead_bytes * mm.weight_bytes
                   / (mm.weight_bytes + mm.activation_bytes))

    # the text encoder runs alone: every other component beside it is weights only
    got = s._resident_beside_streamed(None, comps, {"text_encoder"})
    assert got == weights("transformer") + weights("transformer_2") + weights("vae")
    # a loop component: its loop partner whole, the others weights only
    got = s._resident_beside_streamed(None, comps, {"transformer"})
    assert got == m["transformer_2"].total_bytes + weights("text_encoder") + weights("vae")
    # a flow that declares no phases: every component concurrent, as before
    monkeypatch.setattr(s, "_flow_topology", lambda c: {"flow": {"type": "vlm"}})
    got = s._resident_beside_streamed(None, comps, {"text_encoder"})
    assert got == sum(mm.total_bytes for n, mm in comps if n != "text_encoder")


def test_another_phase_s_activation_peak_does_not_shrink_the_segments(monkeypatch):
    """PixArt-XL-2-1024-MS at 1024x2048 on the Mac's profile and reading (8 192 MB rung): the text
    encoder is streamed. Its VAE — post_loop, another phase — is given a 6 000 MB activation peak,
    still whole under the 7 537 MB usable rung. Reserving that peak beside the text encoder's
    segments leaves no room for the embedding op and refuses; reserving what is live beside them
    (the other components' weights) streams."""
    pin_host(monkeypatch, 24576, 11198, "the Mac's reading this case was measured at")
    impose_rung(monkeypatch, 8192)
    s = PrismSolver()
    real = s._compute_memory

    def heavy_vae(*a, **k):
        out = real(*a, **k)
        if "vae" in out:
            out["vae"].activation_bytes = 6000 * MB
        return out

    s._compute_memory = heavy_vae
    p = s.solve_smart(NBXContainer.load(str(container_root("PixArt-XL-2-1024-MS"))),
                      profile(APPLE_M4_PRO), InputConfig(batch_size=1, height=1024, width=2048),
                      mode="compiled")
    assert p.strategy == "layer_streaming", p.strategy
    parts = getattr(s, "_layer_stream_partitions", None) or {}
    assert set(parts) == {"text_encoder"}, sorted(parts)
