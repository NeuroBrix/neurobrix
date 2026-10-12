"""A streamed component's load is priced one PIECE at a time, never the whole component at once.

Measured 2026-10-08 on the M4 Pro, deepseek-moe-16b-chat native (apple-on-mq20 8635257c),
layer_streaming: the plan priced its host side 65 260 MB, of which "loading 61 670 MB" — the
whole 31 GB container passed through the loader twice (stored + pinned copy) — against a
measured peak memory footprint of 3 606 MB (`/usr/bin/time -l`,
`native_2026_10_08/measure/deepseek-moe-16b-chat.native.log`). The streaming strategy loads a
component segment by segment (`_piece_executor`), so the dearest piece bounds what one load
holds. Over-pricing every streamed plan by its whole container made the price useless as a
gate: Allegro was priced 33 103 MB against 16 986 MB free and kept regardless ("the system may
page").

The oracle takes each streamed component's dearest-piece share of its stored bytes from the
partition the solver chose, and the whole-component figure from the container's shard sizes.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_a_streamed_component_is_loaded_one_piece_at_a_time.py
"""
from __future__ import annotations

from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                              pin_host, profile)

MODEL = "GLM-4.1V-9B-Thinking"


def test_a_streamed_load_is_priced_by_its_dearest_piece(monkeypatch):
    from neurobrix.core.prism import InputConfig, PrismSolver
    from neurobrix.nbx import NBXContainer
    pin_host(monkeypatch, 24576, 17667, "the Mac's reading, 2026-10-08 10:18")
    impose_rung(monkeypatch, 4096)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    container = NBXContainer.load(str(container_root(MODEL)))
    s = PrismSolver()
    p = s.solve_smart(container, profile(APPLE_M4_PRO), InputConfig(batch_size=1), mode="compiled")
    assert p.strategy == "layer_streaming", f"precondition: the Mac's plan streams ({p.strategy!r})"
    shards = container.get_shard_sizes()
    parts = {c: part for c, part in s._layer_stream_partitions.items()
             if c in p.layer_stream_plan and part.total_weight_bytes}
    whole = max(sum(sh.values()) for sh in shards.values() if sh)
    piece = max(sum(shards[c].values()) * max(g.weight_bytes for g in part.segments)
                // part.total_weight_bytes for c, part in parts.items() if shards.get(c))
    streamed_dearest = max((sum(shards[c].values()) for c in parts if shards.get(c)), default=0)
    assert streamed_dearest == whole, "precondition: the dearest component is a streamed one"
    # a component not streamed still passes whole (GLM's model.visual, 1 702 MB)
    unstreamed = max((sum(sh.values()) for c, sh in shards.items() if sh and c not in parts),
                     default=0)
    t = p.host_footprint["transient_bytes"]
    # mq25 merge (Dell, 2026-10-12): the rack's 8ef1165f prices the same load with the loader's
    # concurrency (the dearest segment's mapped pages + io.num_workers x (1 + pinned uploads in
    # flight) largest tensors, read/converted/pinned — measured on the V100 rack, stage B), which this
    # oracle's one-piece-x-2 bound (1 702 MB x 2 here) does not model: 13 396 MB on this plan. The
    # rule-independent claim kept: a streamed component is never priced as a whole pass.
    bound = 2 * max(piece, unstreamed)
    assert bound < whole, "precondition: the bound separates a piece from the whole component"
    assert t < whole, (f"loading priced {t / 2**20:.0f} MB; the whole streamed component is "
                       f"{whole / 2**20:.0f} MB (one piece x 2 = {bound / 2**20:.0f} MB)")
