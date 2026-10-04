"""A refusal says, rung by rung, why a component got no tile — under the compiled engine too.

The tiling reason reached the refusal only through the Triton host-placement line (a host placement
sized on the card). Under the compiled engine a declined component went to the host and the refusal
said nothing of the tile it did not get (Allegro-TI2V re-propagated, 720x1280, 2026-10-04); and
layer streaming recorded nothing at all when its tiler returned a tile that its weights took over
the rung. The doctrine review of 64c24b13 asked for each rung's own reasons beside the figure they
were held to, from a real solve.

Injections, each seen RED: layer streaming's oversized-tile reason removed -> the real-solve case;
the per-rung compiled lines removed from the refusal -> the real-solve and placement cases; the host
rungs not skipped -> the de-duplication case.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory
from neurobrix.nbx.container import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, impose_rung,
                                              pin_dedicated_card, pin_host, profile)

MB = 1024 ** 2


def test_layer_streaming_says_its_tile_and_weights_overflow_the_rung(monkeypatch):
    """A REAL compiled solve. Allegro at 256x640, 88 frames, on a 640 MB rung (589 MB usable): its VAE
    tiles to ~243 MB of activations, and its 423 MB of weights take that tile over the rung; a host too
    small to take what the card cannot leaves the cascade nothing, and the refusal must say what layer
    streaming found — before, that rung recorded nothing."""
    pin_dedicated_card(monkeypatch, 16151, 267, "the rack's card 0, 2026-09-29")
    impose_rung(monkeypatch, 640)
    pin_host(monkeypatch, 4096, 2048, "a host too small to take a component the card cannot")
    small_host = copy.deepcopy(V100_16GB)
    small_host["cpu"]["ram_mb"] = 4096
    c = NBXContainer.load(str(container_root("Allegro")))
    request = InputConfig(batch_size=1, height=256, width=640, num_frames=88, temporal_compression=4)
    with pytest.raises(RuntimeError) as exc:
        PrismSolver().solve_smart(c, profile(small_host), request, mode="compiled")
    msg = str(exc.value)
    line = next((l for l in msg.splitlines() if l.strip().startswith("layer_streaming: no component tile")), "")
    assert "cuda:0 (589 MB usable)" in line and "vae: its tile and weights still need" in line, msg


def _encoder(tmp_path):
    """The frozen-batch encoder of test_a_whole_clip_encoder_tile_asks_no_frame_lattice: its time map
    ((s0*t + 1)//2 + 1)//2 folds the batch into time, and no temporal class reads it."""
    sym = lambda i, tr: {"type": "symbol", "id": i, "trace": tr}
    op = lambda k, l, r, tr: {"type": k, "left": l, "right": r, "trace": tr}
    b, t, h, w = sym("s0", 1), sym("s1", 20), sym("s2", 112), sym("s3", 176)
    t_out = op("floordiv", op("add", op("floordiv", op("add", op("mul", b, t, 20), 1, 21), 2, 10), 1, 11), 2, 5)
    graph = {"input_tensor_ids": ["x"], "output_tensor_ids": ["z"], "ops": {}, "execution_order": [],
             "tensors": {"x": {"shape": [1, 3, 20, 112, 176], "symbolic_shape": {"dims": [b, 3, t, h, w]}},
                         "z": {"shape": [1, 5, 4, 14, 22],
                               "symbolic_shape": {"dims": [b, t_out, 4, op("floordiv", h, 8, 14),
                                                           op("floordiv", w, 8, 22)]}}}}
    comp = tmp_path / "components" / "enc"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(graph))
    (comp / "profile.json").write_text(json.dumps({"config": {"block_out_channels": [1, 2, 3, 4]}}))
    return SimpleNamespace(_cache_path=tmp_path)


def _refusal(s, comp="enc"):
    s._strategies_tried, s._layer_streaming_declined = ["lazy_sequential"], None
    m = SimpleNamespace(weight_mb=244.0, activation_mb=319_500.0, total_mb=319_744.0,
                        weight_bytes=244 * MB, activation_bytes=319_500 * MB, overhead_bytes=0)
    card = SimpleNamespace(device_string="cuda:0", capacity_mb=31_130.0, recommended_mb=None, host_memory=None)
    with pytest.raises(RuntimeError) as exc:
        s._fail_error([(comp, m)], [card])
    return str(exc.value)


def _declined(tmp_path, mode):
    s = PrismSolver()
    s._mode = mode
    s._input_config = InputConfig(batch_size=1, height=720, width=1280, num_frames=88, temporal_compression=4)
    mem = ComponentMemory(component_name="enc", weight_bytes=244 * MB, activation_bytes=319_500 * MB,
                          overhead_bytes=0)
    assert s._spatial_component_tiling(_encoder(tmp_path), "enc", mem, 32768) is None
    return s


def test_a_compiled_placement_rung_names_the_encoders_decline_and_its_figure(tmp_path):
    s = _declined(tmp_path, "compiled")
    s._record_rung_tiling_decline("a component's placement", "cuda:0", 28_639.0, "enc")
    s._tiling_declined["enc"] = "a later rung's reason"            # the last write must not be the one printed
    msg = _refusal(s)
    assert "a component's placement: no component tile on cuda:0 (28,639 MB usable) — enc: no axis" in msg, msg
    assert "(s0*s1)" in msg and "a later rung's reason" not in msg, msg


def test_no_rung_prints_a_components_reason_twice(tmp_path):
    """A rung the Triton host line already printed is not printed again as a compiled rung line."""
    s = _declined(tmp_path, "triton")
    rung = "a component's host placement"
    s._host_device_overflow = {rung: ("cuda:0", 28_639.0, {"enc": 415_350.0})}
    s._snapshot_tiling_declines(rung, {"enc": 415_350.0})
    s._record_rung_tiling_decline(rung, "cuda:0", 28_639.0, "enc")
    msg = _refusal(s)
    assert msg.count("no axis of its output z") == 1, msg
