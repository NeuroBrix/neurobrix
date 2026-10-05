"""Prism prices a weight on the device at the width the engine HOLDS it at, never at its stored width.

Wan2.1-T2V-1.3B-Diffusers stores its umt5 encoder in float32 (21 671 MB, the 4 006 MB token embedding
among it) and runs it at the half compute dtype: the Triton loader takes fp32 under a half compute to
the half (`stored_dtype_in_compute`, the arena's `_target_nbytes`), the ATen loader converts every
float to the plan dtype (`_convert_weights_dtype`). Its arena holds 10 835 MB. The layer-streaming
rung cut its pieces at the STORED bytes (`_weight_sizes_by_component`): on the Mac's 14 687 MB reading
four segments of ~5.7 GB priced, ~2.9 GB held each — twice the arena the plan announced, on every
machine (the previous agent's finding on branch a-unified-phase-is-held-to-its-own-load, 0e24b28c).

The host side keeps the stored width where the loader READS (`component_loads`' triton transient: one
tensor as read and as converted) and takes the held width where it HOLDS (a pinned staging copy).
"""
import json
import sys

import pytest

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism import host_footprint as HF
from neurobrix.core.prism.runtime_widths import held_index_sizes, held_weight_bytes, held_weight_dtype
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, no_door, pin_host, profile

MODEL = "Wan2.1-T2V-1.3B-Diffusers"
MB = 1 << 20


def test_each_engine_holds_a_stored_weight_at_its_loaders_width():
    # Triton: fp32 under a half takes the half; a half takes the compute half; the rest keeps its own.
    assert held_weight_dtype("float32", "float16", "triton") == "float16"
    assert held_weight_dtype("float32", "bfloat16", "triton_sequential") == "bfloat16"
    assert held_weight_dtype("bfloat16", "float16", "triton") == "float16"
    assert held_weight_dtype("float16", "float32", "triton") == "float16"
    assert held_weight_dtype("int64", "float16", "triton") == "int64"
    # ATen: every float to the plan dtype, integers kept.
    assert held_weight_dtype("float16", "float32", "compiled") == "float32"
    assert held_weight_dtype("float32", "float16", "sequential") == "float16"
    assert held_weight_dtype("bool", "float16", "compiled") == "bool"
    assert held_weight_bytes(400, "float32", "float16", "triton") == 200
    assert held_weight_bytes(200, "float16", "float32", "compiled") == 400
    with pytest.raises(ValueError, match="unknown engine"):
        held_weight_dtype("float32", "float16", "metal-magic")
    with pytest.raises(ValueError, match="no stored dtype"):
        held_index_sizes({"w": {"size_bytes": 4}}, "float16", "triton")


def test_the_component_weight_is_its_mixed_index_at_the_held_width():
    """A mixed container (fp32 + bf16 under a fp16 plan): each weight at its own held width — not the
    dominant dtype's factor applied to all of them."""
    from types import SimpleNamespace as NS
    from neurobrix.core.prism.solver import _consumed_weight_bytes
    idx = {"blocks.0.w": {"size_bytes": 400 * MB, "dtype": "float32"},
           "blocks.1.w": {"size_bytes": 200 * MB, "dtype": "bfloat16"},
           "pos.ids": {"size_bytes": 8 * MB, "dtype": "int64"}}
    tensors = {f"param::{n}": {"is_parameter": True, "weight_name": n} for n in idx}
    graph = {"tensors": tensors, "execution_order": ["op0"],
             "ops": {"op0": {"input_tensor_ids": list(tensors)}}}
    comp = NS(graph=graph, weights_index={"tensors": idx})
    assert _consumed_weight_bytes(comp, "float16", "triton") == (200 + 200 + 8) * MB
    assert _consumed_weight_bytes(comp, "float32", "compiled") == (400 + 400 + 8) * MB


def test_a_pinned_copy_holds_the_held_width_and_a_load_reads_the_stored_one():
    from types import SimpleNamespace as NS
    keys = {"te": {"blocks.0.w": 400 * MB, "blocks.1.w": 400 * MB, "embed.w": 1000 * MB}}
    held = {"te": {k: n // 2 for k, n in keys["te"].items()}}
    plan = NS(loading_mode="lazy", components={"te": NS(device="cuda:0", dtype="float16",
                                                          shard_map={"s0": "cpu"})},
              component_memory={})
    f = HF.host_footprint(plan, keys, {"te": {"s0": 1800 * MB}}, "triton", None, {"float16": 2, "float32": 4},
                          lambda k: k.startswith("blocks."), held_sizes=held)
    assert f["steady"]["te"] == 400 * MB, f["steady"]                 # block weights pinned at fp16
    assert f["transient_bytes"] == 2 * 1000 * MB, f["transient_bytes"]  # the largest tensor as read (fp32)


@pytest.fixture(scope="module")
def _wan():
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    root = container_root(MODEL)
    man = json.loads((root / "manifest.json").read_text())
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "a red apple rolling slowly",
                                       "--steps", "4", "--height", "480", "--width", "832", "--triton"])
    return root, request_input_config(args, man, man.get("family"), root)


def test_the_umt5_encoder_is_streamed_in_pieces_of_its_held_bytes(monkeypatch, _wan):
    """The Mac's 14 687 MB reading (test_a_unified_phase_is_held_to_its_own_load): the umt5 encoder is
    streamed, and its segments together hold its arena — 10 835 MB at fp16 — not its 21 671 MB file."""
    root, ic = _wan
    no_door(monkeypatch)
    monkeypatch.delenv("NBX_CENSUS", raising=False)
    monkeypatch.delenv("NBX_CENSUS_DEVICES", raising=False)
    pin_host(monkeypatch, 24576, 14687, "the Mac, 2026-10-05 13:40")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(HF, "process_footprint_now", lambda: 238 << 20)
    s = PrismSolver()
    container = NBXContainer.load(str(root))
    p = s.solve_smart(container, profile(APPLE_M4_PRO), ic, mode="triton")
    assert "text_encoder" in (p.layer_stream_plan or {}), (p.strategy, p.layer_stream_plan)
    dt = p.components["text_encoder"].dtype
    stored = sum(s._weight_sizes_by_component(container)["text_encoder"].values())
    held = sum(s._held_sizes_by_component(container)["text_encoder"].values())
    assert str(dt) in ("float16", "bfloat16") and held * 2 == stored, (dt, held, stored)
    segs = s._layer_stream_partitions["text_encoder"].segments
    priced = sum(seg.weight_bytes for seg in segs)
    assert priced <= held, (f"{len(segs)} segments priced {priced / MB:,.0f} MB, the arena holds "
                            f"{held / MB:,.0f} MB (stored {stored / MB:,.0f} MB)")
    assert priced >= held * 9 // 10, (priced / MB, held / MB)
