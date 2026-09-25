"""A streamed component's plan reserves, beside its pieces, the weights its FLOW reads by name.

Under `layer_streaming` the base executor now holds every non-block weight of a streamed
component (the token embedding, a head, the norms) resident, for the flow's by-name reads
(`LayerStreamingStrategy._ensure_flow_reads`). The pieces were cut against the rung less what
stays beside them — whole components, graph constants, the KV reserve — and nothing counted these
weights: the plan promised room the run did not have. On main (96288dfc) the resident figure
beside the pieces carried the graph constants alone.

The oracle reads the bytes from the component's own `weights_index.json` here, not through the
solver; the solver's resident figure less its graph constants must equal them.

The machine is built (register 102): the Mac's profile and reading, the Mac's rungs.
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import _graph_constant_bytes
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                              pin_host, profile)

# (model, rung MB): rows of the Mac's (df2588e7) at which the LM streams.
CASES = [("granite-speech-3.3-8b", 16384), ("GLM-4.1V-9B-Thinking", 11264),
         ("MiniCPM-o-4_5", 11264), ("Janus-Pro-7B", 12288)]


def _non_block_bytes(root, comp):
    """Every weight of `comp` outside a numbered block, from its index — read here. "Block" is
    the loader's own definition (`_BLOCK_RE`), which is what decides what a base holds."""
    from neurobrix.triton.weight_loader import _BLOCK_RE
    tensors = json.loads((root / "components" / comp / "weights_index.json").read_text())["tensors"]
    return sum(int(v["size_bytes"]) for k, v in tensors.items() if not _BLOCK_RE.search(k))


@pytest.mark.parametrize("model,rung", CASES, ids=[c[0] for c in CASES])
def test_the_plan_reserves_the_flow_read_weights_beside_the_pieces(monkeypatch, model, rung):
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    impose_rung(monkeypatch, rung)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    root = container_root(model)
    dj = json.loads((root / "runtime" / "defaults.json").read_text())
    kw = {k: dj[k] for k in ("height", "width") if isinstance(dj.get(k), int)}
    s = PrismSolver()
    p = s.solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO),
                      InputConfig(batch_size=1, **kw), mode="triton")
    assert p.strategy == "layer_streaming" and p.layer_stream_plan, (
        f"{model} at {rung} MB planned {p.strategy!r}: the Mac's row is not reproduced")
    streamed = list(p.layer_stream_plan)
    graphs = {c: json.loads((root / "components" / c / "graph.json").read_text()) for c in streamed}
    constants = sum(_graph_constant_bytes(graphs[c]) for c in streamed)
    # A flow reads a component by name when its graph takes `inputs_embeds` (the flows'
    # `uses_embeds`), read here from the graph file.
    read = [c for c in streamed if "input::inputs_embeds" in graphs[c]["input_tensor_ids"]]
    expected = sum(_non_block_bytes(root, c) for c in read)
    assert expected > 0, f"precondition: {streamed} carry non-block weights"
    reserved = s._layer_stream_constant_bytes - constants
    assert reserved == expected, (
        f"{model}: {reserved / 2**20:.1f} MB reserved beside the pieces for what the flow reads, "
        f"{expected / 2**20:.1f} MB held (the non-block weights of {streamed})")


def test_a_component_no_flow_reads_reserves_nothing(monkeypatch):
    """PixArt's T5 streamed at 2048x1024 (the Mac's render): its graph takes token ids, no flow
    reads it by name, and its non-block weights (the shared embedding its first op consumes) stay
    with the piece that consumes them. Reserving them beside every piece — or holding the whole
    non-block set of a VAE on its base (SANA-Video: 4 663 MB) — pins what the rung streams."""
    pin_host(monkeypatch, 24576, 11198, "the Mac's reading")
    impose_rung(monkeypatch, 8192)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    root = container_root("PixArt-XL-1024")
    s = PrismSolver()
    p = s.solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO),
                      InputConfig(batch_size=1, height=2048, width=1024), mode="triton")
    assert p.strategy == "layer_streaming" and "text_encoder" in p.layer_stream_plan, p.strategy
    graphs = {c: json.loads((root / "components" / c / "graph.json").read_text())
              for c in p.layer_stream_plan}
    assert not any("input::inputs_embeds" in g["input_tensor_ids"] for g in graphs.values())
    constants = sum(_graph_constant_bytes(g) for g in graphs.values())
    assert s._layer_stream_constant_bytes == constants, (
        s._layer_stream_constant_bytes / 2**20, constants / 2**20)
