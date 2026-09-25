"""Every piece a component is streamed in binds every symbol it uses.

The first gate for `layer_streaming` asserted the PLAN (strategy name, segment count, peak) and
never ran a piece: the recipe, not the dish (register 100). The Mac then rendered the exact plan
it proved — PixArt-XL-1024 at 2048x1024, text encoder in 6 pieces, peak 8 129.3 MB under the
8 192 rung — and the render died after 1 821 s (the Mac, 8e786e70):

    UnboundSymbolError: ZERO FALLBACK: symbol 's1' (seq_len, binds from
    input::attention_mask::dim_1) is not bound at runtime ... Bound: ['s0', 's3'].

Piece 3 uses `s1` and no input handed to it carries `s1`: the hidden states carry `s3`, a second
symbol the trace minted for the same 120-token extent from `input_ids`. Pieces 4-5 lose the batch
symbol too. 26 runs on the Mac failed this way (PixArt-XL-1024, PixArt-Sigma-XL-1024, Flex.1-alpha,
Open-Sora-v2, SANA-Video).

WHAT THIS GATE EXECUTES. The binding every piece performs at runtime, with the engine's own
`SymbolResolver.bind_from_inputs` — the call that raised on the Mac — chained piece to piece:
piece k receives the component's inputs it declares, and the seam tensors shaped by evaluating
their symbolic dims under the WHOLE component's bindings (what the previous piece hands over).
For each piece: every symbol referenced anywhere in it (tensors, op attributes, in every form the
resolvers accept) is BOUND. No weights, no card: the binding, executed, not a plan.

What this does NOT prove, stated: the VALUES bound. The inputs are shaped from the same symbolic
dims the binding reads, so a value comparison would be true by construction (the reviewer's point,
2026-09-24) and is not made; and binding does not depend on size, so the three sizes below guard
only against a trace value leaking into a binding. The values, at batch 1/2/8 in every engine, are
the executed gate's: `tests/regression/test_a_streamed_component_computes_what_it_computes_whole.py`
(whole vs pieces, bit-identical).

A limit, stated: `_walk_symbols` enumerates the reference forms the resolvers accept TODAY, the same
forms `build_segment_graph` walks to choose its carriers. A new form taught to a resolver and to
neither walker would pass here and fail at runtime; the executed gate is the one that would see it.

Sizes: each symbol at its trace value, at 2x, and far (7x + 1), one value per symbol NAME.
"""
from __future__ import annotations

import functools
import json
from types import SimpleNamespace

import pytest

from neurobrix.core.optim.passes.normalize import normalize_for_branch
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.layer_partition import Segment, build_segment_graph
from neurobrix.nbx import NBXContainer
from neurobrix.triton.symbols import SymbolResolver
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                              pin_host, profile)

# (model, height, width, host MB free, rung MB): the Mac's own reading and rung for each case.
CASES = [
    ("PixArt-XL-1024", 2048, 1024, 11198, 8192),        # the Mac's render (8e786e70)
    ("PixArt-Sigma-XL-1024", 2048, 1024, 11198, 8192),
    ("Flex.1-alpha", 1024, 1024, 18186, 16384),
]

# Every container a streamed plan reaches in the two censuses (the Mac's 4a3658d7 reshape census and
# this rack's 2026-09-24 pinned-machine census), at the container's own default request, swept over
# rungs so the rung that streams it is found rather than assumed. A model no rung streams FAILS its
# cell: a case the gate never reaches is not a case it proved.
SWEEP = ["Open-Sora-v2", "SANA-Video_2B_720p_diffusers", "DeepSeek-Coder-V2-Lite-Instruct",
         "Qwen3-30B-A3B-Thinking-2507", "Qwen3-Coder-30B-A3B-Instruct", "Qwen3-Omni-30B-A3B-Instruct",
         "Qwen3-VL-30B-A3B-Thinking", "deepseek-moe-16b-chat", "granite-speech-3.3-8b"]
RUNGS = [4096, 8192, 16384]
# A container whose defaults declare no request is given one explicitly — without it the plan
# refuses before streaming is tried and the cell proves nothing. Open-Sora-v2's 2026-09-13 build
# declares no resolution (Forge dd2fa21 adds the vendor's); its traced extent is used here.
REQUESTS = {"Open-Sora-v2": dict(height=112, width=176, num_frames=51, temporal_compression=4)}
SCALES = {"trace": lambda t: t, "twice": lambda t: 2 * t, "far": lambda t: 7 * t + 1}


def _walk_symbols(obj, out, known):
    """Every symbol reference in every form the engine's resolvers accept (shape_resolver,
    triton/symbols, triton/sequence): a symbol node, a `symbol_id` key, a bare id string."""
    if isinstance(obj, dict):
        if obj.get("type") == "symbol" and obj.get("id"):
            out.add(str(obj["id"]))
        if isinstance(obj.get("symbol_id"), str):
            out.add(obj["symbol_id"])
        for v in obj.values():
            _walk_symbols(v, out, known)
    elif isinstance(obj, list):
        for v in obj:
            _walk_symbols(v, out, known)
    elif isinstance(obj, str) and obj in known:
        out.add(obj)


def _referenced(seg):
    known = set((seg.get("symbolic_context") or {}).get("symbols") or {})
    out = set()
    _walk_symbols(seg["tensors"], out, known)
    _walk_symbols(seg["ops"], out, known)
    return out & known


def _plan(monkeypatch, model, h, w, host_free, rung, request=None):
    pin_host(monkeypatch, 24576, host_free, "the Mac's reading")
    impose_rung(monkeypatch, rung)
    root = container_root(model)
    s = PrismSolver()
    ic = request if request is not None else InputConfig(batch_size=1, height=h, width=w)
    p = s.solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO), ic, mode="triton")
    assert p.strategy == "layer_streaming" and p.layer_stream_plan, (
        f"{model}: planned {p.strategy!r}; this gate judges streamed pieces")
    family = json.loads((root / "manifest.json").read_text()).get("family", "")
    return root, p, family


def _whole_bindings(graph, scale):
    """The whole component bound at one size: every symbol at scale(trace), one value per NAME."""
    syms = (graph.get("symbolic_context") or {}).get("symbols") or {}
    by_name = {}
    for sid, info in syms.items():
        if not isinstance(info.get("trace_value"), int):
            raise AssertionError(f"symbol {sid} declares no trace value: {info}")
        by_name.setdefault(info.get("name") or sid, scale(int(info["trace_value"])))
    return {sid: by_name[info.get("name") or sid] for sid, info in syms.items()}


def _shape(meta, whole):
    """A tensor's runtime shape under the whole component's bindings."""
    r = SymbolResolver({"symbols": {}})
    r._bindings.update(whole)
    dims = ((meta.get("symbolic_shape") or {}).get("dims")) or meta.get("shape") or []
    return tuple(r.resolve(d) for d in dims)


@functools.lru_cache(maxsize=4)
def _normalized(root, comp, family, declared_moe):
    """The graph Prism cuts, once per component: it depends on neither the size nor the rung,
    and the MoE fusion on a 30B graph costs minutes. Callers only read it. `declared_moe` is the
    plan's own MoE declaration for the component (`layer_stream_moe`), as the strategy uses it."""
    raw = json.loads((root / "components" / comp / "graph.json").read_text())
    return normalize_for_branch(raw, "triton", family, declared_moe=declared_moe)


def _check_every_piece(model, root, plan, family, size):
    for comp, bounds in plan.layer_stream_plan.items():
        graph = _normalized(root, comp, family, plan.layer_stream_moe.get(comp))
        whole = _whole_bindings(graph, SCALES[size])
        order_index = {u: i for i, u in enumerate(graph["execution_order"])}
        for k, (first, last) in enumerate(bounds):
            seg = build_segment_graph(graph, Segment(
                index=k, first_op=first, last_op=last,
                op_count=order_index[last] - order_index[first] + 1), order_index)
            inputs = {tid: SimpleNamespace(shape=_shape(seg["tensors"][tid], whole))
                      for tid in seg["input_tensor_ids"]}
            resolver = SymbolResolver(seg["symbolic_context"])
            resolver.bind_from_inputs(inputs, seg["input_tensor_ids"], seg["tensors"])
            bound = resolver.bindings
            used = _referenced(seg)
            unbound = sorted(used - set(bound))
            assert not unbound, (
                f"{model}.{comp} piece {k}/{len(bounds)} at {size} uses {unbound} and none of "
                f"its inputs carries them (sources: "
                f"{[seg['symbolic_context']['symbols'][s]['source'] for s in unbound]})")


@pytest.mark.parametrize("size", list(SCALES))
@pytest.mark.parametrize("model,h,w,host_free,rung", CASES, ids=[c[0] for c in CASES])
def test_every_piece_binds_every_symbol_it_uses_to_the_whole_components_value(
        monkeypatch, model, h, w, host_free, rung, size):
    root, plan, family = _plan(monkeypatch, model, h, w, host_free, rung)
    _check_every_piece(model, root, plan, family, size)


def _default_request(root):
    if root.name in REQUESTS:
        return InputConfig(batch_size=1, **REQUESTS[root.name])
    dj_path = root / "runtime" / "defaults.json"
    dj = json.loads(dj_path.read_text()) if dj_path.is_file() else {}
    kw = {k: dj[k] for k in ("height", "width", "num_frames") if isinstance(dj.get(k), int)}
    if "num_frames" in kw and isinstance(dj.get("temporal_compression_ratio"), int):
        kw["temporal_compression"] = dj["temporal_compression_ratio"]
    return InputConfig(batch_size=1, **kw)


@pytest.mark.parametrize("model", SWEEP)
def test_every_streamed_plan_in_the_catalogue_binds_every_symbol(monkeypatch, model):
    root = container_root(model)
    streamed = 0
    for rung in RUNGS:
        try:
            _, plan, family = _plan(monkeypatch, model, None, None, 18186, rung,
                                    request=_default_request(root))
        except (AssertionError, RuntimeError):
            continue                     # not streamed at this rung (or refused, said elsewhere)
        streamed += 1
        for size in SCALES:
            _check_every_piece(f"{model}@{rung}", root, plan, family, size)
    assert streamed, f"{model}: no rung in {RUNGS} streams it — this cell proved nothing"
