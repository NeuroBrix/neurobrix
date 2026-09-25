"""Execute one component WHOLE and then in the PIECES Prism planned, on this card, and compare.

The dish, not the recipe (vacuous-gates register 100). A streaming plan is proved by running it:
the Mac's PixArt-XL-1024 render planned 6 pieces under the 8 192 rung and died in piece 3 at
runtime (8e786e70), and no plan-level cell could have seen it.

What it does, in one process, one card:
  1. plans the model under the Apple M4 Pro profile, the Mac's host reading injected and the rung
     imposed (the door), so the pieces are exactly the ones the Mac's machine would get;
  2. builds the component's executor the way the runtime does, and runs it WHOLE;
  3. runs the SAME inputs through the real `LayerStreamingStrategy` over the planned pieces,
     one piece's weights resident at a time;
  4. writes both outputs' summary and their difference as JSON.

usage: streamed_component_vs_whole.py MODEL COMPONENT MODE BATCH SEQ HOST_FREE_MB RUNG_MB OUT.json
       [H W]   (the image request the plan is made for; without it the plan is made at batch 1
               with no request size)

The component's inputs: token ids with a full attention mask when it takes exactly those; any
other component (an LM stage fed `inputs_embeds`) gets inputs built from its OWN declared graph
inputs — every symbolic dim bound by the symbol's declared NAME (`batch` -> BATCH, `seq_len`
-> SEQ, any other name refused), floating inputs drawn N(0, 0.02) in float32 (the executor casts
component inputs to its compute dtype), integer inputs counting 0..n-1 along their last axis
(positions), boolean inputs all False. The values are arbitrary; the gate compares the SAME
inputs whole and in pieces, so they need only be finite and in range.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _plan(model, mode, host_free, rung, h, w):
    import pytest
    from neurobrix.core.prism import InputConfig, PrismSolver
    from neurobrix.nbx import NBXContainer
    from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                                  pin_host, profile)
    mp = pytest.MonkeyPatch()
    try:
        pin_host(mp, 24576, host_free, "the Mac's reading")
        impose_rung(mp, rung)
        root = container_root(model)
        s = PrismSolver()
        ic = (InputConfig(batch_size=1, height=h, width=w) if h is not None
              else InputConfig(batch_size=1))
        p = s.solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO), ic, mode=mode)
    finally:
        mp.undo()
    return root, p


def _to_engine(arr, mode, device):
    if mode.startswith("triton"):
        from neurobrix.kernels.nbx_tensor import NBXTensor
        return NBXTensor.from_numpy(np.ascontiguousarray(arr)).to(device)
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _to_numpy(t):
    """Host copy, float32 — the triton tensor's own `numpy()` (R33: no torch on that side)."""
    if type(t).__module__.startswith("torch"):
        return t.detach().float().cpu().numpy()
    from neurobrix.kernels.nbx_tensor import parse_dtype
    if not str(getattr(t, "dtype", "")).endswith(("float32", "int64", "int32", "bool")):
        t = t.to(parse_dtype("float32"))          # bf16/fp16 read as raw bytes otherwise
    return np.asarray(t.numpy(), dtype=np.float32)


def _declared_inputs(dag, by_name, rng):
    """The component's inputs from its own declaration (see the module docstring)."""
    syms = (dag.get("symbolic_context") or {}).get("symbols") or {}
    out = {}
    for tid in dag["input_tensor_ids"]:
        meta = dag["tensors"][tid]
        dims = (meta.get("symbolic_shape") or {}).get("dims") or meta.get("shape") or []
        shape = []
        for d in dims:
            sid = d.get("id") if isinstance(d, dict) else (d if isinstance(d, str) else None)
            if sid is None:
                shape.append(int(d))
                continue
            name = (syms.get(sid) or {}).get("name")
            if name not in by_name:
                raise SystemExit(f"input {tid} dim {sid} is named {name!r}; this harness binds "
                                 f"only {sorted(by_name)} and will not guess another")
            shape.append(int(by_name[name]))
        dt = str(meta.get("dtype") or "")
        if "bool" in dt:
            a = np.zeros(shape, dtype=bool)
        elif "int" in dt:
            a = np.broadcast_to(np.arange(shape[-1], dtype=np.int64), shape).copy()
        else:
            a = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
        out[tid[7:]] = a
    return out


def main():
    model, comp, mode, batch, seq, host_free, rung, out_path = sys.argv[1:9]
    batch, seq, host_free, rung = int(batch), int(seq), int(host_free), int(rung)
    h, w = (int(sys.argv[9]), int(sys.argv[10])) if len(sys.argv) > 10 else (None, None)
    root, plan = _plan(model, mode, host_free, rung, h, w)
    bounds = (plan.layer_stream_plan or {}).get(comp)
    if not bounds:
        raise SystemExit(f"{model}.{comp}: the plan ({plan.strategy}) does not stream this "
                         f"component at rung {rung}; nothing to compare")
    dtype = plan.components[comp].dtype

    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.core.strategies.base import StrategyContext
    from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy
    family = json.loads((root / "manifest.json").read_text()).get("family", "")
    # The card this runs on, from this machine's own detected profile — never a vendor or an
    # architecture written here (R23). The PLAN is the Mac's; the EXECUTION is this card's.
    from neurobrix.core.prism.autodetect import load_default_profile
    here = load_default_profile().devices[0]
    vendor = str(getattr(here, "brand", "") or "").split(".")[-1].lower()
    arch = str(getattr(here, "architecture", "") or "")
    if not vendor or not arch:
        raise SystemExit(f"this machine's profile names no vendor/architecture for its card: {here}")
    device = here.get_device_string()

    def executor():
        ex = GraphExecutor(family=family, vendor=vendor, arch=arch, device=device,
                           dtype=dtype, mode=mode)
        ex.load_graph(root / "components" / comp / "graph.json")
        ex._cache_path = root
        return ex

    rng = np.random.default_rng(0)
    whole_ex = executor()
    wanted = [t[7:] for t in whole_ex._dag["input_tensor_ids"]]
    if sorted(wanted) == ["attention_mask", "input_ids"]:
        arrays = {"input_ids": rng.integers(0, 32000, size=(batch, seq), dtype=np.int64),
                  "attention_mask": np.ones((batch, seq), dtype=np.int64)}
    else:
        arrays = _declared_inputs(whole_ex._dag, {"batch": batch, "seq_len": seq}, rng)
    inputs = {n: _to_engine(arrays[n], mode, device) for n in wanted}

    whole_ex.load_weights(str(root), comp)
    whole = whole_ex.run(dict(inputs))
    # ONE SIDE TWICE before a difference is attributed: the whole component again, same inputs.
    whole_again = whole_ex.run(dict(inputs))
    whole_ex.unload_weights()
    del whole_ex

    base = executor()
    ctx = StrategyContext(strategy_name="layer_streaming", allocations={comp: (device, {})},
                          component_executors={comp: base},
                          runtime_package=SimpleNamespace(cache_path=root),
                          layer_segments={comp: bounds},
                          layer_graphs=dict(plan.layer_stream_graph),
                          layer_moe=dict(plan.layer_stream_moe))
    strat = LayerStreamingStrategy(ctx, "layer_streaming")
    pieces = strat.execute_component(comp, "once", dict(inputs))

    report = {"model": model, "component": comp, "mode": mode, "batch": batch, "seq": seq,
              "request": [h, w], "rung_mb": rung, "pieces": len(bounds), "dtype": dtype,
              "inputs": {n: list(arrays[n].shape) for n in wanted}, "outputs": {}}
    for key, val in whole.items():
        a = _to_numpy(val)
        if key not in pieces:
            report["outputs"][key] = {"missing_from_pieces": True, "whole_shape": list(a.shape)}
            continue
        b = _to_numpy(pieces[key])
        entry = {"whole_shape": list(a.shape), "pieces_shape": list(b.shape)}
        a2 = _to_numpy(whole_again[key])
        entry["whole_vs_whole_max_abs_diff"] = float(np.abs(a.astype(np.float64) - a2).max())
        if a.shape == b.shape:
            entry["mean_abs_diff"] = float(np.abs(a.astype(np.float64) - b).mean())
            entry["rel_l2"] = float(np.linalg.norm(a.astype(np.float64) - b) / max(np.linalg.norm(a.astype(np.float64)), 1e-30))
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            entry.update(max_abs_diff=float(d.max()), bit_identical=bool((a == b).all()),
                         whole_finite=bool(np.isfinite(a).all()), whole_abs_mean=float(np.abs(a).mean()))
        report["outputs"][key] = entry
    Path(out_path).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
