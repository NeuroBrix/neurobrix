"""The kernel-level instances of the nbxtensor benchmark: every compute op of the catalogue at the shapes and dtypes
the certification census recorded, card-less.

Source of truth, in this order:
- the census table of the device (`config/census/<vendor>/<profile>/<rung>.jsonl`): which ops of which model the
  Triton engine launches a kernel for (its rows' `ops`);
- the model's container: each op's type, arguments and attributes (`components/<c>/graph.json`);
- the request the census derives at (`tools/trace_request.gate_request`), bound to each component's symbols the way
  the census binds them (`run.request_input_config` -> `ActivationProfiler.build_symbol_map`; an autoregressive
  language model at its prompt's length), and each tensor's runtime dtype from the census's own derivation
  (`tools/derived_census.derive_component(..., out_dtypes=)`).

One instance = one (op type, input shapes, input dtypes, non-tensor arguments, kwargs); the models and op uids that
run it are listed with it. A component whose symbols the request does not bind (a value-derived extent the census
reaches through its extent sites: a decode step, a codec frame count) is not instanced and is counted by name in the
output's `skipped` record — never benchmarked at an invented shape.

    python benchmarks/nbxtensor_vs_torch/instances.py --census apple/apple_m4_pro/18g --hardware <profile id> \\
        --out benchmarks/nbxtensor_vs_torch/results/apple_m4_pro/instances.jsonl [--models A,B]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

MODE = "triton"


def bind_profile(hardware: str):
    """The device the instances are derived for, bound as the census binds it (`derived_census.derive`): the
    profile, the vendor's key ladders, the wrappers' profile."""
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.kernels import census as _census
    from neurobrix.kernels import wrappers as W
    prof = load_profile(hardware)
    _census._bind_target(hardware, None)
    W.set_hardware_profile(prof)
    return prof


def census_ops(census: str) -> dict:
    """{model: {op uid}} the census table records a Triton kernel for."""
    path = REPO / "src" / "neurobrix" / "config" / "census" / f"{census}.jsonl"
    out = collections.defaultdict(set)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("mode") == MODE:
            out[row["model"]].update(row.get("ops") or [])
    return dict(out)


def _component_bindings(model: str):
    """[(component, graph, {symbol id: value}, compute dtype)] at the model's gate request, or a skip reason per
    component — the census's own binding rule for every flow it binds from the request."""
    import derived_census as DC
    import trace_request as TR
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism.profiler import ActivationProfiler

    request = TR.gate_request(model)
    cache = DC.CACHE / model
    manifest = json.loads((cache / "manifest.json").read_text())
    topo = json.loads((cache / "topology.json").read_text())
    args = create_parser().parse_args(["run", "--model", model, *request, "--triton"])
    ic = request_input_config(args, manifest, manifest.get("family"), cache)
    flow = topo.get("flow") or {}
    gen = flow.get("generation") or {}
    autoregressive = flow.get("type") == "autoregressive_generation"
    head = gen.get("head_component", "lm_head") if autoregressive else None
    prompt_len = DC.prompt_tokens(model, args.prompt) if autoregressive and args.prompt else None
    bound, skipped = [], {}
    for comp_dir in sorted((cache / "components").iterdir()):
        comp = comp_dir.name
        gpath = comp_dir / "graph.json"
        if not gpath.exists():
            continue
        g = json.loads(gpath.read_text())
        table = (g.get("symbolic_context") or {}).get("symbols") or {}
        syms = {}
        if autoregressive:
            for sid, info in table.items():
                if info.get("name") == "batch":
                    syms[sid] = 1
                elif info.get("name") in ("seq_len", "sequence_length"):
                    syms[sid] = 1 if comp == head else int(prompt_len)
        else:
            m = ActivationProfiler(g).build_symbol_map(ic, placement_floor=False)
            syms = {sid: int(m[sid]) for sid in table if m.get(sid) is not None}
        missing = sorted(set(table) - set(syms))
        if missing:
            skipped[comp] = f"symbols {missing} not bound by the request (a value-derived extent)"
            continue
        dtype = (manifest.get("dtype") or g.get("torch_dtype") or "float16")
        bound.append((comp, g, syms, str(dtype).replace("torch.", "")))
    return bound, skipped


def instances_of(model: str, ops_wanted: set, has_native_bf16: bool):
    """{instance key: instance} for one model, and its skipped components."""
    import derived_census as DC
    from neurobrix.kernels import classification as CL
    out, skipped = {}, {}
    bound, skipped = _component_bindings(model)
    for comp, g, syms, cdtype in bound:
        dtypes: dict = {}
        unhandled = collections.Counter()
        try:
            DC.derive_component(model, comp, cdtype, MODE, syms, has_native_bf16, 0, 1, 1, unhandled,
                                out_dtypes=dtypes)
            shape = DC._shape_fn(DC.runtime_graph(model, comp), comp, syms, {}, {})
        except Exception as e:                       # said, per component, never silent
            skipped[comp] = f"derivation refused: {type(e).__name__}: {e}"
            continue
        rg = DC.runtime_graph(model, comp)
        for uid, op in rg["ops"].items():
            if uid not in ops_wanted or not CL.is_triton_op(op["op_type"].split("::")[-1].split(".")[0]):
                continue
            ins = op.get("input_tensor_ids") or []
            try:
                shapes = [list(shape(t)) for t in ins]
            except Exception as e:
                skipped[f"{comp}/{uid}"] = f"shape refused: {e}"
                continue
            dts = [str(dtypes.get(t, rg["tensors"].get(t, {}).get("dtype", "?"))) for t in ins]
            attrs = op.get("attributes") or {}
            spec = {"op_type": op["op_type"], "inputs": ins, "shapes": shapes, "dtypes": dts,
                    "args": attrs.get("args"), "kwargs": attrs.get("kwargs") or {},
                    "attributes": {k: v for k, v in attrs.items() if k != "args"},
                    "symbols": syms, "output_dtypes": op.get("output_dtypes")}
            key = hashlib.sha256(json.dumps([spec["op_type"], shapes, dts, _literal(attrs.get("args")),
                                             spec["attributes"]], sort_keys=True, default=str).encode()).hexdigest()[:16]
            inst = out.setdefault(key, {"key": key, **spec, "sites": []})
            inst["sites"].append(f"{model}/{comp}/{uid}")
    return out, skipped


def _literal(args):
    """The arguments with their tensor ids dropped: two ops differing only in which tensors they read are one
    instance."""
    if isinstance(args, dict):
        if args.get("type") in ("tensor", "tensor_ref"):
            return "T"
        if args.get("type") == "tensor_tuple":
            return ["T"] * len(args.get("tensor_ids") or [])
        return {k: _literal(v) for k, v in args.items() if k != "trace"}
    if isinstance(args, list):
        return [_literal(a) for a in args]
    return args


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--census", required=True, help="<vendor>/<profile>/<rung>, e.g. apple/apple_m4_pro/18g")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="comma-separated; default: every model of the census")
    ap.add_argument("--hardware", required=True, help="the hardware profile id the census was derived for")
    a = ap.parse_args(argv)
    bind_profile(a.hardware)
    from neurobrix.kernels.wrappers import has_native_bf16
    wanted = census_ops(a.census)
    models = a.models.split(",") if a.models else sorted(wanted)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged, skipped = {}, {}
    for model in models:
        try:
            found, sk = instances_of(model, wanted.get(model, set()), has_native_bf16())
        except Exception as e:
            skipped[model] = f"refused: {type(e).__name__}: {e}"
            print(f"[instances] {model}: refused — {e}", flush=True)
            continue
        for k, inst in found.items():
            merged.setdefault(k, {**inst, "sites": []})["sites"].extend(inst["sites"])
        skipped.update({f"{model}/{c}": r for c, r in sk.items()})
        print(f"[instances] {model}: {len(found)} instances, {len(sk)} components skipped", flush=True)
    with open(out, "w") as f:
        for inst in merged.values():
            f.write(json.dumps(inst, default=str) + "\n")
    out.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=1, sort_keys=True))
    print(f"[instances] {len(merged)} instances over {len(models)} models -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
