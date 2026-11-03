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
        --out <campaign>/instances.jsonl [--models A,B]
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



def _runs_a_triton_kernel(CL, op_name: str) -> bool:
    """Whether the op runs a Triton kernel (the classification's TRITON class), an op it does not know being no."""
    try:
        return CL.get_execution_type(op_name) == CL.OpExecution.TRITON
    except KeyError:
        return False


def bind_profile(hardware: str):
    """The device the instances are derived for, bound as the census binds it (`derived_census.derive`): the
    profile, the vendor's key ladders, the wrappers' profile."""
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.kernels import census as _census
    from nbxtensor.ops import wrappers as W
    prof = load_profile(hardware)
    _census._bind_target(hardware, None)
    W.set_hardware_profile(prof)
    return prof


def load_hardware_profile(hardware: str):
    """The machine's hardware profile, read as the engine reads it (core/prism/loader.load_profile), nothing bound."""
    from neurobrix.core.prism.loader import load_profile
    return load_profile(hardware)


def census_rows(census: str) -> list:
    """The census table's Triton rows (`<vendor>/<profile>/<rung>`), models under their names of today."""
    sys.path.insert(0, str(REPO / "tools"))
    from container_renames import current_name
    path = REPO / "src" / "neurobrix" / "config" / "census" / f"{census}.jsonl"
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("mode") == MODE:
            row["model"] = current_name(row["model"])
            out.append(row)
    return out


def census_ops(census: str) -> dict:
    """{model: {op uid}} the census table records a Triton kernel for — under each container's name of today (a row
    written before a rename keeps its former name: read through tools/container_renames.current_name)."""
    sys.path.insert(0, str(REPO / "tools"))
    from container_renames import current_name
    path = REPO / "src" / "neurobrix" / "config" / "census" / f"{census}.jsonl"
    out = collections.defaultdict(set)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("mode") == MODE:
            out[current_name(row["model"])].update(row.get("ops") or [])
    return dict(out)


def plan_dtypes(model: str, request: list, hardware: str) -> dict:
    """{component: compute dtype} of the plan the engine makes for the request on `hardware`, at its whole card —
    `derived_census.plan_record` (`neurobrix run --explain-plan --json`, the census shadow), the census's own source:
    the DtypeEngine's decision on that profile, never the dtype the container stores (bf16 weights compute in fp16 on a
    V100, which has no bf16; the stored dtype formed bf16 keys the engine never launches there, refused
    KeyNotCertified on 16 of 25 nbxtensor instances, 2026-11-03)."""
    import derived_census as DC
    plan = DC.plan_record(model, request, MODE, hardware, None)
    _PLANS[(model, hardware)] = plan
    return {c["name"]: str(c["dtype"]).replace("torch.", "") for c in plan["components"]}


#: {(model, hardware): the plan record `plan_dtypes` read} — the same plan gives the extents and cuts the engine runs
#: each component at (`instances_of`), read once.
_PLANS: dict = {}

#: The metadata ops whose output is a view with OTHER strides than a contiguous input's (a transposed weight: the
#: graph's `aten.t(W)` that every linear reads). A chain of reshapes keeps a contiguous base contiguous and needs no
#: replay; a `contiguous`/`clone` ends the chain (its output is materialised by the engine as it is by the bench).
STRIDED_VIEWS = frozenset({"t", "transpose", "permute", "expand", "slice", "select", "narrow", "as_strided",
                           "diagonal", "unfold", "movedim", "swapaxes", "numpy_T"})
_CHAIN_ENDS = frozenset({"contiguous", "clone"})


def _metadata(CL, name: str) -> bool:
    try:
        return CL.get_execution_type(name) == CL.OpExecution.METADATA
    except KeyError:
        return False


def input_chain(rg: dict, producers: dict, tid: str, CL):
    """(root tensor id, [metadata ops from the root to `tid`]) — the views the graph forms on the way to an op's
    input. Followed while the producer is a metadata op of ONE tensor input; a `contiguous`/`clone` ends it."""
    steps, cur = [], tid
    while cur in producers:
        uid = producers[cur]
        op = rg["ops"][uid]
        name = op["op_type"].split("::")[-1].split(".")[0]
        ins = op.get("input_tensor_ids") or []
        if name in _CHAIN_ENDS or not _metadata(CL, name) or len(ins) != 1:
            break
        steps.append(uid)
        cur = ins[0]
    return cur, list(reversed(steps))


def _component_bindings(model: str, hardware: str):
    """[(component, graph, {symbol id: value}, compute dtype)] at the model's gate request, or a skip reason per
    component — the census's own binding rule for every flow it binds from the request, each component at the
    compute dtype the engine's plan gives it on `hardware` (`plan_dtypes`)."""
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
    dtypes = plan_dtypes(model, request, hardware)
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
        if comp not in dtypes:
            skipped[comp] = "not in the engine's plan for the request (no compute dtype: the plan never runs it)"
            continue
        bound.append((comp, g, syms, dtypes[comp]))
    return bound, skipped


def instances_of(model: str, ops_wanted: set, has_native_bf16: bool, hardware: str):
    """{instance key: instance} for one model, and its skipped components and ops.

    Each component at the extents the engine's PLAN runs it at, as the census derives it (`derive_keys`): its
    temporal stream's chunks and its vendor's or the engine's tiles (`derived_census.cut_bindings`), the stretches it
    runs in slices of a token axis at each slice's length (`sliced_bindings`). An op the plan cuts at op level
    (`plan_tiling`) launches at its tiles' shapes, which no binding gives: it is skipped by name, never benchmarked
    at the whole extent (the Dell's V100 run, 2026-11-03 01:23: whole-clip video VAE keys the plan never forms,
    refused KeyNotCertified). Each input that the graph forms as a strided view (a transposed weight) carries the
    chain of metadata ops from its root, replayed untimed by both contenders (01:58: the bench gave every M <= 4
    GEMM a contiguous [K, N] weight and its addmv paid a copy the engine never makes)."""
    import derived_census as DC
    from neurobrix.kernels import classification as CL
    out = {}
    bound, skipped = _component_bindings(model, hardware)
    plan = _PLANS.get((model, hardware)) or {}
    for comp, g, syms, cdtype in bound:
        unhandled = collections.Counter()
        try:
            cuts = DC.cut_bindings(model, comp, g, syms, plan)
        except Exception as e:                       # said, per component, never silent
            skipped[comp] = f"plan cut refused: {type(e).__name__}: {e}"
            continue
        for comp_v, b in cuts:
            whole = DC.VARIANT not in comp_v
            view, tiled_tf = DC.plan_tiling(plan if whole else {}, comp)
            try:
                sliced = DC.sliced_bindings(model, comp, b, plan, unhandled) if whole else []
            except Exception as e:
                skipped[comp_v] = f"sliced stretches refused: {type(e).__name__}: {e}"
                continue
            in_slices = set().union(*(set(o) for o, _b in sliced)) if sliced else set()
            for uid in sorted(set(tiled_tf) & ops_wanted):
                skipped[f"{comp_v}/{uid}"] = ("the plan runs this op in tiles (op_level_tiling_ops): its launches are "
                                              "the tiles' shapes, not instanced")
            passes = [(b, None, frozenset(set(tiled_tf) | in_slices), frozenset())]
            passes += [(b_, frozenset(o) - frozenset(tiled_tf), frozenset(),
                        frozenset(u for u, pin in o.items() if pin)) for o, b_ in sliced]
            for binding, only, exclude, pins in passes:
                _instances_at(DC, CL, model, comp, comp_v, g, binding, cdtype, has_native_bf16, ops_wanted, view,
                              tiled_tf, only, exclude, pins, out, skipped)
        if unhandled:
            skipped[f"{comp}/unhandled"] = "; ".join(f"{k} x{n}" for k, n in unhandled.items())
    return out, skipped


def _instances_at(DC, CL, model, comp, comp_v, g, syms, cdtype, has_native_bf16, ops_wanted, view, tiled_tf, only,
                  exclude, pins, out, skipped):
    """The instances of one component graph at one binding (`only`: the ops to take, `exclude`: the ops not to)."""
    dtypes: dict = {}
    unhandled = collections.Counter()
    try:
        DC.derive_component(model, comp_v, cdtype, MODE, syms, has_native_bf16, 0, 1, 1, unhandled,
                            tiling=view, tiled_tf=tiled_tf, pins=pins, out_dtypes=dtypes)
        rg = DC.runtime_graph(model, comp_v)
        shape = DC._shape_fn(rg, comp_v, syms, {}, {})
    except Exception as e:                       # said, per component, never silent
        skipped[f"{comp_v}@{sorted(syms.items())}"] = f"derivation refused: {type(e).__name__}: {e}"
        return
    # the dtype facts the engine's dispatcher runs the op under, as the census derived its key: the component's
    # compute dtype, the graph's traced dtype (the AMP cast-back floor) and the precision contract's role for the op
    contract = DC._contract(model, comp_v, rg, cdtype)
    gdtype = str(rg.get("torch_dtype") or g.get("torch_dtype") or "").replace("torch.", "") or None
    producers = {t: u for u, o in rg["ops"].items() for t in (o.get("output_tensor_ids") or [])}

    def dt(t):
        return str(dtypes.get(t, rg["tensors"].get(t, {}).get("dtype", "?")))

    for uid, op in rg["ops"].items():
        if uid not in ops_wanted or uid in exclude or (only is not None and uid not in only) \
                or not _runs_a_triton_kernel(CL, op["op_type"].split("::")[-1].split(".")[0]):
            continue
        ins = op.get("input_tensor_ids") or []
        prelude, roots = [], []
        try:
            shapes = [list(shape(t)) for t in ins]
            for t in ins:
                root, chain = input_chain(rg, producers, t, CL)
                names = {rg["ops"][u]["op_type"].split("::")[-1].split(".")[0] for u in chain}
                if not names & STRIDED_VIEWS:
                    roots.append({"id": t, "shape": list(shape(t)), "dtype": dt(t)})
                    continue
                roots.append({"id": root, "shape": list(shape(root)), "dtype": dt(root)})
                for u in chain:
                    o = rg["ops"][u]
                    a_ = o.get("attributes") or {}
                    prelude.append({"op_uid": u, "op_type": o["op_type"], "args": a_.get("args"),
                                    "attributes": {k: v for k, v in a_.items() if k != "args"},
                                    "inputs": o.get("input_tensor_ids") or [],
                                    "output": (o.get("output_tensor_ids") or [None])[0]})
        except Exception as e:
            skipped[f"{comp_v}/{uid}"] = f"shape refused: {e}"
            continue
        dts = [dt(t) for t in ins]
        attrs = op.get("attributes") or {}
        spec = {"op_type": op["op_type"], "inputs": ins, "shapes": shapes, "dtypes": dts,
                "args": attrs.get("args"), "kwargs": attrs.get("kwargs") or {},
                "attributes": {k: v for k, v in attrs.items() if k != "args"},
                "symbols": syms, "output_dtypes": op.get("output_dtypes"), "op_uid": uid,
                "compute_dtype": cdtype, "graph_dtype": gdtype,
                "contract": {"safe": bool(contract.safe),
                             "island": ("fp32" if uid in contract.fp32_op_uids
                                        else "narrow" if uid in contract.narrow_op_uids else None)},
                "roots": roots, "prelude": prelude}
        # two sites run one instance only when the engine runs them alike: the dtype facts and the views the op
        # reads are part of the key
        views = [[p["op_type"], _literal(p["args"]), p["attributes"]] for p in prelude]
        # (an op that reads no view keeps the key it had before views were recorded: rows already measured stand)
        key = hashlib.sha256(json.dumps([spec["op_type"], shapes, dts, _literal(attrs.get("args")),
                                         spec["attributes"], cdtype, gdtype, spec["contract"],
                                         op.get("output_dtypes")] + ([views] if views else []),
                                        sort_keys=True, default=str).encode()).hexdigest()[:16]
        inst = out.setdefault(key, {"key": key, **spec, "sites": []})
        inst["sites"].append(f"{model}/{comp_v}/{uid}")


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
    from nbxtensor.ops.wrappers import has_native_bf16
    wanted = census_ops(a.census)
    models = a.models.split(",") if a.models else sorted(wanted)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged, skipped = {}, {}
    for model in models:
        try:
            found, sk = instances_of(model, wanted.get(model, set()), has_native_bf16(), a.hardware)
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
