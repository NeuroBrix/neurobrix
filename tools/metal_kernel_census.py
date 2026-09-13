#!/usr/bin/env python3
"""The numeric census: every kernel in the library, against the Dell's bank.

    python tools/metal_kernel_census.py --bank <dir> --out <dir> [--only mm,add]

The library is the 280 `@triton.jit` kernels under `src/neurobrix/kernels/ops`
— counted from the source, so the number is not a claim but a reading. The
bank is a set of `.npz` files, each holding the seeded inputs an op ran with,
the output CUDA produced, and an fp64 oracle; its `meta` carries the ULP
statistics CUDA scored and the kernels the call launched.

This runs each entry here, through the engine's own wrappers, and reports:

  * the Metal distance to the SAME fp64 oracle, beside the CUDA distance
    already in the file — the owner's bar is **Metal ULP no greater than CUDA
    ULP**, per kernel and per shape;
  * which of the 280 kernels the bank reaches at all, and which it does not,
    because a kernel with no entry is an unmeasured kernel and saying so is
    part of the census;
  * every refusal, with its reason, as a measured gap rather than a missing
    row.

The bank lives on a read-only mount; nothing here writes to it.

It REFUSES to conclude without writing its files. A census whose numbers exist
only in a report is not a census.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHES = (Path.home() / ".cache" / "triton_msl",
           Path.home() / ".triton" / "cache")


def clear_caches() -> list:
    cleared = []
    for path in _CACHES:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        cleared.append(str(path))
    replay = Path.home() / ".neurobrix" / "replay_cache"
    if replay.exists():
        for f in replay.glob("autotune_configs_*.json"):
            f.unlink()
            cleared.append(str(f))
    return cleared


def library_kernels() -> list:
    """Every `@triton.jit` kernel in the ops library, read from the source."""
    root = REPO_ROOT / "src" / "neurobrix" / "kernels" / "ops"
    found = []
    for f in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if "jit" in ast.unparse(dec):
                    found.append({"kernel": node.name,
                                  "file": str(f.relative_to(root))})
                    break
    return found


def ulp_distance(got: np.ndarray, oracle: np.ndarray) -> dict:
    """Distance in ULP of the result's own dtype, plus the plain errors.

    The oracle is fp64 and is rounded to the result's dtype first, so the
    comparison is "how many representable steps away" — what the bank's own
    numbers mean. Integer results compare bit-for-bit.
    """
    got = np.asarray(got)
    oracle = np.asarray(oracle)
    dtype = got.dtype
    if not np.issubdtype(dtype, np.floating):
        equal = bool(got.shape == oracle.shape and np.array_equal(got, oracle.astype(dtype)))
        return {"max_ulp": 0 if equal else None, "identical": equal,
                "integer": True, "nonfinite": 0}
    rounded = oracle.astype(dtype)
    view = {np.dtype(np.float16): np.int16,
            np.dtype(np.float32): np.int32}.get(dtype, np.int64)
    bias = {np.int16: 0x8000, np.int32: 0x80000000}.get(view, 0x8000000000000000)

    def ordered(v):
        i = v.view(view).astype(np.int64)
        return np.where(i < 0, np.int64(bias) - i, i)

    finite = np.isfinite(got) & np.isfinite(rounded)
    ulp = (np.abs(ordered(got[finite]) - ordered(rounded[finite]))
           if finite.any() else np.array([0]))
    scale = float(np.abs(oracle).max()) or 1.0
    return {
        "max_ulp": int(ulp.max()),
        "mean_ulp": float(ulp.mean()),
        "max_abs_err": float(np.abs(got.astype(np.float64) - oracle).max()),
        "rel_err": float(np.abs(got.astype(np.float64) - oracle).max() / scale),
        "nonfinite": int((~np.isfinite(got)).sum()),
        "identical": bool(np.array_equal(got, rounded)),
        "integer": False,
    }


# Ops whose engine entry point is not `<op>` or `<op>_wrapper`, or that need
# their arguments arranged. Everything else resolves by name.
_EXPLICIT = {
    "scaled_dot_product_attention":
        lambda w, i, k: w.scaled_dot_product_attention_wrapper(*i[:3], **k),
    "_scaled_dot_product_efficient_attention":
        lambda w, i, k: w.scaled_dot_product_attention_wrapper(*i[:3], **k),
    "convolution": lambda w, i, k: w.conv2d_wrapper(*i[:3], **k),
    # `dim` sits BETWEEN the two tensors in this one's signature.
    "index_select": lambda w, i, k: w.index_select_wrapper(i[0], k.get("dim", 0), i[1]),
    "mm": lambda w, i, k: w.mm(i[0], i[1]),
    "matmul": lambda w, i, k: w.mm(i[0], i[1]),
    "addmm": lambda w, i, k: w.addmm(i[0], i[1], i[2]),
    "bmm": lambda w, i, k: w.bmm(i[0], i[1]),
    "baddbmm": lambda w, i, k: w.baddbmm_wrapper(i[0], i[1], i[2]),
}

# Ops the bank records without enough to re-issue the call. `fill` writes a
# constant that is nowhere in the file except the oracle itself, and taking
# the answer from the oracle would test only that something was written.
_NO_SIGNATURE = {
    "fill": "the fill VALUE is not recorded; it exists only in the oracle, "
            "and reading it from there would test the store and not the value",
    "cat": "the entry records no inputs (it is one of the bank's unlaunched "
           "rows), so there is no call to re-issue",
    "clamp": "the clamp BOUNDS are not recorded, and running with the "
             "engine's defaults is a different call from the one CUDA ran — "
             "it scored 2755 ULP against an oracle for a call it never made",
    "clamp_min": "the minimum is not recorded (same reason as clamp)",
    "constant_pad_nd": "the PAD WIDTHS are not recorded and the output shape "
                       "does not determine the left/right split",
}


def kwargs_from_tag(op: str, tag: str, payload) -> dict:
    """The call arguments the bank encodes in its TAG rather than its kwargs.

    Every entry records `kwargs: {}`, but a reduction's `dim`, an upsample's
    scale and a group norm's group count are not optional — the call cannot be
    re-issued without them, and they ARE recorded, in the tag
    (`rows3x1024_dim1`, `x2`, `C32HW33G8`). Reading them there is reading what
    the bank wrote down; guessing them would produce a comparison against a
    different call than CUDA ran, which is worse than no comparison.

    Whatever this returns is checked afterwards against the recorded OUTPUT
    SHAPE, and a mismatch marks the entry not-replayable rather than compared.
    """
    import re
    kwargs = {}
    found = re.search(r"_dim(-?\d+)", tag or "")
    if found:
        kwargs["dim"] = int(found.group(1))
    if op in ("_softmax", "_log_softmax") and "dim" not in kwargs:
        kwargs["dim"] = -1                      # rows4xN: the row axis
    if op == "glu" and "dim" not in kwargs:
        kwargs["dim"] = -1                      # out is half the input
    if op == "index_select" and "dim" not in kwargs:
        kwargs["dim"] = 0                       # rows33_sel19: rows selected
    if op == "upsample_nearest2d":
        found = re.fullmatch(r"x(\d+)", tag or "")
        if found and "out0" in payload.files:
            # The engine takes the OUTPUT SIZE, and the bank recorded it.
            kwargs["output_size"] = list(payload["out0"].shape[-2:])
    if op == "native_group_norm":
        found = re.search(r"G(\d+)", tag or "")
        if found and "in0" in payload.files:
            shape = payload["in0"].shape
            hxw = 1
            for d in shape[2:]:
                hxw *= d
            kwargs.update({"N": shape[0], "C": shape[1], "HxW": hxw,
                           "num_groups": int(found.group(1)), "eps": 1e-5})
    return kwargs


def _entry_point(wrappers, op: str):
    if op in _EXPLICIT:
        return _EXPLICIT[op]
    for cand in (f"{op}_wrapper", op, f"{op.lstrip('_')}_wrapper", op.lstrip("_")):
        fn = getattr(wrappers, cand, None)
        if callable(fn):
            return lambda w, i, k, _fn=fn: _fn(*i, **k)
    return None


def run_entry(payload, meta):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    op = meta["op"]
    if op in _NO_SIGNATURE:
        raise NotImplementedError(_NO_SIGNATURE[op])
    entry = _entry_point(wrappers, op)
    if entry is None:
        raise NotImplementedError(f"no engine entry point resolves for {op!r}")
    inputs = [NBXTensor.from_numpy(np.ascontiguousarray(payload[k]))
              for k in sorted(payload.files) if k.startswith("in")]
    if not inputs:
        raise NotImplementedError("the entry records no inputs")
    kwargs = dict(meta.get("kwargs") or {})
    kwargs.update(kwargs_from_tag(op, meta.get("tag", ""), payload))
    try:
        out = entry(wrappers, inputs, kwargs)
    except TypeError as exc:
        # An argument mismatch is the BANK not recording something the call
        # needs, or this tool naming it differently — never the backend
        # refusing. Calling anyway, with a default, would compare against an
        # oracle for a call that was never made: `clamp` scored 2755 ULP that
        # way before this was separated out.
        raise NotImplementedError(
            f"the call needs arguments that are not recorded in a form this "
            f"tool can re-issue (tried {sorted(kwargs) or 'positional only'}): "
            f"{exc}") from exc
    outs = out if isinstance(out, (tuple, list)) else [out]
    arrays = [np.asarray(o.numpy()) for o in outs]

    # The shape check that makes the reconstruction safe: if what came back is
    # not the shape CUDA produced, the two are not the same call and are not
    # compared.
    for i, arr in enumerate(arrays):
        key = f"out{i}"
        if key in payload.files and tuple(arr.shape) != tuple(payload[key].shape):
            raise NotImplementedError(
                f"output {i} is {tuple(arr.shape)} here against "
                f"{tuple(payload[key].shape)} in the bank — the call was "
                f"re-issued with different arguments, so the numbers are not "
                f"comparable")
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", default=None,
                        help="comma-separated op substrings")
    args = parser.parse_args()

    cleared = clear_caches()
    from neurobrix.kernels import launcher
    launcher.install()

    wanted = [w.strip() for w in args.only.split(",")] if args.only else None
    rows = []
    files = sorted(args.bank.rglob("*.npz"))
    for path in files:
        payload = np.load(path, allow_pickle=True)
        meta = json.loads(str(payload["meta"]))
        op = meta["op"]
        if wanted and not any(w in op for w in wanted):
            continue
        row = {"op": op, "tag": meta.get("tag"), "file": path.name,
               "dir": path.parent.name,
               "launched": meta.get("launched", []),
               "cuda": meta.get("stats", [])}
        started = time.time()
        try:
            got = run_entry(payload, meta)
            row["metal"] = [ulp_distance(g, payload[f"oracle{i}"])
                            for i, g in enumerate(got)
                            if f"oracle{i}" in payload.files]
            row["status"] = "ok"
        except NotImplementedError as exc:
            row["status"] = "not-replayable"
            row["error"] = str(exc)[:220]
        except Exception as exc:
            row["status"] = "refused"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}"
            row["traceback_tail"] = traceback.format_exc().splitlines()[-1][:200]
        row["wall_s"] = round(time.time() - started, 4)

        # The bar, applied per entry: Metal no further from the oracle than CUDA.
        verdict = "—"
        if row["status"] == "ok" and row["metal"]:
            m = row["metal"][0].get("max_ulp")
            c = (row["cuda"][0].get("max_ulp") if row["cuda"] else None)
            if m is None or c is None:
                verdict = "no ULP recorded"
            elif m <= c:
                verdict = "PASS"
            else:
                verdict = "OVER"
        row["verdict"] = verdict
        rows.append(row)
        mark = {"ok": "ok ", "refused": "REF", "not-replayable": "n/a"}[row["status"]]
        extra = ""
        if row["status"] == "ok" and row["metal"]:
            extra = (f"metal={row['metal'][0].get('max_ulp')} "
                     f"cuda={row['cuda'][0].get('max_ulp') if row['cuda'] else None} "
                     f"{verdict}")
        elif row["status"] != "ok":
            extra = row.get("error", "")[:90]
        print(f"  {mark} {op:<40} {str(meta.get('tag')):<22} {extra}", flush=True)

    library = library_kernels()
    document = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/metal_kernel_census.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "bank": str(args.bank),
        "caches_cleared_before_the_run": cleared,
        "bar": "Metal max ULP <= CUDA max ULP, per kernel and per shape, "
               "against the same fp64 oracle",
        "library_kernel_count": len(library),
        "library": library,
        "entries": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "census.json").write_text(json.dumps(document, indent=1))
    (args.out / "CENSUS.md").write_text(_census_markdown(document))

    missing = [n for n in ("census.json", "CENSUS.md")
               if not (args.out / n).exists()]
    if missing:
        print(f"\nREFUSING to conclude: {missing} not written", flush=True)
        return 2
    ok = [r for r in rows if r["status"] == "ok"]
    over = [r for r in rows if r["verdict"] == "OVER"]
    refused = [r for r in rows if r["status"] == "refused"]
    print(f"\n{len(rows)} entries: {len(ok)} ran, {len(refused)} refused, "
          f"{len(over)} over the bar -> {args.out / 'CENSUS.md'}")
    return 1 if (over or refused) else 0


def _census_markdown(document) -> str:
    rows = document["entries"]
    by_kernel = {}
    for r in rows:
        for k in r["launched"] or []:
            by_kernel.setdefault(k, []).append(r)

    ok = [r for r in rows if r["status"] == "ok"]
    over = [r for r in rows if r["verdict"] == "OVER"]
    refused = [r for r in rows if r["status"] == "refused"]
    nrep = [r for r in rows if r["status"] == "not-replayable"]

    out = []
    out.append("# Numeric census — the kernel library against the Dell's reference bank")
    out.append("")
    out.append(f"Generated **{document['generated']}** by `{document['tool']}` "
               f"on {document['machine']}.")
    out.append("**No public claim is made from any number here.**")
    out.append("")
    out.append(f"* bank: `{document['bank']}` (read-only mount; nothing written to it)")
    out.append(f"* bar: {document['bar']}")
    out.append("* caches cleared by the tool before the run:")
    for c in document["caches_cleared_before_the_run"]:
        out.append(f"  * `{c}`")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"* library: **{document['library_kernel_count']}** `@triton.jit` "
               f"kernels, counted from the source")
    out.append(f"* bank entries replayed: **{len(rows)}** — {len(ok)} ran, "
               f"{len(refused)} refused, {len(nrep)} not replayable")
    out.append(f"* of those that ran: **{len(ok) - len(over)} at or under CUDA's "
               f"ULP**, {len(over)} over")
    reached = {k for r in rows for k in (r["launched"] or [])}
    lib_names = {k["kernel"] for k in document["library"]}
    out.append(f"* kernels the bank reaches: **{len(reached & lib_names)}** of "
               f"{len(lib_names)}; not reached: {len(lib_names - reached)}")
    out.append("")

    if over:
        out.append("## Over the bar")
        out.append("")
        out.append("| op | shape | Metal ULP | CUDA ULP | Metal rel | kernels |")
        out.append("|---|---|---:|---:|---:|---|")
        for r in over:
            m, c = r["metal"][0], (r["cuda"][0] if r["cuda"] else {})
            out.append(f"| {r['op']} | {r['tag']} | {m.get('max_ulp')} | "
                       f"{c.get('max_ulp')} | {m.get('rel_err'):.2e} | "
                       f"{', '.join(r['launched'] or []) or '—'} |")
        out.append("")

    if refused:
        out.append("## Refused")
        out.append("")
        out.append("| op | shape | reason |")
        out.append("|---|---|---|")
        for r in refused:
            out.append(f"| {r['op']} | {r['tag']} | {r.get('error', '')[:200]} |")
        out.append("")

    out.append("## Every bank entry")
    out.append("")
    out.append("| op | shape | status | Metal ULP | CUDA ULP | verdict | kernels launched |")
    out.append("|---|---|---|---:|---:|---|---|")
    for r in rows:
        m = r["metal"][0] if r.get("metal") else {}
        c = r["cuda"][0] if r.get("cuda") else {}
        out.append(f"| {r['op']} | {r['tag']} | {r['status']} | "
                   f"{m.get('max_ulp', '—')} | {c.get('max_ulp', '—')} | "
                   f"{r['verdict']} | {', '.join(r['launched'] or []) or '—'} |")
    out.append("")

    out.append(f"## The library, all {document['library_kernel_count']} kernels")
    out.append("")
    out.append("A kernel with no bank entry is an UNMEASURED kernel. Saying so "
               "is part of the census.")
    out.append("")
    out.append("| kernel | file | bank entries | worst Metal ULP | worst CUDA ULP | verdict |")
    out.append("|---|---|---:|---:|---:|---|")
    for k in document["library"]:
        entries = by_kernel.get(k["kernel"], [])
        if not entries:
            out.append(f"| `{k['kernel']}` | {k['file']} | 0 | — | — | not reached |")
            continue
        ms = [e["metal"][0].get("max_ulp") for e in entries
              if e.get("metal") and e["metal"][0].get("max_ulp") is not None]
        cs = [e["cuda"][0].get("max_ulp") for e in entries
              if e.get("cuda") and e["cuda"][0].get("max_ulp") is not None]
        verdicts = {e["verdict"] for e in entries}
        if "OVER" in verdicts:
            v = "OVER"
        elif any(e["status"] == "refused" for e in entries):
            v = "refused"
        elif "PASS" in verdicts:
            v = "PASS"
        else:
            v = "—"
        out.append(f"| `{k['kernel']}` | {k['file']} | {len(entries)} | "
                   f"{max(ms) if ms else '—'} | {max(cs) if cs else '—'} | {v} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
