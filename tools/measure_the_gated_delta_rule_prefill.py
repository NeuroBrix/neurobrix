#!/usr/bin/env python3
"""The gated delta rule's prefill on one card: the library's serial scan (`nbxtensor.ops.gated_delta_rule.
gated_delta_rule_wrapper`, the state float32) against the vendor's chunked torch function (the PyTorch engine's path),
at the lengths, batches and dtypes asked, at the head geometry asked. A measurement, nothing more: no kernel here.

Per point (L, B, dtype), the same operands to both sides (q, k [B, L, Hk, dk], v [B, L, Hv, dv], beta at the dtype,
g float32, no starting state: a first prefill, as the layer calls it):
- `--warmup` runs of each side, then `--runs` timed runs, the two sides alternated run by run (a drift of the card
  hits both), each run framed by a synchronisation of BOTH runtimes before and of its own after; wall seconds by
  `time.perf_counter`. Median, quartiles, min and max are kept, every time too.
- the peak card memory of each side, by its own allocator, over one call: the peak above the bytes live before it
  (its outputs and its workspace) and the absolute peak.
- the max deviation of each side's output and final state against the float64 reference of L1
  (`nbxtensor.ops.oracles.fp64.delta_rule_oracle`, windowed over heads and batch rows above its cap, never over
  positions), from the very numbers both sides read (the inputs rounded to the dtype on the host).

The vendor function is named by `--vendor module:function`. A function behind a dispatch decorator (one that may
hand the call to another package's kernel) is measured through its `__wrapped__`: the torch body as its file states
it. The file (relative to its package root), its sha256, the function's first line and the package's version are
written into the result: that is the citation, the source is never copied here.

THE VERDICT RULE (written before any run, 2026-11-16; tested on synthetic numbers in
`tests/unit/tools/test_the_prefill_verdict_answers_each_way_on_synthetic_numbers.py`):
1. A point where a side did not run is `no measurement`.
2. A point where a side's output deviation exceeds `--bound-out` (plus half an ulp of the output dtype) or its state
   deviation exceeds `--bound-state` is `not comparable` (speeds are compared at equal correctness only).
3. Otherwise, with q1 and q3 the quartiles of each side's timed runs: the scan WINS the point when its q3 is below
   the vendor's q1, LOSES it when its q1 is above the vendor's q3, and TIES it otherwise.
4. The scan wins a length L when it wins every point at L. A tie is not a win (ruling R2: the Triton path must do
   better than the PyTorch path). A length with a `no measurement` or `not comparable` point has no verdict.
5. The verdict word: LOSES when the scan loses at least one length (the chunked kernel is then carded), WINS when it
   wins every length, NO VERDICT otherwise. The line names the lengths of each kind.

Run (a card; the same file is its own CPU dry run with `--device cpu`, where the scan, which launches on a device
only, is formed and checked but not run, and no timing is claimed):

    python tools/measure_the_gated_delta_rule_prefill.py --device cuda:0 --vendor <module>:<function> \\
        --lengths 512,2048,4096 --batches 1,4 --dtypes float32,float16 --key-heads 16 --value-heads 48 \\
        --key-dim 128 --value-dim 128 --eps 1e-6 --bound-out 4e-6 --bound-state 4e-6 --warmup 3 --runs 20 \\
        --seed 458 --commit "$(git rev-parse HEAD)" --out-dir <folder>
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import time

import numpy as np

RULE = ("per point: a side not run = no measurement; a deviation above its bound (out: bound-out + half an ulp of the "
        "output dtype; state: bound-state) = not comparable; else scan q3 < vendor q1 = win, scan q1 > vendor q3 = "
        "loss, otherwise tie. Per length: win only if every point wins; any no-measurement or not-comparable point = "
        "no verdict; else loss. Word: LOSES if any length is lost, WINS if every length is won, else NO VERDICT.")

#: Half an ulp relative to the largest value of an output stored at this dtype (the device test's ROUNDING).
OUTPUT_ROUNDING = {"float32": 0.0, "float16": 2.0 ** -11}
NUMPY_DTYPE = {"float32": np.float32, "float16": np.float16}


# --- the rule ------------------------------------------------------------------------------------------------------

def spread(times):
    """Median, quartiles, min and max of a side's timed runs, in seconds."""
    t = np.asarray(times, dtype=np.float64)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {"median_s": float(med), "q1_s": float(q1), "q3_s": float(q3), "min_s": float(t.min()),
            "max_s": float(t.max())}


def point_verdict(scan, vendor, dtype, bound_out, bound_state):
    """(kind, why) of one point under the rule: kind in 'no measurement', 'not comparable', 'win', 'loss', 'tie'.
    `scan` and `vendor` are the sides' records: `ran`, `q1_s`, `q3_s`, `dev_out`, `dev_state`."""
    for name, side in (("scan", scan), ("vendor", vendor)):
        if not side["ran"]:
            return "no measurement", f"the {name} did not run: {side['why']}"
    limit_out = bound_out + OUTPUT_ROUNDING[dtype]
    for name, side in (("scan", scan), ("vendor", vendor)):
        if not (side["dev_out"] <= limit_out and side["dev_state"] <= bound_state):
            return "not comparable", (f"the {name}'s deviation out {side['dev_out']:.2e} / state "
                                      f"{side['dev_state']:.2e} exceeds {limit_out:.2e} / {bound_state:.2e}")
    if scan["q3_s"] < vendor["q1_s"]:
        return "win", "scan q3 below vendor q1"
    if scan["q1_s"] > vendor["q3_s"]:
        return "loss", "scan q1 above vendor q3"
    return "tie", "the quartile ranges overlap"


def overall_verdict(points):
    """The verdict over every point (each a dict with `L` and `kind`): {word, wins, losses, none, line}."""
    by_length = {}
    for p in points:
        by_length.setdefault(p["L"], []).append(p["kind"])
    wins, losses, none = [], [], []
    for L in sorted(by_length):
        kinds = by_length[L]
        if any(k in ("no measurement", "not comparable") for k in kinds):
            none.append(L)
        elif all(k == "win" for k in kinds):
            wins.append(L)
        else:
            losses.append(L)
    word = "LOSES" if losses else ("WINS" if wins and not none else "NO VERDICT")
    line = (f"VERDICT: THE SCAN {word}" if word != "NO VERDICT" else "VERDICT: NO VERDICT") + \
        f" (wins at L = {wins or '-'}; loses at L = {losses or '-'}; no verdict at L = {none or '-'})"
    return {"word": word, "wins": wins, "losses": losses, "none": none, "line": line}


# --- the operands --------------------------------------------------------------------------------------------------

def draw(B, L, Hk, Hv, dk, dv, seed):
    """The operands in float32 on the host, drawn as the library's tests draw them (`_delta_rule_draw.draw`): q, k, v
    standard normal, g = -A softplus(n) with A in (0.01, 1) per value head, beta = sigmoid(n); no state."""
    r = np.random.default_rng(seed)
    q = r.standard_normal((B, L, Hk, dk)).astype(np.float32)
    k = r.standard_normal((B, L, Hk, dk)).astype(np.float32)
    v = r.standard_normal((B, L, Hv, dv)).astype(np.float32)
    A = r.uniform(0.01, 1.0, Hv).astype(np.float32)
    g = (-A * np.log1p(np.exp(r.standard_normal((B, L, Hv))))).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-r.standard_normal((B, L, Hv))))).astype(np.float32)
    return q, k, v, g, beta


def rounded(q, k, v, g, beta, dtype):
    """The numbers both sides read: q, k, v, beta stored at `dtype`, g float32."""
    nd = NUMPY_DTYPE[dtype]
    return q.astype(nd), k.astype(nd), v.astype(nd), g, beta.astype(nd)


# --- the two sides -------------------------------------------------------------------------------------------------

def device_index(device):
    if device == "cpu":
        return None
    if not device.startswith("cuda:"):
        raise SystemExit(f"--device {device!r}: this measurement takes 'cpu' (a dry run) or 'cuda:N'")
    return int(device.split(":", 1)[1])


class Scan:
    """The library's scan: operands formed on the device, one call = one launch."""

    def __init__(self, device, eps, scale):
        from nbxtensor import device as D
        from nbxtensor import memory as M
        from nbxtensor.dtype.nbx_dtype import NBXDtype
        from nbxtensor.ops import gated_delta_rule as G
        self.D, self.M, self.G, self.NBXDtype = D, M, G, NBXDtype
        self.idx, self.eps, self.scale = device_index(device), eps, scale
        self.ops = self.out = None
        if self.idx is not None:
            D.select(self.idx)

    def prepare(self, arrays, dtype):
        """The operands at `dtype`; on the host (a dry run) the library's own shape and dtype checks only."""
        dt = getattr(self.NBXDtype, dtype)
        F32 = self.NBXDtype.float32
        q, k, v, g, beta = arrays
        if self.idx is None:
            self.G.gated_delta_rule_shapes(q, k, v, g, beta, None)
            self.G.gated_delta_rule_dtypes(dt, dt, dt, F32, dt, None, F32)
            self.ops = None
            return
        from nbxtensor.tensor import NBXTensor
        self.ops = tuple(NBXTensor.from_numpy_as(a, F32 if a is g else dt, device=self.idx) for a in arrays)

    @property
    def why_not(self):
        return None if self.idx is not None else ("the library's scan launches on a device only; this run's device "
                                                  "is cpu (operands formed and accepted by its shape and dtype checks)")

    def call(self):
        self.out = self.G.gated_delta_rule_wrapper(*self.ops, None, qk_l2norm=True, eps=self.eps, scale=self.scale,
                                                   compute_dtype=self.NBXDtype.float32)

    def sync(self):
        if self.idx is not None:
            self.D.synchronize()

    def peak(self):
        """(peak above the live bytes before one call, absolute peak) by the library's allocator."""
        self.out = None
        self.sync()
        self.M.empty_cache_pool()                   # parked blocks count as live: the call's own bytes start fresh
        self.M.reset_peak_memory(self.idx)
        base = self.M.memory_allocated(self.idx)
        self.call()
        self.sync()
        top = self.M.peak_memory_allocated(self.idx)
        return top - base, top

    def deviations(self, ref_o, ref_s):
        from nbxtensor.ops.oracles import fp64 as O
        o, s = self.out
        return O.deviation_against(o, ref_o), O.deviation_against(s, ref_s)

    def release(self):
        self.ops = self.out = None


class Vendor:
    """The vendor's chunked torch function, called as the layer calls it in a first prefill (q and k expanded to the
    value heads before it, no state, the final state asked, the l2 norm inside)."""

    def __init__(self, spec, device):
        import torch
        self.torch = torch
        mod_name, _, fn_name = spec.partition(":")
        if not fn_name:
            raise SystemExit(f"--vendor {spec!r}: give it as <module>:<function>")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        self.fn = getattr(fn, "__wrapped__", fn)
        self.dev = torch.device(device)
        self.ops = self.out = None
        src = inspect.getsourcefile(self.fn)
        root = mod_name.split(".")[0]
        parts = src.replace(os.sep, "/").split("/")
        rel = "/".join(parts[len(parts) - 1 - parts[::-1].index(root):]) if root in parts else os.path.basename(src)
        with open(src, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        from importlib import metadata
        self.citation = {"named": spec, "measured": f"{self.fn.__module__}.{self.fn.__qualname__}",
                         "through_wrapped": self.fn is not fn, "file": rel, "file_sha256": sha,
                         "first_line": inspect.getsourcelines(self.fn)[1], "package": root,
                         "package_version": metadata.version(root), "torch_version": torch.__version__}

    def prepare(self, arrays, dtype, ratio):
        t = self.torch
        q, k, v, g, beta = (t.from_numpy(np.ascontiguousarray(a)).to(self.dev) for a in arrays)
        if ratio > 1:
            q, k = q.repeat_interleave(ratio, dim=2), k.repeat_interleave(ratio, dim=2)
        self.ops = (q, k, v, g, beta)

    def call(self):
        q, k, v, g, beta = self.ops
        self.out = self.fn(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                           use_qk_l2norm_in_kernel=True)

    def sync(self):
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)

    def peak(self):
        if self.dev.type != "cuda":
            self.call()
            return None, None
        self.out = None
        self.sync()
        self.torch.cuda.reset_peak_memory_stats(self.dev)
        base = self.torch.cuda.memory_allocated(self.dev)
        self.call()
        self.sync()
        top = self.torch.cuda.max_memory_allocated(self.dev)
        return top - base, top

    def deviations(self, ref_o, ref_s):
        from nbxtensor.ops.oracles import fp64 as O
        o, s = (x.float().cpu().numpy() for x in self.out)
        return O.oracle_deviation(o, ref_o), O.oracle_deviation(s, ref_s)

    def release(self):
        self.ops = self.out = None
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()


def device_lines(device, vendor):
    """Each runtime's own line for the card it uses; on a card both must show the same one (by its compute target)."""
    if device == "cpu":
        return {"library": "cpu: the library is not launched", "vendor": "torch on cpu"}
    from nbxtensor import device as D
    t = vendor.torch
    if D.count() < 1 or not t.cuda.is_available():
        raise SystemExit(f"no card: the library sees {D.count()} device(s), torch cuda available = "
                         f"{t.cuda.is_available()}")
    idx = device_index(device)
    p = t.cuda.get_device_properties(idx)
    lib, ven = D.target_name(), f"cuda-{p.major}{p.minor}"
    if lib != ven:
        raise SystemExit(f"the two runtimes do not show the same card: the library compiles for {lib}, torch's "
                         f"device {idx} is {p.name} ({ven})")
    return {"library": f"{lib}, {D.count()} device(s), device {idx}",
            "vendor": f"torch {t.__version__} device {idx}: {p.name} ({ven}), {p.total_memory} bytes"}


# --- the measurement -----------------------------------------------------------------------------------------------

def measure_point(scan, vendor, args, L, B, dtype):
    from nbxtensor.ops.oracles import fp64 as O
    arrays = rounded(*draw(B, L, args.key_heads, args.value_heads, args.key_dim, args.value_dim,
                           args.seed + 1000 * B + L), dtype)
    scan.prepare(arrays, dtype)
    vendor.prepare(arrays, dtype, args.value_heads // args.key_heads)
    ran = scan.why_not is None
    sides = [("scan", scan)] if ran else []
    sides.append(("vendor", vendor))
    for _ in range(args.warmup):
        for _, s in sides:
            s.call()
            s.sync()
    times = {name: [] for name, _ in sides}
    for i in range(args.runs):
        for name, s in (sides if i % 2 == 0 else sides[::-1]):
            for _, other in sides:
                other.sync()
            t0 = time.perf_counter()
            s.call()
            s.sync()
            times[name].append(time.perf_counter() - t0)
    ref_o, ref_s = O.delta_rule_oracle(*(a.astype(np.float32) for a in arrays), None, qk_l2norm=True,
                                       eps=args.eps, scale=args.key_dim ** -0.5)()
    rec = {}
    for name, s in (("scan", scan), ("vendor", vendor)):
        if name not in times:
            rec[name] = {"ran": False, "why": scan.why_not}
            continue
        over, top = s.peak()
        dev_o, dev_s = s.deviations(ref_o, ref_s)
        rec[name] = dict(ran=True, why=None, times_s=times[name], peak_over_inputs_bytes=over, peak_bytes=top,
                         dev_out=float(dev_o), dev_state=float(dev_s), **spread(times[name]))
        s.release()
    scan.release()
    kind, why = point_verdict(rec["scan"], rec["vendor"], dtype, args.bound_out, args.bound_state)
    return {"L": L, "B": B, "dtype": dtype, "scan": rec["scan"], "vendor": rec["vendor"], "kind": kind, "why": why,
            "oracle": getattr(ref_o, "describe", "whole")}


def _ms(side):
    if not side["ran"]:
        return "not run"
    return f"{side['median_s'] * 1e3:.3f} ({side['q1_s'] * 1e3:.3f}-{side['q3_s'] * 1e3:.3f})"


def _mib(side):
    if not side["ran"] or side["peak_over_inputs_bytes"] is None:
        return "-"
    return f"{side['peak_over_inputs_bytes'] / 2 ** 20:.1f} / {side['peak_bytes'] / 2 ** 20:.1f}"


def _dev(side):
    return f"{side['dev_out']:.1e} / {side['dev_state']:.1e}" if side["ran"] else "-"


def table(result):
    head = ("| L | B | dtype | scan ms median (q1-q3) | vendor ms median (q1-q3) | vendor / scan | "
            "scan MiB over inputs / peak | vendor MiB over inputs / peak | scan dev out / state | "
            "vendor dev out / state | point |")
    rows = [head, "|" + "---|" * 11]
    for p in result["points"]:
        s, v = p["scan"], p["vendor"]
        ratio = f"{v['median_s'] / s['median_s']:.2f}" if s["ran"] and v["ran"] else "-"
        rows.append(f"| {p['L']} | {p['B']} | {p['dtype']} | {_ms(s)} | {_ms(v)} | {ratio} | {_mib(s)} | {_mib(v)} | "
                    f"{_dev(s)} | {_dev(v)} | {p['kind']} |")
    note = "DRY RUN on cpu: no timing is claimed." if result["dry_run"] else ""
    return "\n".join(rows + ["", note, result["verdict"]["line"], ""])


def _ints(text):
    return [int(x) for x in text.split(",") if x]


def parse(argv):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--device", required=True)
    a.add_argument("--vendor", required=True, help="<module>:<function> of the vendor's chunked function")
    a.add_argument("--lengths", required=True, type=_ints)
    a.add_argument("--batches", required=True, type=_ints)
    a.add_argument("--dtypes", required=True, type=lambda s: [x for x in s.split(",") if x])
    for name in ("key-heads", "value-heads", "key-dim", "value-dim", "warmup", "runs", "seed"):
        a.add_argument(f"--{name}", required=True, type=int)
    for name in ("eps", "bound-out", "bound-state"):
        a.add_argument(f"--{name}", required=True, type=float)
    a.add_argument("--commit", required=True)
    a.add_argument("--out-dir", required=True)
    args = a.parse_args(argv)
    for d in args.dtypes:
        if d not in NUMPY_DTYPE:
            a.error(f"--dtypes {d!r}: this measurement rounds its inputs on the host, at {sorted(NUMPY_DTYPE)} only")
    if args.value_heads % args.key_heads:
        a.error(f"{args.value_heads} value heads are not a whole multiple of {args.key_heads} key heads")
    if args.runs < 4 or args.warmup < 1:
        a.error("quartiles need at least 4 timed runs and one warm-up run")
    return args


def main(argv=None):
    args = parse(argv)
    vendor = Vendor(args.vendor, args.device)
    lines = device_lines(args.device, vendor)
    print("library:", lines["library"])
    print("vendor:", lines["vendor"])
    scan = Scan(args.device, args.eps, args.key_dim ** -0.5)
    points = []
    for L in args.lengths:
        for B in args.batches:
            for dtype in args.dtypes:
                p = measure_point(scan, vendor, args, L, B, dtype)
                print(f"L={L} B={B} {dtype}: scan {_ms(p['scan'])} ms, vendor {_ms(p['vendor'])} ms, "
                      f"dev scan {_dev(p['scan'])}, vendor {_dev(p['vendor'])}: {p['kind']} ({p['why']})", flush=True)
                points.append(p)
    result = {"commit": args.commit, "dry_run": args.device == "cpu", "rule": RULE, "vendor": vendor.citation,
              "devices": lines, "args": {k: v for k, v in vars(args).items() if k != "out_dir"}, "points": points,
              "verdict": overall_verdict(points)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "gated_delta_rule_prefill.json"), "w") as f:
        json.dump(result, f, indent=1)
    text = table(result)
    with open(os.path.join(args.out_dir, "gated_delta_rule_prefill.md"), "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
