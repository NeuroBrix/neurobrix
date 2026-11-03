"""The markdown summary of one device's benchmark: every cell where nbxtensor loses, named.

A kernel instance is LOST when nbxtensor is wrong or refused, or when its median is above the best torch contender's
that is right (`ok`); the ratio is nbxtensor's median over that best. A model cell is lost when the Triton run failed,
its output disagrees with the eager one beyond identity, or its wall time is above the eager run's. The losses are the
optimisation queue (Hocine's rule 14: Triton must do better than PyTorch).

    python benchmarks/nbxtensor_vs_torch/summary.py --device apple_m4_pro \\
        --kernel <campaign>/kernel.tsv> [--model <campaign>/model.tsv>] --out <summary.md>
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


#: A refusal the bench did not cause and nbxtensor's kernels did not lose: the key is outside the census (a confirmation
#: run never sweeps: `KeyNotCertified`), or the instance is beyond the device memory's op budget.
_KEY_OUTSIDE = ("KeyNotCertified", "CERTIFIED-ONLY")
_MEMORY = ("out_of_class", "out of memory", "OutOfMemory", "ImpossibleExtentError")


def refusal_kind(r) -> str | None:
    """'key' (key outside the census), 'memory' (beyond the device's memory), or None (any other outcome)."""
    text = f"{r.get('status', '')} {r.get('note', '')}"
    if r.get("status") == "out_of_class" or (r.get("status") == "refused" and any(m in text for m in _MEMORY)):
        return "memory"
    if r.get("status") == "refused" and any(m in text for m in _KEY_OUTSIDE):
        return "key"
    return None


def kernel_counts(rows) -> dict:
    """The kernel level in separate columns (the supervision, 2026-11-03 07:35), before any 'lost on N': lost
    (nbxtensor wrong, refused for another reason, or slower than the best right torch contender), refused because its
    key is outside the census, refused for memory, not finished in the run, won."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["key"]][r["contender"]] = r
    out = collections.Counter()
    for key, c in by.items():
        star = c.get("*")
        if star is not None and refusal_kind(star) == "memory":
            out["refused_memory"] += 1
            continue
        nbx = c.get("nbxtensor")
        if nbx is None:
            continue
        kind = refusal_kind(nbx)
        if kind == "key":
            out["refused_key"] += 1
        elif kind == "memory":
            out["refused_memory"] += 1
        elif nbx.get("status") == "not_finished":
            out["not_finished"] += 1
        else:
            out["judged"] += 1
    lost, won, _total = kernel_losses(rows)
    out["lost"], out["won"] = len(lost), won
    return dict(out)


def kernel_losses(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["key"]][r["contender"]] = r
    lost, won, total = [], 0, 0
    for key, c in by.items():
        nbx = c.get("nbxtensor")
        if nbx is None:
            continue
        if refusal_kind(nbx) is not None or nbx.get("status") == "not_finished":
            continue                     # counted apart (`kernel_counts`): neither a loss nor a win
        total += 1
        torch_ok = [r for n, r in c.items() if n != "nbxtensor" and r.get("status") == "ok" and _f(r.get("median_ms"))]
        best = min(torch_ok, key=lambda r: _f(r["median_ms"])) if torch_ok else None
        if nbx.get("status") != "ok":
            lost.append((float("inf"), nbx, best, nbx.get("status")))
        elif best is not None and _f(nbx["median_ms"]) > _f(best["median_ms"]):
            lost.append((_f(nbx["median_ms"]) / _f(best["median_ms"]), nbx, best, "slower"))
        else:
            won += 1
    return sorted(lost, key=lambda t: -t[0]), won, total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.kernel), delimiter="\t"))
    lost, won, total = kernel_losses(rows)
    n = kernel_counts(rows)
    md = [f"# nbxtensor vs torch — {a.device}", "",
          "| judged | won | lost | refused (key outside the census) | refused (memory) | not finished in the run |",
          "|---|---|---|---|---|---|",
          f"| {total} | {won} | {len(lost)} | {n.get('refused_key', 0)} | {n.get('refused_memory', 0)} | "
          f"{n.get('not_finished', 0)} |", "",
          f"Kernel level: of {total} judged instances, nbxtensor right and at least as fast as the best right torch "
          f"contender on {won}; lost on {len(lost)}.", "",
          "| op | shapes | dtypes | sites | nbxtensor | best torch | ratio | why |",
          "|---|---|---|---|---|---|---|---|"]
    for ratio, nbx, best, why in lost:
        md.append(f"| {nbx['op_type']} | {nbx['shapes']} | {nbx['dtypes']} | {nbx['sites']} | "
                  f"{nbx.get('median_ms') or nbx.get('status')} ms | "
                  f"{(best['contender'] + ' ' + best['median_ms'] + ' ms') if best else '-'} | "
                  f"{'-' if ratio == float('inf') else f'{ratio:.2f}x'} | "
                  f"{why}{(': ' + nbx['note'][:120]) if nbx.get('note') and why != 'slower' else ''} |")
    if a.model:
        mrows = list(csv.DictReader(open(a.model), delimiter="\t"))
        by = collections.defaultdict(dict)
        for r in mrows:
            by[r["model"]][r["contender"]] = r
        md += ["", "Model level (entry point; torch.compile has no engine mode):", "",
               "| model | eager s | eager peak MB | nbxtensor s | nbxtensor peak MB | outputs | lost |",
               "|---|---|---|---|---|---|---|"]
        for m, c in sorted(by.items()):
            if m == "*":
                continue
            e, n = c.get("torch_eager", {}), c.get("nbxtensor", {})
            lost_m = (str(n.get("rc")) != "0" or (_f(n.get("wall_s")) or 0) > (_f(e.get("wall_s")) or float("inf"))
                      or (n.get("agrees_with_torch_eager") or "").startswith("differs"))
            md.append(f"| {m} | {e.get('wall_s', '')} | {e.get('peak_mb', '')} | {n.get('wall_s', '')} | "
                      f"{n.get('peak_mb', '')} | {n.get('agrees_with_torch_eager', '')} | {'LOST' if lost_m else ''} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"[summary] {a.device}: kernel {won}/{total} won, {len(lost)} lost -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
