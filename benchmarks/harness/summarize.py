"""Public matrix renderer for a dated benchmark results directory.

Reads benchmarks/results/<date>/*.json cell artifacts (written by
run_bench.py) and emits SUMMARY.md in the same directory:

1. **Capability spotlight first** (supervisor doctrine, 2026-08-08):
   every DNR cell — a competitor that cannot run the row's model on
   this rig at its documented best — is listed with its evidence,
   next to NeuroBrix's own status on the same row. "They cannot run
   model X on V100; we do" is the headline axis; speed follows.
2. One table per (row, config) with the per-column headline metric:
   tokens → median warm tok/s + median TTFT; image/video → median
   wall (+ s/step); stt → median wall + RTF; omni → median wall.
   Peak GPU memory rides every cell (rig-wide sum for the machine
   config).

Usage:
  python benchmarks/harness/summarize.py --date 2026_08_05_ref
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _fmt(v, nd=2, suffix=""):
    return "—" if v is None else f"{v:.{nd}f}{suffix}"


def headline(artifact: dict, metric_class: str) -> str:
    reqs = artifact.get("requests") or []
    peak = artifact.get("peak_gpu_mem_mib")
    mem = f" · {peak / 1024:.1f}G" if peak else ""
    if metric_class == "tokens":
        def rate(r):
            if r.get("tok_per_s") is not None:
                return r["tok_per_s"]
            # Derive from the exact per-token stream-event count when
            # the cell predates the tokens-fallback (same formula).
            ns, w, tt = (r.get("tokens_streamed"), r.get("wall_s"),
                         r.get("ttft_s"))
            if ns and ns > 1 and w and tt is not None and w > tt:
                return (ns - 1) / (w - tt)
            return None
        tps = _median([rate(r) for r in reqs])
        ttft = _median([r.get("ttft_s") for r in reqs])
        return f"{_fmt(tps)} tok/s · TTFT {_fmt(ttft)}s{mem}"
    wall = _median([r.get("wall_s") for r in reqs])
    if metric_class == "stt":
        rtf = _median([r.get("rtf") for r in reqs])
        return f"{_fmt(wall)}s · RTF {_fmt(rtf)}{mem}"
    if metric_class in ("image", "video"):
        sps = _median([r.get("s_per_step") for r in reqs])
        extra = f" · {_fmt(sps)}s/step" if sps else ""
        return f"{_fmt(wall)}s{extra}{mem}"
    return f"{_fmt(wall)}s{mem}"  # omni: wall prompt→artifact


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    import yaml
    rows_cfg = {r["id"]: r for r in yaml.safe_load(
        (REPO / "benchmarks" / "config" / "rows.yml").read_text())["rows"]}
    out_dir = REPO / "benchmarks" / "results" / args.date

    # cells[(row, config)][column] = artifact
    cells: dict = {}
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "env_manifest.json":
            continue
        try:
            a = json.loads(p.read_text())
        except ValueError:
            continue
        if "row" not in a or "column" not in a:
            continue
        cfg = a.get("config", "pinned")
        cells.setdefault((a["row"], cfg), {})[a["column"]] = a

    lines = [f"# Benchmark matrix — {args.date}", ""]

    # ── 1. Capability spotlight (DNR cells lead) ──
    dnr_lines = []
    for (row_id, cfg), cols in sorted(cells.items()):
        nbx_ok = [c for c, a in cols.items()
                  if c.startswith("neurobrix") and a.get("status") == "ok"]
        for col, a in sorted(cols.items()):
            if a.get("status") != "dnr":
                continue
            us = (f"NeuroBrix RUNS it ({', '.join(sorted(nbx_ok))})"
                  if nbx_ok else "NeuroBrix columns pending")
            dnr_lines.append(
                f"- **{row_id}** [{cfg}] — `{col}` **cannot run it on "
                f"this rig**: {a.get('evidence', a.get('error', ''))} "
                f"→ {us}.")
    if dnr_lines:
        lines += ["## What only NeuroBrix runs on this hardware", "",
                  "Every cell below is a documented does-not-run verdict "
                  "for a competitor at its pinned best on the V100 rig.",
                  "", *dnr_lines, ""]

    # ── 2. Per-row metric tables ──
    for (row_id, cfg), cols in sorted(cells.items()):
        mclass = rows_cfg.get(row_id, {}).get("metric_class", "tokens")
        lines += [f"## {row_id} — config: {cfg}", "",
                  "| column | result |", "|---|---|"]
        for col, a in sorted(cols.items()):
            st = a.get("status")
            if st == "ok":
                cell = headline(a, mclass)
            elif st == "dnr":
                cell = f"**DNR** — {a.get('evidence', '')[:160]}"
            else:
                cell = f"error — `{str(a.get('error', ''))[:120]}`"
            lines.append(f"| {col} | {cell} |")
        lines.append("")

    out = out_dir / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[summarize] {sum(len(c) for c in cells.values())} cells "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
