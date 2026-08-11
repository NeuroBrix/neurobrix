#!/usr/bin/env python3
"""Paired ON/OFF delta report for the fusion_vertical campaign.

Reads two dated result dirs (same rows, same source, OFF vs ON arm),
computes per-cell medians of the warm requests and prints the paired
delta table. Median over N warm requests; cold start reported apart.

Usage:
  python3 benchmarks/harness/fusion_delta_report.py \
      --off 2026_08_10_fusion_off --on 2026_08_10_fusion_on
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _median(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def _cell_metrics(d: dict) -> dict:
    reqs = d.get("requests") or []
    out = {"cold_start_s": d.get("cold_start_s")}
    for key in ("wall_s", "ttft_s", "tok_per_s", "s_per_step", "rtf"):
        m = _median([r.get(key) for r in reqs])
        if m is not None:
            out[key] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    args = ap.parse_args()

    off_dir = REPO / "benchmarks" / "results" / args.off
    on_dir = REPO / "benchmarks" / "results" / args.on

    print(f"# fusion_vertical paired delta — OFF={args.off} ON={args.on}")
    print(f"| cell | metric | OFF | ON | delta |")
    print(f"|---|---|---|---|---|")
    rows_seen = 0
    for on_path in sorted(on_dir.glob("*.json")):
        if on_path.name == "env_manifest.json":
            continue
        off_path = off_dir / on_path.name
        if not off_path.exists():
            continue
        try:
            don = json.loads(on_path.read_text())
            doff = json.loads(off_path.read_text())
        except ValueError:
            continue
        if don.get("status") != "ok" or doff.get("status") != "ok":
            print(f"| {on_path.stem} | status | {doff.get('status')} "
                  f"| {don.get('status')} | — |")
            continue
        mon = _cell_metrics(don.get("result", don))
        moff = _cell_metrics(doff.get("result", doff))
        rows_seen += 1
        # EXACT-gate text check (gardien 2026-08-10): the media byte
        # gate covers wav/png; the VLM/LLM rows' exactness lives in the
        # per-request answer_sha256 — compare the sha SETS across arms.
        ron = (don.get("result", don) or {}).get("requests") or []
        roff = (doff.get("result", doff) or {}).get("requests") or []
        sha_on = {r.get("answer_sha256") for r in ron} - {None}
        sha_off = {r.get("answer_sha256") for r in roff} - {None}
        if sha_on or sha_off:
            verdict = ("BYTE-IDENTICAL" if sha_on == sha_off and
                       len(sha_on) == 1 else "MISMATCH")
            print(f"| {on_path.stem} | answer_sha | {len(sha_off)} uniq "
                  f"| {len(sha_on)} uniq | {verdict} |")
        for key in sorted(set(mon) | set(moff)):
            a, b = moff.get(key), mon.get(key)
            if a and b:
                # Positive = ON faster for time metrics; for tok_per_s
                # positive = ON higher (both read as "ON better").
                if key == "tok_per_s":
                    delta = (b - a) / a * 100.0
                else:
                    delta = (a - b) / a * 100.0
                print(f"| {on_path.stem} | {key} | {a:.4g} | {b:.4g} "
                      f"| {delta:+.2f}% |")
    if not rows_seen:
        print("(no paired ok cells found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
