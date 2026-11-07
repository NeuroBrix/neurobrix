#!/usr/bin/env python3
"""The time of every certified kernel key, in ONE dated, cumulative registry (Hocine Benkelaya 2026-11-01 21:51: the figure that
says whether the rewrite wins or loses is read at every gate, not at the end).

    python tools/kernel_times.py --dir src/neurobrix/config/autotune/apple/apple_m4_pro --out <registres>/benchmarks/kernel_times.tsv

One row per (profile, kernel, dtype, key, certification date): the certified config's best time (best_ms), the second
best, its deviation against the fp64 oracle, the backend's hash and the engine version — read from each entry's proof,
never measured here. The file is appended, never rewritten: a key re-certified later is a NEW row (its date differs),
so the history of a key's time stays readable. Rows already present are skipped, so the tool is run after every
certification (and at every gate) without duplicating anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = ["date", "vendor", "profile", "kernel", "dtype", "key", "best_ms", "second_ms", "deviation", "tolerance",
           "backend_hash", "engine_version", "hardware_profile"]


def rows(directory: Path):
    for path in sorted(directory.glob("*.json")):
        doc = json.loads(path.read_text())
        for key, entry in (doc.get("entries") or {}).items():
            proof = entry.get("proof") or {}
            if proof.get("best_ms") is None:
                continue
            yield {"date": proof.get("date"), "vendor": doc.get("vendor"), "profile": doc.get("profile"),
                   "kernel": doc.get("kernel"), "dtype": doc.get("dtype"), "key": key,
                   "best_ms": proof.get("best_ms"), "second_ms": proof.get("second_ms"),
                   "deviation": proof.get("deviation"), "tolerance": proof.get("tolerance"),
                   "backend_hash": (proof.get("backend") or {}).get("backend_hash"),
                   "engine_version": proof.get("engine_version"),
                   "hardware_profile": (proof.get("machine") or {}).get("hardware_profile")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="a certified directory (<vendor>/<profile>)")
    ap.add_argument("--out", required=True, help="the cumulative registry (TSV, appended)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    seen = set()
    if out.is_file():
        with out.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                seen.add((r["profile"], r["kernel"], r["dtype"], r["key"], r["date"]))
    new = [r for r in rows(Path(a.dir)) if (r["profile"], r["kernel"], r["dtype"], r["key"], r["date"]) not in seen]
    write_header = not out.is_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        if write_header:
            w.writeheader()
        w.writerows(sorted(new, key=lambda r: (r["date"] or "", r["kernel"] or "", r["key"])))
    print(f"[kernel_times] {len(new)} new row(s) appended to {out} ({len(seen)} already there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
