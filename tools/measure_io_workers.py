#!/usr/bin/env python3
"""How many parallel reads a machine's weight loaders should keep in flight — MEASURED, for the hardware profile's
`cpu.io_workers` (rule 1: a fact of the machine, never a number written by hand). The measurement is the library's
(`nbxtensor.io.read_rate.measure`: the loaders' reads, each trial's pages evicted first; F_NOCACHE alone kept serving
cached pages on the Mac, 19 064 MiB/s, evicted 32 MiB/s, 2026-10-30).

    python tools/measure_io_workers.py --files '<cache>/<model>/components/*/weights/*.safetensors' --max-gb 2

The answer is the SMALLEST worker count within `--within` (5 %) of the best rate: a worker beyond it holds a shard in
host memory for nothing (Prism's host estimate counts one per worker). A measure whose first and last one-worker
trials disagree is said, and is not to be recorded.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from nbxtensor.io.read_rate import PIECE, measure


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", required=True, help="glob of the shard files to read (the machine's real weights)")
    ap.add_argument("--max-gb", type=float, default=2.0, help="bytes read per trial")
    ap.add_argument("--max-workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--within", type=float, default=0.05)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(a.files))
    if not files:
        print(f"no file matches {a.files!r}", file=sys.stderr)
        return 2
    m = measure(files, int(a.max_gb * (1 << 30)), a.max_workers, a.within)
    print(f"[io_workers] {m['bytes'] / 2**30:.2f} GiB per trial from {len(files)} file(s), pieces of {PIECE >> 20} MiB")
    for c, r in m["rates"].items():
        print(f"[io_workers] {c:>3} worker(s): {r / 2**20:8.1f} MiB/s")
    if not m["steady"]:
        print(f"[io_workers] WARNING: one worker read {m['rates'][1] / 2**20:.0f} then {m['last_one_worker'] / 2**20:.0f} "
              f"MiB/s — the storage was not steady (another reader?); do not record this measure")
    best = max(m["rates"], key=m["rates"].get)
    print(f"cpu:\n  io_workers: {m['workers']}   # measured {time.strftime('%Y-%m-%d')}: "
          f"{m['rates'][m['workers']] / 2**20:.0f} MiB/s, best {m['rates'][best] / 2**20:.0f} MiB/s ({best} workers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
