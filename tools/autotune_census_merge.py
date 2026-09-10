"""Merge autotune replay caches into one census file for `neurobrix autotune certify --census`.

The certifier's census is the machine's replay cache — every key the engine swept at runtime.
Since the retrace gate froze the kernel-config state per gate (2026-09-07), a gate's runtime
sweeps land in the gate's OWN replay cache (`<model dir>/autotune_replay/`), never in the
machine's: Sana 4K's old Triton arm swept 29 keys of 4K convolutions the census had never
seen. This merges every replay cache found under the given roots (and the machine's) into
one census; a key present in several caches keeps the first config seen (the census carries
keys, the certifier measures the configs itself).

    python tools/autotune_census_merge.py --out /path/census.json [--roots validation_outputs/retrace_2026_09_07 ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MACHINE_CACHE = Path.home() / ".neurobrix" / "replay_cache"
CACHE_FILE = "autotune_configs_cuda-70.json"


def find_caches(roots, name: str = CACHE_FILE) -> list:
    """Every replay-cache file under the roots (any depth), the machine's first."""
    found = []
    machine = MACHINE_CACHE / name
    if machine.exists():
        found.append(machine)
    for root in roots:
        for p in sorted(Path(root).rglob(name)):
            if p not in found:
                found.append(p)
    return found


def merge(paths) -> tuple:
    """(census dict, {path: keys it added})"""
    census, added = {}, {}
    for p in paths:
        try:
            doc = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        n = 0
        for k, v in doc.items():
            if k not in census:
                census[k] = v; n += 1
        added[str(p)] = n
    return census, added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="*", default=[], help="directories to search for replay caches (recursively)")
    args = ap.parse_args(argv)
    paths = find_caches(args.roots)
    census, added = merge(paths)
    Path(args.out).write_text(json.dumps(census, indent=0, sort_keys=True))
    for p, n in added.items():
        if n:
            print(f"[census] +{n} key(s) from {p}")
    print(f"[census] {len(census)} keys from {len(paths)} cache(s) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
