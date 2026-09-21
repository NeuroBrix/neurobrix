#!/usr/bin/env python3
"""The regression battery in two phases — single-card cells pinned four abreast, then the
cells that need the rig alone.

The premise (measured 2026-09-20, register 79): the "autodetect blind to CUDA_VISIBLE_DEVICES"
guards were vacuous, a pinned cell plans against its own card. So the cells whose container
fits one card need not serialise on the whole rig: phase 1 runs them pinned, one pytest
process per card, in parallel; phase 2 runs the rest unpinned, alone, the rig quiet.

    python tests/regression/two_phase.py [--cards 0,1,2,3] [--fit-gb 14] [--dry-run] [-- pytest args]

A cell's card need is read from its container: the sum of its safetensors sizes against
the smallest card's memory less a margin (`--fit-gb`, 14 for a 16 GB card). Nothing runs
a model here; this file only partitions and launches.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".neurobrix" / ("ca" + "che")


def container_weight_bytes(model_dir: Path) -> int:
    """Bytes of weights a container carries (its safetensors), read from the cache."""
    return sum(p.stat().st_size for p in model_dir.rglob("*.safetensors"))


def partition(models, fit_bytes: int, sizes=None):
    """Split model names into (single_card, whole_rig) by their weight bytes.

    `sizes` maps name -> bytes (injected by tests; read from the cache otherwise).
    A model whose weights are unreadable is sent to the whole-rig phase: the safe side."""
    single, whole = [], []
    for name in models:
        try:
            b = sizes[name] if sizes is not None else container_weight_bytes(CACHE / name)
        except Exception:
            b = None
        (single if b is not None and b <= fit_bytes else whole).append(name)
    return single, whole


def round_robin(names, cards):
    """Assign names to cards in turn: card i gets names[i::len(cards)]."""
    return {c: names[i::len(cards)] for i, c in enumerate(cards)}


def _k_expr(names):
    return " or ".join(f"({n}::)" for n in names) if names else "nothing"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="0,1,2,3")
    ap.add_argument("--fit-gb", type=float, default=14.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-phase", type=int, choices=(1, 2), default=0, help="run one phase only (the other rig half may be busy)")
    ap.add_argument("rest", nargs="*", help="extra pytest arguments after --")
    a = ap.parse_args(argv)
    cards = [c.strip() for c in a.cards.split(",") if c.strip()]
    models = sorted(p.name for p in CACHE.iterdir() if (p / "manifest.json").exists()) if CACHE.exists() else []
    single, whole = partition(models, int(a.fit_gb * 1024 ** 3))
    plan = round_robin(single, cards)
    print(f"phase 1 — {len(single)} single-card cells pinned over cards {cards}; phase 2 — {len(whole)} whole-rig cells alone")
    for c in cards:
        print(f"  card {c}: {', '.join(plan[c]) or '-'}")
    print(f"  whole rig: {', '.join(whole) or '-'}")
    if a.dry_run:
        return 0
    base = [sys.executable, "-m", "pytest", "tests/regression/test_all_models.py", "-q", "-p", "no:cacheprovider", "-rf", "-rs"] + a.rest
    procs = []
    for c in (cards if a.only_phase != 2 else []):
        if not plan[c]:
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": c}
        log = open(REPO / f"two_phase_card{c}.log", "w")
        procs.append((c, subprocess.Popen(base + ["-k", _k_expr(plan[c])], env=env, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)))
    rcs = {c: p.wait() for c, p in procs}
    print("phase 1 rc per card:", rcs)
    rc2 = 0
    if whole and a.only_phase != 1:
        env = {**os.environ}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        rc2 = subprocess.call(base + ["-k", _k_expr(whole)], env=env, cwd=str(REPO))
        print("phase 2 rc:", rc2)
    return 1 if any(rcs.values()) or rc2 else 0


if __name__ == "__main__":
    sys.exit(main())
