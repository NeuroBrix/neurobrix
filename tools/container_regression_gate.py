#!/usr/bin/env python3
"""Refuse to publish a container that lost ground against the one already served.

WHY IT EXISTS
-------------
On 2026-09-12 two incomplete CogVideoX containers were built and their uploads
started onto a working hub slug. Neither was corrupt and neither was caught by a
gate:

  * 5.45 GB, no backbone — the snapshot's transformer held an index and no
    weights. Caught by eye: 6.3 GB against the 21.6 GB the hub listed.
  * 17.32 GB, half a text encoder — the snapshot held
    `model-00001-of-00002.safetensors` and not the second. Caught by a
    component-by-component comparison against the installed container, run by
    hand five minutes into the upload: text_encoder 8.87 -> 4.66 GB with
    transformer and vae byte-identical.

Both entry doors now exist (`forge build` refuses a snapshot whose index names
shards it does not hold). This is the other half, and it is a different question:
the entry door asks whether the INPUTS are complete, this asks whether the OUTPUT
is a regression against what is already being served.

**The comparison belongs in the tool, not in a person.** The installed container
is a declaration of what this model contains, it is on disk, and it was available
to every command that wrote a new one.

WHAT IT COMPARES, AND WHY NOT A CHECKSUM
-----------------------------------------
Not bytes: a legitimate rebuild changes bytes everywhere. Per-component TOTAL
SIZE, against the installed tree, with growth allowed and shrinkage refused
beyond a tolerance. A component that halves is a missing shard; a component that
vanishes is a demoted module; a component that grows is a bigger build, which is
someone's decision and not a regression.

Usage:
    python tools/container_regression_gate.py MODEL.nbx --model-name NAME
    python tools/container_regression_gate.py MODEL.nbx --model-name NAME --allow-shrink
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

#: A rebuild moves a component's size by rounding, re-shard boundaries and a
#: corrected graph. Two per cent absorbs those; a missing shard is a halving.
TOLERANCE = 0.02


class ContainerRegression(RuntimeError):
    """The new container carries less than the one already installed."""


def component_sizes_in_archive(nbx: Path) -> dict:
    out = collections.Counter()
    with zipfile.ZipFile(nbx) as z:
        for info in z.infolist():
            if info.filename.startswith("components/") and info.filename.count("/") >= 2:
                out[info.filename.split("/")[1]] += info.file_size
    return dict(out)


def manifest_components(nbx: Path) -> set:
    with zipfile.ZipFile(nbx) as z:
        return set((json.loads(z.read("manifest.json")).get("components") or {}).keys())


def component_sizes_installed(cache_root: Path) -> dict:
    comp = cache_root / "components"
    if not comp.is_dir():
        return {}
    return {d.name: sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            for d in comp.iterdir() if d.is_dir()}


def compare(nbx: Path, installed_root: Path, tolerance: float = TOLERANCE) -> dict:
    """{'baseline': bool, 'rows': [...], 'problems': [...]}"""
    new = component_sizes_in_archive(nbx)
    old = component_sizes_installed(installed_root)
    if not old:
        # NO BASELINE IS NOT A PASS. Said in clear, never as silence: a first
        # build of a model has nothing to regress against, and that is a fact
        # about the check, not a verdict about the container.
        return {"baseline": False, "rows": [], "problems": [],
                "note": f"no installed container at {installed_root} — this check "
                        f"has no baseline and adjudicates nothing"}
    rows, problems = [], []
    for name in sorted(set(new) | set(old)):
        o, n = old.get(name, 0), new.get(name, 0)
        ratio = (n / o) if o else float("inf")
        rows.append({"component": name, "installed": o, "new": n, "ratio": ratio})
        if o and n == 0:
            problems.append(f"{name}: present in the installed container "
                            f"({o / 2**30:.2f} GB) and ABSENT from the new one")
        elif o and ratio < 1 - tolerance:
            problems.append(f"{name}: {o / 2**30:.2f} GB -> {n / 2**30:.2f} GB "
                            f"({ratio:.2f}x). A component that shrinks is a missing "
                            f"shard until someone says otherwise")
    return {"baseline": True, "rows": rows, "problems": problems}


def refuse_regression(nbx, model_name: str, cache_root=None,
                      allow_shrink: bool = False) -> dict:
    nbx = Path(nbx)
    root = Path(cache_root or (Path.home() / ".neurobrix" / "cache")) / model_name
    report = compare(nbx, root)
    if not report["baseline"]:
        print(f"   [regression gate] {report['note']}")
        return report
    for r in report["rows"]:
        print(f"   [regression gate] {r['component']:16s} "
              f"{r['installed'] / 2**30:8.2f} GB -> {r['new'] / 2**30:8.2f} GB "
              f"({r['ratio']:.3f}x)")
    if report["problems"] and not allow_shrink:
        raise ContainerRegression(
            "CONTAINER REGRESSION: this build carries less than the container "
            f"already installed for {model_name}:\n"
            + "".join(f"    {p}\n" for p in report["problems"])
            + "  Publishing it would replace a working model with a smaller one. "
              "The deliberate opening is --allow-shrink.")
    if report["problems"]:
        print("   [regression gate] --allow-shrink: "
              + "; ".join(report["problems"]))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nbx")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--allow-shrink", action="store_true")
    args = ap.parse_args()
    try:
        report = refuse_regression(args.nbx, args.model_name, args.cache_root,
                                   args.allow_shrink)
    except ContainerRegression as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if report["baseline"] and not report["problems"]:
        print("   [regression gate] PASS — no component lost ground")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
