#!/usr/bin/env python3
"""Retire from the machine's replay-cache census the keys the certifier declared
UNREACHABLE, and write a dated record of what left and why.

WHY A CENSUS CARRIES KEYS THE ENGINE CANNOT PRODUCE
---------------------------------------------------
The replay cache accumulates every shape key the zoo's runs resolved on this
machine, across engine versions. When a wrapper changes how it computes a key
(2026-09-12: `IEEE_PRECISION` and `PROMOTE_B` now always True on the GEMM class,
and the output dtype widened), the old keys stay in the census, and every
`certify --only-missing` re-tries them, fails to reproduce them, and counts them
as work: 183 of 218 "missing" keys on 2026-09-12, 2.9 % of the census three days
earlier. A census that over-declares its work lies about coverage.

WHO DECIDES A KEY IS UNREACHABLE
--------------------------------
Not this tool. The certifier does, by synthesising inputs from the census key and
reading the key the wrapper computes for them; when the two differ it prints
`UNREACHABLE — the wrapper computed key K' for inputs synthesized from K`. This
tool reads those lines from the certifier's logs (`--from-log`, repeatable) and
retires exactly the (kernel, K) pairs named there. It never infers one.

WHAT IT WRITES
--------------
1. the census file, without the retired keys (a `.bak.<stamp>` copy beside it);
2. a JSON of the retired entries with their configs, under `--record-dir`, so
   the retirement is reversible;
3. an appended, dated paragraph in `docs/reference/census-retirements.md`: the
   count per kernel, the engine that could not produce them, the logs read.

It refuses to touch the census when the logs name nothing, and it refuses to
write a record that claims a retirement it did not perform.

Usage:
    python tools/census_retire_unreachable.py --from-log LOG [--from-log LOG …]
        [--census PATH] [--record-dir DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD_DOC = REPO / "docs/reference/census-retirements.md"
LINE = re.compile(r"\[certify\] (\S+) \S+ .*UNREACHABLE — the wrapper computed key (\(.*?\)) "
                  r"for inputs synthesized from (\(.*?\)):")
#: `--retire-failed`: a key whose inputs cannot even be SYNTHESISED (2026-09-13:
#: two conv2d keys with in_feat_dim=0, "cannot reshape array") is not a shape the
#: engine meets; the certifier names it FAILED with the describe_key text, and
#: the census key is recovered by matching that description against the census.
FAILED_LINE = re.compile(r"\[certify\] (\S+) \S+ (.*?): FAILED — (.*)$")


def unreachable_from_logs(paths) -> dict:
    """{(kernel_short, census_key_text): computed_key_text} named by the certifier."""
    out = {}
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if m:
                out[(m.group(1), m.group(3))] = m.group(2)
    return out


def failed_from_logs(paths, census: dict) -> dict:
    """{(kernel_short, census_key_text): failure} for FAILED keys, recovered by
    describing every census key of that kernel and matching the certifier's text."""
    sys.path.insert(0, str(REPO / "src"))
    from neurobrix.kernels import autotune_certified as C
    from neurobrix.triton import autotune_cache as atc
    tuners = {q.split(".")[-1]: t for q, t in atc._autotuners()}
    out = {}
    wanted = []
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            m = FAILED_LINE.search(line)
            if m:
                wanted.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    if not wanted:
        return out
    for ident in census:
        if "::" not in ident:
            continue
        qual, ktext = ident.split("::", 1)
        short = qual.split(".")[-1]
        tuner = tuners.get(short)
        key = C.parse_key(ktext)
        if tuner is None or key is None:
            continue
        desc = C.describe_key(tuner, key)
        for k, d, why in wanted:
            if k == short and d == desc:
                out[(short, ktext)] = why
    return out


def retire(census: dict, named: dict) -> tuple[dict, dict]:
    """(kept, retired) — retired holds the census entries whose (short kernel,
    key text) the certifier named; a named pair absent from the census is not
    an error, it is simply not there to retire."""
    kept, retired = {}, {}
    for ident, cfg in census.items():
        if "::" not in ident:
            kept[ident] = cfg
            continue
        qual, ktext = ident.split("::", 1)
        if (qual.split(".")[-1], ktext) in named:
            retired[ident] = cfg
        else:
            kept[ident] = cfg
    return kept, retired


def _engine_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log", action="append", required=True)
    ap.add_argument("--census", default=None, help="default: the machine's replay cache")
    ap.add_argument("--record-dir", type=Path, default=None,
                    help="where the reversible JSON of retired entries goes (default: beside the census)")
    ap.add_argument("--record-doc", type=Path, default=RECORD_DOC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retire-failed", action="store_true",
                    help="also retire keys the certifier named FAILED (inputs that cannot be synthesised)")
    args = ap.parse_args()

    if args.census is None:
        sys.path.insert(0, str(REPO / "src"))
        from neurobrix.triton import autotune_cache as atc
        args.census = atc._artifact_path()
    census_path = Path(args.census)
    if not census_path.exists():
        print(f"REFUSED: no census at {census_path}", file=sys.stderr)
        return 1
    named = unreachable_from_logs(args.from_log)
    doc = json.loads(census_path.read_text())
    wrapped = isinstance(doc, dict) and "entries" in doc and isinstance(doc["entries"], dict)
    census = doc["entries"] if wrapped else doc
    if args.retire_failed:
        named.update(failed_from_logs(args.from_log, census))
    if not named:
        print("REFUSED: the logs name no UNREACHABLE key — nothing to retire, nothing written.",
              file=sys.stderr)
        return 1
    kept, retired = retire(census, named)
    per_kernel: dict = {}
    for ident in retired:
        per_kernel[ident.split("::")[0].split(".")[-1]] = per_kernel.get(ident.split("::")[0].split(".")[-1], 0) + 1
    print(f"census {len(census)} keys; the logs name {len(named)}; {len(retired)} present and retired "
          f"{per_kernel}; {len(named) - len(retired)} named but not in the census")
    if not retired:
        print("REFUSED: none of the named keys is in the census — nothing written.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("--dry-run: nothing written.")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup = census_path.with_name(census_path.name + f".bak.{stamp}")
    shutil.copy2(census_path, backup)
    record_dir = args.record_dir or census_path.parent
    record_dir.mkdir(parents=True, exist_ok=True)
    record = record_dir / f"census_retired_{stamp}.json"
    record.write_text(json.dumps({"retired_at": stamp, "engine": _engine_sha(), "census": str(census_path),
                                  "logs": [str(p) for p in args.from_log],
                                  "computed_instead": {k: v for (_, k), v in named.items()},
                                  "entries": retired}, indent=1))
    new_doc = {**doc, "entries": kept} if wrapped else kept
    tmp = census_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(new_doc, indent=1, default=str))
    os.replace(tmp, census_path)

    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    para = (f"\n## {when} — {len(retired)} keys retired, engine `{_engine_sha()}`\n\n"
            f"Named UNREACHABLE by `neurobrix autotune certify` (the wrapper computed a "
            f"different key for inputs synthesised from the census key)"
            + (" or FAILED (inputs that cannot be synthesised)" if args.retire_failed else "") + " in: "
            + ", ".join(f"`{Path(p).name}`" for p in args.from_log) + ".\n\n"
            "| kernel | retired |\n|---|---:|\n"
            + "".join(f"| `{k}` | {n} |\n" for k, n in sorted(per_kernel.items()))
            + f"\nCensus {len(census)} → {len(kept)} keys. Reversible record: `{record}`; "
              f"backup: `{backup.name}`.\n")
    args.record_doc.parent.mkdir(parents=True, exist_ok=True)
    if not args.record_doc.exists():
        args.record_doc.write_text(
            "# Census retirements\n\nKeys removed from a machine's replay-cache census because the "
            "certifier declared them UNREACHABLE — recorded by an older engine whose wrappers "
            "computed the key differently. One dated paragraph per retirement; append, never "
            "edit. Tool: `tools/census_retire_unreachable.py`.\n")
    with args.record_doc.open("a") as f:
        f.write(para)
    print(f"retired {len(retired)}; census now {len(kept)}; record {record}; doc {args.record_doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
