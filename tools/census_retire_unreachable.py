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


def table_keys(table: Path) -> set:
    """{(kernel_short, key text)} every row of a profile's census table names."""
    out = set()
    for line in Path(table).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["kernel"].split(".")[-1], r["key"]))
    return out


def retire_outside(kernel_short: str, entries: dict, keys: set) -> tuple[dict, dict]:
    """(kept, retired) — retired holds the certified entries whose (kernel, key) the
    census table does not name: a certificate for a shape no catalogue container forms
    (the supervisor, 2026-10-07 17:37: the census is the single reference)."""
    kept, retired = {}, {}
    for ktext, entry in entries.items():
        (kept if (kernel_short, ktext) in keys else retired)[ktext] = entry
    return kept, retired


def main_outside_the_table(args) -> int:
    """`--certified-dir DIR --table T`: retire from every `<kernel>.<dtype>.json` of DIR
    the entries T does not name. Writes each file in place (format re-claimed from what
    its entries satisfy), one reversible JSON per run under `--record-dir`, and a dated
    paragraph listing every retired key in `--record-doc`."""
    keys = table_keys(args.table)
    if not keys:
        print(f"REFUSED: the table {args.table} names no key — nothing retired, nothing written.",
              file=sys.stderr)
        return 1
    held = set(args.hold_kernels.split(",")) if args.hold_kernels else set()
    files = [f for f in sorted(Path(args.certified_dir).glob("*.json")) if f.name.split(".")[0] not in held]
    plan, before, after = {}, 0, 0
    for f in files:
        doc = json.loads(f.read_text())
        kept, retired = retire_outside(f.name.split(".")[0], doc.get("entries") or {}, keys)
        before += len(kept) + len(retired); after += len(kept)
        if retired:
            plan[f] = (doc, kept, retired)
    print(f"certified {before} entries in {len(files)} files; table {len(keys)} keys; "
          f"{before - after} outside the table in {len(plan)} files")
    if not plan:
        print("nothing outside the table — nothing written.")
        return 0
    if args.dry_run:
        print("--dry-run: nothing written.")
        return 0
    sys.path.insert(0, str(REPO / "src"))
    from neurobrix.kernels import autotune_certified as C
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    rec_dir = args.record_dir or Path(args.certified_dir)
    rec_dir.mkdir(parents=True, exist_ok=True)
    record = rec_dir / f"certified_retired_{stamp}.json"
    record.write_text(json.dumps({"retired_at": stamp, "engine": _engine_sha(), "table": str(args.table),
                                  "certified_dir": str(args.certified_dir),
                                  "entries": {f.name: r for f, (_, _, r) in plan.items()}}, indent=1))
    for f, (doc, kept, _) in plan.items():
        doc = {**doc, "entries": kept}
        doc["format"] = C.format_for(kept)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, f)
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    para = (f"\n## {when} — {before - after} certified entries retired, engine `{_engine_sha()}`\n\n"
            f"Not named by the census table `{args.table}` ({len(keys)} keys): certificates for shapes no "
            f"catalogue container forms. Directory `{args.certified_dir}`: {before} → {after} entries"
            + (f" (held: {', '.join(f'`{k}`' for k in sorted(held))}, rows known wrong, not judged)" if held else "")
            + ". "
            f"Reversible record: `{record}`.\n\n| file | retired | kept |\n|---|---:|---:|\n"
            + "".join(f"| `{f.name}` | {len(r)} | {len(k)} |\n" for f, (_, k, r) in plan.items())
            + "\n<details><summary>retired keys</summary>\n\n"
            + "".join(f"- `{f.name}` `{k}`\n" for f, (_, _, r) in plan.items() for k in r)
            + "\n</details>\n")
    args.record_doc.parent.mkdir(parents=True, exist_ok=True)
    with args.record_doc.open("a") as fh:
        fh.write(para)
    print(f"retired {before - after}; directory now {after}; record {record}; doc {args.record_doc}")
    return 0


def _engine_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log", action="append")
    ap.add_argument("--certified-dir", type=Path, default=None,
                    help="with --table: retire the certified entries the census table does not name")
    ap.add_argument("--table", type=Path, default=None, help="the profile's census table (jsonl)")
    ap.add_argument("--census", default=None, help="default: the machine's replay cache")
    ap.add_argument("--record-dir", type=Path, default=None,
                    help="where the reversible JSON of retired entries goes (default: beside the census)")
    ap.add_argument("--record-doc", type=Path, default=RECORD_DOC)
    ap.add_argument("--hold-kernels", default=None,
                    help="with --table: comma-separated kernel short names left whole (their table rows are known wrong)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retire-failed", action="store_true",
                    help="also retire keys the certifier named FAILED (inputs that cannot be synthesised)")
    args = ap.parse_args()
    if args.certified_dir or args.table:
        if not (args.certified_dir and args.table) or args.from_log:
            ap.error("--certified-dir and --table go together, without --from-log")
        return main_outside_the_table(args)
    if not args.from_log:
        ap.error("--from-log is required (or --certified-dir with --table)")
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
