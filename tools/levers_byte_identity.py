#!/usr/bin/env python3
"""Byte identity across a lever, at every length the locked protocol uses.

A lever that is meant to be inert has to be shown inert, not asserted: the
same request, on two source trees that differ only by the lever, must produce
the same bytes. Where it does NOT, that row was wrong before — and the row
says which length and which arm.

    python tools/levers_byte_identity.py run --out DIR \
        --before /path/to/tree-without-the-levers \
        --after  /path/to/tree-with-them
    python tools/levers_byte_identity.py table --out DIR

The lengths come from the locked protocol itself (`benchmarks/config/rows.yml`
and `benchmarks/harness/prompts/*.txt`), never from a list written here, so a
row added there is measured here without touching this file. Rows whose model
is not cached on this machine, and rows that need media assets, are named as
skipped with their reason rather than dropped.

Both trees are driven as cold CLI processes with their own `src` on
PYTHONPATH; the three caches go before every run, so a warm artifact from one
tree can never be read by the other.

It writes `identity.json` and `IDENTITY.md`, and refuses to conclude without
them.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.path.expanduser("~")) / ".neurobrix" / "cache"
ARMS = ("compiled", "sequential", "triton", "triton-sequential")

#: Arms whose cost grows with the square of the context (no KV cache: the
#: whole prefix is re-run at every decode step). Named here so a long-context
#: row can leave them out ON PURPOSE and say so, rather than time out.
QUADRATIC_ARMS = ("triton-sequential",)


def clear_caches() -> list:
    """The three caches, before every run — the locked measurement protocol."""
    home = Path(os.path.expanduser("~"))
    cleared = []
    for path in (home / ".cache" / "triton_msl", home / ".triton" / "cache"):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            cleared.append(str(path))
    replay = home / ".neurobrix" / "replay_cache"
    if replay.is_dir():
        for f in replay.glob("autotune_configs_*.json"):
            f.unlink(missing_ok=True)
            cleared.append(str(f))
    return cleared


def load_rows(tree: Path) -> tuple:
    """The locked rows, and the reason each unusable one is unusable."""
    rows_path = tree / "benchmarks" / "config" / "rows.yml"
    if not rows_path.exists():
        raise RuntimeError(f"the locked protocol is not here: {rows_path}")
    try:
        import yaml
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            f"reading the locked rows needs PyYAML ({exc}); it is not "
            f"optional here — the lengths must come from the protocol, not "
            f"from a list in this file.")
    doc = yaml.safe_load(rows_path.read_text())
    rows = doc["rows"] if isinstance(doc, dict) and "rows" in doc else doc
    usable, skipped = [], []
    for row in rows or []:
        model = row.get("neurobrix_model")
        if not model:
            skipped.append({"row": row.get("id"), "why": "no neurobrix_model"})
            continue
        if not (CACHE / model).is_dir():
            skipped.append({"row": row.get("id"), "model": model,
                            "why": "not cached on this machine"})
            continue
        if not row.get("max_new_tokens") or not row.get("prompt"):
            skipped.append({"row": row.get("id"), "model": model,
                            "why": "not a text-generating row (no prompt / no max_new_tokens)"})
            continue
        media = [k for k in ("image", "input_image", "audio", "video") if row.get(k)]
        if media:
            skipped.append({"row": row.get("id"), "model": model,
                            "why": f"needs media inputs: {', '.join(media)}"})
            continue
        usable.append(row)
    return usable, skipped


def context_cases(tree: Path) -> list:
    """The protocol's longer contexts, as (label, text)."""
    out = []
    for name in ("mid_ctx", "long_ctx", "xlong_ctx"):
        path = tree / "benchmarks" / "harness" / "prompts" / f"{name}.txt"
        if path.exists():
            out.append((name, path.read_text()))
    return out


def run_one(tree: Path, model: str, prompt: str, max_tokens: int, arm: str,
            outdir: Path, tag: str, timeout: int) -> dict:
    out_path = outdir / f"out_{tag}.txt"
    out_path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "src")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", model, "--prompt", prompt,
           "--max-tokens", str(max_tokens), "--temperature", "0",
           "--output", str(out_path), f"--{arm}"]
    clear_caches()
    started = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout, cwd=str(tree))
        rc, tail = proc.returncode, (proc.stderr or proc.stdout)[-500:]
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"timeout after {timeout}s"
    text = out_path.read_text() if out_path.exists() else ""
    return {"rc": rc, "wall_s": round(time.time() - started, 3),
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "out_file": out_path.name,
            "error_tail": "" if rc == 0 else tail}


def run(args) -> int:
    before, after = Path(args.before).resolve(), Path(args.after).resolve()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, skipped = load_rows(after)
    if args.models:
        keep = {m.strip() for m in args.models.split(",")}
        for row in list(rows):
            if row["neurobrix_model"] not in keep:
                rows.remove(row)
                skipped.append({"row": row.get("id"), "model": row["neurobrix_model"],
                                "why": "not in --models"})

    cases = []
    for row in rows:
        cases.append({"row": row["id"], "model": row["neurobrix_model"],
                      "length": "row", "prompt": row["prompt"],
                      "max_tokens": int(row["max_new_tokens"])})
        if (row.get("family") or "") == "llm":
            for label, text in context_cases(after):
                cases.append({"row": row["id"], "model": row["neurobrix_model"],
                              "length": label, "prompt": text,
                              "max_tokens": args.long_max_tokens})

    results = []
    for case in cases:
        arms = list(args.arm or ARMS)
        left_out, why = [], ""
        # An arm with no KV cache re-runs the whole prefix at every decode
        # step: N new tokens over a C-token context is O(N*C) forwards. At
        # the row's own budget that is hours per run, so it is left out ON
        # PURPOSE and named, never silently dropped. The head_dim cell
        # measures that arm at the length this campaign exists for.
        if (case["length"] in ("long_ctx", "xlong_ctx")
                or case["max_tokens"] > args.quadratic_max_tokens):
            left_out = [a for a in arms if a in QUADRATIC_ARMS]
            arms = [a for a in arms if a not in QUADRATIC_ARMS]
            if left_out:
                why = (f"no KV cache: {case['max_tokens']} new tokens over a "
                       f"{len(case['prompt'])}-character context is O(N*C) "
                       f"forwards, hours per run; measured instead by "
                       f"tools/head_dim_length_cell.py at head_dim")
        print(f"{case['model']} · {case['length']} · {case['max_tokens']} tokens", flush=True)
        per_arm = {}
        for arm in arms:
            entry = {}
            for label, tree in (("before", before), ("after", after)):
                tag = f"{case['model']}_{case['length']}_{arm}_{label}"
                entry[label] = run_one(tree, case["model"], case["prompt"],
                                       case["max_tokens"], arm, outdir, tag,
                                       args.timeout)
                print(f"    {arm:20s} {label:6s} rc={entry[label]['rc']} "
                      f"sha={entry[label]['sha256'][:8]} "
                      f"{entry[label]['wall_s']}s", flush=True)
            same = (entry["before"]["rc"] == 0 == entry["after"]["rc"]
                    and entry["before"]["sha256"] == entry["after"]["sha256"])
            entry["verdict"] = ("unchanged" if same else
                                "refused" if 0 not in (entry["before"]["rc"], entry["after"]["rc"])
                                else "CHANGED")
            per_arm[arm] = entry
        results.append({**case, "arms": per_arm,
                        "arms_left_out": left_out,
                        "left_out_why": why})

    document = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/levers_byte_identity.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "before_tree": str(before), "before_rev": _rev(before),
        "after_tree": str(after), "after_rev": _rev(after),
        "caches_cleared_before_every_run": clear_caches(),
        "results": results, "skipped": skipped,
    }
    doc = outdir / "identity.json"
    doc.write_text(json.dumps(document, indent=1))
    if not doc.exists():
        raise RuntimeError(f"refusing to conclude: {doc} was not written")
    print(f"\nwritten: {doc}")
    return table(args)


def _rev(tree: Path) -> str:
    out = subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    rev = out.stdout.strip() or "?"
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return rev + (" +local changes" if dirty else "")


def table(args) -> int:
    outdir = Path(args.out)
    doc_path = outdir / "identity.json"
    if not doc_path.exists():
        raise RuntimeError(f"refusing to render a table: {doc_path} does not exist")
    doc = json.loads(doc_path.read_text())
    lines = ["# Byte identity across the levers, at the locked protocol's lengths", "",
             f"Generated **{doc['generated']}** by `tools/levers_byte_identity.py` "
             f"on {doc['machine']}.", "",
             f"* before: `{doc['before_rev']}` — {doc['before_tree']}",
             f"* after:  `{doc['after_rev']}` — {doc['after_tree']}", "",
             "The three caches are cleared before **every** run, so no artifact "
             "built by one tree can be read by the other.", "",
             "| model | length | tokens | arm | before | after | verdict |",
             "|---|---|---:|---|---|---|---|"]
    changed = 0
    for res in doc["results"]:
        for arm, entry in res["arms"].items():
            if entry["verdict"] == "CHANGED":
                changed += 1
            lines.append(
                f"| {res['model']} | {res['length']} | {res['max_tokens']} | "
                f"`--{arm}` | `{entry['before']['sha256'][:8]}` | "
                f"`{entry['after']['sha256'][:8]}` | "
                + ("unchanged" if entry["verdict"] == "unchanged"
                   else f"**{entry['verdict']}**") + " |")
    lines += ["", f"**{changed}** of {sum(len(r['arms']) for r in doc['results'])} "
              f"measured (model, length, arm) cells changed.", ""]
    left = [(r["model"], r["length"], r["arms_left_out"], r["left_out_why"])
            for r in doc["results"] if r["arms_left_out"]]
    if left:
        lines += ["## Arms left out, and why", "",
                  "| model | length | arms | reason |", "|---|---|---|---|"]
        for model, length, arms, why in left:
            lines.append(f"| {model} | {length} | "
                         + ", ".join(f"`--{a}`" for a in arms) + f" | {why} |")
        lines.append("")
    if doc["skipped"]:
        lines += ["## Rows of the locked protocol not measured here, and why", "",
                  "| row | model | reason |", "|---|---|---|"]
        for s in doc["skipped"]:
            lines.append(f"| {s.get('row')} | {s.get('model', '—')} | {s['why']} |")
        lines.append("")
    md = outdir / "IDENTITY.md"
    md.write_text("\n".join(lines) + "\n")
    if not md.exists():
        raise RuntimeError(f"refusing to conclude: {md} was not written")
    print(f"written: {md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--before", required=True)
    r.add_argument("--after", required=True)
    r.add_argument("--models", default=None)
    r.add_argument("--arm", action="append", default=None)
    r.add_argument("--long-max-tokens", type=int, default=16,
                   help="decode budget for the long-context cases")
    r.add_argument("--quadratic-max-tokens", type=int, default=16,
                   help="above this budget the cacheless arms are left out, "
                        "and the table says so")
    r.add_argument("--timeout", type=int, default=3600)
    r.set_defaults(func=run)
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.set_defaults(func=table)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
