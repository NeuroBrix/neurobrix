#!/usr/bin/env python3
"""The regression matrix: every catalogue model x every served mode, TODAY, at a size other
than the trace size, each cell's artefact kept for an outside judgment.

The owner's standing directive (2026-09-26): a model that worked before and fails now is a
REGRESSION measured against its last dated proof, not a new bug, and the answer is global —
one table of the catalogue, causes grouped, one fix per cause at the brick. This tool produces
today's half of that table on the machine it runs on (CUDA here, Metal on the Mac); the last
dated proofs are compiled beside it (`last_proofs.json`), and the two are joined by `table`.

One cell = one `neurobrix run` of the model's judged request (`precision_zoo_campaign.
request_args`: the family's calibration section, its media and its bound — the one brick the
batteries and the retrace gate use) in one mode:

    native            the compiled (ATen) engine, the reference arm
    triton            the Triton engine, compiled sequence
    triton-sequential the Triton engine, op by op

at ONE SIZE OTHER THAN THE TRACE SIZE. The judged requests of the language, speech and upscaler
families already differ from their trace extents (a prompt is not 23 tokens, a clip is not the
trace clip, a 448-pixel image is not the upscaler's trace tile). An image or video request
names no size, and the engine then renders at the container's own size — the traced one
(`resolution.container_size`). So those two families get an explicit size: the container's own
height taken to three quarters on the lattice (64 pixels for an image, 32 for a video), the
width kept — a non-square request away from the trace, the one class that catches a swapped or
frozen spatial axis.

Each cell records: rc, wall time, the engine's own execute time, the artefact's path, size and
sha256, the mechanical judgment (`judge_artefact`: degeneracy and geometry for an image, empty
or single-token for a text), the first error line on failure, and the request it ran. The
CONTENT judgment (an eye for an image, an STT for a WAV, a reader for a text) is written into
the row afterwards by the judge, with the artefact's path, so the table carries links that
open — never a PASS pronounced from rc or from two arms agreeing (R29).

Caches: the Triton compilation cache is the machine's (a kernel compiled is the same kernel);
the replay cache (runtime sweeps of uncertified keys) is owned per CARD, so parallel cards
never write one JSON store at once.

    tools/regression_matrix.py run --models A,B --gpu 1 --out nbx/campaigns/<dated>/matrix
    tools/regression_matrix.py table --out nbx/campaigns/<dated>/matrix [--proofs last_proofs.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import precision_zoo_campaign as Z  # noqa: E402  the judged request, the output kind
from judge_artefact import image_degeneracy, text_degeneracy  # noqa: E402

MODES = {"native": [], "triton": ["--triton"], "triton-sequential": ["--triton-sequential"]}
CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
LATTICE = {"image": 64, "video": 32}


def off_trace_size(model: str, family: str):
    """(height, width) for an image or video request: the container's own size, height at three
    quarters on the family's lattice, width kept. None for the families whose judged request is
    already away from the trace, or when the container states no size (said in the row)."""
    if family not in LATTICE:
        return None
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.resolution.container_size import container_output_size
    pkg = NBXRuntimeLoader().load(str(CACHE / model))
    size = container_output_size(pkg.manifest, pkg.defaults,
                                 pkg.topology.get("components", {}) or {}, pkg.components)
    if size is None:
        # The container states no size: the engine then renders at the family's own default
        # (executor: "a family constant is the last resort"), which is what it was traced at.
        from neurobrix.core.config import get_family_config
        fam_defaults = get_family_config(family).get("defaults") or {}
        if "height" not in fam_defaults or "width" not in fam_defaults:
            return None
        size = (fam_defaults["height"], fam_defaults["width"])
    h, w = (int(v) for v in size)
    step = LATTICE[family]
    h2 = max(step, (h * 3 // 4) // step * step)
    return (h2, w) if h2 != h else (max(step, h - step), w)


def first_error(log: Path) -> str:
    text = log.read_text(errors="replace") if log.exists() else ""
    for pat in (r"(ZERO FALLBACK[^\n]*)", r"((?:Runtime|Value|Shape\w*|OutOfMemory|Key|Index)Error[^\n]*)",
                r"(Traceback[^\n]*)", r"(TIMEOUT[^\n]*)"):
        m = re.findall(pat, text)
        if m:
            return m[-1][:300]
    return text.strip().splitlines()[-1][:300] if text.strip() else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mechanical(path: Path, family: str, expect_hw=None) -> dict:
    """The mechanical half of R29 — the content half is the judge's."""
    if not path.exists():
        return {"missing": True}
    suffix = path.suffix.lower()
    if suffix == ".png":
        return image_degeneracy(path, expect_shape=expect_hw)
    if suffix == ".txt":
        return text_degeneracy(path)
    return {"path": str(path), "bytes": path.stat().st_size}


def run_cell(model: str, mode: str, gpu: str, out: Path, timeout: int, src: Path) -> dict:
    family = Z.family_of(model)
    req = Z.request_args(model, family, [])
    size = off_trace_size(model, family)
    if size is not None:
        req = req + ["--height", str(size[0]), "--width", str(size[1])]
    ext = Z.output_ext(family, req)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    art = d / f"{mode}{ext}"
    log = d / f"{mode}.log"
    if art.exists():
        art.unlink()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(src),
           "NEUROBRIX_REPLAY_CACHE": str(out / f"replay_card{gpu}"), "PYTHONNOUSERSITE": "1"}
    # The cell runs under the interpreter the matrix was launched with (the pinned engine
    # python), written in the row: a matrix measures ONE stack, and the stack is part of the cell.
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model, *req, *MODES[mode],
           "--output", str(art)]
    rc, wall = Z.run(cmd, env, log, timeout)
    tree = src.parent
    row = {"model": model, "family": family, "mode": mode, "gpu": gpu, "rc": rc,
           "wall_s": round(wall, 1), "exec_s": Z.exec_time(log), "request": req,
           "off_trace_size": list(size) if size else None, "log": str(log),
           "python": sys.executable,
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "engine": subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "engine_tree": str(tree)}
    if art.exists() and rc == 0:
        row.update(artefact=str(art), sha256=sha256(art), bytes=art.stat().st_size,
                   mechanical=mechanical(art, family, size))
    else:
        row["error"] = first_error(log) if rc != 0 else "rc 0 and no artefact"
    return row


def cmd_run(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows_card{a.gpu}.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["mode"]))
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        for mode in a.modes.split(","):
            if (model, mode) in done:
                continue
            row = run_cell(model, mode, a.gpu, out, a.timeout, Path(a.src))
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[matrix] {model} {mode} rc={row['rc']} {row.get('wall_s')}s "
                  f"{row.get('error', '')[:120]}", flush=True)
    return 0


def cmd_table(a) -> int:
    out = Path(a.out)
    rows = [json.loads(l) for f in sorted(out.glob("rows_card*.jsonl")) for l in f.read_text().splitlines()]
    proofs = json.loads(Path(a.proofs).read_text()) if a.proofs and Path(a.proofs).exists() else {}
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["mode"]] = r
    lines = ["| model | family | last proof | native | triton | triton-sequential |", "|---|---|---|---|---|---|"]
    for model in sorted(by):
        lp = (proofs.get(model) or {}).get("last_proof") or {}
        cells = []
        for mode in MODES:
            r = by[model].get(mode)
            if r is None:
                cells.append("not run")
            elif r["rc"] != 0:
                cells.append(f"rc {r['rc']}: {r.get('error', '')[:80]}")
            else:
                mech = r.get("mechanical") or {}
                cells.append(("DEGENERATE " + "; ".join(mech.get("reasons", []))[:80]) if mech.get("degenerate")
                             else f"ran {r['wall_s']} s, judge: {r.get('judged', 'pending')}")
        lines.append(f"| {model} | {by[model][next(iter(by[model]))]['family']} | "
                     f"{lp.get('date') or '—'} {lp.get('verdict') or ''} | " + " | ".join(cells) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def catalogue_repo_ids(path: Path) -> dict:
    """cache directory -> repository id, from the catalogue's table (the repository id is the
    name; the directory is the 'cache directory if it differs' column, else the manifest's
    model_name, else the repository's basename — the first that exists in the cache)."""
    out = {}
    for line in path.read_text().splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 9 or not cols[0].isdigit():
            continue
        repo = cols[1].strip("`")
        for cand in (cols[8].strip("`"), cols[2], repo.split("/")[-1]):
            if cand and (CACHE / cand).is_dir():
                out[cand] = repo
                break
    return out


def cmd_export(a) -> int:
    """The joint row schema agreed with the Mac on the peer channel (2026-09-26 17:42 CEST): one
    JSON line per (model, stack, mode). 'native' is written 'compiled'; the container's sha256 is
    its manifest.json's; the verdict is the judge's, 'pending' until an outside judgment is written."""
    out = Path(a.out)
    repos = catalogue_repo_ids(Path(a.catalogue))
    proofs = json.loads(Path(a.proofs).read_text()) if a.proofs and Path(a.proofs).exists() else {}
    rows = [json.loads(l) for f in sorted(out.glob("rows_card*.jsonl")) for l in f.read_text().splitlines()]
    lines = []
    for r in rows:
        manifest = CACHE / r["model"] / "manifest.json"
        judged = r.get("judged")
        # A failed cell is 'pending' until judged: a harness cause (a timeout, an input not fed,
        # a card too small) is 'not-runnable-here(<reason>)', never 'broken' by its rc alone.
        verdict = r.get("verdict") or "pending"
        lp = (proofs.get(r["model"]) or {}).get("last_proof")
        if lp is not None and "stack" not in lp:
            # A proof whose python is unread is no regression baseline: the batteries ran on the
            # old venv until 2026-09-26 (the zoo brick), and an ATen DAG is not stable across torch.
            lp = {**lp, "stack": "unknown"}
        lines.append({
            "repo_id": repos.get(r["model"]),
            "container": r["model"],
            "container_sha256": sha256(manifest) if manifest.exists() else None,
            "stack": a.stack,
            "python": r.get("python"),
            "mode": "compiled" if r["mode"] == "native" else r["mode"],
            "request": " ".join(r["request"]),
            "today": {"date": r["date"], "engine": r["engine"], "rc": r["rc"],
                      "artifact": r.get("artefact"), "judged": judged, "verdict": verdict,
                      "error": r.get("error")},
            "last_proof": lp,
            "regression": r.get("regression"),
            "bisect": r.get("bisect"),
            "cause_class": r.get("cause_class"),
        })
    dest = Path(a.dest)
    dest.write_text("".join(json.dumps(x) + "\n" for x in lines))
    unnamed = sorted({x["container"] for x in lines if x["repo_id"] is None})
    print(f"{len(lines)} rows -> {dest}; containers without a catalogue line: {unnamed or 'none'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True)
    r.add_argument("--gpu", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--modes", default=",".join(MODES))
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--src", default=str(REPO / "src"), help="the engine tree's src the runs import (a frozen worktree)")
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("--proofs", default=None)
    e = sub.add_parser("export", help="the joint row schema shared with the Mac's Metal half")
    e.add_argument("--out", required=True)
    e.add_argument("--catalogue", required=True, help="the Mac's CATALOGUE.md")
    e.add_argument("--proofs", default=None)
    e.add_argument("--dest", required=True)
    e.add_argument("--stack", required=True, choices=("cuda", "metal"),
                   help="the machine's stack, written in every row (the joint table has two halves)")
    a = ap.parse_args()
    return {"run": cmd_run, "table": cmd_table, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
