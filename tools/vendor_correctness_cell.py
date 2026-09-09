#!/usr/bin/env python3
"""Vendor-correctness cell — run the permanent justness column of the protocol.

Reads `benchmarks/harness/vendor_cells.yml` and, for each cell, runs OUR engine
and the ORIGINAL model on its own vendor stack, then compares the two outputs by
the cell's declared metric against the cell's declared bound.

The point of this cell is what it can see that nothing else we own can: our
sequential oracle replays the same graph the runtime executes, so a defect OF
THE GRAPH is identical on both arms and every byte gate reports IDENTICAL. Only
a comparison against the vendor's own stack can catch it.

A cell whose vendor stack is absent reports NOT-RUN with the reason. NOT-RUN is
a finding, not a pass — the summary counts it separately and never folds it into
the green count.

Usage:
    python3 tools/vendor_correctness_cell.py --cells benchmarks/harness/vendor_cells.yml \
        --out validation_outputs/vendor_cells_<date> [--family llm] [--id llm-dense-tinyllama]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Tuple

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── comparison metrics ──────────────────────────────────────────────────────
# Each returns (value, passed). The bound comes from the cell, never from here.

def _words(t: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (t or "").lower())


def m_text_common_prefix_words(ours: str, theirs: str, bound: Any) -> Tuple[Any, bool]:
    a, b = _words(ours), _words(theirs)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n, n >= float(bound)


def m_psnr_db(ours: str, theirs: str, bound: Any) -> Tuple[Any, bool]:
    """Delegates to tools/image_fidelity.py — the metric brick, not a copy."""
    p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "image_fidelity.py"),
                        theirs, ours, "--json"], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr[-400:]}, False
    d = json.loads(p.stdout)
    return d, float(d.get("psnr", 0.0)) >= float(bound)


METRICS = {
    "text_common_prefix_words": m_text_common_prefix_words,
    "psnr_db": m_psnr_db,
}


# ── vendor runners ──────────────────────────────────────────────────────────

def vendor_ollama(cell: Dict[str, Any], defaults: Dict[str, Any],
                  host: str) -> Dict[str, Any]:
    req = cell.get("request", {})
    body = json.dumps({
        "model": cell["vendor"]["ref"],
        "messages": [{"role": "user", "content": req["prompt"]}],
        "stream": False,
        "options": {"temperature": req.get("temperature", 0),
                    "seed": defaults.get("seed", 42),
                    "num_predict": req.get("max_tokens", defaults.get("max_tokens", 48)),
                    # CPU: the cell must never contend with a timed campaign.
                    "num_gpu": 0},
    }).encode()
    r = urllib.request.Request(f"{host}/api/chat", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            d = json.load(resp)
        return {"output": d.get("message", {}).get("content", ""), "error": None}
    except Exception as e:
        return {"output": None, "error": f"ollama unreachable or model absent: {e!r}"}


def vendor_diffusers(cell: Dict[str, Any], defaults: Dict[str, Any],
                     out_dir: str) -> Dict[str, Any]:
    v, req = cell["vendor"], cell.get("request", {})
    venv = v.get("venv", "")
    py = os.path.join(venv, "bin", "python")
    if not os.path.isfile(py):
        return {"output": None, "error": f"vendor venv absent: {venv}"}
    if not os.path.isdir(v["ref"]):
        return {"output": None, "error": f"vendor snapshot absent: {v['ref']}"}
    png = os.path.join(out_dir, f"{cell['id']}_vendor.png")
    cmd = [py, os.path.join(REPO, "tools", "vendor_image_repro.py"),
           "--snapshot", v["ref"], "--prompt", req["prompt"],
           "--seed", str(defaults.get("seed", 42)),
           "--steps", str(req.get("steps", 20)),
           "--guidance", str(req.get("guidance", 4.5)),
           "--height", str(req.get("height", 1024)),
           "--width", str(req.get("width", 1024)),
           "--out", png]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0 or not os.path.isfile(png):
        return {"output": None, "error": f"diffusers run failed: {p.stderr[-500:]}"}
    return {"output": png, "error": None}


def vendor_unavailable(cell: Dict[str, Any], *_args, **_kwargs) -> Dict[str, Any]:
    return {"output": None,
            "error": f"no runner wired for vendor.kind={cell['vendor']['kind']!r} "
                     f"({cell['vendor'].get('ref')}) — declared, not yet runnable"}


# ── our side ────────────────────────────────────────────────────────────────

def run_ours(cell: Dict[str, Any], defaults: Dict[str, Any], mode: str,
             src: str, out_dir: str, timeout: int) -> Dict[str, Any]:
    req = cell.get("request", {})
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-c",
           "import sys; from neurobrix.cli import main; sys.exit(main())",
           "run", "--model", cell["model"], "--seed", str(defaults.get("seed", 42))]
    if "prompt" in req:
        cmd += ["--prompt", req["prompt"]]
    if req.get("temperature") is not None:
        cmd += ["--temperature", str(req["temperature"])]
    if req.get("max_tokens") or defaults.get("max_tokens"):
        cmd += ["--max-tokens", str(req.get("max_tokens", defaults["max_tokens"]))]
    is_image = cell["compare"]["metric"].startswith("psnr")
    png = os.path.join(out_dir, f"{cell['id']}_ours.png")
    if is_image:
        cmd += ["--output", png]
    if mode == "triton":
        cmd.append("--triton")
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"output": None, "error": f"our engine timed out after {timeout}s"}
    if p.returncode != 0:
        return {"output": None, "error": f"our engine failed rc={p.returncode}: "
                                         f"{p.stderr[-500:]}"}
    if is_image:
        return ({"output": png, "error": None} if os.path.isfile(png)
                else {"output": None, "error": "our engine wrote no image"})
    # One extractor for the CLI's framing, shared with the verdict table. The
    # copy that lived here knew "Generated text:" but not the form the CLI
    # actually prints ("Generated <n> tokens"), so a correct TinyLlama
    # generation came back as the whole banner and the cell reported DIVERGES
    # on a harness bug. A cell that cries wolf is worth less than no cell.
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from moe_verdict_table import engine_text
    return {"output": engine_text(p.stdout), "error": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(REPO, "benchmarks", "harness",
                                                    "vendor_cells.yml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="/home/mlops/nbx_converge_mac/src")
    ap.add_argument("--mode", default="triton", choices=["triton", "compiled"])
    ap.add_argument("--family", default=None)
    ap.add_argument("--id", dest="cell_id", default=None)
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.cells))
    defaults = spec.get("defaults", {})
    os.makedirs(args.out, exist_ok=True)

    results = []
    for cell in spec["cells"]:
        if args.family and cell["family"] != args.family:
            continue
        if args.cell_id and cell["id"] != args.cell_id:
            continue
        # A cell whose comparison the harness cannot compute is NOT-RUN, never a
        # pass — the ladder cell below is driven by moe_real_path_check.py.
        metric = cell["compare"]["metric"]
        row: Dict[str, Any] = {
            "id": cell["id"], "family": cell["family"], "model": cell["model"],
            "vendor": cell["vendor"], "metric": metric,
            "bound": cell["compare"]["bound"], "blind_to": cell.get("blind_to", "").strip(),
        }
        t0 = time.time()
        if metric not in METRICS:
            # A cell can declare its own driver when the generic one-shot path
            # cannot express it (the MoE ladder needs eight loads at eight
            # lengths). Naming the command is the honest report; silently
            # counting it green would be the dishonest one.
            driver = cell.get("driven_by")
            row.update(verdict="DELEGATED" if driver else "NOT-RUN",
                       reason=(f"driven by its own tool: {driver}" if driver else
                               f"metric {metric!r} has no implementation in this harness"))
            results.append(row)
            print(f"[cell] {cell['id']:32s} {row['verdict']:9s} ({row['reason']})",
                  flush=True)
            continue

        kind = cell["vendor"]["kind"]
        if kind == "ollama":
            ven = vendor_ollama(cell, defaults, args.ollama_host)
        elif kind == "diffusers":
            ven = vendor_diffusers(cell, defaults, args.out)
        else:
            ven = vendor_unavailable(cell)
        if ven["error"]:
            row.update(verdict="NOT-RUN", reason=ven["error"],
                       seconds=round(time.time() - t0, 1))
            results.append(row)
            print(f"[cell] {cell['id']:32s} NOT-RUN  ({ven['error'][:90]})", flush=True)
            continue

        ours = run_ours(cell, defaults, args.mode, args.src, args.out, args.timeout)
        if ours["error"]:
            row.update(verdict="NOT-RUN", reason=ours["error"],
                       seconds=round(time.time() - t0, 1))
            results.append(row)
            print(f"[cell] {cell['id']:32s} NOT-RUN  ({ours['error'][:90]})", flush=True)
            continue

        value, passed = METRICS[metric](ours["output"], ven["output"],
                                        cell["compare"]["bound"])
        row.update(verdict="AGREES" if passed else "DIVERGES", value=value,
                   ours=ours["output"] if not metric.startswith("psnr") else ours["output"],
                   theirs=ven["output"], seconds=round(time.time() - t0, 1))
        results.append(row)
        print(f"[cell] {cell['id']:32s} {row['verdict']:8s} {metric}={value} "
              f"bound={cell['compare']['bound']}", flush=True)

    path = os.path.join(args.out, "vendor_cells.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    agree = sum(1 for r in results if r["verdict"] == "AGREES")
    diverge = sum(1 for r in results if r["verdict"] == "DIVERGES")
    notrun = sum(1 for r in results if r["verdict"] == "NOT-RUN")
    deleg = sum(1 for r in results if r["verdict"] == "DELEGATED")
    print(f"\n{agree} agree · {diverge} diverge · {notrun} not-run · "
          f"{deleg} delegated (not-run is a finding, never a pass)")
    print(f"written: {path}")
    return 1 if diverge else 0


if __name__ == "__main__":
    raise SystemExit(main())
