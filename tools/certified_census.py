#!/usr/bin/env python
"""The certification CENSUS: every kernel x shape x dtype key the catalogue demands of ONE
hardware profile, read from graphs, never from weights, without a card.

The doctrine (the owner, 2026-09-21): certification has three stages. CENSUS — each machine
enumerates, for its own hardware profile, the keys the catalogue demands; the keys are derived
the way the launcher derives them, through a shape-only pass that carries the machine's plan,
strategy, ladder rung and tile unit. CERTIFICATION — the keys are swept and oracle-proven on
synthetic tensors (`neurobrix autotune certify --census <this file>`). VERIFICATION — one judged
run per served mode at zero miss; a miss is a census defect.

How the shape-only pass works: the engine runs as a SHADOW (`kernels/census.py`, installed by
the CLI under NBX_CENSUS=1): no device memory, no launch, no value, no weight file — the
allocator answers fake pointers, the launcher answers None, every autotuned kernel's `run`
answers None behind the one wrapper that forms and records its key (`_configs.run_with_notice`,
the same door the live record uses), value reads answer zero, and each component's parameters
are shadow tensors shaped from the graph's own param specs. The run is started with NO visible
device (`CUDA_VISIBLE_DEVICES=`), which is the door: whatever the stack does, no context can
exist on a card. The profile is the census's INPUT (`--hardware <profile>`); the plan, the
strategy, the rung and the tile unit are the engine's own for that profile, because the engine
solves them itself from the profile and the graphs.

A frozen dimension where a symbolic one is expected is detected BEFORE any key is harvested
(`tools/where_the_symbol_chain_breaks.py`: a declared input symbol whose chain breaks on a
literal written into a shape, or that no op ever carries): such a container is marked for
retrace and contributes no key — Qwen3-Omni's view carried its trace length 23, and a census
that harvested it would certify shapes the model can only meet at one request. The list this
produces is the hub's honest retrace queue.

The proof (the only honest one): the census of a model already run must contain every key that
model's replay cache holds, and nothing it cannot reach. First landed 2026-09-21: TinyLlama, six
keys, shadow == live served set == replay set. `--prove <replay dir or census file>` runs that
comparison here.

    python tools/certified_census.py --hardware default-c5d28c27 --out nbx/campaigns/.../census.json
    python tools/certified_census.py --hardware default-c5d28c27 --models TinyLlama-1.1B-Chat --prove ~/.neurobrix/replay_cache/...

The request each model is shadowed at is the family's judged request — the calibration section
of `config/families/<family>.yml` plus the campaign's media and bounds (`tools/
precision_zoo_campaign.request_args`, the one brick the retrace gate and the batteries use). A
request-dependent dimension (a prompt's token count, a step count) enters the key exactly as
the launcher forms it: the key is EXACT today, nothing buckets it, so the census covers the
judged requests it is given (`--extra` adds flags to every request; `--requests-json` adds
whole requests per model) and says so in `models[*].requests`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import precision_zoo_campaign as _zoo                       # noqa: E402  (request_args, CACHE)
import where_the_symbol_chain_breaks as _chain              # noqa: E402  (analyse)

CACHE = _zoo.CACHE
FORMAT = "nbx-census/1"
#: The engine modes that hand keys to the Triton launcher — the served modes a census covers.
#: The ATen modes (compiled, sequential) launch no NeuroBrix kernel and form no key.
MODES = {"triton": ["--triton"], "triton-sequential": ["--triton-sequential"]}


def _family(model: str) -> str:
    return json.loads((CACHE / model / "manifest.json").read_text()).get("family", "")


def _device_count(hardware: str) -> int:
    """The profile's device count — the shadow's `device_count()` answer. A profile that does
    not load, or declares no device, refuses here: a one-card census taken in silence for a
    four-card profile would be wrong at every placement."""
    from neurobrix.core.prism.loader import load_profile
    n = len(load_profile(hardware).devices)
    if n < 1:
        raise SystemExit(f"profile {hardware!r} declares no device; the census needs the machine's cards")
    return n


def frozen_dims(model: str) -> list:
    """Rows where a declared input symbol is LOST: its chain breaks on a literal equal to its
    trace value that is not a weight extent (the tool's own 'clean' class — a literal that is
    also a weight extent is read one by one, never counted), or no op ever carries it."""
    rows = []
    for gp in sorted((CACHE / model / "components").glob("*/graph.json")):
        for r in _chain.analyse(gp):
            if r.get("error"):
                # A graph the analyser cannot read is not clean — it is unread. Said as a row
                # of its own; the model is then marked, never harvested as if inspected.
                rows.append({"component": gp.parent.name, "unreadable": r["error"]})
                continue
            b = r.get("first_break")
            lost = r.get("never_carried") or (
                b and b.get("relation") == "v" and not b.get("literal_is_a_parameter_extent"))
            if lost:
                rows.append({"component": gp.parent.name, "symbol": r.get("symbol"), "name": r.get("name"),
                             "trace_value": r.get("trace_value"), "source": r.get("source"),
                             "never_carried": bool(r.get("never_carried")),
                             "first_break": b and {k: b.get(k) for k in ("op_uid", "op_type", "literal", "relation")}})
    return rows


def shadow(model: str, request: list, mode: str, hardware: str, n_dev: int, timeout: int, log_dir: Path) -> dict:
    """One shadow run; returns its keys and its fate. A failure is reported, never folded."""
    rec = log_dir / f"{model}.{mode}.keys"
    log = log_dir / f"{model}.{mode}.log"
    if rec.exists():
        rec.unlink()
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": str(n_dev),
                "NBX_KEY_RECORD": str(rec), "PYTHONPATH": str(REPO / "src")})
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model, *request, *MODES[mode], "--hardware", hardware]
    t0 = time.time()
    with open(log, "w") as fh:
        rc = _zoo.run_group(cmd, env, fh, timeout, cwd=str(REPO))
    keys = rec.read_text().splitlines() if rec.exists() else []
    tail = ""
    if rc != 0:
        lines = [l for l in log.read_text(errors="replace").splitlines() if "Error" in l or "ERROR" in l]
        tail = (lines[-1] if lines else "")[:300]
    return {"mode": mode, "rc": rc, "wall_s": round(time.time() - t0, 1), "keys": keys, "error": tail,
            "command": " ".join(cmd[2:])}


def census_model(model: str, hardware: str, modes: list, extra: list, requests: list, timeout: int,
                 log_dir: Path) -> dict:
    fam = _family(model)
    row = {"family": fam, "status": "ok", "keys": 0, "modes": {}, "requests": [], "frozen": []}
    frozen = frozen_dims(model)
    if any("unreadable" in r for r in frozen):
        row.update(status="unreadable", frozen=frozen)
        return row
    if frozen:
        row.update(status="retrace", frozen=frozen)
        return row
    reqs = requests or [_zoo.request_args(model, fam, list(extra))]
    row["requests"] = [" ".join(r) for r in reqs]
    keys = set()
    for mode in modes:
        for req in reqs:
            res = shadow(model, req, mode, hardware, _device_count(hardware), timeout, log_dir)
            row["modes"].setdefault(mode, []).append({k: v for k, v in res.items() if k != "keys"} | {"keys": len(res["keys"])})
            keys.update(res["keys"])
            if res["rc"] != 0:
                row["status"] = "failed"
    row["keys"] = len(keys)
    row["_keys"] = sorted(keys)
    return row


def directory_idents(vendor_profile: str) -> set:
    """Every ident the certified directory SERVES for `<vendor>/<profile>` — read through the
    engine's own loader and lookup, never the raw files: the loader drops an entry whose proof
    names no card (register 56 — a legacy proof serves no memory class until re-proven), and a
    coverage counted from the raw files read 6 served for a model the directory holds nothing
    for (TinyLlama, 2026-09-21). `any_class=True`: served on SOME memory class of the profile."""
    from neurobrix.kernels import autotune_certified as C
    from neurobrix.triton.autotune_cache import _autotuners
    out = set()
    root = REPO / "src" / "neurobrix" / "config" / "autotune" / vendor_profile
    tuners = {qual: at for qual, at in _autotuners()}
    for p in root.glob("*.json"):
        doc = json.loads(p.read_text())
        qual = doc.get("kernel")
        at = tuners.get(qual)
        if at is None:
            continue
        for ktext in (doc.get("entries") or {}):
            key = C.parse_key(ktext)
            if key is None:
                continue
            try:
                if C.lookup(qual, at, key, any_class=True) is not None:
                    out.add(f"{qual}::{ktext}")
            except Exception:
                continue
    return out


def prove(census: dict, path: str) -> dict:
    """The census must contain every key the replay cache (or census file) holds, and nothing it
    cannot reach — the second half is the directory's job (`census_retire_unreachable.py`); here
    it is the keys the census has that the reference lacks, said in full."""
    src = Path(path)
    if src.is_dir():
        files = sorted(src.glob("*.json"))
        if len(files) != 1:
            raise SystemExit(f"--prove {src}: expected exactly one replay cache file, found {len(files)}")
        src = files[0]
    doc = json.loads(src.read_text())
    ref = set((doc.get("entries", doc) if isinstance(doc, dict) else {}).keys())
    have = set(census["entries"])
    return {"reference": str(src), "reference_keys": len(ref), "census_keys": len(have),
            "missing_from_census": sorted(ref - have), "beyond_reference": sorted(have - ref),
            "identical": ref == have}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hardware", required=True, help="the profile the census is taken for (config/hardware/<id>.yml)")
    ap.add_argument("--models", default=None, help="comma-separated; default: every container in the cache")
    ap.add_argument("--modes", default="triton", help="comma-separated served modes: triton, triton-sequential")
    ap.add_argument("--extra", nargs="*", default=[], help="flags appended to every request")
    ap.add_argument("--requests-json", default=None, help='{"<model>": [[flags...], ...]} — whole requests per model')
    ap.add_argument("--timeout", type=int, default=3600, help="per shadow run, seconds")
    ap.add_argument("--jobs", type=int, default=2, help="shadow runs in flight (CPU-bound, no card)")
    ap.add_argument("--directory", default=None, help="<vendor>/<profile> of the certified directory to measure coverage against")
    ap.add_argument("--prove", default=None, help="a replay cache dir/file or census file the census must contain")
    ap.add_argument("--out", required=True, help="census.json (certifier format: entries keyed by <kernel>::<key>)")
    ap.add_argument("--logs", default=None, help="where the shadow logs and key records go (default: beside --out)")
    a = ap.parse_args()

    models = a.models.split(",") if a.models else sorted(
        p.name for p in CACHE.iterdir() if (p / "manifest.json").exists())
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f"unknown mode {m!r}; served modes are {sorted(MODES)}", file=sys.stderr)
            return 2
    per_model_requests = json.loads(Path(a.requests_json).read_text()) if a.requests_json else {}
    out = Path(a.out)
    log_dir = Path(a.logs) if a.logs else out.parent / (out.stem + "_runs")
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = {}
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futs = {m: pool.submit(census_model, m, a.hardware, modes, a.extra, per_model_requests.get(m), a.timeout, log_dir)
                for m in models}
        for m, f in futs.items():
            rows[m] = f.result()
            r = rows[m]
            print(f"[census] {m:44s} {r['family']:10s} {r['status']:8s} {r['keys']:5d} key(s)"
                  + (f"  frozen: {len(r['frozen'])} symbol(s)" if r["frozen"] else ""), flush=True)

    entries = {}
    for m, r in rows.items():
        for ident in r.pop("_keys", []):
            qual, _, ktext = ident.partition("::")
            e = entries.setdefault(ident, {"kernel": qual, "key": ktext, "models": []})
            e["models"].append(m)
    from neurobrix import __version__ as engine_version
    census = {"format": FORMAT, "hardware": a.hardware, "engine_version": engine_version,
              "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              "modes": modes, "wall_s": round(time.time() - t0, 1),
              "models": rows, "retrace_queue": sorted(m for m, r in rows.items() if r["status"] == "retrace"),
              "failed": sorted(m for m, r in rows.items() if r["status"] in ("failed", "unreadable")),
              "entries": entries}
    if a.directory:
        served = directory_idents(a.directory)
        census["coverage"] = {"directory": a.directory, "served": sum(1 for k in entries if k in served),
                              "to_certify": sorted(k for k in entries if k not in served)}
    if a.prove:
        census["proof"] = prove(census, a.prove)
    out.write_text(json.dumps(census, indent=1))
    n_ok = sum(1 for r in rows.values() if r["status"] == "ok")
    print(f"\n[census] {len(entries)} key(s) from {n_ok} model(s); retrace queue {len(census['retrace_queue'])}; "
          f"failed {len(census['failed'])}; {census['wall_s']} s; written {out}")
    if a.directory:
        c = census["coverage"]
        print(f"[census] directory {a.directory}: {c['served']} served, {len(c['to_certify'])} to certify")
    if a.prove:
        p = census["proof"]
        print(f"[census] proof against {p['reference']}: {'IDENTICAL' if p['identical'] else 'DIFFERS'} — "
              f"{len(p['missing_from_census'])} missing from the census, {len(p['beyond_reference'])} beyond the reference")
        return 0 if p["identical"] else 1
    return 0 if not census["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
