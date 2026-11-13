#!/usr/bin/env python3
"""The symbolic-shapes gate over a whole zoo of built containers: three card-less scanners, one table, one verdict.

Symbolic shapes are re-proved on every built container at each change of the container builder. This gate finds
every container under `--root`, hands each component graph to the three scanners this folder already holds, and
says per container and per scanner whether it is clean:

  * `chain`     — `where_the_symbol_chain_breaks.analyse(graph_path)`: a declared symbol whose chain breaks at an
                  operator that wrote a literal into its output shape, or that no operation carries at all;
  * `broadcast` — `symbolic_broadcast_scan.scan_graph(graph)`: an elementwise op whose inputs broadcast at the
                  build values and not away from them;
  * `collision` — `symbol_collision_census.census_one(graph_path)`: an axis built at a value where two shape rules
                  cannot be told apart (where a defect would be invisible; batch axes excluded as the census does).

The scanners are imported as functions; their logic is not touched here. A scanner is clean on a container when
none of the container's component graphs gives it a finding or an error.

What a container is, as found on disk:
  * a `.nbx` file (a zip holding `components/<name>/graph.json`): its graphs are extracted into a private work
    folder (`--work`, default the system's temporary folder), scanned, and the folder removed;
  * a folder holding `manifest.json` and `components/` (an unpacked cache entry): its graphs are read in place.
Nothing is ever written under `--root`; an output or work path inside it is refused by name.

    python tools/symbolic_zoo_gate.py --root DIR --out zoo.md [--jobs 8] [--baseline OLD.json] [--write-baseline NEW.json]

Writes `--out` (markdown) and the same state as JSON beside it (`zoo.json`). Exit status:
  0  no (container, scanner) pair that was clean in `--baseline` is not clean now;
  1  at least one is (named in the table and on stderr) — a container clean before and now unreadable counts;
  2  refused: no container under `--root`, or a path that would write under it.
A container new since the baseline is reported, not a failure; one the baseline knew and the root no longer holds
is reported as absent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

TOOLS = Path(__file__).resolve().parent
GRAPH = "graph.json"
SCANNER_NAMES = ("chain", "broadcast", "collision")


class Refused(RuntimeError):
    """The gate cannot give a verdict (nothing to scan, or it would write under the root)."""


# ---------------------------------------------------------------------------------------------------------------
# The scanners, as functions: graph path -> {"clean": bool, "counts": {name: int}}.

def _load_tool(module_name: str):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))          # the scanners import their sibling `atelier`
    mod = sys.modules.get(module_name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(module_name, TOOLS / f"{module_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def scan_chain(graph_path: Path) -> dict:
    rows = _load_tool("where_the_symbol_chain_breaks").analyse(graph_path)
    errors = [r for r in rows if r.get("error")]
    good = [r for r in rows if not r.get("error")]
    breaks = [r for r in good if r.get("first_break")]
    at_build = [r for r in breaks if r["first_break"].get("relation") == "v"
                and not r["first_break"].get("literal_is_a_parameter_extent")]
    never = [r for r in good if r.get("never_carried")]
    counts = {"symbols_followed": len(good), "breaks": len(breaks), "breaks_literal_is_build_value": len(at_build),
              "never_carried": len(never), "errors": len(errors)}
    return {"clean": not (breaks or never or errors), "counts": counts,
            "errors": [r["error"] for r in errors]}


def scan_broadcast(graph_path: Path) -> dict:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    found, unevaluable = _load_tool("symbolic_broadcast_scan").scan_graph(graph)
    return {"clean": not found, "counts": {"breaks": len(found), "unevaluable_ops": unevaluable}}


def scan_collision(graph_path: Path) -> dict:
    rows, accepted = _load_tool("symbol_collision_census").census_one(graph_path)
    errors = [r for r in rows if r.get("error")]
    axes = [r for r in rows if not r.get("error")]
    flags = [cls for r in axes for cls, _ in r.get("flags", [])]
    counts = {"blind_axes": len(axes), "arithmetic": flags.count("arithmetic"),
              "weight_extent": flags.count("weight-extent"), "batch_axes_excluded": len(accepted),
              "errors": len(errors)}
    return {"clean": not rows, "counts": counts, "errors": [r["error"] for r in errors]}


REAL_SCANNERS: Dict[str, Callable[[Path], dict]] = {
    "chain": scan_chain, "broadcast": scan_broadcast, "collision": scan_collision}


# ---------------------------------------------------------------------------------------------------------------
# Finding the containers.

def find_containers(root: Path) -> tuple:
    """([(key, kind, path)], [unreadable folder, relative]) under root; hidden folders are not entered."""
    found, unreadable = [], []

    def onerror(exc: OSError):
        unreadable.append(os.path.relpath(exc.filename, root) if exc.filename else str(exc))

    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if "manifest.json" in filenames and "components" in dirnames:
            found.append((here.relative_to(root).as_posix() or ".", "folder", here))
            dirnames[:] = []
            continue
        for f in sorted(filenames):
            if f.endswith(".nbx"):
                rel = (here / f).relative_to(root)
                key = rel.parent.as_posix() if f == "model.nbx" else rel.as_posix()
                found.append((key, "nbx", here / f))
    return sorted(found), sorted(unreadable)


def _graphs_of(kind: str, path: Path, work: Path) -> tuple:
    """(component -> graph path, reason not scanned or None). Extracts `.nbx` graphs into `work`."""
    if kind == "folder":
        graphs = {g.parent.name: g for g in sorted((path / "components").glob(f"*/{GRAPH}"))}
    else:
        graphs = {}
        try:
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    parts = info.filename.split("/")
                    if len(parts) != 3 or parts[0] != "components" or parts[2] != GRAPH:
                        continue
                    comp = parts[1]
                    if not comp or comp in (".", "..") or "\\" in comp:
                        return {}, f"unreadable: member {info.filename!r} names no component folder"
                    dest = work / "components" / comp / GRAPH
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 22)
                    graphs[comp] = dest
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
            return {}, f"unreadable: {type(exc).__name__}: {exc}"
    if not graphs:
        return {}, "not scanned: no components/<name>/graph.json in the container"
    return graphs, None


#: The scanners of the running gate; set before the workers fork, so they inherit them rather than receive them.
_SCANNERS: Dict[str, Callable[[Path], dict]] = {}


def scan_container(job: tuple) -> dict:
    """One container through every scanner. job = (key, kind, path, work_root)."""
    key, kind, path, work_root = job
    scanners = _SCANNERS
    t0 = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="zoo_", dir=str(work_root)))
    try:
        graphs, reason = _graphs_of(kind, Path(path), work)
        rec = {"container": key, "kind": kind, "scanned": reason is None, "reason": reason,
               "components": sorted(graphs), "scanners": {}}
        if reason is None:
            for name, fn in scanners.items():
                agg = {"clean": True, "counts": {}, "errors": [], "unclean_components": []}
                for comp, gp in graphs.items():
                    try:
                        res = fn(gp)
                    except Exception as exc:  # a scanner that cannot read a graph: named, never a clean verdict
                        res = {"clean": False, "counts": {"errors": 1},
                               "errors": [f"{type(exc).__name__}: {exc}"]}
                    for k, v in (res.get("counts") or {}).items():
                        agg["counts"][k] = agg["counts"].get(k, 0) + int(v)
                    agg["errors"] += [f"{comp}: {e}" for e in res.get("errors") or []]
                    if not res["clean"]:
                        agg["clean"] = False
                        agg["unclean_components"].append(comp)
                rec["scanners"][name] = agg
        rec["clean"] = rec["scanned"] and all(s["clean"] for s in rec["scanners"].values())
        rec["seconds"] = round(time.monotonic() - t0, 2)
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------
# The verdict against a baseline.

def compare(state: dict, baseline: Optional[dict]) -> dict:
    """Per container: regressions (scanners clean in the baseline and not now), new, absent."""
    out = {"regressions": {}, "new": [], "absent": []}
    if baseline is None:
        return out
    before = baseline.get("containers") or {}
    now = state["containers"]
    for key, rec in now.items():
        old = before.get(key)
        if old is None:
            out["new"].append(key)
            continue
        lost = []
        for name, s in (old.get("scanners") or {}).items():
            if not s.get("clean"):
                continue
            cur = rec["scanners"].get(name)
            if not rec["scanned"]:
                lost.append(f"{name} (container {rec['reason']})")
            elif cur is None:
                lost.append(f"{name} (not run this time)")
            elif not cur["clean"]:
                lost.append(name)
        if lost:
            out["regressions"][key] = lost
    out["absent"] = sorted(k for k in before if k not in now)
    out["new"].sort()
    return out


def summary(state: dict) -> dict:
    recs = list(state["containers"].values())
    scanned = [r for r in recs if r["scanned"]]
    per = {}
    for name in state["scanner_names"]:
        rs = [r["scanners"][name] for r in scanned if name in r["scanners"]]
        counts: Dict[str, int] = {}
        for s in rs:
            for k, v in s["counts"].items():
                counts[k] = counts.get(k, 0) + v
        per[name] = {"clean_containers": sum(1 for s in rs if s["clean"]),
                     "unclean_containers": sum(1 for s in rs if not s["clean"]), "counts": counts}
    return {"containers_found": len(recs), "scanned": len(scanned),
            "not_scanned": len(recs) - len(scanned),
            "clean": sum(1 for r in recs if r["clean"]), "per_scanner": per,
            "unreadable_folders": len(state["unreadable_folders"])}


# ---------------------------------------------------------------------------------------------------------------
# The table.

def _cell(s: Optional[dict], name: str) -> str:
    if s is None:
        return "-"
    c = s["counts"]
    mark = "clean" if s["clean"] else "FOUND"
    if name == "chain":
        detail = (f"{c.get('breaks', 0)} breaks ({c.get('breaks_literal_is_build_value', 0)} at build value), "
                  f"{c.get('never_carried', 0)} never carried")
    elif name == "broadcast":
        detail = f"{c.get('breaks', 0)} breaks, {c.get('unevaluable_ops', 0)} unevaluable"
    elif name == "collision":
        detail = (f"{c.get('blind_axes', 0)} blind axes (arith {c.get('arithmetic', 0)}, "
                  f"weight {c.get('weight_extent', 0)})")
    else:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
    if c.get("errors"):
        detail += f", {c['errors']} errors"
    return f"{mark}: {detail}"


def markdown(state: dict) -> str:
    sm, cmp_ = state["summary"], state["comparison"]
    names = state["scanner_names"]
    lines = [f"# Symbolic zoo gate — {state['root']}", "",
             f"- containers found: {sm['containers_found']}; scanned: {sm['scanned']}; "
             f"not scanned: {sm['not_scanned']}; clean on every scanner: {sm['clean']}",
             f"- jobs: {state['jobs']}; wall time: {state['wall_seconds']} s"]
    for name in names:
        p = sm["per_scanner"][name]
        lines.append(f"- `{name}`: clean on {p['clean_containers']}, findings on {p['unclean_containers']}; "
                     + ", ".join(f"{k} {v}" for k, v in sorted(p["counts"].items())))
    if state["unreadable_folders"]:
        lines.append(f"- folders the gate could not enter ({len(state['unreadable_folders'])}): "
                     + ", ".join(f"`{u}`" for u in state["unreadable_folders"]))
    if state.get("baseline"):
        lines.append(f"- baseline `{state['baseline']}`: {len(cmp_['regressions'])} regression(s), "
                     f"{len(cmp_['new'])} new, {len(cmp_['absent'])} absent")
        for key, lost in sorted(cmp_["regressions"].items()):
            lines.append(f"  - REGRESSION `{key}`: no longer clean on {', '.join(lost)}")
        for key in cmp_["absent"]:
            lines.append(f"  - absent since the baseline: `{key}`")
    lines += ["", "| container | scanned | components | " + " | ".join(names) + " | clean | vs baseline |",
              "|---|---|---|" + "---|" * len(names) + "---|---|"]
    for key, r in sorted(state["containers"].items()):
        if key in cmp_["regressions"]:
            vs = "REGRESSION: " + ", ".join(cmp_["regressions"][key])
        elif key in cmp_["new"]:
            vs = "new"
        else:
            vs = "" if not state.get("baseline") else "held"
        scanned = "yes" if r["scanned"] else f"no — {r['reason']}"
        cells = [_cell(r["scanners"].get(n), n) for n in names]
        lines.append(f"| `{key}` | {scanned} | {len(r['components'])} | " + " | ".join(cells)
                     + f" | {'yes' if r['clean'] else 'no'} | {vs} |")
    errs = [(k, n, e) for k, r in sorted(state["containers"].items())
            for n, s in r["scanners"].items() for e in s.get("errors") or []]
    if errs:
        lines += ["", "## Scanner errors", ""] + [f"- `{k}` `{n}`: {e}" for k, n, e in errs]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------

def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_gate(root: Path, out: Path, jobs: int = 8, baseline: Optional[Path] = None,
             write_baseline: Optional[Path] = None, work: Optional[Path] = None,
             scanners: Optional[Dict[str, Callable[[Path], dict]]] = None) -> int:
    scanners = dict(scanners or REAL_SCANNERS)
    root = Path(root)
    work = Path(work or tempfile.gettempdir())
    json_out = out.with_suffix(".json")
    for label, p in (("--out", out), ("the JSON beside --out", json_out), ("--write-baseline", write_baseline),
                     ("--work", work)):
        if p is not None and _inside(Path(p), root):
            raise Refused(f"{label} {p} is under --root {root}: the gate never writes under the root it scans")
    if not root.is_dir():
        raise Refused(f"--root {root} is not a folder")
    base = None
    if baseline is not None:
        try:
            base = json.loads(Path(baseline).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Refused(f"--baseline {baseline}: cannot be read ({type(exc).__name__}: {exc})") from None
    containers, unreadable = find_containers(root)
    if not containers:
        raise Refused(f"--root {root} holds no container (no .nbx file, no folder with manifest.json and "
                      f"components/). A gate over zero containers is not a clean gate.")
    work.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="symbolic_zoo_gate_", dir=str(work)))
    # where_the_symbol_chain_breaks reads its default cache at import; the cache this gate scans is its work root.
    os.environ.setdefault("NEUROBRIX_CACHE_DIR", str(work_root))
    t0 = time.monotonic()
    try:
        _SCANNERS.clear()
        _SCANNERS.update(scanners)
        tasks = [(k, kind, str(p), str(work_root)) for k, kind, p in containers]
        if jobs <= 1:
            recs = [scan_container(t) for t in tasks]
        else:
            with multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
                recs = []
                for rec in pool.imap_unordered(scan_container, tasks):
                    recs.append(rec)
                    print(f"[{len(recs)}/{len(tasks)}] {rec['container']}: "
                          f"{'clean' if rec['clean'] else ('not scanned' if not rec['scanned'] else 'findings')}"
                          f" ({rec['seconds']} s)", file=sys.stderr, flush=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    state = {"root": str(root), "jobs": jobs, "scanner_names": list(scanners),
             "wall_seconds": round(time.monotonic() - t0, 1), "unreadable_folders": unreadable,
             "baseline": str(baseline) if baseline else None,
             "containers": {r["container"]: r for r in sorted(recs, key=lambda r: r["container"])}}
    state["summary"] = summary(state)
    state["comparison"] = compare(state, base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(state), encoding="utf-8")
    json_out.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    if write_baseline is not None:
        Path(write_baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(write_baseline).write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    sm = state["summary"]
    print(f"containers found {sm['containers_found']}, scanned {sm['scanned']}, clean {sm['clean']}, "
          f"wall {state['wall_seconds']} s; table {out}, JSON {json_out}")
    regs = state["comparison"]["regressions"]
    for key, lost in sorted(regs.items()):
        print(f"REGRESSION {key}: no longer clean on {', '.join(lost)}", file=sys.stderr)
    return 1 if regs else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="the folder holding the built containers (only read)")
    ap.add_argument("--out", required=True, help="markdown table; the JSON is written beside it")
    ap.add_argument("--jobs", type=int, default=8, help="containers scanned in parallel")
    ap.add_argument("--baseline", default=None, help="the JSON of a previous run; exit 1 on any clean pair lost")
    ap.add_argument("--write-baseline", default=None, help="record the current state as a baseline JSON here")
    ap.add_argument("--work", default=None, help="where graphs are extracted (default: the temporary folder)")
    a = ap.parse_args(argv)
    try:
        return run_gate(Path(a.root), Path(a.out), jobs=a.jobs,
                        baseline=Path(a.baseline) if a.baseline else None,
                        write_baseline=Path(a.write_baseline) if a.write_baseline else None,
                        work=Path(a.work) if a.work else None)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
