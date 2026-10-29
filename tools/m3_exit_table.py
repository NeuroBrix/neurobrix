"""M3's exit table: what Prism predicts a Triton component holds against what the run measured it held.

M3 closes when Prism prices every Triton component by the allocator's replay of its walk (`held_peak_bytes`: the walk
through the device pool and the hot loop's deferred queue, `--explain-plan` "the allocator's replay holds N MB at its
peak") instead of the profiled activations times the vendor file's `memory.triton_arena_activation_factor`. The
replay may replace the factor only where it agrees with what the runs measured, cell by cell: this table is that
comparison, read from a gate's own rows — the gate IS the benchmark (rule 3), nothing is run on a card here.

For every Triton row of the gate (`rows_card*.jsonl`, the harness's output):
  predicted  the row's own command (its model, request, profile and flags) with `--explain-plan`, card-less under the
             census shadow (`NBX_CENSUS=1`, no card visible): per component its weights, its walked activations, the
             replay's held peak and its pool ratio over the walk;
  measured   the row's log, the Triton release line of each component: "held at most N MB from the driver since its
             load (B MB of it held before)" — the device allocator's window the load opened, live and pooled bytes;
             the component's own hold is N - B. A component released more than once (a VAE run twice) keeps its most.
  the factor the arena figure the plan uses today: weights + activations x the planned device's vendor factor.

One row per (cell, component), appended as each cell finishes (a long table never loses a finished cell), then a
summary line per cell. The explain processes run in parallel on the hardware profile's compute threads
(`cpu.compute_threads`, rule 1), never on a count read from the machine.

    tools/m3_exit_table.py --rows GATE_DIR_OR_ROWS.jsonl [...] --tree REPO --out TABLE.tsv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atelier  # noqa: E402

COLUMNS = ("model", "mode", "strategy", "component", "placement", "device", "weights_mb", "activations_mb", "walk_mb", "replay_held_mb",
           "pool_ratio", "arena_factor", "arena_mb", "measured_held_mb", "measured_before_mb", "measured_own_mb",
           "replay_over_measured", "arena_over_measured", "run_peak_mb", "row_date", "row_engine", "explain_rc")

#: `solver.explain_plan`'s component line
_PLAN_LINE = re.compile(
    r"^\s+(?P<comp>\S+)\s+->\s+(?P<dev>\S+?)(?:\s+sharded)?\s+weights (?P<w>\d+) MB \+ activations (?P<a>\d+) MB"
    r" \+ overhead \d+ MB(?:; the allocator's replay holds (?P<held>\d+) MB at its peak"
    r"(?: \(the pool x(?P<ratio>[\d.]+) over the walk\))?)?")
#: the plan's strategy line, and the line under a component that says it is not held whole on its device
_STRATEGY_LINE = re.compile(r"^strategy\s+(?P<s>\S+)")
_NOT_WHOLE = (("zero3:", "zero3"), ("expert store:", "expert_store"))
#: the run's device peak (`[Peak] device_mb=N`; `na` on a card-less run)
_PEAK_LINE = re.compile(r"^\[Peak\] device_mb=(?P<mb>\d+)", re.M)
#: `graph_executor._cuda_peak_since_load`'s Triton release line
_RELEASE_LINE = re.compile(
    r"\[Triton\] '(?P<comp>[^']+)': released \d+ weights, held at most (?P<held>\d+) MB from the driver since its load"
    r"(?: \((?P<before>\d+) MB of it held before\))?")


class ExitTableRefused(RuntimeError):
    """The table cannot be made from what it was given (named)."""


def gate_rows(paths: List[str]) -> List[dict]:
    """The Triton rows of the gate files or directories `paths` (a directory: its `rows_card*.jsonl`). An input that
    names no file, or files with no Triton row, is refused by name — an empty table is never a green one."""
    files: List[str] = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "rows_card*.jsonl"))) if os.path.isdir(p) else [p]
    missing = [f for f in files if not os.path.isfile(f)]
    if not files or missing:
        raise ExitTableRefused(f"no gate rows: {missing or paths} (a gate directory holds rows_card*.jsonl)")
    rows = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if str(r.get("mode", "")).startswith("triton"):
                    rows.append(r)
    if not rows:
        raise ExitTableRefused(f"{files}: no Triton row — the table compares the Triton allocator's replay")
    latest: Dict[tuple, dict] = {}
    for r in rows:                                    # a cell replayed in the gate: its last row is the gate's
        latest[(r["model"], r["mode"])] = r
    return list(latest.values())


def strategy(explain_text: str) -> Optional[str]:
    """The plan's strategy (`strategy` line of `--explain-plan`)."""
    for line in explain_text.splitlines():
        m = _STRATEGY_LINE.match(line)
        if m:
            return m["s"]
    return None


def predicted(explain_text: str) -> Dict[str, dict]:
    """Per component, the plan's figures from `--explain-plan`'s output, and its placement: `whole` unless the line
    under it says its weights live elsewhere (`zero3`, `expert_store`) — the arena factor prices only a component held
    whole, so only those rows can agree or disagree with it."""
    out: Dict[str, dict] = {}
    last = None
    for line in explain_text.splitlines():
        m = _PLAN_LINE.match(line)
        if m:
            last = m["comp"]
            out[last] = {"device": m["dev"], "weights_mb": int(m["w"]), "activations_mb": int(m["a"]),
                         "replay_held_mb": int(m["held"]) if m["held"] else None,
                         "pool_ratio": float(m["ratio"]) if m["ratio"] else None, "placement": "whole"}
            continue
        if last is not None and line.startswith("   ") and not line.lstrip().startswith(("-", "->")):
            for word, placement in _NOT_WHOLE:
                if line.strip().startswith(word):
                    out[last]["placement"] = placement
        elif not line.startswith("  "):
            last = None
    return out


def run_peak(log_text: str) -> Optional[int]:
    """The run's device peak (`[Peak] device_mb=N`), or None."""
    peaks = [int(m["mb"]) for m in _PEAK_LINE.finditer(log_text)]
    return max(peaks) if peaks else None


def measured(log_text: str) -> Dict[str, dict]:
    """Per component, the most it held from the driver over its loads, and what was held before that load."""
    out: Dict[str, dict] = {}
    for m in _RELEASE_LINE.finditer(log_text):
        held, before = int(m["held"]), int(m["before"] or 0)
        cur = out.get(m["comp"])
        if cur is None or held - before > cur["held"] - cur["before"]:
            out[m["comp"]] = {"held": held, "before": before}
    return out


def explain_argv(command: str) -> List[str]:
    """The row's own command, asked for its plan: `--explain-plan` added, nothing else changed."""
    argv = shlex.split(command)
    if "neurobrix" not in argv or "run" not in argv:
        raise ExitTableRefused(f"not an entry-point command: {command[:200]}")
    return argv + ["--explain-plan"]


def arena_factor(hardware_id: str, device: str) -> Optional[float]:
    """The vendor file's measured `memory.triton_arena_activation_factor` of `device` in the profile `hardware_id`."""
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.core.config.loader import vendor_config_for_device
    prof = load_profile(hardware_id)
    idx = int(device.split(":")[1]) if device.startswith("cuda:") else None
    devs = [d for d in prof.devices if idx is None or d.index == idx]
    if not devs:
        return None
    d = devs[0]
    brand = getattr(getattr(d, "brand", None), "value", getattr(d, "brand", None))
    v = ((vendor_config_for_device(brand, d.architecture, getattr(d, "compute_capability", None),
                                   model=getattr(d, "name", None)).get("memory") or {})
         .get("triton_arena_activation_factor"))
    return float(v) if v is not None else None


def cell_profile(row: dict, tree: Path, _seen: Dict[str, tuple] = {}) -> tuple:
    """(profile id, rung MB) the row's cell ran under: the profile its command names, else the one the runtime
    resolves on the row's card (`regression_matrix.cell_profile`, the harness's own function, detection only)."""
    import regression_matrix as RM
    argv = shlex.split(row["command"])
    if "--hardware" in argv:
        from neurobrix.core.prism.loader import load_profile
        from neurobrix.core.prism.structure import memory_class_from_profile_id
        hw = argv[argv.index("--hardware") + 1]
        cls = memory_class_from_profile_id(load_profile(hw).id)
        if cls is None:
            raise ExitTableRefused(f"{hw}: its memory class cannot be read from its id")
        return hw, int(cls) * 1024
    gpu = str(row.get("gpu", ""))
    if gpu not in _seen:
        _seen[gpu] = RM.cell_profile(gpu, tree / "src")
    return _seen[gpu]


def table_rows(row: dict, explain_text: str, explain_rc: int, log_text: str, hardware_id: str) -> List[dict]:
    plan, meas = predicted(explain_text), measured(log_text)
    strat, peak = strategy(explain_text), run_peak(log_text)
    out = []
    for comp in sorted(set(plan) | set(meas)):
        p, m = plan.get(comp, {}), meas.get(comp)
        f = arena_factor(hardware_id, p["device"]) if p.get("device") else None
        walk = (p["weights_mb"] + p["activations_mb"]) if p else None
        arena = round(p["weights_mb"] + p["activations_mb"] * f) if p and f is not None else None
        own = (m["held"] - m["before"]) if m else None
        out.append({"model": row["model"], "mode": row["mode"], "strategy": strat, "component": comp,
                    "placement": p.get("placement"), "device": p.get("device"),
                    "weights_mb": p.get("weights_mb"), "activations_mb": p.get("activations_mb"), "walk_mb": walk,
                    "replay_held_mb": p.get("replay_held_mb"), "pool_ratio": p.get("pool_ratio"),
                    "arena_factor": f, "arena_mb": arena,
                    "measured_held_mb": m["held"] if m else None, "measured_before_mb": m["before"] if m else None,
                    "measured_own_mb": own,
                    "replay_over_measured": (round(p["replay_held_mb"] / own, 3)
                                             if own and p.get("replay_held_mb") is not None else None),
                    "arena_over_measured": round(arena / own, 3) if own and arena is not None else None,
                    "run_peak_mb": peak, "row_date": row.get("date"), "row_engine": row.get("engine"),
                    "explain_rc": explain_rc})
    return out


def _explain(row: dict, tree: Path, python: str, scratch: Path, hw: str, rung_mb: int) -> tuple:
    """The row's plan as the harness asks it (`regression_matrix.plan_host_need`): card-less on the census path, at
    the cell's profile and its card's rung."""
    argv = explain_argv(row["command"])
    argv[0] = python                                  # the engine's interpreter, pinned (environments.md)
    if "--hardware" not in argv:
        argv += ["--hardware", hw]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NBX_CENSUS="1", NBX_CENSUS_DEVICES="1",
               NBX_PRISM_BUDGET_MB=str(rung_mb), PYTHONNOUSERSITE="1",
               PYTHONPATH=f"{tree / 'src'}:{tree / 'nbxtensor' / 'src'}",
               NEUROBRIX_REPLAY_CACHE=str(scratch / f"replay_{row['model']}_{row['mode']}"))
    p = subprocess.run(argv, cwd=tree, env=env, capture_output=True, text=True, timeout=1200)
    return p.returncode, p.stdout + p.stderr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", required=True, help="gate directories or rows_card*.jsonl files")
    ap.add_argument("--tree", required=True, help="the repository whose Prism predicts")
    ap.add_argument("--out", required=True, help="the table (TSV), under the workshop's writable roots")
    ap.add_argument("--python", default=None, help="the engine interpreter (default: atelier's `engine`)")
    a = ap.parse_args(argv)
    rows = gate_rows(a.rows)
    out = atelier.writable(a.out)
    tree = Path(a.tree).resolve()
    python = a.python or str(atelier.python("engine"))
    profiles = {(r["model"], r["mode"]): cell_profile(r, tree) for r in rows}
    from neurobrix.core.prism.loader import load_profile
    threads = {load_profile(h).cpu.compute_threads for h, _ in profiles.values()}   # the profile's threads (rule 1)
    workers = min(min(threads), len(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
    scratch = Path(tempfile.mkdtemp(dir=atelier.tmp()))
    print(f"[exit table] {len(rows)} Triton cell(s), {workers} explain worker(s) (cpu.compute_threads) -> {out}",
          flush=True)

    def one(row):
        hw, rung = profiles[(row["model"], row["mode"])]
        rc, text = _explain(row, tree, python, scratch, hw, rung)
        evidence = out.with_suffix(".explain") / f"{row['model']}.{row['mode']}.txt"   # the plan the row is read against
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(text)
        log = Path(row["log"]).read_text(errors="replace") if row.get("log") and Path(row["log"]).is_file() else ""
        lines = table_rows(row, text, rc, log, hw)
        with open(out, "a") as f:                     # appended as each cell finishes
            for r in lines:
                f.write("\t".join("" if r[c] is None else str(r[c]) for c in COLUMNS) + "\n")
        worst = [r["replay_over_measured"] for r in lines if r["replay_over_measured"] is not None]
        print(f"[exit table] {row['model']} {row['mode']}: explain rc {rc}, {len(lines)} component(s), replay/measured "
              f"{min(worst) if worst else '-'}..{max(worst) if worst else '-'}"
              + ("" if log else "  [no log: nothing measured]"), flush=True)
        return rc

    try:
        with ThreadPoolExecutor(workers) as ex:
            rcs = list(ex.map(one, rows))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)    # the workshop's tmp is emptied after each job
    bad = sum(1 for rc in rcs if rc != 0)
    print(f"[exit table] done: {len(rows)} cell(s), {bad} explain(s) failed -> {out}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
