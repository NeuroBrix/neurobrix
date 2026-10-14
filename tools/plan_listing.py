"""Prism LISTS, without executing anything, the models whose plan differs between hardware profiles.

The validation regime (the owner, 2026-10-13 14:20, precision 18:05): a model validates ONCE per
distinct Prism plan, not per card class. A gate runs the card's real class; the other class's cell is
owed only where the plan differs (strategy, loading, placement, expert store, tiling, kv cache). On
2026-10-13 16:5x this listing (then a campaign script over 196 explained plans) found 25 of 49
models plan-identical between the rack's 16 GB and 32 GB profiles, and a second scheduler placed
their cells on any card.

How: for each model, mode and profile, the cell's own command line (`regression_matrix.cell_argv`:
the gate's request) with `--hardware <profile> --explain-plan --json` — the plan the runtime would
receive, no weight loaded, no kernel run — and the plan's SIGNATURE compared across the profiles. A
profile whose explanation is refused leaves the model UNKNOWN, named with the refusal; never classed.

Outputs (in --out): plan_identical.txt, plan_differs.txt, plan_unknown.txt (one model per line) and
plans.json ({model: {mode: {profile: signature | {"refused": reason}}}}).

usage: plan_listing.py --models A,B --profiles P1,P2 --src <tree>/src --out DIR [--modes native,triton]
       [--python PY] [--timeout S] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import regression_matrix as RM  # noqa: E402

PYTHON = os.environ.get("NEUROBRIX_PYTHON", sys.executable)

#: What of an explained plan decides "the same plan": the strategy, the loading mode, each
#: component's placement (device count, dtype, expert store), the op-level tiling, the conv3d chunks,
#: the component tiling and the KV cache. The figures (planned MB) are not compared: two rungs of one
#: strategy are one plan for the cell.
SIGNATURE_KEYS = ("strategy", "loading_mode", "op_level_tiling", "conv3d_chunks", "component_tiling", "kv_cache")


def signature(plan: Dict[str, Any]) -> str:
    comps = sorted((c.get("name"), len(c.get("devices") or []), c.get("dtype"),
                    json.dumps(c.get("expert_store"), sort_keys=True))
                   for c in plan.get("components") or [])
    return json.dumps([[plan.get(k) for k in SIGNATURE_KEYS], comps], sort_keys=True)


def explain_argv(model: str, mode: str, profile: str) -> List[str]:
    """The cell's own argv with its output replaced by the explanation: one definition of the request."""
    argv = RM.cell_argv(model, mode, "/dev/null")
    i = argv.index("--output")
    return argv[:i] + argv[i + 2:] + ["--hardware", profile, "--explain-plan", "--json"]


def _explain(argv: List[str], src: Path, python: str, timeout: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """The explanation under the census shadow (`derived_census.shadow_env`): at the profile, never at
    the room, no device visible — the derivation's own way."""
    from derived_census import shadow_env
    p = subprocess.run([python, "-m", "neurobrix", *argv], capture_output=True, text=True, env=shadow_env(src),
                       timeout=timeout, cwd=str(src.parent))
    out = p.stdout
    i = out.find("{")
    if p.returncode != 0 or i < 0:
        tail = (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ["(no output)"])[-1]
        return None, f"rc {p.returncode}: {tail[:200]}"
    try:
        return json.loads(out[i:]), ""
    except json.JSONDecodeError as e:
        return None, f"no JSON in the explanation ({e})"


def classify(plans: Dict[str, Dict[str, Dict[str, Any]]], profiles: List[str]) -> Dict[str, List[str]]:
    """{identical|differs|unknown: [models]} from {model: {mode: {profile: signature | {"refused": why}}}}."""
    out = {"identical": [], "differs": [], "unknown": []}
    for m in sorted(plans):
        sigs = plans[m]
        if any(isinstance(v, dict) for mode in sigs.values() for v in mode.values()):
            out["unknown"].append(m)
            continue
        same = all(len({sigs[mode][p] for p in profiles}) == 1 for mode in sigs)
        out["identical" if same else "differs"].append(m)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated container names")
    ap.add_argument("--profiles", required=True, help="comma-separated hardware profile ids (config/hardware/<id>.yml)")
    ap.add_argument("--modes", default="native,triton")
    ap.add_argument("--src", required=True, help="the engine tree's src/ the plans are explained with")
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    models = [m for m in a.models.split(",") if m.strip()]
    profiles = [p for p in a.profiles.split(",") if p.strip()]
    modes = [m for m in a.modes.split(",") if m.strip()]
    if len(profiles) < 2:
        raise SystemExit("REFUSED: a listing compares at least two profiles")
    src = Path(a.src).resolve()
    jobs = [(m, mode, p) for m in models for mode in modes for p in profiles]

    def one(job):
        m, mode, p = job
        plan, why = _explain(explain_argv(m, mode, p), src, a.python, a.timeout)
        return job, (signature(plan) if plan is not None else {"refused": why})

    plans: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for (m, mode, p), sig in ex.map(one, jobs):
            plans.setdefault(m, {}).setdefault(mode, {})[p] = sig
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = classify(plans, profiles)
    for k, v in classes.items():
        (out / f"plan_{k}.txt").write_text("".join(x + "\n" for x in v))
    (out / "plans.json").write_text(json.dumps({"profiles": profiles, "modes": modes, "plans": plans}, indent=1, sort_keys=True))
    print(f"[plan_listing] {len(models)} models x {len(modes)} modes x {len(profiles)} profiles: "
          f"{len(classes['identical'])} identical, {len(classes['differs'])} differ, {len(classes['unknown'])} unknown"
          + (" (" + "; ".join(f"{m}: " + next(v["refused"] for mode in plans[m].values() for v in mode.values() if isinstance(v, dict))[:80]
                               for m in classes["unknown"]) + ")" if classes["unknown"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
