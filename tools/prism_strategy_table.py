#!/usr/bin/env python3
"""What strategy Prism picks for every model of the local zoo, and on what budget.

Prism's decision is a CPU decision: a container's metadata, a hardware profile read from
YAML, an activation estimate. So it can be read for the whole zoo without touching a card,
and two trees can be compared line by line — which is what a change to a STRATEGY BUDGET
owes: not "nothing moved" (inertia would mean keeping the bug) but, for every model that
moves, the numbers that moved it.

    python3 tools/prism_strategy_table.py --hardware v100-32g > after.tsv
    PYTHONPATH=<other tree>/src python3 tools/prism_strategy_table.py --hardware v100-32g > before.tsv
    diff before.tsv after.tsv

Columns: model, strategy, loading_mode, planned MB, sum(weights) MB, max(weight+activation)
MB, max(activation) MB, capacity MB. The two middle columns are the two budgets a
single-GPU rung can be written with; the capacity is what they are compared against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# APPENDED, not inserted at 0: this tool is run against ANOTHER tree by PYTHONPATH (that is
# the whole point of comparing two trees line by line), and inserting its own src first made
# both arms import the same engine and report a perfectly identical table.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


_SG = {}


def _watch_single_gpu(PrismSolver) -> None:
    """Record the single-GPU rung's own verdict, once per process.

    The strategy a model ends on is the cascade's answer; what a change to THIS rung's
    budget moves is whether the rung accepted at all. A table that showed only the final
    strategy could report "nothing moved" while the rung's verdict had flipped under a
    later filter — so the verdict is read where it is made.
    """
    if getattr(PrismSolver, "_nbx_watched", False):
        return
    orig = PrismSolver._try_single_gpu

    def wrapped(self, sorted_comps, comp_mem, devices, shard_sizes, profile, container):
        r = orig(self, sorted_comps, comp_mem, devices, shard_sizes, profile, container)
        _SG["verdict"] = "accepted" if r is not None else "refused"
        return r

    PrismSolver._try_single_gpu = wrapped
    PrismSolver._nbx_watched = True


def one(name: str, hardware: str, cache: Path) -> str:
    from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
    from neurobrix.nbx import NBXContainer

    _watch_single_gpu(PrismSolver)
    _SG.pop("verdict", None)

    container = NBXContainer.load(str(cache / name))
    profile = load_profile(hardware)
    defaults_path = cache / name / "runtime" / "defaults.json"
    d = json.loads(defaults_path.read_text()) if defaults_path.exists() else {}
    family = None
    try:
        family = container.metadata.get("family")
    except Exception:
        pass
    fam = {}
    if family:
        try:
            from neurobrix.core.config import get_family_defaults
            fam = get_family_defaults(family) or {}
        except Exception:
            fam = {}
    cfg = InputConfig(
        batch_size=2,
        height=d.get("height") or fam.get("height") or 1024,
        width=d.get("width") or fam.get("width") or 1024,
        dtype="float16",
        vae_scale=d.get("vae_scale_factor", 8),
        num_frames=d.get("num_frames"),
        temporal_compression=d.get("temporal_compression_ratio", 4),
    )
    solver = PrismSolver()
    plan = solver.solve_smart(container, profile, cfg)
    cm = plan.component_memory or {}
    sum_w = sum(m.weight_mb for m in cm.values())
    max_peak = max((m.weight_mb + m.activation_mb for m in cm.values()), default=0.0)
    max_act = max((m.activation_mb for m in cm.values()), default=0.0)
    # The capacity the rungs actually compare against: the solver's own device view, minus
    # the blanket driver/library reserve it subtracts before accepting a single-GPU plan.
    devs = solver._prepare_devices(profile)
    cap = (devs[0].capacity_mb - solver.oom_reserve_mb) if devs else 0
    return (f"{name}\t{plan.strategy}\t{plan.loading_mode}\t{plan.total_memory_mb:.0f}\t"
            f"{sum_w:.0f}\t{max_peak:.0f}\t{max_act:.0f}\t{cap:.0f}\t"
            f"{_SG.get('verdict', 'not-reached')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", default="v100-32g")
    ap.add_argument("--cache", default=str(Path.home() / ".neurobrix" / "cache"))
    ap.add_argument("--models", default="", help="comma-separated; default = the whole cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    names = ([m for m in args.models.split(",") if m] if args.models
             else sorted(p.name for p in cache.iterdir() if (p / "manifest.json").exists()))
    print("model\tstrategy\tloading_mode\tplanned_mb\tsum_weights_mb\tmax_component_mb\tmax_activation_mb\tusable_mb\tsingle_gpu_rung")
    for name in names:
        try:
            print(one(name, args.hardware, cache), flush=True)
        except BaseException as e:                       # a container this profile refuses is a ROW
            msg = str(e).replace("\t", " ").replace("\n", " ")[:160]
            print(f"{name}\tREFUSED\t-\t-\t-\t-\t-\t-\t-\t{type(e).__name__}: {msg}", flush=True)
            if os.environ.get("NBX_PRISM_TABLE_TRACE"):
                traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
