#!/usr/bin/env python3
"""What Prism decides on the Apple profile when a model does not fit.

The Apple machine has ONE device and its memory is unified: 24 GB shared with
the host. A model larger than that cannot be resident, so the question is not
"does it fit" but "what does the solver do about it, and does the plan it
returns actually respect the budget it was given".

Reads the imported model caches in place; nothing is written to the mount and
nothing is imported locally. Weights are never loaded — Prism plans from the
component metadata and the shard sizes.

    python tools/prism_apple_budget.py --out validation_outputs/<dated>/ \
        --model Allegro --model DeepSeek-Coder-V2-Lite-Instruct
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import traceback
from pathlib import Path

CACHE = Path.home() / "Mounts" / "Super-NeuroBrix-Cache"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    sys.path.insert(0, "src")
    from neurobrix.nbx import NBXContainer
    from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.core.config import get_family_defaults

    prof = load_profile(get_or_create_default_profile())
    budget_mb = sum(d.memory_mb for d in prof.devices)
    rows = []
    for name in args.model:
        path = args.cache / name
        rec = {"model": name, "cache": str(path)}
        try:
            disk_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            rec["cache_gb"] = round(disk_bytes / 2**30, 2)
            c = NBXContainer.load(str(path))
            man = c.get_manifest() or {}
            rec["family"] = man.get("family")
            rec["dtype"] = man.get("dtype")
            rec["components"] = [x.name for x in c.get_neural_components()]
            # The InputConfig MUST mirror what the executor actually runs at,
            # or Prism sizes activations for a resolution nobody asked for. A
            # first pass here used a flat 1024x1024 and made Wan2.1's VAE
            # 26865 MB, of which 25344 was activation -- an 8x over-estimate
            # that turned a model into a refusal. run.py documents this exact
            # trap; this is its fallback chain: defaults.json -> family -> 1024.
            cached_defaults = {}
            dj = path / "runtime" / "defaults.json"
            if dj.exists():
                cached_defaults = json.loads(dj.read_text())
            fam_defaults = get_family_defaults(rec["family"]) if rec["family"] else {}
            height = cached_defaults.get("height") or fam_defaults.get("height") or 1024
            width = cached_defaults.get("width") or fam_defaults.get("width") or 1024
            icfg = InputConfig(
                batch_size=2,  # CFG effectively doubles batch, as run.py does
                height=height, width=width, dtype="float16",
                vae_scale=cached_defaults.get("vae_scale_factor", 8),
                num_frames=cached_defaults.get("num_frames"),
                temporal_compression=cached_defaults.get("temporal_compression_ratio", 4),
            )
            rec["input_config"] = {"batch_size": 2, "height": height, "width": width,
                                   "num_frames": cached_defaults.get("num_frames"),
                                   "source": ("defaults.json" if cached_defaults.get("height")
                                              else ("family config" if fam_defaults.get("height")
                                                    else "fallback 1024"))}
            solver = PrismSolver()
            plan = solver.solve(c, prof, icfg)
            # What the solver DECIDED to tile, before the plan filters it.
            # solver.py keeps only entries whose device startswith "cuda",
            # under a comment that says "landed on a GPU" -- so on mps/hip/xpu
            # a decision that was made is then dropped. Recording both sides is
            # the only way to tell "discarded" from "never decided".
            rec["tiling_decided_by_solver"] = sorted(
                (getattr(solver, "_component_tiling", {}) or {}).keys())
            rec["strategy"] = plan.strategy
            rec["loading_mode"] = plan.loading_mode
            rec["selection_reason"] = plan.selection_reason
            rec["total_memory_mb"] = round(float(plan.total_memory_mb), 1)
            rec["cpu_ram_mb"] = int(plan.cpu_ram_mb)
            rec["component_memory_mb"] = {
                k: round(float(getattr(v, "total_mb", getattr(v, "weights_mb", 0))), 1)
                for k, v in (plan.component_memory or {}).items()
            }
            # Where each component actually landed. The strategy name does not
            # say this, and it is the only thing that answers "will it fit".
            rec["placement"] = {
                k: {"devices": list(v.devices), "memory_mb": round(float(v.memory_mb), 1),
                    "strategy": v.strategy, "sharded": bool(v.sharded)}
                for k, v in (plan.components or {}).items()}
            # total_memory_mb is a SUM over components whatever the strategy
            # (solver.py: `total_mb += mem.total_mb`). For an EAGER strategy the
            # sum is what has to be resident; for a LAZY one the peak is
            # max(component), so comparing the sum to the device budget would
            # condemn a plan that is correct. Compare the right one.
            per_comp = [v["memory_mb"] for v in rec["placement"].values()] or [0.0]
            on_device = [v["memory_mb"] for v in rec["placement"].values()
                         if not any(d.startswith("cpu") for d in v["devices"])] or [0.0]
            rec["resident_mb"] = (round(sum(on_device), 1)
                                  if plan.loading_mode == "eager" else round(max(on_device), 1))
            rec["resident_basis"] = ("sum of device-resident components (eager)"
                                     if plan.loading_mode == "eager"
                                     else "largest device-resident component (lazy)")
            rec["fits_device_budget"] = rec["resident_mb"] <= budget_mb
            rec["sum_all_components_mb"] = round(sum(per_comp), 1)
            # On UNIFIED memory the offload rungs buy nothing: a component
            # placed on "cpu" sits in the same 24 GB the device is using. The
            # solver has no concept of this -- it treats cpu.ram_mb as a second
            # pool -- so the device-only figure above can call a plan safe when
            # on this machine it is not. Both are recorded; neither is dropped.
            off = [v["memory_mb"] for v in rec["placement"].values()
                   if any(d.startswith("cpu") for d in v["devices"])]
            rec["offloaded_to_host_mb"] = round(sum(off), 1)
            rec["unified_resident_mb"] = round(rec["resident_mb"] + sum(off), 1)
            rec["fits_unified_budget"] = rec["unified_resident_mb"] <= budget_mb
            # A component whose ACTIVATION dominates can be planned with
            # tiling, which cuts what is actually resident. Without these two
            # fields "resident_mb exceeds the budget" is not a finding, it is
            # an unread plan -- the solver carries them for exactly this case.
            rec["runtime_op_tiling"] = {
                k: (list(v)[:6] if isinstance(v, (list, dict)) else str(v)[:120])
                for k, v in (plan.runtime_op_tiling or {}).items()}
            rec["component_tiling"] = {
                k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else str(v)[:120]
                for k, v in (plan.component_tiling or {}).items()}
            rec["tiled_components"] = sorted(
                set(rec["runtime_op_tiling"]) | set(rec["component_tiling"]))
            rec["status"] = "planned"
        except Exception as e:
            rec["status"] = "raised"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback_tail"] = traceback.format_exc()[-600:]
        rows.append(rec)
        print(f"{name}: {rec.get('strategy', rec['status'])}")

    args.out.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/prism_apple_budget.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "profile": {"id": prof.id, "vendor": getattr(prof, "vendor", None),
                    "total_vram_gb": prof.total_vram_gb,
                    "device_budget_mb": budget_mb,
                    "devices": len(prof.devices),
                    "memory": "unified with the host"},
        "note": "model caches read in place; nothing written to the mount, "
                "nothing imported locally, no weights loaded",
        "results": rows,
    }
    (args.out / "records.json").write_text(json.dumps(doc, indent=2))

    md = ["# Prism on the Apple profile, at and over the memory budget", "",
          f"Generated **{doc['generated']}** by `{doc['tool']}` on {doc['machine']}.",
          "**No public claim is made from any number here.**", "",
          f"* profile: `{prof.id}` — vendor {doc['profile']['vendor']}, "
          f"{prof.total_vram_gb:.1f} GB, {len(prof.devices)} device, unified with the host",
          f"* device budget: **{budget_mb} MB**",
          f"* {doc['note']}", "",
          "| model | cache GB | family | resolution | strategy | loading | on device MB | fits device | to host MB | fits UNIFIED |",
          "|---|---:|---|---|---|---|---:|---|---:|---|"]
    for r in rows:
        if r["status"] != "planned":
            md.append(f"| {r['model']} | {r.get('cache_gb','?')} | {r.get('family','?')} | "
                      f"**{r['status']}** | — | — | {r.get('error','')[:60]} |")
        else:
            ic = r.get("input_config", {})
            md.append(f"| {r['model']} | {r['cache_gb']} | {r['family']} | "
                      f"{ic.get('height','?')}x{ic.get('width','?')} ({ic.get('source','?')}) | "
                      f"`{r['strategy']}` | {r['loading_mode']} | {r['resident_mb']} | "
                      f"{'yes' if r['fits_device_budget'] else '**NO**'} | "
                      f"{r['offloaded_to_host_mb']} | "
                      f"{'yes' if r['fits_unified_budget'] else '**NO**'} |")
    md += ["", "**resident MB** is the sum of device-resident components for an",
           "eager plan and the largest one for a lazy plan — `total_memory_mb`",
           "is a sum whatever the strategy, so comparing IT to the budget would",
           "condemn a lazy plan that is correct.", "",
           "## Where each component landed", ""]
    for r in rows:
        if r["status"] == "planned":
            _d = r.get("tiling_decided_by_solver") or []
            _t = r.get("tiled_components") or []
            if _d and not _t:
                md.append(f"> Tiling was DECIDED for {', '.join(_d)} and then dropped "
                          f"from the plan — `solver.py` keeps only components whose "
                          f"device starts with `cuda`, and this one is on `mps:0`.")
                md.append("")
            md.append(f"**{r['model']}** — {r['resident_basis']}"
                      + (f" · tiled: {', '.join(_t)}" if _t else " · no tiling planned"))
            md.append("")
            md.append("| component | devices | MB | strategy |")
            md.append("|---|---|---:|---|")
            for k, v in r["placement"].items():
                md.append(f"| {k} | {', '.join(v['devices']) or 'cpu'} | "
                          f"{v['memory_mb']} | {v['strategy']} |")
            md.append("")
    md += ["", "## Why each strategy was chosen", ""]
    for r in rows:
        if r["status"] == "planned":
            md += [f"**{r['model']}** — `{r['strategy']}`, loading `{r['loading_mode']}`",
                   "", f"> {r['selection_reason'] or '(the plan carried no reason)'}", ""]
    (args.out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print(f"\n{len(rows)} models -> {args.out}/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
