"""Zoo sweep — Phase 1 report-only driver of the OptimizationEngine.

Walks every installed build (a directory holding topology.json under
the engine cache root), runs the GraphAnalyzer on every component
graph, and writes the gains map: one dated JSON artifact per model
plus a synthesis table. Transforms nothing, loads no GPU — pure CPU
graph analysis.

Usage:
    python -m neurobrix.core.optim.sweep --out <artifact_dir> \
        [--root <cache_root>] [--models m1,m2] [--date YYYY_MM_DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import GraphAnalyzer
from .policy import PASS_REGISTRY

DEFAULT_ROOT = Path.home() / ".neurobrix" / "cache"


def discover_models(root: Path, only: set[str] | None) -> list[Path]:
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "topology.json").exists():
            if only and d.name not in only:
                continue
            out.append(d)
    return out



def _family_of(model_dir: Path) -> str:
    """Family from the build manifest — the same key the runtime reads."""
    mf = model_dir / "manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text()).get("family", "?")
        except Exception:
            pass
    return "?"


def sweep_model(model_dir: Path, out_dir: Path) -> dict:
    """Analyze every component graph of one build; return the summary row."""
    model = model_dir.name
    comp_summaries = []
    totals: dict[str, int] = {}
    removable: dict[str, int] = {}
    n_ops_total = 0
    n_suspects = 0
    stored_order_total = 0
    exec_order_total = 0
    reports_payload = []

    # THE EXECUTED SURFACE, not the stored one (measured 2026-08-19).
    # A findings count over the on-disk graph is a phantom for any MoE
    # build: the runtime fuses the expert subgraphs into one
    # `custom::moe_fused` op, and the ops it absorbs never launch. On
    # the canonical int4 decode row the executed sequence is 4,490 ops
    # out of 115,419 stored (-96%); deepseek-moe 2,683 of 18,610. A map
    # that counts the stored graph promises a pass work it can never
    # do — so the sweep applies the same fusion the runtime applies,
    # and records BOTH numbers so the gap itself stays visible.
    fam = _family_of(model_dir)

    for gpath in sorted(model_dir.rglob("graph.json")):
        raw = gpath.read_bytes()
        graph = json.loads(raw)
        n_stored_order = len(graph.get("execution_order") or [])
        try:
            from neurobrix.core.runtime.graph.moe_fusion import (
                detect_and_fuse_moe)
            graph = detect_and_fuse_moe(graph, fam, declared=True)
        except Exception:
            # A build the fusion does not recognise analyses as-is; the
            # two order counts then agree and the row says so.
            pass
        n_exec_order = len(graph.get("execution_order") or [])
        component = graph.get("component_name") or gpath.parent.name
        rep = GraphAnalyzer(model, component, graph, raw).run()
        counts = rep.counts()
        opsrem = rep.ops_removable()
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        for k, v in opsrem.items():
            removable[k] = removable.get(k, 0) + v
        n_ops_total += rep.n_ops
        stored_order_total += n_stored_order
        exec_order_total += n_exec_order
        n_suspects += len(rep.suspects)
        comp_summaries.append({
            "component": component,
            "n_ops": rep.n_ops,
            "counts": counts,
            "ops_removable": opsrem,
            "n_suspects": len(rep.suspects),
            # Both surfaces, so the phantom gap stays visible per
            # component rather than hiding inside a total.
            "n_ops_stored_order": n_stored_order,
            "n_ops_executed_order": n_exec_order,
        })
        reports_payload.append(rep.to_dict())
        del graph, rep, raw  # bound memory across large video graphs

    payload = {
        "model": model,
        "n_ops_stored_order": stored_order_total,
        "n_ops_executed_order": exec_order_total,
        "n_components": len(comp_summaries),
        "n_ops_total": n_ops_total,
        "totals": totals,
        "ops_removable": removable,
        "n_suspects": n_suspects,
        "components": comp_summaries,
        "reports": reports_payload,
    }
    out_path = out_dir / f"{model}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    return {
        "model": model,
        "n_components": len(comp_summaries),
        "n_ops": n_ops_total,
        "totals": totals,
        "ops_removable": removable,
        "n_suspects": n_suspects,
    }


def write_summary(rows: list[dict], out_dir: Path, date: str) -> None:
    cats = sorted(PASS_REGISTRY)
    lines = [
        f"# OptimizationEngine — Phase 1 gains map ({date})",
        "",
        "Report-only sweep of every installed build: sites the declared",
        "pass classes could act on. Counts are FINDINGS (sites); the",
        "ops-removable table estimates eliminated ops/launches per pass.",
        "No transformation was applied.",
        "",
        "## Findings per model (sites)",
        "",
        "| model | ops | " + " | ".join(cats) + " |",
        "|---|---|" + "|".join(["---"] * len(cats)) + "|",
    ]
    for r in sorted(rows, key=lambda r: -r["n_ops"]):
        cells = [str(r["totals"].get(c, 0)) for c in cats]
        lines.append(
            f"| {r['model']} | {r['n_ops']} | " + " | ".join(cells) + " |"
        )
    grand = {c: sum(r["totals"].get(c, 0) for r in rows) for c in cats}
    grem = {c: sum(r["ops_removable"].get(c, 0) for r in rows) for c in cats}
    lines += [
        "| **TOTAL** | "
        + str(sum(r["n_ops"] for r in rows))
        + " | "
        + " | ".join(str(grand[c]) for c in cats)
        + " |",
        "",
        "## Ops removable per pass (zoo total)",
        "",
        "| pass | gate | sites | ops removable |",
        "|---|---|---|---|",
    ]
    for c in cats:
        p = PASS_REGISTRY[c]
        lines.append(
            f"| {c} | {p.gate_class.value} | {grand[c]} | {grem[c]} |"
        )
    n_susp = sum(r.get("n_suspects", 0) for r in rows)
    lines += [
        "",
        "## Build-side suspects (NOT optimization sites)",
        "",
        f"{n_susp} frozen-bound signatures surfaced (slice end == trace "
        "extent of a symbolic dim — the slice-attrs class). These are "
        "coverage bugs to fix at the source; per-model op_uids in each "
        "model's JSON under `reports[].suspects`.",
        "",
        "| model | suspects |",
        "|---|---|",
    ] + [
        f"| {r['model']} | {r.get('n_suspects', 0)} |"
        for r in sorted(rows, key=lambda r: -r.get("n_suspects", 0))
        if r.get("n_suspects", 0)
    ] + [
        "",
        f"Models swept: {len(rows)}. Pass versions: "
        + ", ".join(f"{c}=v{PASS_REGISTRY[c].version}" for c in cats)
        + ".",
        "",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--date", type=str, required=True)
    args = ap.parse_args(argv)

    only = set(filter(None, args.models.split(","))) or None
    # Structural write scope: Phase 1 artifacts land under a
    # validation_outputs/ tree only — never inside the engine cache or
    # any infrastructure mount.
    if "validation_outputs" not in args.out.resolve().parts:
        ap.error("--out must live under a validation_outputs/ directory")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_dir in discover_models(args.root, only):
        try:
            row = sweep_model(model_dir, args.out)
        except Exception as e:  # a broken build must not sink the map
            print(f"[sweep] {model_dir.name}: FAILED {e}", file=sys.stderr)
            row = {"model": model_dir.name, "n_components": 0,
                   "n_ops": 0, "totals": {}, "ops_removable": {},
                   "error": str(e)}
        rows.append(row)
        print(f"[sweep] {row['model']}: ops={row['n_ops']} "
              f"totals={row['totals']}")
    write_summary(rows, args.out, args.date)
    print(f"[sweep] {len(rows)} models -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
