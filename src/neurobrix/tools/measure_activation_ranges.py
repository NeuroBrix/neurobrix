"""measure_activation_ranges — decide per-component activations_fp16_safe.

Phase 1 doctrine: the per-component flag `activations_fp16_safe` in
`forge/config/model_registry.yml` opts a model in to the cast-back
behavior on AMP_FP32_OPS reductions (rms_norm, div). Enabling the flag
when activations are NOT fp16-safe risks silent fp16 saturation. This
tool measures the actual range distribution by running compiled mode
(numerically authoritative) and emits a per-component verdict:

    - max(abs) < 30000               → safe : suggest activations_fp16_safe: true
    - 30000 <= max(abs) < 50000      → marginal : flag+visual review required
    - max(abs) >= 50000              → unsafe : keep activations_fp16_safe: false

Usage:
    python -m neurobrix.tools.measure_activation_ranges \\
        --model Sana_1600M_4Kpx_BF16 \\
        --prompt "red apple" --steps 4 \\
        --component vae

The tool runs `neurobrix run` in compiled mode under the
`NBX_TRACE_RANGES=rms_norm,div,convolution` env probe (built into
core/runtime/graph/compiled_sequence.py) and post-processes the
emitted TSV (default `/tmp/nbx_ranges.tsv`).

Outputs the verdict to stdout and a registry-snippet that can be copied
into `forge/config/model_registry.yml`.
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


SAFE_MAX = 30_000.0
MARGINAL_MAX = 50_000.0
FP16_MAX = 65_504.0
DEFAULT_RANGE_LOG = "/tmp/nbx_ranges.tsv"
DEFAULT_OPS = "rms_norm,div,convolution"


def parse_log(path: Path) -> List[Tuple[str, float, str]]:
    """Return list of (op_type, max_abs, dtype) rows from the TSV log."""
    rows: List[Tuple[str, float, str]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("op_idx"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                rows.append((parts[2], float(parts[3]), parts[4] if len(parts) > 4 else "?"))
            except (ValueError, IndexError):
                continue
    return rows


def per_op_stats(rows: List[Tuple[str, float, str]]) -> Dict[str, Dict[str, float]]:
    """Build per-op-type statistics (count, max, p99, p95, p50, min)."""
    by_op: Dict[str, List[float]] = defaultdict(list)
    for ot, ma, _dt in rows:
        by_op[ot].append(ma)
    out: Dict[str, Dict[str, float]] = {}
    for ot, vals in by_op.items():
        s = sorted(vals)
        n = len(s)
        if n == 0:
            continue
        out[ot] = {
            "count": float(n),
            "max": s[-1],
            "p99": s[min(n - 1, int(n * 0.99))],
            "p95": s[min(n - 1, int(n * 0.95))],
            "p50": s[n // 2],
            "min": s[0],
        }
    return out


def verdict(global_max: float) -> str:
    if global_max < SAFE_MAX:
        return "SAFE"
    if global_max < MARGINAL_MAX:
        return "MARGINAL"
    return "UNSAFE"


def render_report(model: str, component: str, stats: Dict[str, Dict[str, float]]) -> str:
    if not stats:
        return f"[{model}/{component}] no rows logged — verify --component matches and ops are present in the graph"
    lines = []
    header = f"{'op_type':<45s} {'count':>6s} {'max':>14s} {'p99':>14s} {'p95':>14s} {'p50':>14s} {'min':>14s}"
    lines.append(header)
    lines.append("-" * len(header))
    global_max = 0.0
    for ot in sorted(stats):
        st = stats[ot]
        lines.append(
            f"{ot:<45s} {int(st['count']):>6d} {st['max']:>14.6g} {st['p99']:>14.6g} "
            f"{st['p95']:>14.6g} {st['p50']:>14.6g} {st['min']:>14.6g}"
        )
        if st["max"] > global_max:
            global_max = st["max"]
    v = verdict(global_max)
    lines.append("")
    lines.append(f"global max(abs) = {global_max:.6g}  ({100*global_max/FP16_MAX:.1f}% of fp16 max {FP16_MAX:.0f})")
    lines.append(f"verdict: {v}")
    if v == "SAFE":
        lines.append(f"  → suggest registry: components.{component}.activations_fp16_safe: true")
    elif v == "MARGINAL":
        lines.append(f"  → marginal margin (>{SAFE_MAX:.0f}). Visual validation required before opt-in.")
    else:
        lines.append(f"  → keep activations_fp16_safe: false. Output > {MARGINAL_MAX:.0f} risks fp16 saturation.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model name (snapshot dir name)")
    ap.add_argument("--prompt", default="red apple", help="Prompt or input text")
    ap.add_argument("--steps", type=int, default=4, help="Diffusion steps for short representative run")
    ap.add_argument("--component", default="(all)",
                    help="Component name to filter the verdict (informational; the run measures the whole pipeline)")
    ap.add_argument("--ops", default=DEFAULT_OPS, help="Comma-separated bare op names to trace")
    ap.add_argument("--log", default=DEFAULT_RANGE_LOG, help="Path to range log TSV")
    ap.add_argument("--neurobrix", default="/home/mlops/ml/venv/bin/neurobrix", help="neurobrix CLI path")
    ap.add_argument("--output", default="/tmp/measure_activation_ranges.png", help="Disposable output path")
    ap.add_argument("--skip-run", action="store_true", help="Skip the neurobrix run and just analyze an existing log")
    args = ap.parse_args()

    log_path = Path(args.log)

    if not args.skip_run:
        if log_path.exists():
            log_path.unlink()
        env = os.environ.copy()
        env["NBX_TRACE_RANGES"] = args.ops
        env["NBX_RANGE_LOG"] = str(log_path)
        cmd = [
            args.neurobrix, "run",
            "--model", args.model,
            "--prompt", args.prompt,
            "--steps", str(args.steps),
            "--output", args.output,
        ]
        print(f"[measure_activation_ranges] running: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"[measure_activation_ranges] neurobrix run exited {proc.returncode}", file=sys.stderr)

    rows = parse_log(log_path)
    stats = per_op_stats(rows)
    print(render_report(args.model, args.component, stats))


if __name__ == "__main__":
    main()
