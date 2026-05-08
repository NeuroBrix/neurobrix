"""Offline op-by-op DAG divergence analysis: sequential vs triton_sequential.

Parses two existing VALUE_TRACE logs (no re-run, no NeuroBrix
modification) and identifies the FIRST op where the triton_sequential
path diverges materially from the sequential PyTorch oracle.

Both logs trace the same ATen DAG executed op-by-op on the same model
(Sana_1600M_4Kpx_BF16) with the same prompt and seed; only the backend
kernel differs (PyTorch eager / cuDNN-cuBLAS vs NBX Triton). Sequential
produces a coherent PNG; triton_sequential produces structured-noise
garbage. The first op where samples cross a "real divergence" threshold
(sign flip on a non-trivial value, or > 5% relative error on positions
0-3) is the bug source — either intrinsically wrong on this shape, or
correct in isolation but receiving inputs subtly wrong from upstream.

Position-4 (= n-1 logical index) is excluded from the divergence
metric: that sample showed inconsistent reads across runs in earlier
diagnostic work and may still produce false positives. Positions 0-3
sample (0, n//8, n//4, n//2) and are deterministic.

Output: validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/diff_dag_op_by_op_report.md
"""
import re
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System")
LOG_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05"
SEQ_LOG = LOG_DIR / "sequential_4kpx_values_v2.log"
TRI_LOG = LOG_DIR / "sana4kpx_triseq_values_v5.log"
OUT_MD = LOG_DIR / "diff_dag_op_by_op_report.md"

# Tolerance bands (positions 0-3 only).
# Two-tier classification:
#   noise: fp16 quantization-level drift, expected between PyTorch
#          eager (cuDNN/cuBLAS) and NBX Triton on the same DAG.
#   real:  divergence that cannot be explained by fp16/fp32 noise on
#          values of magnitude > REAL_ABS_FLOOR (sign flip, or both
#          relative >= REAL_REL AND absolute >= REAL_ABS_MIN).
NOISE_ABS = 1e-3
NOISE_REL = 0.01
REAL_REL = 0.05        # 5% relative
REAL_ABS_MIN = 0.01    # absolute diff must also exceed this to count as "real"
REAL_ABS_FLOOR = 0.01  # ignore relative threshold when |seq| < this (avoid div-by-tiny)

LINE_RE = re.compile(
    r"VALUE_TRACE path=\S+ op_idx=(\d+) op_uid=(\S+) samp=(.+?)\]\s*$"
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse(line: str):
    """Return (op_idx, op_uid, samples_list_or_None)."""
    m = LINE_RE.search(line)
    if not m:
        return None
    op_idx = int(m.group(1))
    op_uid = m.group(2)
    samp_raw = m.group(3)
    if any(s in samp_raw for s in (
            "int_dtype", "no_floats", "empty", "ERR_",
            "None", "tuple_empty", "unknown")):
        return op_idx, op_uid, None
    nums = [float(x) for x in NUM_RE.findall(samp_raw)]
    return op_idx, op_uid, nums


def load_log(path: Path):
    out = []
    for line in path.open():
        if "VALUE_TRACE" not in line:
            continue
        e = parse(line)
        if e is not None:
            out.append(e)
    return out


def classify(seq: list, tri: list):
    """Compare positions 0-3 of two 5-vectors. Skip position 4 (known
    n-1 fingerprint artifact). Return (kind, max_abs_diff, max_rel_diff,
    sign_flips).

    kind ∈ {"none", "noise", "borderline", "real"}.
    """
    if seq is None or tri is None:
        return "skip", 0.0, 0.0, 0
    if len(seq) < 4 or len(tri) < 4:
        return "skip", 0.0, 0.0, 0
    seq4 = seq[:4]
    tri4 = tri[:4]
    abs_diffs = [abs(a - b) for a, b in zip(seq4, tri4)]
    max_abs = max(abs_diffs)
    rel_diffs = [
        abs(a - b) / max(abs(a), REAL_ABS_FLOOR)
        for a, b in zip(seq4, tri4)
    ]
    max_rel = max(rel_diffs)
    sign_flips = sum(
        1 for a, b in zip(seq4, tri4)
        if (a > 0 and b < 0 and abs(a) > REAL_ABS_FLOOR and abs(b) > REAL_ABS_FLOOR)
        or (a < 0 and b > 0 and abs(a) > REAL_ABS_FLOOR and abs(b) > REAL_ABS_FLOOR)
    )
    if max_abs == 0.0:
        return "none", 0.0, 0.0, 0
    if max_abs <= NOISE_ABS and max_rel <= NOISE_REL:
        return "noise", max_abs, max_rel, sign_flips
    # Real = sign flip on a non-tiny value, OR (relative >= 5% AND
    # absolute >= REAL_ABS_MIN). The absolute floor prevents
    # classifying fp16 quantization noise on tiny values (e.g.
    # 0.00157 vs 0.001658 = 5.6% rel) as "real divergence".
    if sign_flips >= 1:
        return "real", max_abs, max_rel, sign_flips
    if max_rel >= REAL_REL and max_abs >= REAL_ABS_MIN:
        return "real", max_abs, max_rel, sign_flips
    return "borderline", max_abs, max_rel, sign_flips


def detect_component_boundaries(stream):
    """Identify trace lines where a new component begins (op_idx resets
    to 0 after having grown beyond a threshold). Useful to attribute a
    line to text_encoder / transformer iter k / vae."""
    bounds = [0]
    last_idx = -1
    for i, e in enumerate(stream):
        if e is None:
            continue
        idx = e[0]
        if idx == 0 and last_idx > 100:
            bounds.append(i)
        last_idx = idx
    return bounds


def component_label(line_idx: int, bounds: list, total: int):
    """Best-effort labeling of a trace line by component."""
    # Sana 4Kpx structure observed: text_encoder (3449 ops),
    # transformer iter 0..11 (~2344 each), vae (~737).
    if not bounds:
        return "?"
    if line_idx < bounds[1] if len(bounds) > 1 else total:
        return "text_encoder"
    # Trailing component is vae
    if len(bounds) >= 13 and line_idx >= bounds[-1]:
        return "vae"
    # Middle: transformer iter 0..N
    for k, b in enumerate(bounds[1:-1] if len(bounds) >= 13 else bounds[1:], start=0):
        next_b = bounds[k + 2] if k + 2 < len(bounds) else total
        if b <= line_idx < next_b:
            return f"transformer iter {k}"
    return "?"


def main():
    print(f"loading {SEQ_LOG.name}...")
    seq_stream = load_log(SEQ_LOG)
    print(f"loading {TRI_LOG.name}...")
    tri_stream = load_log(TRI_LOG)
    n = min(len(seq_stream), len(tri_stream))
    print(f"comparing {n} aligned trace positions")

    bounds = detect_component_boundaries(seq_stream)
    print(f"component boundaries: {bounds[:6]}... ({len(bounds)} total)")

    counts = {"none": 0, "noise": 0, "borderline": 0, "real": 0, "skip": 0}
    real_ops = []          # all "real" divergence events with metadata
    borderline_ops = []
    by_op_uid_real = {}    # op_uid -> count of real divergences
    misaligned = 0
    first_sign_flip = None      # decisive bug indicator
    first_abs_gt_1 = None       # first large-magnitude divergence
    first_rel_gt_50pct = None   # first catastrophic relative divergence

    for i in range(n):
        a = seq_stream[i]
        b = tri_stream[i]
        if a is None or b is None:
            counts["skip"] += 1
            continue
        a_idx, a_uid, a_vec = a
        b_idx, b_uid, b_vec = b
        if a_idx != b_idx or a_uid != b_uid:
            misaligned += 1
            continue
        kind, max_abs, max_rel, sf = classify(a_vec, b_vec)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "real":
            real_ops.append((i, a_idx, a_uid, a_vec, b_vec,
                              max_abs, max_rel, sf))
            by_op_uid_real[a_uid] = by_op_uid_real.get(a_uid, 0) + 1
            if first_sign_flip is None and sf >= 1:
                first_sign_flip = (i, a_idx, a_uid, a_vec, b_vec,
                                   max_abs, max_rel, sf)
            if first_abs_gt_1 is None and max_abs >= 1.0:
                first_abs_gt_1 = (i, a_idx, a_uid, a_vec, b_vec,
                                  max_abs, max_rel, sf)
            if first_rel_gt_50pct is None and max_rel >= 0.5:
                first_rel_gt_50pct = (i, a_idx, a_uid, a_vec, b_vec,
                                       max_abs, max_rel, sf)
        elif kind == "borderline":
            borderline_ops.append((i, a_idx, a_uid, a_vec, b_vec,
                                    max_abs, max_rel, sf))

    print(f"\nsummary:")
    for k in ("none", "noise", "borderline", "real", "skip"):
        print(f"  {k}: {counts[k]}")
    print(f"  misaligned (op_idx/op_uid mismatch): {misaligned}")

    # Compose markdown report
    out_lines = []
    out_lines.append("# Op-by-op DAG divergence: sequential 4Kpx vs triton_sequential 4Kpx")
    out_lines.append("")
    out_lines.append("Generated by `tools/diff_dag_op_by_op.py`. No NeuroBrix changes; offline analysis of existing VALUE_TRACE logs.")
    out_lines.append("")
    out_lines.append(f"- Sequential log: `{SEQ_LOG.name}` ({len(seq_stream)} VALUE_TRACE lines)")
    out_lines.append(f"- Triton-seq log: `{TRI_LOG.name}` ({len(tri_stream)} VALUE_TRACE lines)")
    out_lines.append(f"- Compared positions: {n} aligned by `(op_idx, op_uid)`")
    out_lines.append(f"- Position-4 (n-1) excluded from metrics (known fingerprint artifact); positions 0-3 = `(0, n//8, n//4, n//2)`.")
    out_lines.append("")
    out_lines.append("## Tolerance bands")
    out_lines.append("")
    out_lines.append(f"- **noise** (fp16 quantization): max_abs ≤ {NOISE_ABS} AND max_rel ≤ {NOISE_REL}")
    out_lines.append(f"- **real divergence**: ≥1 sign flip on |seq|>{REAL_ABS_FLOOR} OR max_rel ≥ {REAL_REL}")
    out_lines.append(f"- **borderline**: between the two")
    out_lines.append("")
    out_lines.append("## Distribution")
    out_lines.append("")
    out_lines.append(f"| classification | count | % of total |")
    out_lines.append(f"|---|---|---|")
    total_compared = sum(counts[k] for k in ("none", "noise", "borderline", "real"))
    for k in ("none", "noise", "borderline", "real", "skip"):
        pct = (100.0 * counts[k] / max(total_compared, 1)) if k != "skip" else 100.0 * counts[k] / max(n, 1)
        out_lines.append(f"| {k} | {counts[k]} | {pct:.2f}% |")
    out_lines.append(f"| misaligned | {misaligned} | {100.0 * misaligned / max(n, 1):.2f}% |")
    out_lines.append("")

    out_lines.append("## TOP-20 first real divergence events (in trace order)")
    out_lines.append("")
    out_lines.append("Position 0..3 only. `max_d` = max_abs_diff, `rel` = max relative diff, `sf` = sign flips, `comp` = component.")
    out_lines.append("")
    out_lines.append("| trace_line | op_idx | op_uid | comp | max_d | rel | sf | seq[0..3] | tri[0..3] |")
    out_lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, idx, uid, sv, tv, ma, mr, sf in real_ops[:20]:
        comp = component_label(i, bounds, n)
        sv_str = "[" + ", ".join(f"{x:.4g}" for x in sv[:4]) + "]"
        tv_str = "[" + ", ".join(f"{x:.4g}" for x in tv[:4]) + "]"
        out_lines.append(f"| {i} | {idx} | `{uid}` | {comp} | {ma:.4g} | {mr:.3f} | {sf} | {sv_str} | {tv_str} |")
    out_lines.append("")

    out_lines.append("## Tiered first-event milestones")
    out_lines.append("")
    out_lines.append("Each row identifies the FIRST trace-order occurrence of a specific severity. These three milestones together narrate how drift accumulates from noise to catastrophic divergence.")
    out_lines.append("")
    out_lines.append("| milestone | trace_line | op_idx | op_uid | comp | max_d | rel | sf |")
    out_lines.append("|---|---|---|---|---|---|---|---|")
    for label, evt in (("first real (rel≥5% AND abs≥0.01) OR sign flip", real_ops[0] if real_ops else None),
                       ("first sign flip", first_sign_flip),
                       ("first max_abs ≥ 1.0", first_abs_gt_1),
                       ("first max_rel ≥ 50%", first_rel_gt_50pct)):
        if evt is None:
            out_lines.append(f"| {label} | — | — | — | — | — | — | — |")
            continue
        i, idx, uid, sv, tv, ma, mr, sf = evt
        comp = component_label(i, bounds, n)
        out_lines.append(f"| {label} | {i} | {idx} | `{uid}` | {comp} | {ma:.4g} | {mr:.3f} | {sf} |")
    out_lines.append("")

    if first_sign_flip:
        i, idx, uid, sv, tv, ma, mr, sf = first_sign_flip
        comp = component_label(i, bounds, n)
        out_lines.append("### Detail: first sign-flip op")
        out_lines.append("")
        out_lines.append(f"- **trace_line**: {i}")
        out_lines.append(f"- **op_idx**: {idx}")
        out_lines.append(f"- **op_uid**: `{uid}`")
        out_lines.append(f"- **component**: {comp}")
        out_lines.append(f"- **sequential samples**: `{sv}`")
        out_lines.append(f"- **triton_sequential samples**: `{tv}`")
        out_lines.append("")

    if first_abs_gt_1:
        i, idx, uid, sv, tv, ma, mr, sf = first_abs_gt_1
        comp = component_label(i, bounds, n)
        out_lines.append("### Detail: first max_abs ≥ 1.0")
        out_lines.append("")
        out_lines.append(f"- **trace_line**: {i}")
        out_lines.append(f"- **op_idx**: {idx}")
        out_lines.append(f"- **op_uid**: `{uid}`")
        out_lines.append(f"- **component**: {comp}")
        out_lines.append(f"- **sequential samples**: `{sv}`")
        out_lines.append(f"- **triton_sequential samples**: `{tv}`")
        out_lines.append("")

    if real_ops:
        i, idx, uid, sv, tv, ma, mr, sf = real_ops[0]
        comp = component_label(i, bounds, n)
        out_lines.append("## FIRST REAL DIVERGENCE (chronological)")
        out_lines.append("")
        out_lines.append(f"- **trace_line**: {i}")
        out_lines.append(f"- **op_idx**: {idx}")
        out_lines.append(f"- **op_uid**: `{uid}`")
        out_lines.append(f"- **component**: {comp}")
        out_lines.append(f"- **max_abs_diff (pos 0-3)**: {ma:.6g}")
        out_lines.append(f"- **max_rel_diff (pos 0-3)**: {mr:.4f}")
        out_lines.append(f"- **sign flips (pos 0-3)**: {sf}")
        out_lines.append(f"- **sequential samples**: `{sv}`")
        out_lines.append(f"- **triton_sequential samples**: `{tv}`")
        out_lines.append("")
        out_lines.append("This op is either the **bug source itself** (computes wrong on this runtime input/shape combo despite passing isolated microtests with random inputs) OR a **propagator** (faithful kernel, but receives upstream input already subtly drifted). To distinguish: walk back through the borderline-divergence ops immediately preceding this real-divergence op — if a chain of borderline accumulates into one real, the source is upstream.")
    out_lines.append("")

    out_lines.append("## Borderline divergence preceding the first real (last 10 before)")
    out_lines.append("")
    if real_ops:
        first_real_line = real_ops[0][0]
        preceding_borderline = [b for b in borderline_ops if b[0] < first_real_line][-10:]
        if preceding_borderline:
            out_lines.append("| trace_line | op_idx | op_uid | comp | max_d | rel | seq[0..3] | tri[0..3] |")
            out_lines.append("|---|---|---|---|---|---|---|---|")
            for i, idx, uid, sv, tv, ma, mr, sf in preceding_borderline:
                comp = component_label(i, bounds, n)
                sv_str = "[" + ", ".join(f"{x:.4g}" for x in sv[:4]) + "]"
                tv_str = "[" + ", ".join(f"{x:.4g}" for x in tv[:4]) + "]"
                out_lines.append(f"| {i} | {idx} | `{uid}` | {comp} | {ma:.4g} | {mr:.3f} | {sv_str} | {tv_str} |")
        else:
            out_lines.append("(none — first real divergence is not preceded by borderline drift)")
    out_lines.append("")

    out_lines.append("## Real divergences grouped by op_uid (TOP 15)")
    out_lines.append("")
    out_lines.append("| op_uid | count |")
    out_lines.append("|---|---|")
    for uid, c in sorted(by_op_uid_real.items(), key=lambda x: -x[1])[:15]:
        out_lines.append(f"| `{uid}` | {c} |")
    out_lines.append("")

    OUT_MD.write_text("\n".join(out_lines))
    print(f"\nreport written: {OUT_MD}")


if __name__ == "__main__":
    main()
