"""Cross-variant DAG divergence: Sana 4Kpx vs Sana 1024.

The single-variant `diff_dag_op_by_op.py` flagged 53% of ops as "real"
divergence on BOTH Sana 4Kpx (PNG garbage) AND Sana 1024 (PNG coherent).
That confirms the 5-position fingerprint threshold catches baseline
cuDNN-vs-Triton fp16 noise indistinguishably from real bug-amplifying
divergence.

This tool isolates the SHAPE-DEPENDENT signal by comparing per-op
divergence statistics between the two variants. For each op_uid that
appears in both, we compare the maximum relative divergence (sequential
vs triton_seq) at Sana 4Kpx versus at Sana 1024:

  rel_4kpx / rel_1024 > 10  →  this op shows shape-dependent extra
                                divergence beyond baseline noise; it's
                                a candidate bug source or amplifier.
  ratio ~ 1                 →  op behaves the same numerically across
                                resolutions; benign noise.

For each op_uid, we keep the MAX over all its occurrences in each
variant (the worst-case divergence). The narrative of "where does
4Kpx-specific extra divergence first appear in trace order" is the
real bug signature.

Inputs (4 logs, all post-fingerprint-fix):
  - sequential_4kpx_values_v2.log        (Sana 4Kpx eager / cuDNN)
  - sana4kpx_triseq_values_v5.log        (Sana 4Kpx Triton)
  - sequential_1024_values_clean.log     (Sana 1024 eager / cuDNN)
  - triton_seq_1024_values_clean.log     (Sana 1024 Triton)

Output: validation_outputs/.../diff_dag_cross_variant_report.md
"""
import re
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System")
LOG_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05"

LOGS = {
    "4kpx_seq": LOG_DIR / "sequential_4kpx_values_v2.log",
    "4kpx_tri": LOG_DIR / "sana4kpx_triseq_values_v5.log",
    "1024_seq": LOG_DIR / "sequential_1024_values_clean.log",
    "1024_tri": LOG_DIR / "triton_seq_1024_values_clean.log",
}
OUT_MD = LOG_DIR / "diff_dag_cross_variant_report.md"

LINE_RE = re.compile(
    r"VALUE_TRACE path=\S+ op_idx=(\d+) op_uid=(\S+) samp=(.+?)\]\s*$"
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse(line: str):
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


def load(path: Path):
    out = []
    for line in path.open():
        if "VALUE_TRACE" not in line:
            continue
        e = parse(line)
        if e is not None:
            out.append(e)
    return out


def per_op_max_divergence(seq_stream, tri_stream):
    """Return dict {(op_idx, op_uid): (max_abs, max_rel,
    first_trace_line)} computing max over all aligned occurrences."""
    out = {}
    n = min(len(seq_stream), len(tri_stream))
    for i in range(n):
        a = seq_stream[i]
        b = tri_stream[i]
        if a is None or b is None:
            continue
        a_idx, a_uid, a_vec = a
        b_idx, b_uid, b_vec = b
        if a_idx != b_idx or a_uid != b_uid:
            continue
        if a_vec is None or b_vec is None:
            continue
        if len(a_vec) < 4 or len(b_vec) < 4:
            continue
        # Use positions 0-3 only.
        a4 = a_vec[:4]
        b4 = b_vec[:4]
        abs_diffs = [abs(x - y) for x, y in zip(a4, b4)]
        rel_diffs = [abs(x - y) / max(abs(x), 0.01)
                     for x, y in zip(a4, b4)]
        max_abs = max(abs_diffs)
        max_rel = max(rel_diffs)
        key = (a_idx, a_uid)
        prev = out.get(key)
        if prev is None or max_abs > prev[0]:
            out[key] = (max_abs, max_rel, i)
    return out


def detect_first_component_boundary(stream):
    """Find first op_idx reset after text_encoder = transformer iter 0
    boundary; useful for locating where bug-relevant ops live."""
    last_idx = -1
    for i, e in enumerate(stream):
        if e is None:
            continue
        idx = e[0]
        if idx == 0 and last_idx > 100:
            return i
        last_idx = idx
    return len(stream)


def main():
    print("loading 4 logs...")
    streams = {k: load(p) for k, p in LOGS.items()}
    for k, s in streams.items():
        print(f"  {k}: {len(s)} VALUE_TRACE lines")

    print("computing per-op max divergence (4Kpx)...")
    div_4kpx = per_op_max_divergence(streams["4kpx_seq"], streams["4kpx_tri"])
    print(f"  unique (op_idx, op_uid) pairs: {len(div_4kpx)}")
    print("computing per-op max divergence (1024)...")
    div_1024 = per_op_max_divergence(streams["1024_seq"], streams["1024_tri"])
    print(f"  unique (op_idx, op_uid) pairs: {len(div_1024)}")

    # Cross-variant comparison: for each op present in both, compute
    # rel_ratio = max_rel_4kpx / max(max_rel_1024, 1e-6).
    common_keys = set(div_4kpx.keys()) & set(div_1024.keys())
    cross = []
    for key in common_keys:
        a4, r4, line4 = div_4kpx[key]
        a1, r1, line1 = div_1024[key]
        rel_ratio = r4 / max(r1, 1e-6)
        abs_ratio = a4 / max(a1, 1e-6)
        cross.append({
            "op_idx": key[0],
            "op_uid": key[1],
            "rel_4kpx": r4,
            "rel_1024": r1,
            "abs_4kpx": a4,
            "abs_1024": a1,
            "rel_ratio": rel_ratio,
            "abs_ratio": abs_ratio,
            "line_4kpx": line4,
        })

    # Order by trace_line to find FIRST op with anomalously high
    # 4Kpx divergence vs its 1024 counterpart.
    cross_sorted_by_line = sorted(cross, key=lambda x: x["line_4kpx"])

    # Filter: shape-specific bugs have rel_ratio >= 10 AND
    # rel_4kpx >= 0.5 (not just amplified noise on tiny values).
    significant = [c for c in cross_sorted_by_line
                   if c["rel_ratio"] >= 10.0 and c["rel_4kpx"] >= 0.5]

    # Also: ops with absolute divergence in 4Kpx that 1024 doesn't have
    # (abs_4kpx > 1.0 AND abs_1024 < 0.01).
    abs_specific = [c for c in cross_sorted_by_line
                    if c["abs_4kpx"] >= 1.0 and c["abs_1024"] < 0.01]

    # Top by rel_ratio (regardless of trace order) to see the most
    # shape-dependent ops.
    top_by_ratio = sorted([c for c in cross if c["rel_4kpx"] >= 0.5],
                          key=lambda x: -x["rel_ratio"])[:20]

    # Locate where transformer iter 0 starts in 4Kpx trace.
    te_end_4kpx = detect_first_component_boundary(streams["4kpx_seq"])

    out_lines = []
    out_lines.append("# Cross-variant DAG divergence: Sana 4Kpx vs Sana 1024")
    out_lines.append("")
    out_lines.append("Generated by `tools/diff_dag_cross_variant.py`. Compares the divergence pattern (sequential vs triton_seq) between Sana 4Kpx (PNG garbage) and Sana 1024 (PNG coherent), per op_uid.")
    out_lines.append("")
    out_lines.append("**Why this view**: the single-variant analysis flagged ~53% of ops as 'real' divergence on BOTH variants — i.e. cuDNN-vs-Triton fp16 noise is the dominant signal. Shape-specific bugs are visible only as **delta between variants**: ops where 4Kpx divergence is >> 1024 divergence.")
    out_lines.append("")
    out_lines.append("Position 0..3 only. `rel` = max_rel_diff over an op's occurrences. `rel_ratio = rel_4kpx / rel_1024`.")
    out_lines.append("")
    out_lines.append(f"- text_encoder ends at 4Kpx trace line {te_end_4kpx} (transformer iter 0 starts there)")
    out_lines.append(f"- common (op_idx, op_uid) pairs across variants: {len(common_keys)}")
    out_lines.append("")
    out_lines.append("## Significant shape-specific divergences (rel_ratio ≥ 10 AND rel_4kpx ≥ 0.5), in trace order")
    out_lines.append("")
    out_lines.append(f"Count: **{len(significant)}**.")
    out_lines.append("")
    if significant:
        out_lines.append("| 4Kpx_line | op_idx | op_uid | rel_4kpx | rel_1024 | rel_ratio | abs_4kpx | abs_1024 |")
        out_lines.append("|---|---|---|---|---|---|---|---|")
        for c in significant[:30]:
            out_lines.append(f"| {c['line_4kpx']} | {c['op_idx']} | `{c['op_uid']}` | {c['rel_4kpx']:.3f} | {c['rel_1024']:.3f} | {c['rel_ratio']:.1f} | {c['abs_4kpx']:.4g} | {c['abs_1024']:.4g} |")
    out_lines.append("")
    out_lines.append("## Absolute-magnitude shape-specific divergences (abs_4kpx ≥ 1.0 AND abs_1024 < 0.01)")
    out_lines.append("")
    out_lines.append(f"Count: **{len(abs_specific)}**. These are ops where 4Kpx shows large absolute drift but 1024 stays quiet.")
    out_lines.append("")
    if abs_specific:
        out_lines.append("| 4Kpx_line | op_idx | op_uid | abs_4kpx | abs_1024 | rel_4kpx | rel_1024 |")
        out_lines.append("|---|---|---|---|---|---|---|")
        for c in abs_specific[:30]:
            out_lines.append(f"| {c['line_4kpx']} | {c['op_idx']} | `{c['op_uid']}` | {c['abs_4kpx']:.4g} | {c['abs_1024']:.4g} | {c['rel_4kpx']:.3f} | {c['rel_1024']:.3f} |")
    out_lines.append("")
    out_lines.append("## TOP-20 ops by rel_ratio (most shape-dependent, regardless of trace order)")
    out_lines.append("")
    if top_by_ratio:
        out_lines.append("| op_idx | op_uid | rel_4kpx | rel_1024 | rel_ratio | abs_4kpx | abs_1024 |")
        out_lines.append("|---|---|---|---|---|---|---|")
        for c in top_by_ratio:
            out_lines.append(f"| {c['op_idx']} | `{c['op_uid']}` | {c['rel_4kpx']:.3f} | {c['rel_1024']:.3f} | {c['rel_ratio']:.1f} | {c['abs_4kpx']:.4g} | {c['abs_1024']:.4g} |")
    out_lines.append("")

    if significant:
        c = significant[0]
        out_lines.append("## FIRST shape-specific divergence (chronological)")
        out_lines.append("")
        out_lines.append(f"- **trace_line (4Kpx)**: {c['line_4kpx']}")
        out_lines.append(f"- **op_idx**: {c['op_idx']}")
        out_lines.append(f"- **op_uid**: `{c['op_uid']}`")
        out_lines.append(f"- **max_rel_diff at 4Kpx**: {c['rel_4kpx']:.4f}")
        out_lines.append(f"- **max_rel_diff at 1024**: {c['rel_1024']:.4f}")
        out_lines.append(f"- **rel_ratio (4Kpx / 1024)**: {c['rel_ratio']:.2f}")
        out_lines.append(f"- **max_abs_diff at 4Kpx**: {c['abs_4kpx']:.4g}")
        out_lines.append(f"- **max_abs_diff at 1024**: {c['abs_1024']:.4g}")
        if c["line_4kpx"] >= te_end_4kpx:
            out_lines.append("- **component**: transformer/vae (post text_encoder)")
        else:
            out_lines.append("- **component**: text_encoder")
        out_lines.append("")
        out_lines.append("This is the FIRST op in trace order where Sana 4Kpx triton diverges from its sequential reference SIGNIFICANTLY MORE than Sana 1024 triton diverges from its sequential reference. By construction, this excludes baseline cuDNN-vs-Triton fp16 noise common to both variants and isolates the shape-dependent bug signature.")
    else:
        out_lines.append("## NO significant shape-specific divergence found")
        out_lines.append("")
        out_lines.append("With thresholds rel_ratio ≥ 10 AND rel_4kpx ≥ 0.5, no op shows 4Kpx-specific extra divergence beyond the 1024 baseline. This suggests the bug is NOT in any individual op's numerics but in how outputs assemble — final layout, RNG, stride reshape, scheduler integration. Recommend inspecting the transformer→VAE boundary or VAE final convolution.")
    out_lines.append("")

    OUT_MD.write_text("\n".join(out_lines))
    print(f"\nreport: {OUT_MD}")


if __name__ == "__main__":
    main()
