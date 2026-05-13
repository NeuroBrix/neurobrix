# P-PRISM-NEVER-REFUSE v2 — matrice cible binaire 4 configs × 4 modes — état progression

Date: 2026-05-13. Tracking the 16 cells of `4 hardware configs × 4
execution modes` mandated by P-PRISM-NEVER-REFUSE v2. State synthesises
S1 through S5 acquis with explicit honest accounting for upstream-blocked
cells.

Configs:
- `32g` — 1× V100 32 GiB
- `16g` — 1× V100 16 GiB
- `2×16g` — 2× V100 16 GiB
- `cpu` — pure CPU (cpu-only-x86, 40-core Xeon, 256 GiB)

Modes: `compiled` / `sequential` / `triton` / `triton_sequential`.

## State (2026-05-13, post S3 closure)

| Config × Mode      | compiled | sequential | triton | triton_sequential |
|--------------------|---|---|---|---|
| 32g                | ✓ | ✓ | ✓ | ✓ |
| 16g                | ✓ (S1 hybrid) | ✓ (S1 hybrid) | ⏳ S5 | ⏳ S5 |
| 2×16g              | ⏳ S4 | ⏳ S4 | ⏳ S4 | ⏳ S4 |
| cpu                | ✓ S1 | ✓ S2 | ⏸ S3 upstream | ⏸ S3 upstream |

Legend: ✓ validated · ⏳ pending sub-chantier · ⏸ upstream prerequisite
missing (escalated per mandate "épuisement technique" clause)

### Achievable within this mandate: **14/16**

The two ⏸ cells (`cpu × triton`, `cpu × triton_sequential`) are
upstream-blocked and excluded honestly per the mandate's escalation
clause. See "Upstream-blocked cells" below.

## Sub-chantier history

### S1 — P-RUNTIME-HYBRID-DEVICE-DISPATCH (closed)

Commit `de5fb9e`. Hybrid CPU+GPU placement (`lazy_sequential`)
unblocked 16 GiB compiled + sequential cells. CPU-pure compiled cell
validated at the same time (TinyLlama haiku, Sana 1024 red apple).

### S2 — P-NATIVE-SEQUENTIAL-CPU-DEBUG (closed)

Commits `8b4d020` (fix), `a3d2248` (anti-régression matrix). RoPE
cache slice end fix in `_patch_seq_len_in_ops` + new
`_adapt_seq_dependent_weights` mirror of `update_seq_dependent_-
constants`. Unblocks `--sequential` autoregressive decode for any
model with `cos_cached` / `sin_cached` named buffers. TinyLlama
sequential validated CPU + GPU; full anti-régression matrix
(compiled / sequential / triton-sequential + Sana 1024 / PixArt-XL)
all green. Artefacts: `validation_outputs/p_prism_never_refuse_s2/`.

### S3 — P-TRITON-CPU-INTEGRATION (closed at Stage 1; escalated)

Commits `b3e479f` (install gate + coverage doc + initial docs),
`d0974e6` (docs corrected to build-from-source reality).

**Closure verdict**: Stage 1 (gate + docs + coverage notes) shipped.
Stages 2-5 (DeviceAllocator CPU branch, wrapper routing,
TritonSequence/Sequential CPU awareness, R29 validation matrix) NOT
implemented.

**Escalation rationale** (per mandate clause "épuisement technique"):
- Upstream `triton-lang/triton-cpu` has NO PyPI wheel (verified by
  feasibility probe + WebFetch on the upstream README, 2026-05-13).
  Install is build-from-source only: `git clone + LLVM build + 30
  min triton-cpu build`, ~15 GB peak disk, with open upstream issue
  #233 (torch 2.6+ build incompatibility) risking failure outright.
- Total real time on Dell with no guarantee ≈ 2-3 h. Validation on
  one host would not generalise to open-source users without that
  same heavyweight build, so success on one Dell environment is not
  a reproducible S3 closure.
- Implementing S3 stages 2-5 blind against the upstream contract
  without a working triton-cpu install = fictitious chantier
  pattern interdit par doctrine (R2).
- NeuroBrix will NOT publish triton-cpu wheels under its own name
  (R25: no internal fork / vendor-strict).
- Numerical gap on fp16 (triton-cpu #147 Dot3D, #222 masked GEMM)
  is independently blocked upstream regardless of wheel state.

Both ⏸ cells in the matrix carry this rationale via the backlog
chantier `P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP` (see Backlog
below).

### S5 — P-DC-AE-RESIDUAL-CHAIN-TILING (NEXT)

Scope: 16 GiB single-GPU cells (compiled / sequential / triton /
triton_sequential) for Sana 4Kpx. Refactor the DC-AE residual chain
to free 3× 4 GiB co-resident tensors (silu::24 op 707 site:
`add::86::out_0` + `conv::63::out_0` + `silu::24::out_0`). Estimated
300-500 lines.

**Reordered before S4** based on the factual VAE-alone peak
measurement at commit `e88eca0` (Q4 case 2): VAE intrinsic peak
> 16 GiB regardless of placement, so 2×16 GiB cells (S4 Gap A
target) cannot be unlocked by inter-component placement alone —
the VAE first has to fit within any single 16 GiB GPU. See
`p_prism_never_refuse_v2_s4_measurement.md`.

### S4 — P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT (after S5)

Scope: 4 cells of `2×16g × {compiled, sequential, triton,
triton_sequential}`. **Gap A only** per mandate Q4 decision —
make Prism's solver / activation estimator produce a real
multi-GPU `component_placement` (cuda:0 + cuda:1) when N devices
≥ 2 and each component fits individually, instead of the current
cascade landing on `component_placement_lazy` with all components
on cuda:0. Activation peak per component is the discriminating
input. Estimated 150-300 lines in Prism solver. Gap B (per-op
cross-device split) is explicitly out of scope and tracked as
backlog `P-OP-LEVEL-CROSS-DEVICE-SPLIT`.

## Upstream-blocked cells (S3 escalation rows)

| Cell | Status | Root cause |
|------|--------|-----------|
| `cpu × triton` | ⏸ | upstream `triton-cpu` no wheel + fp16 Dot3D #147 |
| `cpu × triton_sequential` | ⏸ | identical to above |

Alternative for end-users today: `--compiled` mode on a CPU profile.
Mature PyTorch CPU backend; both cells already proven via S1 (TinyLlama
haiku + Sana 1024 red apple on cpu-only-x86).

## Backlog (chantiers opened by this mandate, not part of v2 closure)

- **P-DC-AE-RESIDUAL-CHAIN-TILING** — S5 work (300-500 lines).
- **P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP** — triggers when EITHER
  (a) upstream `triton-lang/triton-cpu` publishes PyPI wheels for
  the platforms NeuroBrix targets, OR (b) NeuroBrix makes a
  separate doctrinal decision to embark an optional backend that
  requires end-user build-from-source. Re-tests S3 stages 2-5 at
  that point. Also re-tests upstream issues #147, #222 and flips
  the marker constants in `src/neurobrix/triton/cpu_backend.py`
  when they close.
- **P-TRITON-CPU-FP16-UPSTREAM-FOLLOWUP** — folded into the
  P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP above (same monitoring
  surface: upstream `triton-cpu` issue tracker). One re-test
  cadence, not two.
- **P-OP-LEVEL-CROSS-DEVICE-SPLIT** — Gap B, opened by S4 doctrine
  decision Q4 (option a). Per-op cross-device split for models
  whose individual components do NOT fit in any single available
  GPU (e.g. Qwen3-30B-A3B class on 2× consumer 16 GiB). PCIe
  activation transfer per op = perf regression on non-NVLink
  hosts → must remain a last-resort fallback, not a standard
  feature. Open when a concrete model demands it.

## Reference — upstream issues citation index

The S3 escalation rationale relies on several `triton-cpu` open
upstream issues. Their durable home is `S3_READINESS_AND_PLAN.md`
under the same directory tree (one level under
`validation_outputs/p_prism_never_refuse_s2/`) — keep that file even
when S3 closes so the references survive. Quick map:

- triton-cpu #147 — fp16 Dot3D accuracy gap (open).
- triton-cpu #222 — `make_block_ptr` GEMM does not handle masks.
- triton-cpu #229 — AVX512-BF16 matmul perf / tuning (perf only).
- triton-cpu #233 — torch 2.6+ build incompatibility on the
  build path (install-time only).

## Next session entry point

1. Read `validation_outputs/p_prism_never_refuse_v2_s4_measurement.md`
   for the empirical reorder rationale.
2. Begin S5 by inspecting the Sana 4Kpx VAE DAG at
   `~/.neurobrix/cache/Sana_1600M_4Kpx_BF16/components/vae/graph.json`
   around `silu::24` (execution_order index ≈ 707) to map the full
   residual chain bounds: which `conv::*` op forks off, which
   intermediate `silu::*` ops bridge it, where `add::*` merges back,
   and what tensor shapes are at play. Project memory has the names
   (`add::86::out_0`, `conv::63::out_0`, `silu::24::out_0`) but the
   full chain shape signature and bounds need DAG-level confirmation
   before writing detection code. Treat the project-memory pointer
   as a starting probe, not a complete pattern spec.
3. The existing `OpLevelTilingEngine` (`core/module/tiling_engine.py`)
   already supports fusion pairs, single-op tiling, in-place
   residual adds, and pixel-shuffle broadcast chains via
   `_detect_pixel_shuffle_broadcast_chains`. S5 adds a fifth
   pattern: long-chain band-streaming. Mirror that file's existing
   detection structure for the new chain pattern; then add the
   tiled wrapper(s) under `kernels/ops/`. R30 / R33 / R34 all apply
   (triton mode mirrors required, no torch in triton path,
   detection by structural signature not by model name).
4. Anti-regression matrix to preserve after S5: Sana 1024 BF16 ×4
   modes, PixArt-XL/Sigma, TinyLlama, Sana 4Kpx 32g (the run that
   produced `p_prism_never_refuse_v2_s4_sana4kpx_32g_reference.png`
   — that PNG is the numerical-equivalence target for S5).
