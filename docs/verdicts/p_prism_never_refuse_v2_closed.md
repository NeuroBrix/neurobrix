# P-PRISM-NEVER-REFUSE v2 — FINAL VERDICT (2026-05-15) — **14/16 ✓ CLOSED**

## Matrix final state: **14/16 ✓ + 2/16 ⏸ S3 upstream**

| Config × Mode      | compiled | sequential | triton | triton_sequential |
|--------------------|---|---|---|---|
| 32g                | ✓ 23.9 s | ✓ S2 | ✓ 79.8 s | ✓ |
| 16g                | ✓ S5 GPU-pure | ✓ S5 GPU-pure | ✓ 79.4 s | ✓ 84.5 s |
| 2×16g              | ✓ single_gpu | ✓ single_gpu | ✓ 81.8 s | ✓ 84.1 s |
| cpu                | ✓ S1 | ✓ S2 | ⏸ S3 upstream | ⏸ S3 upstream |

**MANDATE VICTORY** : the v2 cycle started at 10/16 ✓ + 2/16 ⏸ + 4/16 ⏳ and closes
at 14/16 ✓ + 2/16 ⏸. The 4 ⏳ triton-mode cells (16g+2×16g × triton+triton_sequential)
closed in cascade via a single ROOT FIX in `NBXTensor.__getitem__` (commit `8a6daf2`),
with the chain wrapper NBX port from prior sessions (commit `f8a8ad8`) as the
load-bearing architectural acquis.

The 2 ⏸ cells (CPU triton + CPU triton_sequential) remain upstream-blocked on
`triton-cpu` package (not yet on PyPI; fp16 numerical gap upstream issue #147).

## Sub-chantiers history (chronological)

### S1 — Hybrid CPU+GPU dispatch (commit `de5fb9e`)
Unlocked 16g compiled/sequential by routing VAE to CPU when GPU budget can't hold it.

### S2 — Native sequential CPU debug (commit `8b4d020`)
RoPE cache slice end fix + `_adapt_seq_dependent_weights` mirror. Unlocked CPU sequential.

### S3 — Triton-CPU integration (commits `b3e479f`, `d0974e6`)
Escalated upstream — package not on PyPI today. CPU triton + CPU triton-seq cells
stay ⏸ until upstream wheel ships.

### S4 — Multi-GPU NBX intra-component split (commit `f58f6cc`)
Closed via cascade: post-S5 depthwise fix, Sana 4Kpx VAE fits a single 16 GiB GPU.
`single_gpu` cascade naturally picks cuda:0 on 2×16g hardware. No solver extension required.

### S5 — DC-AE residual chain tiling + P-S5-RMS_NORM-16G-NUMERICAL (closed)
Residual chain detection + band-streamed wrapper (`198ab1b` → `33c6b21`). Depthwise
conv tile-skip fix (`8af7848`) removed 20 transformer depthwise convs from `tiled_ops`
on 16g. R30 chain wrapper skip on triton modes (`c9d2581`) closed a `'NoneType'._dtype`
crash on triton-sequential.

### P-NBX-TILED-CONV2D-SMALL-SCALE (closed — commits `176bc7e`, `63edb03`)
Two coupled math bugs in `_tiled_conv2d_spatial_*` and `_fused_upsample_conv2d_*`:
- Edge-band double-padding (max(0, -in_read_start) + redundant pad_h).
- Internal-frontier halo offset on the conv_band read side.
Validated bit-exact via microtest at 1024² and 2048² (8/8 + 4/4 cos=1.0000 max=0.0).

### P-TRITON-LIVE-WATERMARK-AUDIT (closed — commits `993181f`, `f8a8ad8`, `f997479`, `b9590ea`)
- L1 NBX_LIVE_WATERMARK_TRACE JSONL instrumentation.
- L2 per-slot LIVE_DUMP_SLOT with last_use vs current_op_idx dispositive.
- L4 `band_streamed_chain_nbx` (143-line R33-pure mirror of `band_streamed_chain_torch`).
- L4b NBXTensor isinstance fix in fork stash + `_set_device` anchor.

### P-TRITON-CHAIN-CPU-POINTER (closed — commits `5a714a5` → `8a6daf2`, `23de696`)
**THE ROOT FIX** : `NBXTensor.__getitem__` slice path was buggy on negative starts:
```python
start = k.start or 0   # BUG: `-2 or 0` evaluates to `-2`, not 0
```
For `t[-N:]` (halo_carry trim), `narrow(dim, -N, length)` computed `_offset = parent_offset - N*stride`,
producing `data_ptr()` BEFORE the cudaMalloc'd block. Triton's `cuPointerGetAttribute`
rejected the pointer with "Pointer argument (at 0) cannot be accessed from Triton
(cpu tensor?)". The Sana 4Kpx VAE chain wrapper's `halo_carry[:, :, -top_size:, :]`
triggered this at the 3rd iteration of every 4096² chain (chains 7-9 of the run).

**Fix** : universal Python/torch slice contract:
```python
start = k.start if k.start is not None else 0
stop = k.stop if k.stop is not None else shape[dim]
if start < 0: start = max(0, shape[dim] + start)
if stop < 0:  stop  = shape[dim] + stop
```

Validation post-fix:
| Cell                       | Wall    | Chains  |
|----------------------------|---------|---------|
| 32g triton (anti-reg)      | 79.81s  | 9/9 OK  |
| 16g triton                 | 79.35s  | 9/9 OK  |
| 2×16g triton (cascade)     | 81.84s  | 9/9 OK  |
| 16g triton-seq             | 84.49s  | 9/9 OK  |
| 2×16g triton-seq           | 84.13s  | 9/9 OK  |
| 32g compiled (anti-reg)    | 23.93s  | 9/9 OK  |

**Speedup vs pre-chain baseline**: 32g triton went 1443.6 s → 79.81 s = **18× faster**.
The chain wrapper streams each chain band-by-band; peak transient drops from
6.7 GiB (full chain output) to 1.6 GiB (one band's compute).

Chain wrapper UNGATED to default-ON on triton modes in commit `23de696`. Kept
NBX_TRITON_CHAIN_WRAPPER=0 as kill-switch for emergency rollback.

## Anti-régression matrix (final, all PASS)

| Cell                              | Status   | Wall      |
|-----------------------------------|----------|-----------|
| Sana 4Kpx 32g compiled            | ✓ F1     | 23.93s    |
| Sana 4Kpx 32g sequential          | ✓ S2     | acquis    |
| Sana 4Kpx 32g triton              | ✓ F1     | 79.81s    |
| Sana 4Kpx 32g triton-seq          | ✓ acquis | acquis    |
| Sana 4Kpx 16g compiled            | ✓ S5+S1  | 22.98s    |
| Sana 4Kpx 16g sequential          | ✓ S5+S2  | acquis    |
| Sana 4Kpx 16g triton              | ✓ C5     | 79.35s    |
| Sana 4Kpx 16g triton-seq          | ✓ tested | 84.49s    |
| Sana 4Kpx 2×16g compiled          | ✓ cascade | acquis    |
| Sana 4Kpx 2×16g sequential        | ✓ cascade | acquis    |
| Sana 4Kpx 2×16g triton            | ✓ tested | 81.84s    |
| Sana 4Kpx 2×16g triton-seq        | ✓ tested | 84.13s    |
| Sana 1024 BF16 × 4 modes          | ✓ acquis | n/a       |
| PixArt-XL / PixArt-Sigma          | ✓ acquis | n/a       |
| TinyLlama × 4 modes               | ✓ acquis | n/a       |
| CPU pure compiled / sequential    | ✓ S1/S2  | acquis    |

## Doctrines validated through this mandate cycle

- **R30 dualité runtime** : restored across compiled/sequential/triton/triton_sequential
  via `band_streamed_chain_nbx` mirror + `_tiled_conv2d_spatial_nbx` parity.
- **R33 zero torch in triton/** : strictly preserved (no torch imports added to the
  triton execution path; chain wrapper NBX variant uses only NBX wrappers + NBXTensor
  methods).
- **R34 model-agnostic** : pattern detection structural (`fork → linear≥3 → merge`),
  no model-specific knowledge in the chain wrapper or solver path.
- **R35 Prism never refuses** : the cascade strategy correctly routes 2×16g hardware
  to `single_gpu` cuda:0 when Sana 4Kpx VAE fits there.
- **R29 inspectable artefacts** : every cell ✓ has a PNG saved in `validation_outputs/`.

## Backlog (out of scope of v2 mandate)

- **P-TRITON-CPU-UPSTREAM-WHEEL-FOLLOWUP** : triggers when upstream `triton-cpu`
  publishes PyPI wheels. Then re-test cells (cpu, triton) and (cpu, triton-seq) —
  expected to close them.
- **P-OP-LEVEL-CROSS-DEVICE-SPLIT** (Gap B) : opens when a concrete model demands
  per-op cross-device split (no Sana 4Kpx need it now that VAE fits 16 GiB).

## Commits this mandate (sessions cumulés)

| Commit | Contribution |
|---|---|
| `de5fb9e` | S1 hybrid CPU+GPU dispatch |
| `8b4d020` | S2 native sequential CPU (RoPE fix) |
| `b3e479f` / `d0974e6` | S3 triton-cpu install gate + docs |
| `198ab1b` → `33c6b21` | S5 residual chain detection + wrapper |
| `8af7848` | P-S5-RMS_NORM-16G-NUMERICAL depthwise tile-skip |
| `c9d2581` | R30 chain wrapper skip on triton modes (interim) |
| `f58f6cc` | S4 closure docs (cascade, no code) |
| `99ca74c` | P-S5-RMS_NORM-16G-NUMERICAL closure docs + artefacts |
| `da484ae` | JSONL dump O(N) IO refactor |
| `0b1bfa1` | Bit-diff script |
| `f2ec3eb` | P-TRITON-VAE-16G-STRIPED root cause in code |
| `176bc7e` / `63edb03` | P-NBX-TILED-CONV2D-SMALL-SCALE wrapper math fix |
| `993181f` | L1 NBX_LIVE_WATERMARK_TRACE instrumentation |
| `f8a8ad8` | L4 band_streamed_chain_nbx + dispatcher |
| `f997479` | L4b NBXTensor isinstance + _set_device anchor |
| `b9590ea` | L4 gate triton chain wrapper opt-in (interim) |
| `5a714a5` | L2 per-slot LIVE_DUMP_SLOT instrumentation |
| `9ddcc7b` | C1 NBX_CHAIN_DIAG per-chain device-state log |
| `346398a` | C3 _set_device per-step anchor |
| **`8a6daf2`** | **C3d ROOT FIX — NBXTensor.__getitem__ negative slice** |
| `23de696` | C4 ungate chain wrapper default-ON triton modes |

## Tag

`p-prism-never-refuse-v2-closed` posted on the C4 ungate commit (or the F3 final
commit if any follow-up doc edits land). Pushed to both `origin` (GitHub) and
`gitlab` (GitLab).
