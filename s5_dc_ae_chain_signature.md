# S5 — DC-AE residual chain signature (factual DAG inspection)

Performed on commit `06c0a44`. Source DAG:
`~/.neurobrix/cache/Sana_1600M_4Kpx_BF16/components/vae/graph.json`.

## 1. Empirical peak in the VAE DAG

Static liveness analyzer over the full execution_order. Tensor sizes
computed from `tensors[tid].shape × dtype_size`. graph.json reports
**fp32** shapes — at runtime fp16/bf16 these are halved.

Top peak (fp32 in graph.json) = **28.0 GiB at exec idx 699
(aten.convolution::62)**. Runtime bf16 = **14.0 GiB**. Three co-resident
tensors at peak:

| Tensor | Shape | fp32 GiB | bf16 GiB |
|---|---|---|---|
| `aten.upsample_nearest2d::4::out_0` | [1, 256, 4096, 4096] | 16.0 | 8.0 |
| `aten.convolution::62::out_0`        | [1, 128, 4096, 4096] | 8.0  | 4.0 |
| `aten.add::85::out_0`                | [1, 256, 2048, 2048] | 4.0  | 2.0 |

This is NOT the silu::24 op-707 site noted in earlier project memory
— that site has a different signature (see section 2). Both sites are
real peaks in the same VAE; this is the bigger one.

## 2. Two distinct chain patterns surfaced

### Pattern A — upsample→conv adjacency (idx 698-699)

```
add::85  ──[698]→  upsample_nearest2d::4  ──[699]→  convolution::62
                          (8 GiB bf16)              (4 GiB bf16 — last 4096²
                                                     channel-reducing conv)
```

Already a **fusion-pair target** of the existing
`OpLevelTilingEngine.fusion_pairs` mechanism — `solver.py:_detect_op_
level_tiling_pairs` produces `(upsample_uid, conv_uid, tile_factor)`
entries for this exact adjacency when the upsample output exceeds
0.25 × component_vram. When the fusion is wired (interceptors active),
the 8 GiB upsample buffer is replaced by per-band 8/tile_factor GiB,
and peak at this site drops from 14 GiB (bf16) to ~6.5 GiB +
band-transient.

**Open question**: on `--hardware v100-16g`, Prism's cascade lands on
`lazy_sequential` with VAE → cpu (S1 hybrid). For `--compiled` and
`--sequential` this is the answer (VAE runs on PyTorch CPU). For
`--triton` this violates R33 (zero torch in triton). The triton 16g
cell needs Prism to NOT fall back to a CPU-placed VAE — instead, try
`single_gpu` + op-level fusion of the upsample→conv pairs.

### Pattern B — residual fork→long-chain→merge (idx 705-711)

```
                pixel_shuffle::4 ──┐
                                   ├─[705]→ add::86 ───┐   (residual fork)
convolution::62 ───────────────────┘                   │
                                                       │
            [706]→ convolution::63 ←─────────────────────┘
                            │
            [707]→ silu::24
                            │
            [708]→ convolution::64
                            │
            [709]→ permute::84 (NCHW → NHWC)
                            │
            [710]→ rms_norm::24
                            │
            [711]→ add::88 ←──── add::86::out_0 (residual merge)
```

All intermediate tensors at this site share shape **[1, 128, 4096, 4096]
= 4.0 GiB bf16 each**. At op 707 (silu::24), three of them are
simultaneously live: `add::86::out_0` (kept alive until op 711 for the
residual merge), `convolution::63::out_0` (the chain step in), and
`silu::24::out_0` (the chain step out). **Co-residence at op 707 = 12
GiB bf16**.

This is what S5 must address. None of the existing tiling_engine
patterns target it:
- `fusion_pairs` requires an upsample→conv adjacency
- `tiled_ops` is single-op tiling (doesn't reach the 3-way co-residence
  problem)
- `inplace_adds` could reuse `add::86::out_0` as the output buffer of
  `add::88` — saves 4 GiB on the output side but does NOT solve the
  3-way mid-chain co-residence
- `_detect_pixel_shuffle_broadcast_chains` already detected and absorbed
  the `expand→clone→view→pixel_shuffle` decomposition upstream (idx
  701-704), so `pixel_shuffle::4::out_0` is a sentinel proxy and NOT
  contributing to the peak; this is acquis, not the gap

## 3. Pattern B universal signature for R34-conforming detection

The detection criterion must be structural, not Sana-specific:

> **A "long residual chain" is a subgraph that satisfies all of:**
>
> 1. A producer op (the **fork**) whose output `T_base` is consumed by
>    BOTH the first op of a linear chain AND the last op of that chain
>    (the **merge** — an `aten::add` consuming `T_base` as one of its
>    two inputs).
> 2. Between fork and merge: a linear chain of N ≥ 3 ops where each op
>    has a single consumer (the next op in the chain), and the last op
>    in the chain feeds the merge's other input.
> 3. All intermediate tensors in the chain (including `T_base` and the
>    merge output) have spatial output bytes ≥ **threshold** (suggested
>    threshold: `0.10 × component_vram`, so on a 16 GiB GPU the chain
>    triggers when intermediate ≥ ~1.6 GiB).
> 4. All chain intermediate shapes are **spatially band-tileable**:
>    same H, W as the merge output (no spatial reduction or upsample
>    inside the chain — those would belong to a different chain
>    pattern).

In Sana 4Kpx VAE:
- Fork = `add::86` (idx 705), output [1,128,4096,4096], 4 GiB bf16
- Chain ops = `convolution::63` (706), `silu::24` (707),
  `convolution::64` (708), `permute::84` (709), `rms_norm::24` (710)
- Merge = `add::88` (idx 711), output [1,128,4096,4096], 4 GiB bf16
- All intermediate shapes consistent at [1,128,4096,4096], 4 GiB each
- N = 5 chain ops, all single-consumer

This same signature appears at 2048², 1024², 512² resolutions in the
VAE decoder (4 DC-AE blocks at progressively reduced scales). The
2048² block has its peak around `rms_norm::21` idx 678 (the OOM site
in the empirical 1×16g probe). Each lower-scale block has 4× less
peak so 2048² is ~6 GiB bf16, 1024² is ~3 GiB, 512² is ~1.5 GiB.
Detection with the 0.10×component_vram threshold (1.6 GiB on 16 GiB
GPU) would trigger on 4096² and 2048² blocks; lower scales pass
through unchanged.

**Anti-pattern check (does NOT match)**:
- Sana 1024 VAE has the same DC-AE structure but at 1024² output,
  the largest intermediate is [1,128,1024,1024] = 256 MiB bf16 — well
  below the 1.6 GiB threshold. Chain not triggered. ✓
- PixArt-XL / PixArt-Sigma use a different VAE (KL-AE, not DC-AE) with
  no `add → conv → silu → conv → permute → rms_norm → add` signature.
  Chain not triggered. ✓
- TinyLlama / autoregressive LLMs have no 2D spatial residual chains.
  Chain not triggered. ✓

## 4. S5 implementation strategy

### 4.1 — Chain band-streaming contract

For a detected chain `(fork, [chain_ops], merge)`:

For each band of H rows `[b_start, b_end)` of the merge output:

1. Determine the corresponding band of `T_base` that the chain reads —
   this is `[b_start - halo_top, b_end + halo_bottom]` where halo
   accumulates the spatial extent each `aten::convolution` in the chain
   contributes (`(kernel_size − 1) // 2` per conv on each side).
2. Slice `T_base` for that input band → 1 band of input.
3. Execute the chain on the band:
   - `conv::63(band)` → band of conv::63 output (with halo expansion)
   - `silu::24(band)` → band of silu::24 output (in-place OK, pointwise)
   - `conv::64(band)` → band of conv::64 output
   - `permute(band)` → view, zero-copy
   - `rms_norm::24(band)` → band of rms_norm::24 output (in-place OK,
     pixel-wise reduction along feature dim is band-safe)
4. Slice the corresponding output band of `T_base` (rows [b_start, b_end)
   without halo, since merge is pointwise).
5. `add(band_chain_output, T_base_output_band)` → write to output rows
   [b_start, b_end) of the merge output buffer.

Live tensor footprint per band:
- `T_base` (4 GiB) — full, alive whole chain
- `merge_output` (4 GiB) — full, being filled
- 1 band of chain intermediate (1/tile_factor × 4 GiB, ≈ 1 GiB at tile=4)
- Total ≈ 9 GiB + weights (~0.5 GiB chain weights) + driver (~3 GiB)
  ≈ **12-13 GiB on 16 GiB GPU**. Fits.

### 4.2 — Detection wired into OpLevelTilingPlan

Add a new field `residual_chains: List[Dict]` alongside the existing
`fusion_pairs`, `tiled_ops`, `inplace_adds`, `pixel_shuffle_broadcast_-
chains`. Each entry stores: fork_uid, merge_uid, chain_uids list,
tile_factor.

Detection happens in `OpLevelTilingEngine._detect_residual_chains`
(new method, mirrors the structure of
`_detect_pixel_shuffle_broadcast_chains`). Called from
`register_into_graph_executor` alongside the existing detections.

### 4.3 — Interceptor wiring

`register_op_uid_interceptor` (existing universal hub) registers an
interceptor on `fork_uid` (`aten::add`) that, when called, executes
the entire band-streamed chain and stores the result in
`merge_uid`'s output slot directly, marking all chain intermediates
+ `merge_uid` itself as already-computed (so subsequent op dispatch
skips them via the existing fusion-proxy mechanism).

Mirror needed for R30: both compiled (`graph_executor` native path)
and triton (`triton_sequential` and `TritonSequence` hot loop) must
recognise the proxy and skip the absorbed ops. The existing
`FusionUpsampleProxy._nbytes = 0` sentinel pattern is the template.

### 4.4 — Tiled chain wrappers

Two implementations to satisfy R30 dual-branch:

- **Compiled / sequential (PyTorch ATen)**: New helper
  `_band_streamed_residual_chain_torch(t_base, chain_spec,
  tile_factor)` in `kernels/ops/residual_chain.py` (new file). Uses
  `F.conv2d`, `F.silu`, custom `rms_norm` via existing kernel —
  consistent with the dual-implementation pattern in
  `fused_upsample_conv.py`.
- **Triton (triton + triton_sequential)**: New helper
  `_band_streamed_residual_chain_nbx(t_base_nbx, chain_spec,
  tile_factor)` in the same file. Uses the existing `conv2d_wrapper`,
  `rms_norm_wrapper` Triton wrappers. R33 zero-torch preserved.

### 4.5 — Cascade decision

Prism solver's cascade currently lands on `lazy_sequential` (VAE on
CPU) for v100-16g Sana 4Kpx. For triton/triton-sequential modes,
this is R33 incompatible. The cascade must prefer `single_gpu` +
op-level tiling (with residual chains + fusion pairs) over CPU
placement when the runtime mode is one of triton-pure.

Plan: add a `_runtime_mode` hint to `PrismSolver.solve_from_container`
input or detect it from the call site (executor.factory passes it
when constructing the solver context). The solver cascade then skips
`lazy_sequential` strategies that would place GPU components on CPU
when `_runtime_mode in {"triton", "triton_sequential"}`. This is a
minimal, R34-conforming change because it's based on a structural
mode flag, not a model name.

## 5. Validation plan (R32 / R29)

Reference equivalence target:
`validation_outputs/p_prism_never_refuse_v2_s4_sana4kpx_32g_reference.png`
(coherent red apple from Sana 4Kpx on v100-32g, 22.7 s, 31.7 GiB peak).

Cells to validate post-S5:
1. Sana 4Kpx 1×16g `--triton` → PNG (R29) + nvidia-smi peak < 16 GiB.
2. Sana 4Kpx 1×16g `--triton-sequential` → PNG (R29) + nvidia-smi peak
   < 16 GiB.
3. (Bonus, expected free) Sana 4Kpx 1×16g `--compiled` and
   `--sequential` continue to work via S1 hybrid; the residual chain
   detection is structural and applies in compiled mode too if Prism
   chooses single_gpu + op-level tiling there.

Anti-régression matrix (no regression on any of these):
- Sana 4Kpx 32g × 4 modes (POINT 7 acquis).
- Sana 1024 BF16 × 4 modes.
- PixArt-XL / PixArt-Sigma × 4 modes (compiled/sequential at least).
- TinyLlama × 4 modes.
- CPU-pure compiled (S1 acquis) and sequential (S2 acquis).

R32: bit-exact OR cosine ≥ 0.99 vs 32g reference. Band-streaming with
sum-of-pointwise ops is bit-exact within an op. Conv with halo is
bit-exact when halo size matches kernel radius. rms_norm reduction is
per-pixel so band-stream is bit-exact. Therefore **bit-exact expected**
unless tile_factor causes accumulation order changes — none expected
in this chain.
