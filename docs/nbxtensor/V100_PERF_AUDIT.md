# V100 performance audit — where the time goes on the m8n8k4 matrix unit, and what it does not reach

2026-11-03. Tesla V100-SXM2 16 GB (cards 0, 1) and 32 GB (cards 2, 3) at the protocol clock (877 / 1290 MHz), engine
tree `a803946a` (nbxtensor), Triton 3.8.0, the rack's generated profiles `default-ff6008b7` (16 GB) and
`default-5ca6057c` (32 GB). Asked by the owner (2026-11-03 00:36, 07:32, 07:36): every operation that can run on the
V100's matrix unit runs there, and every loss to torch is explained before the library and the stable are published.

This first part covers the matrix-product operations. The second part, the operations no matrix unit accelerates
(layer_norm, group_norm, softmax, embedding: bound by memory bandwidth, where the gain comes from fusion and memory
access), is written from the kernel-level benchmark's summary.

## Method

- **Through the entry point.** Each model profile is `python -m neurobrix run` with the gate cell's own command
  (request, flags, profile, `--certified-only`), recorded with Nsight Systems 2023.2
  (`nsys profile --trace=cuda --cuda-graph-trace=node`). The tables are `nsys stats --report cuda_gpu_kern_sum`.
  Every output was read (text) or written (video, rc 0); none of these runs is a gate cell.
- **Static where nothing needs a card.** Routing, census coverage and the key each instance forms are read from the
  launcher's own functions (`launch_keys.matrix_unit_operands`, `tools/derived_census.derive_component`).
- **Kernel timings** are medians of 7 calls after 2 warm-up calls, the device synchronised around each, the FMA
  baseline served from the certified directory (`NBX_AUTOTUNE_CERTIFIED_ONLY=1`).

## 1. Triton has no tensor-core path on sm_70; the engine has its own

Triton 3.8.0 compiles an fp16 `tl.dot` (`[64, 32] x [32, 64]`) for sm_70 to 1 024 `fma` PTX instructions and no
`mma`. Other V100 projects report the same and write their own `mma.m8n8k4` kernels
([V100 flash-attention](https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1),
[TensorFold #264](https://github.com/ashhart/TensorFold/pull/264)). This is why nbxtensor carries its own m8n8k4
kernels (`ops/mma_m8n8k4.py`: CUTLASS sm70 fragments, fp16 × fp16 → fp32, fp32 operands as hi + lo), chosen by the
profile's `matrix_unit` rows (`config/vendors/nvidia/volta.yml`), never by a card name:

| operation | kernel on the unit | operands |
|---|---|---|
| mm, addmm, bmm, baddbmm (M ≥ 64) | `matmul_m8n8k4_kernel` | fp16 native; fp32 / bf16 split (three MMAs) |
| conv2d (implicit GEMM) | `conv2d_m8n8k4_kernel` | fp16 native |
| scaled dot-product attention | `flash_attention_m8n8k4_kernel` | fp16 native; fp32 split, rows per head dimension |

## 2. Which bench instances take the unit

Every GEMM instance of the kernel-level benchmark with at least 64 output rows takes the unit (mm 69, addmm 39,
bmm 11). The 279 that do not all have M < 64, where the GEMV and batched kernels own the rows by design: a row
band mostly empty leaves the unit idle (`matrix_unit_operands`). No GEMM with a unit row available runs on the FMA
path. A smaller `matrix_unit.mm` row (block_m 16 or 32) would bring M = 16–63 to the unit; at M ≤ 16 such a GEMM is
bound by its weight bytes (about M flop per byte read), so the gain is expected at M = 32–63 and is to be measured.

The census confirms it from the other side: a launch on the unit forms no autotune key, so every autotuned GEMM or
convolution key of the volta census is a launch the unit does not take. On the 16 GB class:

| kernel (dtype) | census rows | why it is not on the unit |
|---|---|---|
| `conv2d_forward_kernel` fp32 | 2 920 | the convolution carries fp16 operands only (section 5) |
| `baddbmm_kernel` fp32, fp32 × fp16 | 2 016 + 840 | LLM attention scores, M = batch × heads under 64 |
| `conv2d_forward_kernel` fp16 | 602 | vocoder conv1d (H = 1, L ≈ 56–62): under the smallest tile |
| `matmul_kernel` fp32 × fp16, fp16 | 388 + 210 | M = 16–24 (short prefixes), under the smallest tile (64) |

## 3. Where a gate request spends its GPU time

| model (class) | GPU kernel time | first kernel | second | decode attention |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite (16 GB), 32 tokens | 11.28 s | `expert_miss_copy` 10.49 s (93 %) | `fused_moe` 0.22 s (prefill) | `paged_decode_attn` 0.012 s |
| Qwen3-30B-A3B (32 GB), 32 tokens | 4.70 s | `expert_miss_copy` 3.52 s (75 %) | `fused_moe` 0.41 s (prefill) | `paged_decode_attn` 0.016 s |
| CogVideoX-2b (16 GB), 2 steps | 121.9 s | `conv2d_forward_kernel` fp32 76.4 s (63 %) | `flash_attention_m8n8k4` 19.6 s | — |
| Allegro (32 GB), 2 steps | 238.9 s | `conv2d_forward_kernel` fp32 122.0 s (51 %) | `matmul_m8n8k4` 31.5 s, flash 31.4 s, `cat_copy` 21.6 s | — |

The mixture-of-experts decode is bound by the host link: the experts the card does not hold are copied at every
token (`expert_miss_copy`), and the decode's own MoE kernels (`moe_gateup_vec`, `moe_down_split_vec`: one token per
expert, a GEMV) and attention (T_q = 1) are bound by memory bandwidth. A matrix-unit variant of `flash_decode` or of
the decode MoE kernels would save under 0.3 % of these requests. The prefill `fused_moe` is compute-bound and is the
next candidate for the unit (0.2–0.4 s per request on these two models).

## 4. The video VAEs: fp32 convolutions on the FMA path

The video VAEs compute their convolutions in fp32 (their stored dtype), and `conv2d_m8n8k4` serves fp16 operands
only, so these convolutions run `conv2d_forward_kernel` on the FMA units — 51 % to 63 % of a video request.

## 5. The fp32 convolution through the split, measured

The convolution's implicit GEMM was given the GEMM's split (each K tile of A and B staged as hi + lo under a
power-of-two scale per output position and per output channel, three MMAs, fp32 accumulation). Its parity test
holds the GEMM's contract (error per output position no larger than the FMA path's plus four fp32 epsilons, on
census shapes and range edges; it fails when the lo term is removed). Measured on card 0 against the certified FMA
path, on the census's own fp32 shapes:

| shape (N, C, H, W → O, kernel) | FMA | best split tile (128 × 64 × 32, 4 × 1 warps) | ratio |
|---|---|---|---|
| 12, 256, 160², → 256, 3 × 3 | 46.2 ms (7.8 TFLOP/s) | 49.5 ms | 0.93 |
| 12, 512, 80², → 512, 3 × 3 | 46.8 ms | 47.4 ms | 0.99 |
| 24, 128, 320², → 128, 3 × 3 | 92.8 ms | 103.0 ms | 0.90 |
| 12, 256, 160², → 512, 3 × 3 | 92.1 ms | 100.0 ms | 0.92 |
| 12, 512, 160², → 256, 1 × 1 | 10.3 ms | 7.6 ms | 1.36 |
| 24, 128, 320², → 3, 3 × 3 | 18.0 ms | 34.6 ms (64 × 32 × 32) | 0.52 |

Other tiles measured: 64 × 64 × 32 (2 × 2), 64 × 32 × 32 (2 × 1), 32 × 64 × 32 (1 × 2), 64 × 128 × 32 (2 × 2),
64 × 64 × 16 (2 × 2), all slower than 128 × 64 × 32 on the 3 × 3 shapes. On Volta, three MMAs plus the hi/lo staging of
the gathered input tile cost as much as the FMA path's fp32 convolution, already at half the fp32 peak.

The owner's decision (2026-11-03 10:32): route the 1 × 1 convolutions only, by a measured row. On the census's fp32
1 × 1 classes, at the best row (128 × 64 × 16, 4 × 1 warps):

| shape (N, C, H, W → O) | FMA | split | ratio |
|---|---|---|---|
| 1, 384, 60 × 104 → 1152 | 1.50 ms | 0.77 ms | 1.95 |
| 1, 384, 60 × 104 → 384 | 0.71 ms | 0.45 ms | 1.58 |
| 2, 384, 120 × 208 → 768 | 4.05 ms | 2.76 ms | 1.47 |
| 12, 512, 160² → 256 | 10.29 ms | 6.72 ms | 1.53 |
| 12, 256, 160² → 256 | 8.92 ms | 5.17 ms | 1.72 |
| 24, 128, 320² → 128 | 10.65 ms | 8.47 ms | 1.26 |
| 2, 192, 120 × 208 → 384 | 1.27 ms | 1.25 ms | 1.01 |
| 1, 32, 60 × 104 → 32 | 0.29 ms | 0.27 ms | 1.07 |
| 1, 16, 60 × 104 → 16 | 0.40 ms | 1.10 ms | 0.36 |

The profile carries the row (`matrix_unit.fp32_split.conv`, kernel 1 × 1); a split convolution takes the unit only
where a row exists for its window and its tile fills (M ≥ block_m, output channels per group ≥ block_n — the 16-channel
case keeps the FMA path), one function deciding for the wrapper, its cost and the census
(`launch_keys.matrix_unit_conv_operands`). The 3 × 3 convolutions keep the FMA path. On CogVideoX-2b (32 GB, the gate
request, through the entry point) the time to output went from 137.3 / 135.4 s to 135.8 / 130.7 s, the video the same
to the eye (PSNR 54.4 dB between the two). Whether cuDNN's own fp32 3 × 3 beats this FMA path is read from the
kernel-level benchmark: if it does, the 3 × 3 is a loss to audit and a better split kernel returns to the plan.

## 6. Open

- Allegro's transformer `addmm::13` (`[72000, 2304] × [2304, 9216]`) reaches the unit on the split rows with an fp32
  activation, while its graph is fp16 along the whole chain, its calibration record peaks at 42.8 and its precision
  contract narrows it. Whether the engine's launch carries fp32 there (three MMAs where one suffices) or only the
  benchmark's derivation does is read on a real launch next.
- The small-M addmm's per-row weight copy (112 µs a row) was the benchmark's own: the graph passes the weight as a
  transposed view. The benchmark now carries each input's strides.

## 2. The kernel-level benchmark, measured (2026-11-03)

`benchmarks/nbxtensor_vs_torch` (tool 637e781b), every compute op of the census at the plan's extents, each instance
run alone on its card: nbxtensor through the engine's sequential dispatcher (`TritonSequentialDispatcher`, the op's own
compute dtype and precision contract, certified-only), torch in eager and in `torch.compile` default mode, the result
judged against a float64 reference before any time is compared. `torch.compile` max-autotune was measured for the first
instances and then stopped (its autotune ran for hours on the large video convolutions, the owner's decision of 15:30):
for the rest the column says it was not run. The tables are in the register (`nbxtensor_vs_torch.v100-*.summary.md`).

| class | judged | won | lost | refused: key outside the census | refused: memory | not finished |
|---|---|---|---|---|---|---|
| V100 16 GB | 1 422 | 35 | 1 387 | 113 | 32 | 0 |
| V100 32 GB | 1 432 | 39 | 1 393 | 113 | 3 | 0 |

Where nbxtensor loses (16 GB; ratio = nbxtensor's median over the best right torch contender's):

| family | instances | won | median ratio | 90th percentile |
|---|---|---|---|---|
| convolution | 511 | 6 | 2.99 | 8.49 |
| group_norm | 154 | 3 | 1.65 | 3.28 |
| mm / addmm | 269 | 15 | 4.1–4.8 | 8.5–10.7 |
| rms_norm / layer_norm | 214 | 0 | 2.6–5.1 | 5.7–11.7 |
| sum / mean / softmax | 113 | 0 | 3.5–5.9 | 8.6–10 |
| scaled_dot_product_attention | 25 | 0 | 12.2 | 17.0 |
| weight_norm | 16 | 0 | 30.4 | 36.4 |

Read with what it measures — **these tables compare torch with the benchmark's per-op dispatch path, not with the
engine** (which replays its compiled sequence without that per-op cost): a tool defect, to be replaced by a table with
the kernel time and the dispatch time in separate columns for both libraries (the supervisor 2026-11-03 17:25).

- **A per-call cost dominates the small instances.** nbxtensor's fastest instance takes 0.10 ms and its tenth percentile
  0.27 ms, where torch eager's are 0.033 and 0.054 ms: the sequential dispatcher's Python and launch path costs
  0.1–0.25 ms per op (the 2 x 2304 addmm, profiled: 0.68 ms of host time around 0.02 ms of kernel). That cost is the
  bench's path, the one `--triton-sequential` runs; the engine's `--triton` mode runs a compiled sequence whose decode
  steps replay their launches. Most of the 2–5x ratios on norms, reductions and small GEMMs are this cost, not the
  kernels.
- **Kernels that lose on their own**, read here and audited one by one: the transposed 1-D convolutions of the vocoders
  (up to 500x: `[1, 768, 184]` upsampled by 8, 103 ms against 0.21 ms), weight_norm (a 1.6 ms floor), the attention at
  12x, and the depthwise convolutions — fixed since (section 7, 2a2d69a4; these tables predate it).
- **Two instances are judged wrong**, both classes: an fp16 `mul` by -3.39e38 (float32's lowest, a masking constant —
  -inf in fp16, which torch produces too, against a finite float64 reference: the comparison's, not a defect), and an
  fp32 attention of GLM-4.1V's and MiniCPM-o's language models under a contract that narrows it to fp16 (relative error
  1.4e-2 against fp32's tolerance 2e-3): whether the tolerance or the kernel is wrong is open, with the tool's author.

## 7. Operations no matrix unit accelerates (first entries)

- **depthwise_conv2d** (2026-11-03): x8 behind torch on SANA-Video's 11 200-channel 3 × 3 (3.07 against 0.38 ms in the
  kernel-level benchmark). Cause read in the kernel: its tile put the channels on the fastest axis, H × W elements apart
  in NCHW memory — no load coalesced. The (channels × rows × columns) tile with the width fastest: 2.63 → 0.33 ms
  (529 GB/s), 41.2 → 6.4 ms, 18.9 → 4.2 ms on the giant shapes; one-row tiles for the vocoders' conv1d depthwise
  (the census's majority) 1.6–2.4×. Parity against fp64 at the fp32 level; the 5 reference outputs byte-identical
  (Kokoro-82M and PixArt-Sigma use it); orpheus-3b byte-identical, its first audio 3.68 → 1.85 s; SANA-Video's 2-step
  video the same to the eye (PSNR 43.1 dB between the two). Every depthwise key (323 per class) re-certified.
- **The expert cache's policy** (2026-11-03): online LFU measured against LRU through the entry point — DeepSeek-Coder-V2-Lite
  16 GB −10 % per token, Qwen3-30B-A3B 32 GB a loss (two runs at 246 and 323 ms where LRU never passed 195): LRU stays.
  A frequency carried from another prompt lost to LRU in simulation (routing does not carry across prompts).
