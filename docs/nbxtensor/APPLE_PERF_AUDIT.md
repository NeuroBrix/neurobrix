# Apple performance audit — where nbxtensor loses to torch on the M4 Pro

2026-11-03. Apple M4 Pro (16-core GPU, 24 GB unified, macOS 27.0.1), triton-ext AppleGPU backend
(`b06cbb0d`), torch 2.14.0 (MPS) as the contender. Asked by the owner (2026-11-03 00:36) before any
publication of the M4 benchmark: the four kernel families nbxtensor lost by the widest margin on the
first Apple run — convolution (×17), embedding (×9), layer_norm (×9) and small-M addmm (×7–9; part of it the bench's own copy, see that section).

## Method

- **One instance per family**, the one the first run lost worst. Each is run through the engine's own
  dispatcher, exactly as the bench times it. The bench drives one instance per process through
  `profile_one.py`, 10–20 timed calls after warm-up.
- **Certified-only.** Every nbxtensor key was served from the certified directory: no sweep, no miss.
- **GPU time is measured, not estimated.** Each run was recorded with Instruments' *Metal System
  Trace* (`xctrace record --template 'Metal System Trace'`). The tables read here are the
  `metal-gpu-intervals` exports: per command buffer, the GPU's execution time and Instruments'
  *CPU to GPU Latency* (commit to GPU start).
- **Torch was traced the same way**, MPS eager on the same instance and inputs.
- **Launch identification.** `NBX_EXT_DEBUG=1` lists every launch of one call (kernel name,
  geometry). `NBX_SYNC_TIMING=1` splits each launch into drain, submit and synchronize.
- **The machine ran nothing else.** The campaign's kernel step was stopped first (see "A bench
  defect found on the way").
- **Where the evidence lives.** Traces, exports, logs and the two drivers are in the campaign
  `campagnes/2026_11_02_m4_apple/audit/`; the instances file is `instances.v2.jsonl`.

The four smoke medians on the engine's path reproduce the first run's numbers within 3–7 %:

| family | instance (key) | nbxtensor (first run) | nbxtensor (audit) | torch eager |
|---|---|---|---|---|
| convolution | `[20,4,90,160] x [2304,4,2,2]`, stride 2, bias, bf16 (`599807e9`) | 52.27 ms | 50.93 ms | 3.10 ms |
| embedding | table `[32128,4096]` bf16, indices `[1,512]` (`5826b14f`) | 2.42 ms | 2.25 ms | 0.26 ms |
| layer_norm | `[2,36000,2304]` bf16 (`9a36f9ba`) | 30.28 ms | 28.15 ms | 3.39 ms |
| addmm | `[2304] + [2,2304] x [2304,2304]` bf16 (`2628b953`) | 2.30 ms | 2.27 ms | 0.26 ms |

## The four losses in one table (GPU measurements, per call)

| family | nbxtensor command buffers per call | GPU time per launch | CPU→GPU latency per launch | MPS, per call |
|---|---|---|---|---|
| convolution | conv kernel + bias kernel | 7.21 ms + **38.14 ms** | 2.70 + 2.47 ms | 1 buffer, 2.79 ms GPU, 0.10 ms latency |
| embedding | embedding kernel | 0.053 ms | **2.46 ms** | 1 buffer, 0.080 ms GPU, 0.14 ms latency |
| layer_norm | fp32 copy of the input + fp32 layer_norm | 4.52 + 6.15 ms | **8.57 + 11.25 ms** | 1 buffer, 3.19 ms GPU, 0.09 ms latency |
| addmm, M = 2 | per ROW: strided copy of the whole weight (the bench's, see below) + addmv + one blit | (0.92 + 0.29 + 0.01 ms) × 2 | (0.24 + 0.18 + 0.08 ms) × 2 | 1 buffer, 0.095 ms GPU |

Two causes run through all four. Each family then has one cause of its own.

## Cross-cutting cause 1 — the launcher maps every operand again at every launch (category b)

**Measure.** CPU→GPU latency is proportional to the bytes a launch binds. Five launches sit between
7.5 and 9.7 ms per GiB:

| launch | bytes bound | latency | ms per GiB |
|---|---|---|---|
| embedding | table 257 MB + output 4 MB | 2.46 ms | 9.7 |
| conv bias | output 332 MB | 2.47 ms | 7.5 |
| conv | input 2.3 MB + output 332 MB | 2.70 ms | 8.1 |
| layer_norm copy | 331 + 662 MB | 8.57 ms | 9.0 |
| layer_norm kernel | 662 + 662 MB + stats | 11.25 ms | 8.4 |

The same buffers allocated by torch's MPS allocator start in 0.09–0.34 ms. This repeats the
engine's own earlier measurement of the wrap: 10.9 ms at 1 GiB and 46.7 ms at 4 GiB (2026-10-04,
quoted in `triton_ext_driver._buffer_for`).

**Cause.** `nbxtensor/backend/triton_ext_driver.py` binds a pointer by building a NEW zero-copy
Metal buffer at every launch. `_buffer_for` calls `metal_native.wrap`, which calls
`newBufferWithBytesNoCopy`. Metal maps that memory into the GPU's address space before the first
command buffer that uses it can start, so each launch pays the mapping for every operand.

The driver wraps from the tensor's own address because triton-ext binds every argument at offset 0
(`metal_native.m`, `setBuffer:… offset:0`). A wrap therefore cannot be kept per allocation and
reused for a view inside it.

**Share of each loss.** All of the embedding's (2.46 of 2.25 ms wall: the kernel itself runs
0.053 ms, faster than MPS's 0.080 ms). 70 % of layer_norm's. 10 % of the convolution's.

**Proposed fix.** Map once per allocation, not once per launch:

1. Keep one Metal buffer per allocator segment for the segment's life.
2. Bind a tensor as (that buffer, its offset).
3. Release the wrap when the allocator frees the segment.

The offset binding is a small upstream change to triton-ext's `metal_native`: an argument carrying
(buffer, offset), passed to `setBuffer:offset:atIndex:`. It goes upstream, not into a fork (R25).
The driver's `pinned_addresses` scope already keeps one whole-allocation wrap alive for
mixture-of-experts bands; this generalises that lifetime.

**Expected.** Embedding 2.25 → ~0.15 ms. Layer_norm −~19 ms. Conv −~5 ms. Every launch on a large
tensor, in every model, gains the same way.

**Cost.** ~2 days: the upstream PR, the driver, the allocator hook and a test that sees the mapping
gone in a trace. **Who.** Mac (driver, upstream PR); the Dell for the allocator's release hook
(nbxtensor/memory is his zone).

## Cross-cutting cause 2 — every launch waits for the GPU (category b)

**Measure.** `NBX_SYNC_TIMING`: each launch is drain → submit → synchronize. The synchronize is
the GPU's full execution plus the latency above.

On the small addmm, the host leaves 0.4 ms between the end of one command buffer and the start of
the next: 3 command buffers per row, 6 per call.

**Cause.** `triton_ext_driver.py:341-345` calls `_native().synchronize()` after every launch, a
`waitUntilCompleted` on the last command buffer (`metal_native.m:84`). triton-ext's own launcher
commits and returns without waiting (`encodeDispatch`, `metal_native.m:367-410`); the wait is
NeuroBrix's. It exists because NeuroBrix's blits run on another queue, and Metal does not order
two queues without an event. The swin2SR NaN reads of 2026-09-17 are the measured reason, kept in
the driver's comment.

Apple's guidance is the opposite pattern:
- batch encoders into fewer command buffers;
- replace `waitUntilCompleted` by `MTLSharedEvent`, which has lower overhead and fewer timeline
  gaps (WWDC22 *Scale compute workloads across Apple GPUs*; Tech Talk *Metal Compute on MacBook
  Pro*).

Others measured the same cost:
- TornadoVM: command-buffer completion reaches the CPU 60–90 µs after the GPU's own signal
  (PR #24, Sep 2026);
- gprMax issue #897: one blocking command buffer per stage, Metal slower than the CPU.

**Proposed fix.** Order the two queues on the GPU instead of the host:
- either encode NeuroBrix's blits on the queue the kernels use;
- or signal and wait an `MTLSharedEvent` between the queues.

Then synchronize only where the host reads device memory. The swin2SR ordering case stays the
test that must stay green.

**Expected.** For large ops it is irrelevant. For decode it is ~0.1–0.4 ms per launch: thousands
of launches per token.

**Cost.** ~1–2 days. **Who.** Mac (driver and the blit path in nbxtensor/backend).

## Convolution — the bias pass, then a quarter-used K tile (categories c, a)

**Measure.** On the GPU, 45.4 ms per call against MPS's 2.79 ms. Traced without its bias, the
same certified launch takes 7.21 ms; that matches its certificate (10.08 ms on 2026-10-22, which
included the mapping). The other **38.14 ms** is `conv2d_bias_inplace_kernel`: one in-place
`out[i] += bias[(i // HW) % C]` over 166 M elements.

**Cause (c, the bias).** The bias kernel indexes in int64 (`program_id(0).to(tl.int64)`) and
computes one integer division and one modulo per element. Apple GPUs have no integer divide
instruction: it is an injected subroutine, slow with a variable divisor. Apple says so in WWDC16
*Advanced Metal Shader Optimization*. A 64-bit operation also costs about 4× a 32-bit one
(philipturner/metal-benchmarks).

332 MB read and written in 38 ms is 17 GB/s, against the 210–220 GB/s the same machine's copies
reach (below). MPSGraph applies the bias inside its convolution.

**Cause (a, the conv kernel, 7.21 ms against MPS's 2.79 ms for conv + bias).** The certified
configuration is `BLOCK_SIZE_BHW 64, OUTF 32, INF 16, SPATIAL_FAST, 2 warps`. With 4 input
channels:
- each of the kh·kw = 4 inner steps issues a `tl.dot` whose K tile (16) is 75 % masked zeros;
- the instance's whole reduction, C·kh·kw = 16, would fill one tile.

**Cause (d, the census).** The certificate benched 2 of its 64 candidates (`benched: 2`); the
screen kept two. Whether another candidate beats 7.2 ms is not measured here.

**Proposed fix.**
1. Apply the bias in the conv kernel's epilogue, as the matrix-unit conv already does
   (`HAS_BIAS`): it removes the pass and one bf16 rounding. Output bytes of biased convolutions
   change in the last bit (one rounding instead of two), so the models with biased convolutions
   are re-judged by their outputs.
2. For small-C convolutions, an implicit GEMM whose K is C·kh·kw in one tile.
3. Re-sweep this key's 64 candidates.

**Expected.** 50.9 → ~10 ms with fix 1. Then ~7 ms with the mapping fix. ≤ 3 ms is the target
for fixes 2–3, to be measured.

**Cost.** Fix 1 half a day plus the re-judging. Fix 2 about a day plus its census rows and
certificates. Fix 3 minutes of machine.

**Who.** Mac (nbxtensor/ops kernels).

## Embedding — the kernel wins, the launch loses (category b only)

**Measure.** One launch per call. The kernel runs 0.053 ms on the GPU, against MPS's 0.080 ms.
The call costs 2.25 ms because its command buffer starts 2.46 ms after commit: the 257 MB table
is wrapped again at every launch.

**Cause.** Cross-cutting cause 1, nothing else. MPS runs `index_select`'s dense Metal kernel
(`aten/src/ATen/native/mps/operations/Indexing.mm`, torch `08187d9e`) on a buffer mapped once.

**Proposed fix, cost, who.** Cross-cutting cause 1. Nothing family-specific.

## Layer_norm — an fp32 copy of the input and fp32 bytes throughout (categories c, b)

**Measure.** Two launches per call:
- `triton_copy_kernel` widens the bf16 input to fp32: 4.52 ms GPU, 993 MB moved, 221 GB/s;
- the fp32 layer_norm reads 662 MB and writes 662 MB: 6.15 ms, 217 GB/s.

Both kernels run at the machine's bandwidth. MPS moves 663 MB in 3.19 ms (208 GB/s) with one
launch. Each of our launches then starts 8.6–11.3 ms after commit (cross-cutting cause 1).

**Cause (c).** `aten::native_layer_norm` is an AMP fp32 op. Under bf16 compute, the Triton dtype
engine's `_wrap_fp32_internal_compute_dtype_output` casts the input to fp32 as a separate pass. It
keeps an fp32 output: this component's graph was traced in fp16, and under bf16 compute the AMP
rule writes bf16 only on a bf16 graph (`triton/dtype.py`). So the call moves 2.3 GB where MPS moves 0.66 GB.

MPS's kernel (`aten/src/ATen/native/mps/kernels/LayerNorm.metal`, `layer_norm_single_row`) does
it differently:
- one threadgroup per row;
- each thread holds its 4 elements in registers;
- it reads the half input once, accumulates in fp32 and writes the half output once.

**Cause (a, d).** Not the kernel: both of ours run at bandwidth. The certified tile pads 2304 to
4096 lanes, so 44 % of lanes are masked. At this bandwidth-bound size that costs nothing
measurable.

**Proposed fix.** Widen inside the kernel: no fp32 copy of the input, one launch. This is the
copy lever of 2026-09-07; the `rms_norm` kernel can already widen on load. It is switched off
there (`rms_norm._nbx_widens_on_load = False`), because the widened kernel is another compilation
than the fp32-input one the certified rows ran: 12–15 % of elements differed at T5's shapes
(2026-09-08). For layer_norm, the widened kernel is proven against the fp64 oracle and certified
at the half-input key before it lands.

The output dtype stays the DtypeEngine's decision, and this audit does not change it. But the
fp32 output costs a further 2× on the write side, so it is a question for the dtype owner. The
fp32 output is read from the measure: the two launches' times are in the ratio 1.36, the ratio of
their bytes with an fp32 output is 1.33, and with a bf16 output it would be 1.0.

**Expected.** 28.2 → ~15 ms. ~6 ms after the mapping fix. ~3.5 ms if the output may be bf16.

**Cost.** Half a day plus the oracle proof and the affected models' 5-sha. **Who.** Mac (dtype
and kernels).

## Addmm, M ≤ 4 — a row loop of GEMVs below bandwidth (category a; the copy seen was the bench's)

**Measure.** For M = 2, the bench's call costs three launches per row:
- `triton_strided_copy_kernel` over all 5 308 416 weight elements: 0.92 ms, 23 GB/s;
- `triton_addmv_kernel`: 0.29 ms, 36 GB/s on the 10.6 MB weight;
- a blit storing the row.

Two rows make 2.4 ms of GPU time. MPS runs the whole addmm in 0.095 ms, 112 GB/s.

**The copy is the bench's, not the engine's** (the Dell's V100 reading, 2026-11-03 01:58, verified
here in the graph). In Allegro's transformer the weight operand is `aten.t(W)`, a transposed VIEW
of the stored `[2304, 2304]` parameter. The small-M path of `linalg_wrappers.addmm` hands
`addmv_wrapper` `b.t()`, which is `W` itself, contiguous: `.contiguous()` copies nothing. The bench
gave `b` as a contiguous `[K, N]` array instead, so `b.t()` was strided and was copied once per row.

What the engine pays is the rest, an estimate from the measured parts:
- two addmv launches at 0.29 ms and two blits on the GPU, ~0.6 ms;
- each launch's own latency (0.08–0.18 ms) and host dispatch on top.

The Dell measured the host share at ~0.68 ms of 0.81 ms on the V100 (sequential dispatcher). It
should be measured in the compiled Triton path too before it is called an engine loss.

**Cause (a, d).**
- The addmv kernel reaches 36 GB/s on its certified tile, far below the copies' 220 GB/s. Its
  certificate is the next thing to read.
- The row loop launches one GEMV per row. Each re-reads the weight; the second read may hit the
  system cache, and this audit does not measure that.

**Proposed fix.** One small-M kernel for M ≤ 4:
- it reads the weight once, through the strides it arrives with;
- all M rows accumulate in fp32 and the bias goes into the epilogue;
- no per-row loop, no blit, one launch.

It is a new kernel key, so it needs census rows and certificates.

**Expected.** ~1.2 ms on the engine's path (estimate) → ~0.3 ms. Below torch's 0.26 ms once
cross-cutting cause 2 is fixed. Decode in every language model runs this path, once per linear per
token.

**Cost.** About a day plus census and certification. **Who.** Mac (nbxtensor/ops).

**The bench follows.** An instance must carry its inputs' strides, the views the graph forms (a
transposed weight), or every GEMM with M ≤ 4 pays a copy the engine does not. That is a harness
fix owed before the M ≤ 4 rows of the Apple table are read.

## A bench defect found on the way, fixed before this audit

The first Apple kernel run's nbxtensor column did not dispatch every op the way the engine does:

- **What went wrong.** The bench took the dispatcher's compute dtype from the container's record
  of each op's output, the trace dtype. The conv above ran in fp16 and the VAE convs stored fp32
  where the engine runs bf16. Those keys were not in the certified directory, so they were swept
  at runtime and timed: 11 of the first 13 instances.
- **The fix.** Both halves landed: `72b2f4c7`, the plan's component dtype (the Dell), and
  `ae750530`, the op's dispatch facts plus a certified-only worker (the Mac). The run was
  stopped and relaunched on them.
- **What it does not change.** The four instances of this audit are served from the certified
  directory on the corrected path. Their medians move by 3–7 %, so three of these losses are the
  engine's own. The fourth, the small-M addmm, carries a copy the engine does not make (its section).

## What triton-ext's Apple backend says about itself

- **Upstream main** holds the minimal backend (#127, 2026-09-14). Its README's known limitations
  are float64 (computed silently in f32) and the `num_warps` cap; it says nothing about
  performance.
- **Open issues.** None of the backend's issues (#130, #147, #156 closed; #181 open, i64 offsets
  truncated by chained `addptr`) concern launch cost or throughput.
- **The served line** (`b06cbb0d`) lowers `tt.dot` to 8×8 simdgroup matrix MMA. Its runtime
  commits one command buffer per kernel call and does not wait.
- **Conclusion.** Neither cross-cutting cause is triton-ext's; both are in NeuroBrix's driver.
  The offset binding is the one upstream addition they need.

## Order proposed

| # | fix | families | estimated gain on these instances | cost | who |
|---|---|---|---|---|---|
| 1 | map once per allocation (offset binding upstream + per-segment wrap) | all | emb −2.4 ms, LN −19 ms, conv −5 ms | ~2 d | Mac + Dell (allocator hook) |
| 2 | conv bias in the kernel epilogue | conv | −38 ms | ½ d + re-judging | Mac |
| 3 | layer_norm widened in-kernel | layer_norm | −13 ms | ½ d + oracle proof | Mac |
| 4 | small-M addmm in one kernel (and the bench's instance strides) | addmm | ~1.2 → ~0.3 ms per call (estimate); decode-wide | ~1 d + census | Mac |
| 5 | no host wait after each launch (queues ordered on the GPU) | all, decode most | 0.1–0.4 ms per launch | 1–2 d | Mac |
| 6 | small-C conv implicit GEMM, re-sweep of the key | conv | 7.2 → ≤ 3 ms (target) | ~1 d | Mac |

## Sources

- Apple, *Scale compute workloads across Apple GPUs* (WWDC22): <https://developer.apple.com/videos/play/wwdc2022/10159/>
- Apple, *Metal Compute on MacBook Pro* (Tech Talk): <https://developer.apple.com/videos/play/tech-talks/10580/>
- Apple, *Advanced Metal Shader Optimization* (WWDC16, integer division): <https://developer.apple.com/videos/play/wwdc2016/606/>
- Apple, Metal Programming Guide, command submission: <https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html>
- philipturner/metal-benchmarks (64-bit integer cost on Apple GPUs): <https://github.com/philipturner/metal-benchmarks>
- TornadoVM PR #24 (completion latency after the GPU signal): <https://github.com/kotselidis/TornadoVM/pull/24>
- gprMax issue #897 (blocking command buffer per stage): <https://github.com/gprMax/gprMax/issues/897>
- triton-ext AppleGPU backend and its issues: <https://github.com/triton-lang/triton-ext/tree/main/backend/AppleGPU>, <https://github.com/triton-lang/triton-ext/issues/156>, <https://github.com/triton-lang/triton-ext/issues/181>
- PyTorch MPS at `08187d9e`: `aten/src/ATen/native/mps/operations/{Convolution,Normalization,Indexing}.mm`, `aten/src/ATen/native/mps/kernels/LayerNorm.metal`
