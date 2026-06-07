# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **chatterbox triton: the TTS-LLM flow handler now mirrors the oracle context
  assembly instead of hand-rolling it** (was producing `*BANG*`/`*Ballon*`
  garbage in both triton modes). The triton `tts_llm` handler fabricated the
  t3 backbone context in numpy — a 2-frame hand-rolled conditioning, plain
  tokenization (6 vs 9 tokens), no classifier-free guidance, and a zeroed
  vocoder reference — feeding the backbone a context that did not match the
  reference path. It now: (1) runs the traced **cond_enc graph** (speaker +
  Perceiver resampler + emotion) for a `[1,34,1024]` conditioning instead of a
  2-frame numpy stand-in; (2) reproduces the model's `start_text_token`/
  `stop_text_token` + space-marker tokenization contract; (3) applies
  **sequential CFG** via the centralized triton CFG engine (guidance from the
  data-driven cascade); (4) feeds the embedded reference-voice conditioning to
  the vocoder instead of zeros. A shared seeded sampler makes both triton modes
  reproducible. The triton context now byte-matches the reference
  (`Initial context (1,44,1024)`, cond=34/text=9/speech=1, CFG on). The handler
  is chatterbox-only, so other TTS models are unaffected; the path stays
  torch-free (numpy CPU glue + graph execution + NBXTensor).

- **Kokoro-82M closed 4/4 (was 2/4): the pytorch-sequential & triton-sequential
  op-by-op paths now run the full prosody-predictor + istftnet decoder.** Two
  sequential-dispatcher completeness fixes (R30 parity — the compiled/triton
  paths already handled these op forms):
  (1) `sequential_dispatcher` now recomputes **1-D** upsample output_size from the
  scale factor for `upsample_nearest1d`/`upsample_linear1d` (the 2-D
  multi-resolution recompute already existed). Kokoro's F0/N branches upsample the
  duration-expanded sequence; the trace-time output_size was `2*trace_len`, so at
  a different runtime `sum(durations)` it mismatched the actual input (124 vs 68 →
  add broadcast crash). The scale (2.0 / 300.0 / 1/300) is read positionally per
  op (linear1d carries an `align_corners` bool before the scale).
  (2) `tensor_resolver` now parses a **complex scalar** arg (`1j`, the imaginary
  unit in the istftnet iSTFT `mul`) — PyTorch ATen handles complex natively, the
  parser just lacked the type.

### Changed

- **openaudio dual_ar sampler is now deterministically seeded (shared across all
  4 modes).** Both `core/flow/dual_ar.py` and `triton/flow/dual_ar.py` previously
  sampled with an UNSEEDED RNG (torch `multinomial` / `np.random.choice`), so the
  4 modes drew different acoustic tokens run-to-run and the STT-substring harness
  was flaky (triton "Hello, wall" vs the others' "Hello world"). Both paths now
  use a deterministic `np.random.RandomState` seeded from `global.seed` (default
  `_DUALAR_SEED=1234`, identical literal in both files) with a byte-identical
  numpy sampling algorithm — same seed ⇒ same draws ⇒ at fp16-close logits the 4
  modes sample identical tokens. Result: openaudio **4/4 under sampling** (not
  just greedy). R27/R28 reproducibility; the pytorch path keeps its compute in
  torch and only the (CPU, numpy) sampling glue is shared-by-duplication.
- **Triton path is now torch-free end to end:** removed the last `import torch`
  from the triton compute path — the `_vram_probe` telemetry in
  `triton/flow/iterative_process.py` used `torch.cuda.memory_allocated()`; it now
  reads `DeviceAllocator.memory_allocated()` (the NBX live-bytes watermark).

### Added

- **VibeVoice cross-engine validation enabler (shared seeded diffusion noise +
  first-K latent dump).** The pytorch and triton `next_token_diffusion` handlers
  now both draw diffusion init noise from the SAME `np.random.RandomState(seed)`
  with the same per-step draw (byte-identical noise across engines, verified
  max|diff|=0.0) — replacing the pytorch path's `torch.randn(generator=)`. A
  gated diagnostic (`NBX_VV_DUMP_LATENTS=<path.npy>`, `NBX_VV_DUMP_K`) dumps the
  first-K diffusion latents per mode. This is the doctrine-correct validation for
  a numerically-chaotic feedback model: triton-seq must match the pytorch-seq
  oracle on the first-K latents (before the feedback loop amplifies fp16 diffs),
  not on the noisy full-decode STT substring. numpy is CPU glue; the pytorch
  compute path is otherwise untouched.
- **Triton (zero-torch) `next_token_diffusion` flow handler (VibeVoice).**
  `triton/flow/next_token_diffusion.py` mirrors the compiled
  `NextTokenDiffusionEngine` on the NBXTensor substrate: the LM / diffusion head /
  acoustic+semantic tokenizers / connectors run through their component graphs as
  NBXTensor, the diffusion sampler is the zero-torch `TritonDPMSolverPPScheduler`
  + `TritonCFGEngine` guidance, and init noise comes from a seeded numpy RNG
  uploaded as an NBXTensor (numpy is CPU orchestration glue only — prompt
  assembly, embedding-table lookups, scalar latent scaling — as in
  `triton/flow/tts_llm.py`). The executor now dispatches `next_token_diffusion`
  to this handler in triton / triton_sequential mode instead of raising. Component
  names and input contracts validated against the VibeVoice .nbx topology.

### Fixed

- **Triton `clamp` rejected tensor bounds; `addmm` rejected N-D activations.**
  `clamp()` did `float(min_val)` unconditionally, so an `aten::clamp.Tensor`
  whose min/max BOUND is itself an `NBXTensor` (e.g. the Flex.1-alpha DiT) raised
  `float() argument must be ... not 'NBXTensor'`. Tensor bounds now route through
  elementwise `maximum_wrapper`/`minimum_wrapper` (R33-pure); scalar bounds keep
  the fused kernel; mixed (one tensor, one scalar) handled per side. `addmm()`
  hard-unpacked `M, K = a.shape`, so a >2-D activation raised "too many values to
  unpack"; it now flattens the leading dims to 2-D, runs the 2-D kernel, and
  restores the leading shape (mirror of `matmul()`'s ND×2D path). Both are
  generic catalogue fixes; the diffusion scheduler-split re-validation is
  unaffected (Sana / PixArt-XL / PixArt-Sigma triton all produce coherent PNG).
- **Triton RoPE positions were frozen at 0 during KV-cache decode for models
  with no `position_ids` input (orpheus).** Such models derive RoPE positions
  from an internal `aten::arange(0, seq_len)`; in single-token KV-cache decode
  `seq_len=1`, so the graph emits `arange(0,1)=[0]` every step — every decoded
  token is RoPE-encoded at absolute position 0, the cache fills with mis-rotated
  keys, and generation degrades into a non-terminating ramble (orpheus triton
  STT "I'm not speaking, but I'm not speaking to you…" vs the correct
  "Hello world!"). The op-by-op triton-*sequential* oracle recomputes the full
  growing context (`arange(0,N)`) and so stayed correct, masking the bug behind
  a passing mode. Fix: `TritonAttentionInterceptor.intercept_arange` shifts the
  arange window START to `cache_len` during decode (`arange(0,1)` →
  `arange(cache_len, cache_len+1)`), preserving output SIZE / symbolic shape —
  the exact mirror of the compiled-core `kv_cache_wrapper.intercept_arange`, the
  missing R30 half on the triton side. Registered ONLY when `not uses_abs_pos`
  (no `position_ids` input); models that drive RoPE from a `position_ids` input
  (TinyLlama, chatterbox t3, MoE LLMs) are untouched (TinyLlama R23
  byte-identical). Closes orpheus triton 4-mode (whisper STT "Hello world!").
- **Triton `aten::fill` silently no-op'd for non-zero values and corrupted
  strided targets.** `NBXTensor.fill_` only handled `value==0` (byte memset);
  the non-zero branch was a `TODO` that returned the tensor unfilled (= garbage),
  and `_meta_fill` mutated its input *in place* — which on a non-contiguous slice
  (e.g. the s3gen `aten.fill` target `aten.slice::…`) wrote the wrong addresses
  under flat indexing. Fix: `fill_` non-zero now uses the pure-Triton
  `fill_kernel` (with a contiguity guard that scatters into strided views), 0-dim
  tensor values are collapsed via `.item()`, and `_meta_fill` returns a FRESH
  contiguous tensor (correct functional `aten::fill` semantics, no input
  mutation). Surfaced by chatterbox s3gen.
- **Triton `aten::broadcast_tensors` returned a nested list instead of unpacked
  outputs.** The handler was `lambda *t: t`, so a `TensorList` input `[a, b]` came
  back as `([a, b],)` — `out_0` resolved to the whole Python list and a downstream
  consumer (`lt` / `where` / `stack`) hit `'list' object has no attribute '_dtype'`.
  Replaced with a real wrapper that broadcasts every input to their common
  (numpy-style) shape and returns them unpacked, one tensor per output slot.
- **Triton `aten::upsample_nearest1d` choked on the `.vec` list scale factor.**
  `upsample_nearest1d.vec` passes `scale_factors` as a 1-element list `[s]`, which
  the wrapper forwarded into the 2D path's scalar `float(scales_w)` → `TypeError`.
  Unwrap the single-element list to a scalar; the scale-recompute path then sizes
  the output from the live input, which is what a variable-length audio vocoder
  (HiFiGAN/iSTFTNet/DAC/S3Gen) needs. (Both surfaced by chatterbox s3gen.)
- **Triton `gather` out-of-bounds on tensors with ≥2 outer dims (CUDA-700).**
  `gather_wrapper` passed `stride(0)` as the kernel's flattened-outer stride, but
  `gather_kernel`'s `outer_idx` flattens *every* dim before the gather dim, so the
  correct stride is `prod(shape[dim:])` (= `shape[dim]·inner_size`). `stride(0)`
  over-counts by `prod(shape[1:dim])`; with ≥2 outer dims every `outer_idx≥1`
  store ran past the buffer end — a destructive out-of-bounds write the driver
  reported as CUDA error 700 at the next synchronising op. Surfaced by the
  chatterbox s3gen T5 relative-position gather `[1,8,180,359]` with index dim=3
  (1440× out of bounds). Byte-identical for `dim≤1` (a single outer dim), so every
  previously-passing gather is unchanged; verified against `torch.gather` on
  dim=1, multi-outer dim=2, and the chatterbox shape.

### Added

- **Separate zero-torch Triton scheduler subtree (`triton/scheduler/`).** The
  Triton diffusion path previously borrowed the PyTorch scheduler via an
  `nbx_to_torch → driver.step` round-trip in `triton/flow/iterative_process.py`
  — the last torch dependency on the Triton compute path. Added a fully separate
  Triton scheduler (`TritonDPMSolverPPScheduler` + `TritonSchedulerFactory`),
  a byte-for-byte mirror of `core/module/scheduler/diffusion/dpm_solver_pp.py`
  with torch replaced by: numpy for the CPU noise schedule (betas/alphas_cumprod/
  sigmas/timesteps, computed once), Python `math` for the per-step scalar
  coefficients (sigma_t/alpha_t/lambda/h/r0), and NBXTensor for the per-step
  latent arithmetic (the only tensor operands). No torch import anywhere in the
  Triton scheduler. Validated bit-equivalent to the PyTorch DPM++ scheduler:
  matching timesteps and a per-step latent trajectory within 3.5e-6 (fp32) over a
  20-step v-prediction/cosine/order-2 run. This is the PyTorch/Triton code-path
  separation requirement — the two paths now share only the CLI entry +
  orchestrator executor. (Flow-matching routes through DPM++ with
  `use_flow_sigmas=True`, matching the core factory.)
- **Triton FlowEuler scheduler + iterative_process rewired off the PyTorch
  scheduler.** Added `TritonFlowEulerScheduler` (numpy timesteps, Python-float
  dt, NBXTensor latent) — flow-matching Euler for Sana/Flex, validated
  bit-equivalent to the core FlowEuler (timesteps 1e-7, trajectory 1.4e-6). The
  orchestrator (`executor._setup_modules`) now picks the scheduler by mode —
  triton modes get `TritonSchedulerFactory`, PyTorch gets `SchedulerFactory` —
  and `triton/flow/iterative_process.py` calls the NBXTensor-native scheduler
  directly (the `nbx_to_torch → driver.step` / `scale_model_input` crutch is
  removed). The triton diffusion compute path is now zero-torch end to end.
  Both triton schedulers expose `timesteps` as NBXTensor `[1]` scalars (numpy kept
  internal for indexing) so the loop iterator / CFG engine / DiT component receive
  tensor-like timesteps exactly as the PyTorch path did (fixes a `numpy.int64 has
  no attribute 'dim'` crash in the triton CFG at Sana step 0).
- **Triton reflect/replicate `aten::pad` routing + `NBXTensor.ones`/`ones_like`.**
  The generic `pad_wrapper` (for `aten::pad`, which carries its mode as a runtime
  arg rather than lowering to `aten::reflection_padNd`) only handled
  `mode='constant'` and raised for `reflect`/`replicate`; it now routes by padded-
  dim count to the existing R33-pure `reflection_pad1d/2d` (and `replication_*`)
  wrappers. And `NBXTensor.ones`/`ones_like` were documented in the engine
  contract but never implemented (only `empty`/`zeros`/`*_like` existed), so the
  istft window-envelope `NBXTensor.ones(...)` call hit `AttributeError`; added,
  backed by the pure-Triton `fill_kernel`. Together these take the chatterbox
  s3gen vocoder from crash (reflect-pad → istft `ones` → `aten::fill`) to full
  end-to-end execution in triton mode.
- **Triton `aten::norm` (L2) kernel** — composed R33-pure from existing kernels
  (`mul` + `sum` + `sqrt`, with fp32 accumulation so a long-row reduction does
  not overflow fp16). Unblocks weight-normed convolutions in triton mode (the DAC
  codec decoder). Only p=2 (the weight-norm case) is wired; other p raise.
- **Triton convolution dilation support.** The conv kernel sized the output with
  the dilation factor (right, smaller output shape) but gathered the kernel taps
  at contiguous offsets, ignoring dilation — a dilated convolution silently read
  the wrong receptive field and produced garbage. The kernel now spaces the taps
  by `dilation` (`tap*dilation` in the input-index math); dilation defaults to 1,
  so non-dilated convolutions are byte-identical. Unblocks the OpenAudio DAC
  vocoder, whose residual units use the classic 1/3/9 dilation stack (its triton
  audio was a low-energy noise wash before — STT "*crickets*", now "Hello world!").

### Added

- **Triton `aten::stft` / `aten::istft`.** Short-time Fourier transform and its
  inverse, built on the existing rfft/irfft DFT-matmul path: `stft` =
  `unfold` framing + window + rfft + transpose; `istft` = irfft + windowed
  overlap-add (via the existing `unfold_backward` scatter) + window-energy
  normalisation + centre-trim. Model-agnostic, R33-pure. Unblocks the
  chatterbox/CosyVoice s3gen HiFiGAN/iSTFTNet vocoder (n_fft=16, hop=4).
- **Triton `aten::conv_transpose1d`.** Thin adapter over the existing transposed
  conv kernel (which already handles the 1D case via an H=1 unsqueeze); only the
  positional-arg order differs from the aten signature. Used by HiFiGAN-style
  upsamplers.

### Fixed

- **Triton `aten::matmul` crashed on ≥4-D batched inputs.** The general batched
  branch passed raw 4-D tensors to `bmm` (which is strictly 3-D → "too many
  values to unpack"). It now collapses the leading batch dims into one, runs
  `bmm`, and restores the batch shape (with simple batch broadcasting). 3-D
  matmul is unchanged (byte-identical); verified exact vs torch on a 4-D case.
- **Triton `aten::layer_norm` returned a 3-tuple where the graph expects one
  output.** The high-level `layer_norm` op was dispatched to the
  `native_layer_norm` wrapper (which also returns mean/rstd); a single-output
  graph then stored the whole tuple, so a downstream op received a tuple. It now
  routes to a wrapper returning just the normalised output. `native_layer_norm`
  (3 outputs) is unchanged.
- **Triton `aten::clip`** now dispatches to `clamp` (it is an alias).
- **RNN (LSTM/GRU) dtype mismatch in sequential/triton modes.** The AMP input
  cast only converted top-level tensor arguments, but `aten::lstm`/`aten::gru`
  pass their hidden state (`[h0, c0]`) and parameters as nested *lists* of
  tensors. The input was cast to fp16 while the hidden/weight lists stayed fp32,
  so PyTorch rejected the op ("input Half × hidden Float"). The cast now recurses
  into list/tuple arguments; flat-argument ops are unaffected (byte-identical).
- **Triton DualAR (OpenAudio) generated near-silent audio — full generation port.**
  The triton `dual_ar` flow ran only the slow (semantic) autoregressive pass and
  then fed the decoder semantic token embeddings via a numpy shortcut, never
  running the fast/depth AR for the residual codebooks nor the RVQ dequantize —
  the decoder received the wrong features. It now mirrors the reference flow:
  slow backbone AR + fast/depth AR (9 residual codebooks) + `codec.quantizer`
  RVQ decode, all on `NBXTensor` (R33-pure; orchestration/sampling in numpy at
  the boundary). Triton greedy codes are now byte-identical to the PyTorch
  reference across all positions and codebooks; STT of the audio reads the prompt.
- **Triton DualAR ignored CLI sampling overrides.** The flow read the embedded
  `defaults` for temperature/top-p/repetition-penalty/max-tokens, so
  `--temperature 0` (greedy) silently sampled at the default 0.7. It now reads the
  `global.*` overrides first (mirroring the reference flow), restoring
  deterministic greedy decoding.
- **`isin` returned a non-boolean tensor.** The membership op staged its result
  through `uint8`, so a downstream boolean negation (`~mask`) missed its "is this
  a bool?" guard and did a raw byte complement (`~0 → 255`) instead of the logical
  NOT — corrupting the OpenAudio DualAR attention mask (first triton-vs-reference
  divergence, localized op-by-op). `isin` now returns a true boolean tensor so
  every bool-aware consumer (negation, `masked_fill`, `where`) sees it correctly.
- **Triton audio output not saved.** The decoder output is an `NBXTensor`, but
  the audio post-processing searched only for `torch.Tensor` waveforms, so it
  never bound `global.output_audio` and the save raised. The post-processor now
  converts the chosen waveform candidate to torch at the mode boundary (shape is
  read without converting, so only the matching waveform is materialised) and
  also looks under the `codec.decoder.output_0` key. Compiled mode is unchanged.
- **Triton mode crashed (CUDA error 700) on boolean-mask `masked_fill`.** An
  attention `masked_fill(~mask, value)` is captured as an in-place assignment
  with a boolean mask; the triton scatter wrapper mis-read the 1-byte mask as
  integer positions and wrote far out of bounds, corrupting the CUDA context
  (surfacing later as a spurious "GPU malloc failed"). A mask whose shape is a
  leading-dim prefix of the target, with a scalar fill, now routes to the
  `masked_fill` kernel (detected by dtype name, robust to the mask being typed
  bool/uint8 or a raw triton dtype). Unblocks the OpenAudio DualAR backbone in
  triton mode.
- **Sequential (op-by-op) mode silently dropped in-place index assignments,**
  producing silent/garbage audio for codec models. An assignment such as
  `dst[:, 0] = value` is captured as an in-place write into a *view* of the
  destination; sequential mode executed it with the functional copy (which
  returns a new tensor and never mutates), so the destination kept its original
  value. The destination of the OpenAudio DAC codec's code tensor stayed all
  zeros → every codebook lookup hit row 0 → silence. Sequential now performs the
  write in place (matching compiled mode), so the view-aliased mutation
  propagates. This affects every model that does `tensor[idx] = value` (OpenAudio,
  Kokoro, Chatterbox, SANA-Video). OpenAudio sequential mode now produces
  word-perfect audio, matching compiled mode (the op-by-op reference oracle).
- **Runtime AMP parity:** the per-op AMP path used by sequential/triton modes now
  mirrors the compiled path's fp16 squaring guard — a hand-rolled
  `mean(x*x)` norm variance is upcast to fp32 so it cannot overflow fp16 and
  collapse the norm to zero.

### Changed

- **OpenAudio TTS decodes its audio codes in a single pass.** The `dual_ar`
  flow now dequantizes (`codec.quantizer`) and decodes (`codec.decoder`) the
  whole generated code sequence at once instead of slicing it into fixed-length
  blocks, which previously cut the codec transformer's `window_size=128`
  cross-frame attention and left inter-block seams. The quantizer features are
  bound to `model.output_0` so the single full-sequence decoder pass resolves
  its input via the topology connection. Mirrored in compiled and triton modes
  (`core/flow/dual_ar.py`, `triton/flow/dual_ar.py`, R30). Validated:
  word-perfect STT at a generation length different from calibration.

### Fixed

- **Reshape crash in the OpenAudio audio decoder at non-calibration lengths.**
  A feature/channel dimension whose value happened to equal a length-dependent
  size at the calibration length was incorrectly treated as variable, so it
  mis-scaled when the generated audio differed in length and a reshape failed.
  The runtime now keeps such a dimension fixed only when it is both an
  architectural constant and fully explained by fixed input dimensions, leaving
  genuinely variable spatial/sequence dimensions untouched (Sana and
  granite-speech outputs unchanged).

- **SDPA Q/K/V dtype alignment in the efficient/flash sequential path.** The
  standard `scaled_dot_product_attention` dispatch already cast Q/K/V to the
  narrowest common dtype, but the `_scaled_dot_product_efficient_attention` /
  `_scaled_dot_product_flash_attention` branch only cast the attn_mask. When
  upstream AMP leaves V in fp32 while Q/K are fp16 (openaudio DualAR backbone),
  torch SDPA rejects the mixed dtypes. Mirrored the alignment into the
  efficient/flash branch, guarded on mismatch (no-op when uniform, so the
  efficient-backend models that already pass — whisper, Voxtral, canary — are
  unchanged). Unblocks openaudio sequential end-to-end.

### Added

- **`NBX_DECODE_BOUND=N` — universal bounded-decode harness (diagnostics).**
  When set, every autoregressive / TTS / audio_llm / dual_ar / encoder_decoder
  decode loop hard-caps its step count to N, in BOTH execution modes (core +
  triton, R30). Lets op-by-op triton-vs-oracle diffs (the 4-mode method) run on
  a 5-10 token window in seconds instead of the full 2048-token generation
  (openaudio dual_ar: 8 tokens in 8.5 s vs ~24 min for 2048 op-by-op). New
  pure-Python helper `core/runtime/decode_bound.py` (`os` only, zero torch —
  R33-safe to import from `triton/flow/`); applied at the `max_tokens`
  computation of all 10 decode loops. Gated, default-off, zero semantic/runtime
  impact unset. NOTE: also fixed a latent asymmetry — `triton/flow/dual_ar.py`
  read `max_tokens` from defaults only (ignoring the CLI `global.max_tokens`
  that the core dual_ar honoured); both now flow through `decode_bound`.

- **`aten::_weight_norm` implemented as a Triton meta-op.** Vocoders' weight-
  normalized convolutions (chatterbox s3gen, HiFi-GAN-style) emit
  `aten::_weight_norm(v, g, dim)`, which the triton path lacked (`[triton]
  Missing op: aten::_weight_norm`). Implemented in `dispatch.py` as
  `w = v · g / ‖v‖`, with `‖v‖` the L2 norm over all dims except `dim`
  (keepdim), via the existing mul/sum/sqrt/div wrappers — pure-Triton, no new
  `@triton.jit` kernel. Resolves the P-TRITON-CHATTERBOX missing-op blocker.
  **CONFIRMED executing**: a patient run reached `[s3gen] Running vocoder` (after
  the t3_cfg decode hit eos at 221 tokens) and ran past `_weight_norm` with no
  error — the meta-op executes correctly (chatterbox's `weight_g` is `[512,1,1]`,
  broadcast correct). chatterbox's NEXT triton blocker is a separate missing op,
  `aten::stft` (vocoder STFT) — full WAV/STT still pending that.

- **`NBX_DECODE_PROGRESS=<file>` gated diagnostic** in the compiled + triton
  `encoder_decoder` flows AND the triton `autoregressive` generator
  (`generator.py`) — writes (buffer-immune) the encoder output stats and a
  per-decode-step trajectory (token count, last token, done flag; last-token
  hidden for encoder_decoder), for encoder sanity + triton-vs-compiled
  first-divergence localization + decode liveness/loop detection. Off by
  default, zero impact when unset; numpy at the dump boundary only (R33-clean).
  This is the toolkit that root-caused the whisper efficient-SDPA causal-mask
  bug and is reused across the audio triton sweep.

- **`NBXTensor.repeat` (aten::repeat)** — tiling (numpy.tile / torch repeat),
  R33-pure (view → expand → contiguous → reshape). Was missing entirely
  (`'NBXTensor' object has no attribute 'repeat'`), crashing any triton model
  whose graph emits `aten::repeat` — surfaced by the parakeet conformer encoder.
  Bit-exact vs torch across shapes / extra leading dims.

### Changed

- **Block-attention windowing (`num_blocks = ceil(seq/W)`) is now resolved from
  trace-emitted symbolic graph dims in every execution mode**, replacing a
  compiled-only runtime pre-compilation pass (`_symbolize_data_dependent_attrs`,
  retired from `compiled_sequence.py`). The graph now arrives with symbolic
  windowing dims and the shared `SymbolicShapeResolver` resolves them, so flexible
  frame counts no longer depend on a mode-specific runtime patch. The general
  cross-branch / seq_len passes are unchanged (they consume whatever symbolic
  exprs are present). Granite Speech `--compiled` transcribes correctly with a
  matching re-trace; sequential-path resolution of these dims is in progress.
- **parakeet RNNT decoder LSTM runs through the triton-pure `lstm_wrapper`**
  instead of a hand-rolled NumPy cell — removes the NumPy compute debt in
  `triton/flow/rnnt.py` (the greedy loop stays Python; only the single-step
  input + h/c cross the boundary, weights converted once). Per-step
  unidirectional multi-layer output is bit-exact vs `torch.nn.LSTM`
  (max|diff|=0). NOTE: parakeet triton end-to-end is still wrong due to a
  separate, pre-existing conformer-ENCODER bug (only now reachable since the
  `repeat` fix cleared the crash that previously masked it) — the next
  triton-sweep step.

### Fixed

- **`triton_sequential` dropped SDPA `scale`/`is_causal` kwargs → 8x-wrong
  attention scale in Whisper-style audio encoders.** `_dispatch_sdpa` read
  `attn_mask`/`dropout_p`/`is_causal`/`scale` POSITIONALLY from `inputs` and
  ignored the op's graph kwargs. Whisper-style encoders (Voxtral `audio_tower`)
  carry `scale=1.0` and `is_causal=false` as **kwargs** with only q/k/v
  positional (the encoder pre-scales Q, so SDPA scale must be 1.0). Reading
  positionally only fell back to the wrapper default `scale = 1/sqrt(head_dim)
  = 0.125` — an 8x-wrong attention scale → garbage encoder output → garbage
  audio_embeds → the LM emitted a generic answer ("You're welcome!") instead of
  transcribing. Fix: `_pos_or_kw` (positional-OR-kwarg) in all three SDPA
  branches, mirroring the compiled path's `compiled_kwargs` forwarding. Same
  class as the rms_norm-eps bug below (a seq dispatcher special-case dropping
  the graph's data-driven kwargs); also explains why orpheus's efficient-SDPA
  seemed fine (its dropped scale default happened to equal the graph's
  1/sqrt(128)). After the fix Voxtral triton-sequential transcribes correctly
  (full sentence, matching the PyTorch-seq oracle and whisper ground-truth);
  the audio_tower SDPA op now matches the compiled path bit-for-bit
  (NBX_OP_FINGERPRINT). Surfaced by genuinely re-verifying `--triton-sequential`
  (the prior audio sweep validated `--triton`/compiled only).

- **`triton_sequential` dropped the `epsilon` kwarg on `custom::rms_norm` →
  wrong RMSNorm eps (1e-6 instead of the model's 1e-5).** The sequential
  dispatcher's `custom::rms_norm` special-case called `func(*inputs)`,
  forwarding only positional tensors and silently discarding the op's
  attribute kwargs — where the model's real `rms_norm_eps` lives (graph
  `custom::rms_norm` attributes carry `{"epsilon": 1e-05}` for Llama-family
  models). The `rms_norm` wrapper then fell back to its `eps=1e-6` default.
  The other three modes (PyTorch-seq via `sequential_dispatcher`, PyTorch-
  compiled, and triton-compiled via `compiled_kwargs`) all forward the
  epsilon correctly, so `triton_sequential` was the lone outlier. On the
  first RMSNorm over small-magnitude embeddings `mean(x²)` is near the eps
  scale, making 1e-5 vs 1e-6 a ~10% denominator swing that compounds through
  every layer: on TinyLlama greedy decode the four modes agreed for steps
  0-6 then triton_sequential flipped a near-tie argmax at step 7 (2215 vs the
  reference 2319) and diverged. Localized by an op-fingerprint prefill diff
  (first diverging op = `custom::rms_norm::0`, identical bit-exact input,
  divergent fp32 output). Fix: forward the resolved attribute kwargs in the
  `custom::rms_norm` branch, mirroring the dispatcher's generic path. After
  the fix all four modes produce byte-identical greedy tokens on TinyLlama.
  Affects every model carrying `custom::rms_norm` (Llama-family: TinyLlama,
  orpheus, granite, chatterbox, openaudio, deepseek-moe, Qwen, Sana text
  encoder); LayerNorm models (whisper, Voxtral, canary) are unaffected,
  which is why they passed the triton sweep untouched.

- **`NBX_DISABLE_ROPE_FUSION=1` diagnostic gate** (triton-compiled `sequence.py`)
  — leaves the HF-Llama rotate_half RoPE chain as native ATen ops instead of
  fusing into `custom::rope_fused`, to isolate fused-rope numerical divergence
  from other compiled-path effects (parallel to `NBX_DISABLE_AUTOTUNE`). Gated,
  default-off. Used to rule the fused rope OUT as the cause of the orpheus
  triton-compiled decode divergence (output was byte-identical fused vs unfused).

- **`NBX_OP_FINGERPRINT` is now emitted by the `triton_sequential` path too**
  (graph_executor sequential dispatch loop), mirroring the compiled
  `TritonSequence` emit. Closes an R30 diagnostic asymmetry — the sequential
  dispatcher previously could not be op-diffed against the compiled hot-loop,
  the very comparison that root-caused the rms_norm eps bug above. Same record
  schema (`{i, op_uid, op_type, tid, shape, dtype, sha}`), gated, default-off,
  zero runtime impact.

- **Scalar-fill ops (`masked_fill`/`fill`/`index_fill`) overflowed fp16 in the
  dynamically-dispatched modes.** A graph may carry a bf16/fp32 mask sentinel
  (e.g. bf16-min ≈ -3.39e38) as the Python scalar fill value; on fp16 hardware
  the fill tensor is bf16→fp16, and converting that scalar to fp16 raises "value
  cannot be converted to type at::Half without overflow" (granite-speech
  conformer attention, in `--sequential` / `--triton`). The scalar is an op
  attribute, not a tensor input, so the per-input AMP casts never saw it. The
  DtypeEngine (single dtype authority) now clamps the scalar to
  `torch.finfo(fill_dtype)` when the fill tensor is fp16/bf16 — numerically inert
  versus the oracle (masked positions are ~0 after softmax either way) and a
  no-op for in-range scalars. R23: whisper `--triton` byte-identical.

- **PEFT/LoRA models with unmerged adapters now run, with the runtime kept
  fully LoRA-agnostic.** A model published with unmerged LoRA adapters
  (`canary-qwen-2.5b`: a frozen Qwen base + `q_proj`/`v_proj` adapters,
  r=128 α=256) ships three tensors per adapted projection — `base_layer`,
  `lora_A`, `lora_B` — while its graph references a single clean weight. Such
  models are now delivered with the adapters already folded into one weight per
  projection (`W = W_base + (α/r)·(B·A)`, a frozen post-training constant), so
  the execution engine binds plain weights with **zero** knowledge of LoRA
  structure. The runtime weight binder accordingly no longer special-cases PEFT
  wrapper names (`base_layer`/`base_model`); that model-structure knowledge has
  been removed from the engine, keeping it model-agnostic (the only weight-key
  helper that remains is the general prefix-hierarchy suffix match). **Validated:**
  canary-qwen-2.5b produces the byte-identical correct vendor transcript in all
  four execution modes (compiled / sequential / triton / triton_sequential,
  transcript md5 `ffcb30ea…`). Folding existing adapters is weight preparation
  of a frozen constant — **not fine-tuning** (that capability is untouched and
  reserved for its own future work).

- **Triton: `aten::_safe_softmax` was a missing op.** The triton dispatch mapped
  `softmax` / `_softmax` → `w.softmax` but not `_safe_softmax` (PyTorch's
  masked-safe softmax — returns 0 for fully-`-inf` rows instead of NaN). Any
  model whose graph emits it crashed with `[triton] Missing op:
  aten::_safe_softmax` (granite-speech conformer attention). Mapped it to
  `w.softmax` and added it to the TRITON op classification. For non-fully-masked
  attention it is identical to softmax; the `dtype=None` 3rd arg is harmless
  (mirrors `_softmax`'s `half_to_float`). (Fully-`-inf`-row safety would need a
  dedicated kernel, deferred until a model exercises that condition.)

- **Triton autoregressive `_tokenize` ignored `chat_mode` → wrong prompt for
  TTS LMs.** The triton flow (`triton/flow/autoregressive.py`) applied the
  generic HF `apply_chat_template` whenever the tokenizer exposed it, ignoring
  the model's `chat_mode` flag. Models with `chat_mode=False` (orpheus,
  openaudio — whose prompt is the bare templated text) got the full chat
  template (system prompt + role markers): orpheus's "Hello world." became a
  **39-token** prompt instead of the correct **4 tokens**, so the prefill
  consumed a different prompt and the entire decode was garbage (degenerate
  tokens, never reached eos). Gated the chat-template path on `chat_mode` and
  mirrored the compiled `TextProcessor.tokenize` basic-encode path
  (`add_special_tokens=True`). orpheus triton now produces a prefill hidden
  (h_norm 108.06 vs compiled 108.08, head byte-identical) and decode tokens
  (128009,128260,128261,128257,…) matching the compiled oracle. Shared fix for
  any `chat_mode=False` autoregressive model in triton.

- **Triton repetition-penalty sampler segfault (H2D cudaMemcpy raw host
  pointer).** `_apply_repetition_penalty` (triton/samplers.py) wrote the
  penalty-adjusted logits back to the GPU with the host source passed as a bare
  `logits_np.ctypes.data` — a Python int. A ctypes call with no `argtypes`
  coerces it to a 32-bit C int, truncating the 64-bit host address → invalid
  source pointer → segfault. Any triton autoregressive model with
  `repetition_penalty != 1.0` crashed at the first sampling step (orpheus TTS).
  Wrapped the host src in `ctypes.c_void_p` (the D2H reads already wrapped both
  ends, which is why only this H2D write crashed). Removed a dead
  `DeviceAllocator` import at the same site.

- **Triton autoregressive KV-cache interceptor: efficient/flash/cudnn SDPA
  arg-binding crash.** `TritonAttentionInterceptor.intercept` (kv_cache.py) has
  the plain-SDPA signature, but the autoregressive flow registered it for ALL
  SDPA variants. The efficient/cudnn ops pass `is_causal` at arg[6] and `scale`
  as a kwarg, so binding them to the plain signature raised
  `intercept() got multiple values for argument 'scale'` — crashing any
  efficient-backend causal decoder that uses the KV cache (orpheus / openaudio
  TTS). Added per-variant `intercept_efficient` / `intercept_flash` remap methods
  and routed each SDPA op type to the matching one (added cudnn to the
  intercepted set) — mirrors the compiled side's per-variant interceptors
  (kv_cache_wrapper.py:622). Plain `scaled_dot_product_attention` routing is
  unchanged → R30-safe for TinyLlama / Voxtral / canary.

- **Triton `--triton`: `aten::_scaled_dot_product_efficient_attention` (and
  `_flash` / `_cudnn`) dropped the causal mask + used scale=1.0.** These
  fused-backend SDPA variants have a shifted positional signature vs plain SDPA
  (an extra `compute_log_sumexp` bool at arg[4] pushes `is_causal` to arg[6],
  `scale` to arg[7]). The `--triton` dispatch (`dispatch.py`) mapped them
  *directly* to `scaled_dot_product_attention_wrapper`, so a positional call
  mis-read `is_causal` (from `dropout_p`→False — causal mask silently dropped)
  and `scale` (from `is_causal`→1.0). Invisible at seq_len=1 (single-element
  softmax), it corrupted every seq_len≥2 forward → constant-token garbage on any
  causal decoder traced with the efficient backend (whisper-large-v3-turbo
  produced `516` repeated). Fixed with a `_meta_sdpa_efficient` remap shim
  mirroring `triton_sequential` (sequential.py:198) — closes the R30 asymmetry.
  Cross-attention (non-causal, 5-arg) and plain `scaled_dot_product_attention`
  are unchanged; compiled never uses this dispatch (R23-safe). whisper-large-v3-
  turbo triton now transcribes byte-identical to the compiled oracle.

- **Triton `index_select` / `aten::index`: enforce integer indices.**
  `index_select_wrapper` passed the index tensor straight to the kernel, whose
  pointer arithmetic `inp + (rows*N + indices)` is only valid for integer
  offsets. The whisper decoder reaches `aten::index` (via `_meta_index`) with
  integer-VALUED `position_ids` tagged float32, so the kernel failed to compile
  (`pointer<fp16> and float32`). Cast a floating index to int64 at this single
  choke point both triton modes funnel through (R30-symmetric), matching torch
  (index_select requires a Long); compiled never touches this wrapper so it is
  R23-safe. Unblocks the whisper-large-v3-turbo triton decode crash (the
  transcription is still incorrect — a separate decoder compute bug, under
  investigation in the same sweep step).

- **Triton `encoder_decoder` flow (whisper): stale `dispatch_op` import + 3-D
  lm_head.** The flow imported `dispatch_op` (removed; the current API is
  `dispatch(op)(...)`), and fed a 3-D `[B,T,H]` hidden state to the strictly-2-D
  `mm` kernel (`too many values to unpack`). Both fixed (flatten leading dims for
  the lm_head matmul, unflatten the result). whisper-large triton now runs the
  encode+decode end-to-end; the transcription is still empty (a separate decode
  issue — next sweep step).

- **Triton `bitwise_not` on a BOOL tensor is the logical NOT, not the bitwise
  complement.** The kernel did `~x`; for bool (0/1) that yields `~1=254` / `~0=255`
  — both non-zero, so the result read back as an **all-True** bool tensor. Bool is
  now routed through `logical_not` (integer dtypes keep the `~x` complement).
  Surfaced by the parakeet conformer attention padding mask `~(arange < len)`: an
  all-True mask made `masked_fill` overwrite every score with -1e4 → uniform
  softmax → dead self-attention → garbage transcription. With this + the new
  `aten::repeat` + the LSTM-on-triton switch, **parakeet-tdt-1.1b triton
  transcribes byte-identical to the compiled oracle** (closed).

- **Triton `abs` of a complex tensor returns the real magnitude (and stops a
  heap corruption).** `abs_wrapper` had no complex branch: for complex64 it ran
  the element-wise float kernel over the interleaved `[real, imag]` storage —
  computing `|interleaved float|` instead of `sqrt(re²+im²)`, mis-typing the
  output as complex, and (numel-vs-byte mismatch) corrupting the host heap
  ("corrupted double-linked list"). Exposed by the Kokoro iSTFT source STFT
  (`abs(_fft_r2c(...))`) feeding the generator noise-convolutions. Now computes
  the magnitude R33-pure from the stride-2 real/imag views; bit-exact vs torch
  at all batch sizes (`sin`/`cos`/`mul`/`exp`/`angle` already had complex
  branches — `abs` was the gap).

- **Triton `remainder` (`%`) follows the divisor's sign (torch / Python),
  not C `fmod`.** The kernel used Triton's `%`, which follows the dividend's
  sign, so negative inputs never wrapped into `[0, divisor)` — `remainder(-0.1,
  1)` returned `-0.1` instead of `0.9`. The Kokoro iSTFT SineGen wraps an
  unconstrained (sometimes negative) phase with `% 1`; the negative phases
  stayed negative and corrupted the harmonic source. Now `a - floor(a/b)*b`;
  bit-exact vs torch for floats and integers.

- **Triton `bilinear2d` upsample grid no longer swaps the OH/OW axes.** The
  launch grid passed `(cdiv(OH,BX), cdiv(OW,BY), …)` while the kernel indexes
  `program_id(0)→ow`, `program_id(1)→oh`. Invisible for square outputs (the
  image-upscaler case, OH==OW), but for non-square outputs most output columns
  never received a program and were left unwritten — breaking every 1-D-as-2-D
  use (`upsample_linear1d`, H=1), e.g. the Kokoro iSTFT SineGen phase resample
  (256↔76800). Now `(cdiv(OW,BX), cdiv(OH,BY), …)`; bit-exact for square,
  non-square, and 1-D. With these two fixes the Kokoro-82M triton decoder
  produces intelligible speech (whisper STT: "Hello, world.").

- **Triton `batch_norm` instance-norm path: null pointers for absent
  weight/bias/running, never the input tensor.** `batch_norm_wrapper` substituted
  the input `x` for absent `weight`/`bias`/`running_mean`/`running_var`. The kernel
  guards each with `if *_pointer:`, so a non-null `x` defeated the guard — it read
  `x` as the affine scale/shift and wrote momentum-blended running stats back into
  the input. `instance_norm` / AdaIN (training=True, no affine, no running buffers
  — the StyleTTS2 AdainResBlk1d norm used throughout the Kokoro predictor and
  decoder) was silently corrupted; standard `batch_norm` (real buffers) never
  tripped it. Now passes null pointers and guards the running-stats update with
  `if running_mean_pointer:`. Eval `batch_norm` (running+affine) stays
  byte-identical; `instance_norm` is bit-exact vs torch across all Kokoro shapes.

- **Triton float-math unary ops promote integer input to float (PyTorch type
  promotion).** `rsqrt`/`sqrt`/`log`/`exp2`/`reciprocal`/`erf`/`tan` preserved an
  integer input dtype through `NBXTensor.empty_like`, computing in int and
  truncating: `rsqrt(int64 2)` returned `0` instead of `0.7071`. The Kokoro
  AdainResBlk1d residual `(h + sc) * rsqrt(2)` (the `2` traced as an int64 scalar;
  torch auto-promotes) collapsed every block output to zero. A shared
  `_promote_int_unary` upcasts integer/bool input to float32; no-op for floats.
  With both fixes the Kokoro-82M triton predictor matches the compiled oracle
  op-for-op (embedding → CNN → BiLSTM → duration → alignment → F0/N blocks).

- **Triton metadata `expand` / `view` / `reshape` now resolve trace-baked shapes
  against the runtime numel** — mirror of the compiled `_make_expand` /
  `_make_view_reshape` runtime fixes that the triton metadata-op port had dropped.
  The graph bakes trace-time shapes; a variable-length model (e.g. the Kokoro
  predictor) can have a different runtime length on a dimension left concrete in
  the packaged graph. `expand` now uses the input's actual non-1 dim when it
  differs from the baked target (the only valid `expand` resolution);
  `view`/`reshape` infer the single changed dimension when the baked product
  mismatches the input numel (NBXTensor's `view` re-strides blindly without
  validating, so the mismatch is detected proactively). Footprint-screened
  (`NBX_DEBUG_META_RESOLVE`): zero activations on TinyLlama (LLM, runtime seq !=
  trace seq) and Sana 1024 (diffusion) — byte-identical there; fires only on the
  variable-length case it fixes. Restores R30 compiled/triton parity.

### Fixed

- **Triton 1-D ZIP-archive constant loading uses the declared (pickled view) shape,
  not the raw storage size.** `_load_constant_triton` "trusted the
  bytes" for a 1-D constant whose torch.save storage held more elements than the
  declared shape — but `torch.load` (native path) reconstructs from the pickled
  shape (a view into the storage), not the storage size. The Kokoro iSTFT window is
  a [20] view into a 21-float storage; triton yielded [21] (decoder frame*window
  broadcast failed) vs native [20]. Now: when the storage is at least the declared
  size, slice to the declared element count (matching torch.load); only a genuinely
  shorter storage falls back to its length. Restores native/triton parity.

### Fixed

- **Triton `_fft_r2c` accepts the ATen `int[]` `dim` argument** (was indexing a
  tuple with the list). Partial — the FFT wrappers remain radix-2 pow2-only; the
  Kokoro iSTFT (n_fft=20, non-pow2) needs a DFT-via-matmul path (next).

### Added

- **Triton complex arithmetic** (`mul` by the imaginary unit `1j`, `real·complex`,
  `complex·complex`; complex `exp` = e^a(cos b + i sin b)) and **`unfold_backward`**
  (overlap-add scatter, pure @triton.jit) — the iSTFT phase-reconstruction +
  overlap-add. NBXTensor gains a `numpy()` output-boundary method; `save_waveform`
  duck-types torch vs NBXTensor. All isolation-validated vs torch (mul·1j exact,
  exp(i·x) 6e-8, mul(mag,e) 1.5e-8, unfold_backward 4.8e-7). Complex branches are
  no-ops for real operands (R23-safe). `_fft_c2r` now accepts the ATen int[] dim arg.


- **NBXTensor complex64/complex128 support** (the declared-but-stubbed complex
  dtype, now completed in the engine): `view_as_real`/`view_as_complex` (zero-copy
  reinterprets, interleaved [real,imag] like numpy `<c8`), `.real`/`.imag` strided
  views, and a real `complex(real,imag)` builder (`complex_wrapper`, was a stub
  that dropped the imaginary part). `angle` now computes `atan2(imag,real)` via a
  new libdevice atan2 kernel (was a stub returning the imag part).
- **Triton non-power-of-2 FFT (DFT-via-matmul)** for `_fft_r2c` / `_fft_c2r`. The
  radix-2 butterfly only handles pow2 lengths; an iSTFT head with n_fft=20 needs a
  DFT computed by matrix multiply (cos/sin basis matrices @ frames → complex pair;
  Hermitian-symmetric inverse for c2r). New R33-pure `kernels/ops/dft.py` basis
  kernels. `fft_r2c` now also returns the FULL complex pair on the pow2 path (it
  dropped the imaginary part before). Model-agnostic (routes on N, never on model
  identity); validated vs torch.fft on N=20 (non-pow2) and N=16 (pow2): r2c/angle/
  c2r/round-trip all <=1e-4.


- **Triton audio flow: fixed-length-decoder chunking** (`_try_chunked_forward`,
  triton-pure mirror of the compiled `AudioFlow._try_chunked_forward`). An iSTFT
  vocoder / codec decoder bakes its window-norm divisor + `as_strided` framing at
  the trace frame count, so it must run at exactly the graph seq_len. A longer
  runtime input is split into trace-length blocks (the primary frame input chunked;
  frame-dependent aux inputs chunked synchronously, detected by runtime-dim !=
  graph-dim; static inputs like the style vector passed whole), each block decoded,
  and the waveforms concatenated. Data-driven (triggers only on a 3D input whose
  runtime seq_len differs from the trace seq_len) — no model-name branching. R33:
  NBXTensor narrow/zeros/cat + the existing boundary `_torch_to_nbx` for the torch
  voicepack style; zero torch compute.


- **Triton transposed convolution** (`aten::convolution` with `transposed=True`,
  1D and 2D, groups + dilation aware). `conv2d_wrapper` previously dropped the
  `transposed`/`output_padding` flags when delegating 1D convs, so a
  `ConvTranspose1d` was computed as a regular strided conv — *halving* the length
  instead of *doubling* it (Kokoro F0/N ProsodyPredictor depthwise pool: 62->31
  instead of 62->124). Wired the existing scatter `conv_transpose2d_kernel`
  (extended with groups via `C_in_per_g`/`C_out_per_g`, dilation, and a scalar
  weight-load fix; `groups=1` reduces to the original behaviour) through a new
  `conv_transpose_wrapper`. Unit-tested vs torch `F.conv_transpose1d/2d`
  (depthwise + groups=1 + 2D), max|diff| 0.008 (fp16 ULP).
- **Triton `reflection_pad1d`** — R33-pure narrow+flip+cat, the last-dim case of
  `reflection_pad2d_wrapper` (vocoder conv/iSTFT pre-pad).

### Fixed

- **Triton `upsample_bilinear2d` kernel crash + `upsample_linear1d` length.** The
  bilinear kernel called `.to()` on a specialized Python int (`(IH-1).to(...)`),
  crashing whenever exercised; replaced with arithmetic float promotion robust to
  both Python-int and tl-scalar args. `upsample_linear1d` now prefers the scale
  factor over the baked trace `output_size` (recompute from the live input length,
  like `upsample_nearest2d_wrapper` already does) so a variable-length audio
  decoder upsamples by the right ratio instead of desyncing from its sibling
  nearest upsample. Bit-identical when trace == runtime.


- **Triton mode now executes LSTM (`aten::lstm`) models** via a pure-Triton LSTM
  kernel — NBXTensor + Triton wrappers (matmul/sigmoid/tanh), zero torch, zero
  NumPy compute; bidirectional, multi-layer, batch-first. Validated bit-close to
  the compiled reference (max|diff| ~3e-4) and end-to-end on Kokoro (per-phoneme
  durations match the reference implementation element-wise).

### Fixed

- **Triton `round` and `repeat_interleave` now work.** `round` referenced a Triton
  math symbol absent in the installed version (it was crashing); it now uses the
  round-half-to-even libdevice primitive (matches the reference). `repeat_interleave`
  is now dispatched by argument shape — the single-tensor "interleaved indices"
  overload (used by Kokoro's duration→alignment build) was misrouted to the
  scalar-repeats path — and its tensor path no longer calls non-existent tensor
  methods.

- **VibeVoice-1.5B text-to-speech is now supported** via a next-token-diffusion
  generation flow. An autoregressive language model emits a control token per
  step; on each speech step a diffusion head samples one acoustic latent
  (classifier-free guidance, prompt-faithful by default), which is decoded to a
  waveform chunk and re-encoded as semantic feedback to the model. Produces
  24 kHz speech that transcribes back to the input text (STT-validated).

### Fixed

- **Triton mode now promotes sequence-length-dependent `expand` sizes given in
  the wrapped-list form, matching compiled mode.** The triton symbolic-promotion
  pass handled only raw-list expand sizes, leaving the wrapped form's trace-time
  length literal — so models with such an expand (e.g. a BERT-style token-type
  embedding, or a RoPE position expand) failed or diverged at a runtime length
  different from the trace length. Triton now mirrors the compiled promotion,
  restoring four-mode parity.

- **Kokoro-82M text-to-speech now runs end-to-end on the native forward path for
  prompts of any length.** The runtime previously padded or truncated every
  prompt to a fixed 23-phoneme length — collapsing every prompt to one fixed
  duration and silently cutting longer text — and ran the prosody predictor via a
  hand-rolled fallback. It now feeds the true phoneme sequence so the sequence
  length is bound per utterance, and chunks the iSTFT decoder across fixed frame
  blocks so utterances of any length synthesize. Combined with corrected
  prosody-predictor durations, the output now matches the reference
  implementation in per-phoneme duration and total length, and is STT-correct for
  short and long prompts (validated for 12-, 48-, and 172-phoneme inputs).

- **Models with structurally-repeated layers** (e.g. tokenizer stacks whose
  layers share identical per-layer names) could silently load some weights into
  the wrong positions, corrupting their output. Such models now load their
  weights correctly; models that already loaded correctly are unaffected.

- **Windowed-attention projectors (Q-Former style) now compute the correct
  number of windows for input sequences whose length is not a multiple of the
  window size.** Such a projector pads its sequence up to a multiple of a window
  size and reshapes into `ceil(seq / window)` windows. The runtime previously
  derived the window count from a floor of the trace-time padding, so the query
  and key attention branches disagreed on the window count and the projector
  attention failed with a tensor-size mismatch on any input whose length was not
  a multiple of the window size. The window count and the dynamic pad are now
  modelled consistently, and the windowed dimension propagates correctly through
  the projector's output reshapes.

### Changed

- **RMSNorm fp32-precision handling is now centralized and shared across
  execution modes.** The fp32 variance upcast that protects RMSNorm against
  fp16 overflow was previously duplicated in three places (the compiled fused
  path, the op-by-op sequential path, and the VibeVoice speech path); the three
  copies are now a single source, guaranteeing identical numerical behaviour in
  every mode. Compute-dtype resolution in the audio flows likewise now reads
  through the dtype engine instead of local copies of the dtype map. No change
  to output for any model.

- **Kokoro-82M no longer pads short-prompt audio to a fixed ~10 s
  window.** The native predictor stage previously force-scaled the
  predicted phoneme durations so they summed to exactly the traced
  128-frame decoder window, stretching a short prompt several-fold
  (e.g. "Hello world." filled ~3 s of elongated phonemes). Durations
  that already fit the window are now kept as predicted and the
  synthesised silent tail is cropped to the spoken content, in both
  compiled and `--triton` modes.

### Changed

- **Chatterbox TTS conditioning now runs the model's speaker + Perceiver
  conditioning encoder instead of a hand-rolled approximation.** The runtime
  previously hand-rolled the conditioning and skipped the Perceiver resampler,
  producing a 2-token conditioning instead of the model's 34-token one (1
  speaker + 32 Perceiver + 1 emotion); the speech model then over-generated
  (1000+ tokens of garbage). The conditioning is now produced by running the
  embedded conditioning-encoder component end-to-end on the default voice,
  matching the vendor reference, so the speech model generates the correct
  ~1 s of tokens. The vocoder is fed the reference voice from the same embedded
  conditioning, and `--reference-audio` is wired through the CLI. Coherent
  vocoder audio is still blocked on a separate sequence-length symbolic
  limitation (tracked as P-SYMBOLIC-ARANGE-SUM-FROM-ITEM).

### Fixed

- **Orpheus TTS now produces intelligible speech instead of noise.** The SNAC
  audio codec expects each 7-token frame de-interleaved into its three
  hierarchical codebook levels in a specific order (level 0: position 0; level
  1: positions 1 and 4; level 2: positions 2, 3, 5, 6). The decoder split the
  frame contiguously (positions 1-2 to level 1, 3-6 to level 2), scrambling the
  hierarchy and producing noise. The de-interleaving now matches the model's
  layout, and the synthesized speech transcribes back correctly.

- **A sequence length frozen inside an `expand` size is now made dynamic at
  runtime, like it already was for `view`/`reshape`.** The symbolic-shape
  promotion unwrapped the `{type:list,value:[...]}` size form for
  `view`/`reshape` but not for `expand`, so an `expand` whose size carried the
  trace-time sequence length (e.g. the rotary-embedding position expansion
  `[1, 1, seq]`) kept that length literal. At decode the single new position
  was then broadcast back to the full trace length, corrupting the rotary
  embeddings and the per-step query/key. `expand` now unwraps and promotes the
  same way; non-sequence models are unaffected (the path only runs when the
  graph has sequence-length symbols).

- **Autoregressive models whose graph computes positions with a two-argument
  range (`arange(0, seq_len)`, the Llama `cache_position` form) no longer
  produce an empty position range at decode.** The KV-cache decode position
  shift assumed the single-argument `arange(seq_len)` form and mis-read the
  two-argument form's first argument as the length, yielding an empty range
  that collapsed the rotary-embedding table and the per-step query/key. The
  shift now handles both forms; single-argument behaviour is unchanged.

- **Models with cross-attention (different query and key/value sequence
  lengths) no longer crash with a shape-mismatch error inside attention.**
  The runtime's attention layout-fixup keyed its "is this key transposed?"
  decision on the sequence axis, which is only reliable for self-attention
  (equal query/key lengths). For cross-attention — a Perceiver resampler with
  32 latent queries over 150 prompt keys, or any encoder/decoder cross-attend
  — it wrongly reshaped a correctly-laid-out tensor and aborted. The decision
  now keys on the head-dimension axis, which is invariant across both cases;
  self-attention behaviour is unchanged.

- **Audio TTS models with an LLM backbone no longer crash with an
  out-of-memory error on the text prompt.** The text was tokenized with
  padding up to the model's full context length, so a short prompt was
  expanded to tens of thousands of tokens; a language model whose attention
  mask scales with sequence length then tried to allocate a mask of
  `context_length²` (VibeVoice: a 131072×131072 mask ≈ 16 GiB) and ran out of
  memory before producing anything. The prompt is now tokenized at its actual
  length.

- **Models with hand-written RMSNorm/LayerNorm no longer produce garbage on
  fp16 hardware (V100) from variance overflow.** Such norms compute the
  variance as `mean(x * x)`; the `x * x` squaring overflows fp16 (max 65504)
  for any activation magnitude above ~256, collapsing the normalisation to
  zero. The dtype engine now upcasts the squaring to fp32 on fp16 hardware
  (and leaves it untouched on bf16 hardware, where the exponent range matches
  fp32 and overflow cannot occur). This is the protection that lets OpenAudio
  generate correct speech (below), and pre-empts the same failure in any other
  model that squares large activations.

- **OpenAudio (Fish-Speech dual-AR) TTS now produces correct speech.**
  Previously the model emitted a constant carrier tone and never stopped
  (running to the token limit); once stopping was corrected it then drifted
  to unrelated words. The dual-AR generation path is fixed end to end: the
  slow backbone now sees the full generated context each step and emits the
  stop token on time, the fast/depth transformer generates the residual
  acoustic codebooks correctly, and a half-precision overflow in the depth
  transformer's normalisation (which collapsed the codebooks) was eliminated.
  "Hello world." now transcribes back as "Hello world!" via STT, in compiled
  mode. The CLI `--temperature` flag (and `--top-p` / `--repetition-penalty`)
  is now honoured for this model — `--temperature 0` gives deterministic
  greedy decoding — instead of always using the embedded defaults.

- **Kokoro-82M now produces intelligible speech.** Previously the
  output was babbling / unintelligible (e.g. "Hello world." came out
  as garbled vowels). Two bugs in the native predictor/text-encoder
  path were corrected: the phoneme padding mask was inverted (so the
  model ran on padding instead of the real tokens), and the text
  encoder's convolution block applied its normalisation and activation
  in the wrong order and with the wrong activation slope. "Hello world."
  now transcribes back as "Hello world." via STT. Prompts longer than
  the model's traced ~23-phoneme input window are still truncated
  (their surviving words are correct); lifting that limit is tracked
  separately.

- **Audio TTS runs no longer write a duplicate `output_<model>.wav`
  in the current working directory.** The compiled-mode and
  shared triton-mode audio flow handlers (`core/flow/audio.py`,
  `core/flow/audio_utils.py`) each saved the waveform directly to
  a hardcoded `output_<model>.wav` path relative to cwd in
  addition to the CLI's family-aware `save_audio` writer at
  `--output`. With `--output` passed, this produced two files
  (the requested one and the stray); without `--output`, it
  produced the legacy default-name file. The flow handlers now
  only deposit the waveform into the variable resolver; the CLI
  is the single writer (`output_dispatch.save_audio`). One
  `SAVED:` print instead of two.

### Added

- **PixArt and Sana now run in `--triton` mode without manual
  per-component flags.** The hardware allocator detects components
  at structural risk of fp16 overflow (build-time graph dtype is
  fp32, the model family is image or video, the conv2d-cascade
  count and conv2d-vs-attention ratio mark the component as VAE-class,
  and the target hardware does not natively support bf16) and pins
  those components to fp32 automatically. PixArt-XL-2 / PixArt-Sigma /
  Sana 1024 produce coherent images in `--triton` out of the box;
  previously the VAE saturated to NaN. Other families (LLM, audio,
  upscaler, multimodal, …) are unaffected by design; manual
  `requires_fp32_compute` continues to work as an explicit override
  and is honored on top of the auto-detect. `NBX_DISABLE_AUTO_FP32=1`
  bypasses the auto-detect for diagnosis (manual flag remains
  honored).

- **`NBX_DTYPE_CLAMP_DIAG=1` diagnostic.** When the dtype engine
  narrows an fp32/fp64/bf16 value to fp16 at an `aten::_to_copy`
  boundary cast and the source actually exceeds the fp16
  representable range (±65504), the engine clips pre-cast to ±65504
  to avoid the alternative of saturating to ±Inf and propagating
  NaN downstream. Enabling this env var logs a one-shot line per
  call site (target / passthrough branch) the first time the clamp
  is actually exercised, with the source dtype, shape, and max-abs
  value — useful when investigating activation-overflow symptoms
  on a new model. Default-off, zero runtime cost.

- **SwinIR classical super-resolution (x2 / x4)** is now supported
  via `neurobrix upscale`, across all four execution modes
  (compiled / sequential / triton / triton-sequential) with
  numerically equivalent output.
- **Per-component fp32 compute opt-in.** Architectures whose
  activation range structurally exceeds the fp16 representable
  range (e.g. deep transformer super-resolution stacks without
  inter-block normalisation) can now declare
  `requires_fp32_compute` so Prism pins those components to
  float32 regardless of the hardware preferred dtype, instead of
  producing NaN/garbage at fp16. Strictly opt-in — existing
  models are unaffected.

### Added

- **Audio-conditioned LLM models now run in `--triton` mode.** Models
  that transcribe/answer from audio via an encoder→projector→LLM
  pipeline (e.g. Voxtral) previously only worked in the default
  engine. The `--triton` path now produces output byte-identical to
  the default engine for Voxtral (validated on a reference clip,
  greedy decoding).

### Fixed

- **LLM/MoE models in `--triton` are now deterministic run-to-run on
  V100-class GPUs.** Greedy generation of the same model and prompt
  could produce different text on consecutive runs: the attention
  kernel used on these GPUs was not bit-reproducible for some model
  shapes. Attention now routes to a bit-reproducible computation on
  the affected hardware whenever it is memory-affordable, so repeated
  runs are byte-identical. Larger image-diffusion models are
  unaffected (unchanged path).

- **MoE models in `--triton` now handle deactivated experts the same
  way as the default engine.** When a Mixture-of-Experts router
  skips an expert, that expert's accumulation into the combined
  output was being dropped in `--triton` mode (the deactivated-path
  was nulled instead of passing the running accumulator through),
  diverging from the default engine. Triton MoE output now mirrors
  the default-engine semantics for skipped experts.

- **MoE models in `--triton` are now deterministic run-to-run.**
  Mixture-of-Experts expert aggregation accumulated through a
  non-deterministic atomic add, so two identical greedy runs of the
  same model and prompt could produce different text. Aggregation is
  now a fixed-order deterministic reduction: repeated runs are
  byte-identical, matching the default engine.

- **Prefetch queue saturation is now logged, and is interruptible.**
  The component prefetch wrapped its enqueue in a bare `except:`
  that swallowed everything — including `KeyboardInterrupt` /
  `SystemExit` — and silently served the component uncached. It now
  catches only queue-full, logs a warning, and lets every other
  exception (and Ctrl-C) propagate.

- **An unknown input-synthesis method is now a clear error instead
  of a silently missing input.** A synthesis rule naming an
  unregistered method was skipped silently, leaving the input slot
  unset so the downstream component consumed garbage and produced
  silently wrong output. It now raises a descriptive ZERO-FALLBACK
  error listing the known methods.

- **A corrupt VAE profile is now a clear error instead of a silently
  wrong image.** When a model's VAE `profile.json` existed but failed
  to parse, the output processor silently fell back to
  `clamp_before_normalize=False`, producing an out-of-range image
  with no diagnostic. It now raises a descriptive error; an absent
  profile remains a legitimate defaults path.

- **`index_put` / `index_put_` scatter writes are no longer silently
  dropped in Triton modes.** Both ops were mapped to identity
  functions, so any model whose graph performs an indexed scatter
  write (MoE-v2 expert-output aggregation, KV-cache indexed writes,
  masked scatter) produced silently wrong output in `--triton` /
  `--triton-sequential` — no crash, no warning. A real Triton
  scatter kernel is now wired for the common case (one integer
  index on the leading dim, with/without accumulate). Unsupported
  advanced-indexing forms now raise a clear error instead of
  silently mis-scattering. Compiled mode was unaffected.

- **`linspace` now returns correct values in Triton modes.**
  Models whose graph contains `aten::linspace` (diffusion / video
  timestep schedules, positional grids) silently received
  uninitialised memory in `--triton` / `--triton-sequential` —
  no crash, no NaN, just wrong output. The Triton linspace kernel
  is now wired (bit-exact vs the reference in fp32, ≤1 ULP in
  fp16/bf16, exact endpoints). Compiled mode was unaffected.

- **Compiled mode no longer crashes on models with trace-time
  orphan scalar constants.** The constant-slot pre-population in
  the zero-overhead compiled path was unreachable dead code, so
  graphs containing a Python-scalar constant captured without
  embedded data (e.g. an attention-mask construction loop) left
  that arena slot unset and the consuming op received `None` →
  crash. The pre-population now runs as part of weight binding
  with a 0-dim scalar default (matching the sequential reference
  path), so these models execute correctly in compiled mode.
  Regular and KV-cache weight slots are unaffected.

- **Audio model loading is now portable across hosts**
  (`core/flow/audio.py`, `core/flow/audio_utils.py`,
  `triton/flow/audio.py`): the three `_find_model_config_path`
  variants no longer fall back to a non-portable per-container
  path entry. The runtime now resolves audio preprocessing
  configuration strictly from the `modules/processor/` or
  `modules/tokenizer/` directories embedded inside the .nbx,
  making audio models loadable on any host (Linux / macOS /
  Windows). Containers produced by older builds that do not
  embed these directories need to be re-imported to remain
  loadable.

- **Tiled conv2d output alignment — `_tiled_conv2d_spatial_*` and
  `_fused_upsample_conv2d_*` wrappers** (`kernels/ops/fused_upsample_conv.py`):
  two coupled math bugs caused band outputs to be shifted by `pad_h`
  rows on edge bands and by `halo_top` rows on internal-frontier
  bands. Edge bands double-counted the image-edge padding (the
  `max(0, -in_read_start)` term already provides it). Internal
  bands wrote `conv_band[:band_h]` to output positions
  `[oh_start:oh_start+band_h]` without offsetting by `halo_top`,
  so each internal-frontier band placed F.conv2d output row
  `oh_start+halo_top-1` at output row `oh_start`. Effect on
  Sana 4Kpx 16g compiled: 50 tiled VAE convs each shifted the
  output by 1 row, producing visually-coherent-but-shifted PNGs.
  Validated by `scripts/microtest_tiled_conv2d_small_scale.py`
  sweeping `(kh ∈ {1,3,5}, pad ∈ {0,1,2}, tile_factor ∈ {1,2})`
  — torch path post-fix is bit-exact vs `F.conv2d`. NBX paths
  (`_tiled_conv2d_spatial_nbx`, `_fused_upsample_conv2d_nbx`)
  fixed in a follow-up commit when the pattern-replace missed
  the two sites where `nbx_add(bias)` is interpolated between
  the `actual_band_h` cap and the indexed-write — post-fix 8/8
  OK, cos=1.0000 max_abs=0.0000. P-NBX-TILED-CONV2D-SMALL-SCALE
  steps 1 + 2.

## 2026-05-10 — P-SANA-4KPX-RUNTIME POINT 8 closure factuelle (audit perf compiled vs sequential)

Audit profile-driven du gap perf triton compiled (hot-loop
TritonSequence) vs triton_sequential (TritonSequentialDispatcher).
Mesures sur 1× V100 32 GiB, `NBX_DISABLE_AUTOTUNE=1` :

| | wall compiled | wall sequential | gap |
|---|---|---|---|
| Sana 1024 | 70.35 s | 73.91 s | −4.8 % |
| Sana 4Kpx | 511.78 s | 513.96 s | **−0.4 %** |

GPU util sustained pendant Sana 4Kpx triton = **83.5 % avg avec 67.5 %
du temps actif >90 %** → workload **compute-bound**. Borne haute
arithmétique du speedup compiled vs sequential = 100/83.5 = **1.20×**
< cible mandate 1.5× → cible **structurellement inatteignable**.
Toutes les hypothèses H1–H5 (silent fallback, no fusion,
contiguous-guard cost, defensive sync, degenerate hot-loop) sont
invalidées par lecture code + corrélation mesures (voir
`validation_outputs/p_sana_4kpx_runtime/point8_compiled_perf_audit/diagnostic_par_hypothese.md`).
Aucune modification de code. Closure factuelle conforme au mandate.
Pistes backlog : `P-TRITON-FUSED-KERNELS`, `P-CUDA-GRAPHS`,
autotune-ON re-mesure baseline.

### Added

- **Hybrid CPU+GPU runtime dispatch — Prism plans with mixed CPU/GPU
  components now execute correctly** (`core/runtime/executor.py`,
  `core/strategies/lazy_sequential.py`, `cli/__init__.py`):
  the executor previously bypassed strategy-level input prep for
  `lazy_sequential`, so a CPU component consuming the output of a GPU
  producer would either stall in implicit transfer or raise a
  device-mismatch error. The fix routes mixed-device plans through
  `strategy.execute_component`, which transfers inputs to each
  component's device before calling `executor.run`. The hybrid
  routing fires when (a) the active strategy is `lazy_sequential` or
  `cpu_execution`, or (b) the plan places one component on CPU and
  another on a GPU device. The CLI entry point also configures
  `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `os.cpu_count()`
  before any torch import so MKL/oneDNN can parallelise the CPU
  portion of a hybrid run. **User-visible effect**: Sana 4Kpx on
  1× V100 16 GiB no longer fails — the VAE runs on host CPU with
  text_encoder and transformer on GPU. Wall-time is dominated by
  the 4096×4096 CPU VAE decode (≥30 min on 40-core hosts; in line
  with Doctrine R35 perf-libre principle).

- **Hybrid CPU+GPU placement — components that overflow the GPU route
  to CPU automatically** (`core/prism/solver.py`,
  `config/hardware/v100-16g.yml`): `_place_component` (the per-component
  cascade used by the `lazy_sequential` strategy) gains a Strategy 4
  CPU fallback. Components that don't fit any single GPU even with
  zero3 weight offload are placed on host CPU; smaller siblings stay
  on GPU. Result: `lazy_sequential` now produces hybrid plans like
  `vae → cpu, text_encoder → cuda:0, transformer → cuda:0` on a
  V100 16 GiB system running Sana 4Kpx. Also adds a 3 GiB
  driver/library overhead reserve (`_OOM_RESERVE_MB`) to both
  `_place_component` Strategy 1 and `_try_single_gpu` cold-mode
  budget, so the planner no longer accepts plans that fit the
  activation estimator but OOM at runtime inside conv outputs.
  The `v100-16g` hardware profile now carries a generic `cpu:`
  section (16-core, 128 GiB, AVX2-class) so the new strategy has a
  RAM budget to consult on minimal profiles. **User-visible effect**:
  on hardware where a single component (typically the diffusion
  VAE at high resolution) exceeds GPU VRAM, the model still produces
  output — VAE runs on CPU while the GPU handles smaller components.
  Wall-time is unbounded (4Kpx VAE on CPU is genuinely a 30-60min
  workload on 16-core hosts) but availability is guaranteed.

- **CPU-only execution support — Prism never refuses on hosts without
  GPUs** (`core/strategies/cpu_execution.py`, `core/prism/solver.py`,
  `core/prism/structure.py`, `config/hardware/cpu-only-x86.yml`):
  new `cpu_execution` strategy added as the last entry of every
  Prism cascade (single-GPU profiles, multi-GPU profiles, and brand-
  new pure-CPU profiles). When the cascade reaches it, every
  component runs entirely on host RAM via the PyTorch ATen native
  CPU dispatcher (`--compiled` and `--sequential` modes). The
  strategy validates that `sum(component totals) <= cpu.ram_mb * 0.7`
  at planning time and selects "cpu" as the placement device. Thread
  configuration is wired automatically from the host profile (cores,
  threads, architecture, features) via the existing
  `apply_cpu_config` path — no hardcoded thread counts. **User-visible
  effect**: NeuroBrix users on developer machines, CI runners, and
  any host without a GPU can now run any model end-to-end via
  `--hardware cpu-only-x86` (or any future CPU-only profile).
  Wall-time is unbounded but the pipeline always produces output.
  Validated: TinyLlama-1.1B-Chat-v1.0 generates a coherent 39-token
  haiku in 6.03 s on a 40-core Xeon Gold 6230 + 256 GiB RAM via
  `--compiled` mode. Triton-CPU integration for the `--triton` /
  `--triton-sequential` modes is a follow-up. No GPU code paths are
  affected — `cpu_execution` scores below every GPU strategy and is
  only chosen when nothing else fits or when the profile reports
  zero GPUs.

- **Op-level memory diagnostic** (`core/runtime/graph_executor.py`):
  new `NBX_LIVENESS_AUDIT_AT_OP_UID=<op_uid>` environment variable
  for the `triton_sequential` execution path. When set, prints at
  the moment the matching op is about to execute: the device-tracked
  live memory counter, the per-tid breakdown of the tensor store
  (sorted by size), and any tensors held past their graph-declared
  last use. Useful for users diagnosing GPU memory pressure on large
  diffusion models (e.g. Sana 4Kpx) — points to the exact op where
  intermediate tensors accumulate and whether the runtime liveness
  matches the graph declaration. Inactive by default (env var unset
  → zero overhead).

## 2026-05-11 — POINT 10 P-PRISM-NEVER-REFUSE remontée condition #2

Investigation extensive du blocker Sana 4Kpx sur 1× et 2× V100 16 GiB.
**Cible binaire NON ATTEINTE** ; remontée selon condition de sortie #2
(blocker architectural >200 lignes hors scope). 6 stratégies testées
end-to-end (`single_gpu`, `single_gpu_lifecycle`, `lazy_sequential`,
`zero3`, pool-tuned `triton_sequential`, `--sequential` PyTorch native +
`expandable_segments`) ; **toutes OOM au même endroit** : `aten.convolution::62`
au runtime peak ~17 GiB (live ~13 GiB + 4 GiB conv output structurel).
Le post-POINT-9 estimator prédit 12 GiB pour VAE, cohérent avec le live
mesuré ; **l'estimator n'est pas le bug**, c'est la model size structurelle
qui dépasse 16 GiB hardware. Audit doctrinal model-agnostic : zero violation
dans le code actif (uniquement des commentaires historiques mentionnant
des noms de modèles). Audit cascade : 9 stratégies effectivement câblées
(message d'erreur misleading qui listait 5/9 est fixé dans cette session).
**Acquis 32 GiB POINTS 7-9 préservés** ; aucune modification de runtime.
Chantiers backlog ouverts : `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT`
(pour 2× 16 GiB), `P-PRISM-CPU-FALLBACK-EXECUTION` (pour 1× 16 GiB,
nouveau, pour respecter pleinement la doctrine "Prism never refuses").

## 2026-05-11 — P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE landed (POINT 9)

`PrismSolver._compute_memory` désormais tiling-aware via two-pass dans
`estimate_peak_memory`. Trois nouveaux mécanismes substituent les
sentinels runtime au peak estimation :
(1) `zero_alloc_uids` — upsamples en fusion pair (`FusionUpsampleProxy`,
0 byte) et chaînes pixel_shuffle broadcast-aware F2a (expand stride-0,
clone `BroadcastClonePyroxy`, view pass-through) marquées 0 byte ;
(2) `inplace_adds` — adds résiduels avec liveness in-place aliasés
(output share buffer with reused input ; reverse-map `frees_at`
gère les last-uses étendus via alias) ;
(3) `force_compute_dtype_for_fp` — override des fp meta dtypes (graph
traced fp32) par le runtime compute_dtype (fp16) pour activations.
Mesure Sana 4Kpx VAE : peak estimé pre-fix **28 GiB** → post-fix
**12 GiB** (−57%) cohérent avec runtime mesuré 16.6 GiB.
Fix config bonus : `config/hardware/v100-16g-x2-01.yml`
`preferred_dtype: float16` ajouté (manquant — fallback fp32 inflait
de 2× toutes les estimations multi-GPU). Anti-régression matrice
4/4 cellules vertes (Sana 1024 / Sana 4Kpx 32g / PixArt-XL / TinyLlama).
Cible binaire Sana 4Kpx FULL sur 1× V100 16 GiB **non atteinte** —
runtime peak structurel ~17 GiB > 16 GiB hardware ;
intra-component VAE split `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT`
ouvert au backlog (explicitement out-of-scope POINT 9 par mandate).

## 2026-05-10 — P-SANA-4KPX-RUNTIME fully closed (full pipeline validation)

Total scope closed. Full pipeline Sana 4Kpx (text_encoder →
transformer 12 steps → VAE) produit pomme rouge sur 1× V100 32 GiB
en triton_sequential (510 s) ET triton compiled (515 s), peak VRAM
**16.6 GiB / 32.5 GiB** (éliminé mécaniquement par les fixes
numériques POINTS 1-6 sans toucher au memory pool). Anti-régression
matrice **10/10 cellules**. Tag élargi `p-sana-4kpx-runtime-fully-closed`
sur `90ac662` distingue cette clôture finale du tag numérique
`p-sana-4kpx-runtime-closed` sur `a862fe0` (POINT 6 H2). Backlog
ouvert : `P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE` (estimator
tiling-aware pour VRAM-contraintes ≤ 16 GiB) et
`P-MULTI-GPU-NBX-ADAPTER` (priorité abaissée — 1× 32 GiB suffit en
production).

## 2026-05-10 — P-SANA-4KPX-RUNTIME POINT 7 closure totale (full pipeline + anti-régression)

POINT 7 ferme le scope restant après POINTS 1-6 H2. Sana 4Kpx FULL
pipeline (text_encoder → transformer 12 steps → VAE) produit une
pomme rouge cohérente en **triton_sequential** (510 s) ET en
**triton compiled** (515 s) sur 1× V100 32 GiB, stratégie Prism
`single_gpu`. Peak VRAM mesuré = **16.6 GiB / 32.5 GiB** (51 % du
budget) — la mesure historique conv::62 OOM 26+8 GiB est éliminée
par les fixes POINTS 1-6, sans toucher au memory pool. Matrice
anti-régression : **10/10 cellules numériquement vertes** (Sana 1024
4 modes, PixArt-XL + PixArt-Sigma triton_seq, TinyLlama triton_seq,
Sana 4Kpx VAE-iso triton_seq, Sana 4Kpx FULL triton_seq + triton
compiled). Découverte factuelle sur configs ≤ 16 GiB :
`PrismSolver` estime activations VAE Sana 4Kpx = 28 GiB worst-case
sans intégrer l'op-level tiling kernel-embedded (runtime réel
16.6 GiB) → rejette au planning avec `ZERO FALLBACK` avant que le
tiling ne puisse engager. Ouvre deux chantiers backlog :
`P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE` (estimator tiling-aware)
et `P-MULTI-GPU-NBX-ADAPTER` (déjà nommé). Artefacts R29 :
`validation_outputs/p_sana_4kpx_runtime/point7_full_closure/`.

## 2026-05-10 — P-SANA-4KPX-RUNTIME closed (numerical correctness)

Sana 4Kpx VAE in triton_sequential mode now produces coherent output.
Chantier closed via 9 structured commits (POINTS 1-6 H2, refs `ea8e8e2`
through `a862fe0`, tag `p-sana-4kpx-runtime-closed`). Root cause final:
missing `.contiguous()` after NBXTensor slice in tiled conv wrappers —
flat-indexed downstream wrappers were reading wrong memory addresses on
non-contiguous views. Structural acquisitions: TritonDtypeEngine
matured (registry-resolved `activations_fp16_safe` flag now
runtime-effective for the first time), halo-based tiled conv2d (50-75×
divergence reduction), contiguous-guard pattern documented as
architectural rule. Full pipeline 4Kpx remains blocked by live-watermark
memory gap — separate chantier `P-TRITON-LIVE-WATERMARK-AUDIT`.

## [Unreleased]

### Fixed

- **`--sequential` / `--triton_sequential` autoregressive decode raised IndexError on the very first decode step for any model using a pre-allocated RoPE cache** (`src/neurobrix/core/runtime/graph_executor.py`, `src/neurobrix/core/runtime/graph/sequential_dispatcher.py`): TinyLlama-style models store `cos_cached` / `sin_cached` as full `[max_position, head_dim]` buffers and the traced graph slices them to `[trace_seq_len, head_dim]` before indexing by `position_ids`. The seq-len patcher rewrote the slice `end` to runtime_seq_len, which collapsed the slice to `[1, head_dim]` during decode (runtime_seq_len=1) — then `aten::index(sliced, position_ids=[N])` for any N≥1 hit `IndexError: index N is out of bounds for dimension 0 with size 1`. The fix mirrors the compiled-mode rewrite in `compiled_sequence._promote_seq_len_scalars_to_symbolic`: when the slice's source weight name matches `cos_cached` / `sin_cached`, the `end` is forced to the full table dim so absolute-position indexing works at any decode step. The sequential-side analogue of `update_seq_dependent_constants` (`_adapt_seq_dependent_weights` + `_recompute_rope_seq_dependent`) is wired for the alternative `constant_T_*` RoPE storage convention. **User-visible effect**: `neurobrix run --sequential` for TinyLlama (and any LLM with named RoPE-cache buffers) now produces coherent text on both GPU and pure-CPU hosts; previously it raised on token 1. Anti-régression: TinyLlama sequential on CPU and GPU both validated 8-token coherent (see `validation_outputs/p_prism_never_refuse_s2/`).

- **NBX tiled conv2d backends produced 94% garbage on Sana 4Kpx triton path — `_fused_upsample_conv2d_nbx` and `_tiled_conv2d_spatial_nbx`** (`src/neurobrix/kernels/ops/fused_upsample_conv.py`): The two NBX band-streaming conv backends sliced their input along the H dim with `input[:, :, pre_start:pre_end, :]`. On an NCHW tensor that slice produces a **non-contiguous NBX view** (the C stride keeps its original H-based value while the slice covers a smaller H), violating the contiguous-storage assumption baked into every downstream Triton wrapper (`upsample_nearest2d_wrapper`, `conv2d_wrapper`, `constant_pad_nd_wrapper`). The torch backends always called `.contiguous()` at the equivalent points; POINT 5's NBX port (commit `ad9b7a3`) ported the halo logic but missed the post-slice `.contiguous()`. Microtest on Sana 4Kpx conv::55 (fusion site, fp16, 1×512×2048×2048 output) measured pre-fix max_abs_diff=167 vs torch reference of max_abs=154 with 94% of elements differing by >1.0 absolute across all tile_factors (2/4/8/16). Post-fix max_abs_diff=0.125 (one fp16 ULP) at every tile_factor. R33 compliance: `NBXTensor.contiguous()` is a legitimate NBXTensor method that materializes via the existing `_strided_copy` Triton kernel, NOT the forbidden `torch.Tensor.contiguous()`. Same defensive pattern as `group_norm_wrapper` (`wrappers.py:2169`). **User-visible effect**: Sana 4Kpx VAE-isolation triton_sequential now produces a coherent red apple PNG (POINT 6 victory in `validation_outputs/p_sana_4kpx_runtime/triton_dtype_engine_point6/sana4kpx_vae_iso_post_point6_RED_APPLE.png`), supersedes all prior "horizontal RGB bands" / "vertical streaks" partial-progress outputs. Sana 1024 and PixArt-XL triton anti-regression preserved (red apples). The residual full-Sana-4Kpx pipeline blocker is now the documented `P-TRITON-LIVE-WATERMARK-AUDIT` (memory, not numerics) — structurally orthogonal.

- **NBX `add_inplace_nbx` silent corruption on non-contiguous targets** (`src/neurobrix/kernels/wrappers.py`): the in-place add kernel uses flat 1D `tl.load/store(ptr + offset, ...)` indexing which assumes the target tensor is contiguous in memory. When Prism's last-use liveness analysis flagged an `aten.add` as in-place-eligible and the target happened to be a non-contiguous view (permute / transpose / strided slice), the flat offsets resolved to wrong memory addresses — silent corruption with no crash. Fix: at function entry, fall back to the standard non-in-place `add` (which goes through `_prepare_binary` + `expand+contiguous`) when `target.is_contiguous()` is false. Contiguous tensors keep the fast in-place path. **User-visible effect on Sana 4Kpx VAE-isolation**: the residual `aten.add::69` divergence cluster at op_idx 649+ (1.3% rel vs PyTorch oracle pre-fix) is eliminated — the in-place add path is now bit-equivalent to the non-in-place reference on any input shape. R30: in-place add is triton-only — compiled path uses torch's native in-place which already handles strided views correctly, so no symmetric edit needed.

- **NBX tiled conv2d band-streaming halo bug — `_tiled_conv2d_spatial_nbx` and `_fused_upsample_conv2d_nbx`** (`src/neurobrix/kernels/ops/fused_upsample_conv.py`): the two NBX backends of the band-streamed conv used `padding=(pad_h, pad_w)` on every band — including internal frontiers — so the kernel filled zeros at every band boundary instead of reading the next band's real rows (halo). The torch backend (`_tiled_conv2d_spatial_torch`, `_fused_upsample_conv2d_torch`) had used the halo-based algorithm correctly from day one (slice with halo rows, asymmetric F.pad only at image edges, conv with `padding=(0,0)`). The NBX backend now mirrors that algorithm via `constant_pad_nd_wrapper` for asymmetric H pad. Measured on Sana 4Kpx VAE: pre-fix had `max_abs_diff` ~50 (98%+ of elements differed) at conv::36/41/46 between modes; post-fix `max_abs_diff` ~0.7-1.4 (fp16 ULP range, ~50-75× reduction). Validated bit-equivalent within fp16 ULP to the torch path. R33 zero-torch preserved. **User-visible effect**: any model with diffusion-style spatial conv ops large enough to trigger op-level tiling (Sana 4Kpx and similar) gets bit-equivalent values between sequential and triton modes for those convs.

- **Per-component registry flags (`activations_fp16_safe` and any future `model_registry.yml` flags) now actually take effect at runtime** (`src/neurobrix/core/runtime/factory.py`, `src/neurobrix/core/runtime/graph_executor.py`): the runtime-direct registry lookup at `graph_executor.py:1925` (compiled mode, "déjà branché" Phase 1 doctrine) and the parallel sequential-mode lookup at line 1567 both read from `getattr(self._pkg, 'cache_path', None)`. **`self._pkg` was never an attribute on `GraphExecutor`** — only on the serving engine — so `_model_name` resolved to `None`, `get_component_flag` silently returned its `default=False`, and every `activations_fp16_safe: true` annotation in `model_registry.yml` was a decorative no-op until now. `ExecutorFactory.create` now poses `executor._cache_path = cache_path` after construction; both flag inits read `self._cache_path` instead. **User-visible effect**: Sana 1024 / Sana 4Kpx VAE+transformer (already annotated `activations_fp16_safe: true`) now actually get the unified `AMP_FP32_OPS` cast-back from POINT 2 (rsqrt / exp / log / *_norm / softmax / rms_norm / div) at compute_dtype, halving the VAE post-loop activation chain VRAM footprint by default — no `NBX_ACTIVATIONS_FP16_SAFE=1` env needed anymore. LLMs (no annotation) stay at default False → no behavior change. Validated end-to-end without env: `dtype_walk_4kpx_post_point2bis_NO_env.tsv` shows `rms_norm::0..3` tri-side outputs as `float16` (cast-back applied via registry-resolved flag); divergence count 291 matches the env-override walk; full anti-regression matrix preserved (Sana 1024 ×2 modes + PixArt-XL/Sigma + TinyLlama + Sana 4Kpx VAE-iso). Audit `grep self._pkg src/neurobrix/core/runtime/graph_executor.py` confirms only the two flag-init call sites referenced the missing attribute — no other plumbing-related bugs of this class.

### Changed

- **TritonDtypeEngine: unified `AMP_FP32_OPS` cast-back path, gated solely by `activations_fp16_safe` registry flag** (`src/neurobrix/triton/dtype.py`, `src/neurobrix/triton/sequential.py`, `src/neurobrix/core/runtime/graph_executor.py`): The previous opt-in design had two superposed gates — first `op_name in AMP_FP32_OPS`, then a separate `op_name in _AMP_FP32_OPS_OPT_IN_CAST_BACK` membership check that fragmented the doctrine: only `rms_norm` and `div` were eligible for cast-back, while every other AMP_FP32 op (`rsqrt`, `exp`, `log`, `layer_norm`, `batch_norm`, `softmax`, etc.) was hardcoded to keep fp32 output regardless of the model's annotation. The voie uniforme drops the second gate: every `AMP_FP32_OPS` member now goes through the same `_wrap_fp32_internal_compute_dtype_output` path that reads `activations_fp16_safe` at call time and casts back to compute_dtype iff True. `rms_norm` (a NeuroBrix custom op, not in PyTorch's `AT_FORALL_FP32`) is added to `AMP_FP32_OPS` so the unified pipeline applies to it. `TritonSequentialDispatcher.__init__` now accepts `activations_fp16_safe` and propagates it to the wrapper-global state (`_w.set_activations_fp16_safe`) at construction, mirroring `TritonSequence.run()`'s set/restore (without the try/finally because sequential mode doesn't nest within compiled mode). **Behaviour**: models without an annotation (LLMs by default) keep `activations_fp16_safe=False` → output stays fp32 → PyTorch-oracle parity preserved. Models annotated `activations_fp16_safe: true` (Sana 4Kpx VAE, Sana 1024 VAE) get cast-back across the full AMP_FP32 surface — VRAM-preserving fp16 throughput. **Anti-regression validated**: Sana 1024 ×2 modes / PixArt-XL / PixArt-Sigma all still produce coherent red apple PNGs; TinyLlama LLM still produces coherent prompt-faithful text; Sana 4Kpx VAE-isolation evolves from monochromatic green grid (post-POINT 1) to colored RGB band pattern (post-POINT 2), confirming color-channel reconstruction is now partially flowing through (Famille B / TilingEngine cluster grid signature still active per the next plan point). **Pre-existing limitation surfaced**: the registry lookup at `graph_executor.py:1925` (and the new sibling at line 1567 for sequential mode) reads from `self._pkg` which is not actually set on `GraphExecutor` — only on the serving engine. Both flag inits silently fall back to default `False`. Workaround: `NBX_ACTIVATIONS_FP16_SAFE=1` env override (validated working). Real fix is a follow-up factory wiring (out of P-SANA-4KPX-RUNTIME POINT 2 scope, escalated separately).

### Fixed

- **Triton-mode component-entry input dtype mismatch with PyTorch oracle (Sana / PixArt / DC-AE VAE chains)** (`src/neurobrix/triton/dtype.py`, `src/neurobrix/triton/sequence.py`, `src/neurobrix/triton/sequential.py`, `src/neurobrix/core/runtime/graph_executor.py`): Triton modes (compiled and sequential) were entering the runtime with the raw input dtype declared in the graph (typically fp32 for activation inputs like `input::z`), while the PyTorch sequential path casts each input to the component's compute_dtype at entry via `DtypeEngine`. On a fp16 component (Volta diffusion VAE), this asymmetry put `input::z` and the dtype-passthrough metadata chain (`unsqueeze → expand → clone → view`) on different precision tracks for the first ~100 ops of the VAE. The triton path now mirrors PyTorch's behaviour at the same logical boundary: a new `TritonDtypeEngine.cast_runtime_inputs(input_map, graph_tensors)` reads each input's declared graph dtype and casts to the engine's compute_dtype when the graph dtype is floating-point (preserves int64 / bool unchanged). It is invoked from `TritonSequence.bind_inputs` (compiled) and via a new `TritonSequentialDispatcher.bind_inputs` delegate at the triton_sequential store-loading boundary. **User-visible effect**: Sana 1024 / PixArt-XL / PixArt-Sigma triton paths produce the same coherent PNG as before (no regression — `validation_outputs/p_sana_4kpx_runtime/triton_dtype_engine_point1/`); Sana 4Kpx VAE-isolation remains green texture pending the TilingEngine cluster fix at op_idx 675-696 (separate point in the same plan). Universal: any model with fp32-tagged input::* tensors that should run at compute_dtype benefits without per-model annotation.

- **rms_norm and bias-add no longer materialize 8 GiB transients on Sana 4Kpx VAE** (`src/neurobrix/kernels/wrappers.py`, `src/neurobrix/kernels/ops/add.py`): two related triton-mode peak-VRAM fixes for the post-pixel_shuffle chain. (1) `rms_norm` now detects when the input was non-contiguous (e.g. the NHWC permute view in DC-AE blocks) and writes its output in place into the freshly allocated contiguous copy that `x.contiguous()` produced — the kernel reads each tile fully before storing, so input==output is per-tile safe and the second 8 GiB allocation is avoided. (2) `add` now routes the common bias-broadcast pattern (a multi-dim tensor + a 1D bias whose size matches the last dim) through a new `add_bias_broadcast_kernel` that reads `bias[offset % feat_dim]` directly instead of materializing an 8 GiB contiguous expand of the bias. Together with the upstream loop-variable fix, these advance Sana 4Kpx triton compiled by 27 ops past the previous OOM site, all the way to the final output convolution `conv::69` (the residual VAE pressure between consecutive 8 GiB tensors at 4096² is still the structural ceiling on V100). Universal: any model using rms_norm on a permuted view, or any add of a 4D activation with a 1D bias, benefits.

- **Triton-mode VRAM watermark drift on long op sequences (8 GiB per stale loop iteration on Sana 4Kpx)** (`src/neurobrix/triton/sequence.py`): a Python loop-variable retention in the per-op cleanup paths kept the most recently freed intermediate alive past its expected last-use, drifting the live VRAM watermark upward across consecutive ops with no cleanup work of their own. On Sana 4Kpx VAE this drift was 8 GiB per affected step. The fix explicitly drops the lingering reference at five cleanup sites, releasing the tensor at the same point its slot is logically dead. **User-visible effect**: Sana 4Kpx triton compiled now advances past the previous OOM at `aten.convolution::64`. The fix is model-agnostic — any model whose triton sequence has stretches of ops without cleanup work benefits from a tighter live-VRAM curve.

### Added

- **Pixel-shuffle broadcast-aware kernel via clone interceptor (F2a Approche C, eliminates 8 GiB clone materialization on Sana 4Kpx)** (`src/neurobrix/core/module/tiling_engine.py`, `src/neurobrix/kernels/ops/fused_upsample_conv.py`, `src/neurobrix/kernels/ops/pixel_shuffle.py`, `src/neurobrix/kernels/wrappers.py`): PyTorch decomposes `pixel_shuffle` as `unsqueeze -> expand -> clone -> view -> pixel_shuffle`. The clone in that chain materializes a contiguous copy of the broadcast view at full size — on Sana 4Kpx VAE the expand `(1,256,1,2048,2048) -> (1,256,2,2048,2048)` followed by `clone` allocates 8 GiB of duplicated channel data per call, even though the broadcast itself is metadata-only. NBX's `expand` already returns a stride-0 view (`kernels/nbx_tensor.py:1581`) that aliases the pre-expand tensor, so we can read through it directly. New static-detection step in `OpLevelTilingEngine._detect_pixel_shuffle_broadcast_chains` finds `expand -> clone -> view -> pixel_shuffle` chains in the DAG by comparing expand input/output shapes (the dim that grew from `1` to `factor` is the broadcast dim). For each match the engine wires three op_uid interceptors: clone wraps the stride-0 view in a `BroadcastClonePyroxy` sentinel (zero compute, `_nbytes=0` to be a no-op in the deferred-free accounting), view forwards the proxy unchanged, and pixel_shuffle reads through a new `pixel_shuffle_broadcast_aware_kernel` that uses the 5D strides (the `stride_v_b == 0` aliases the broadcast index for free) and writes a clean 4D contiguous output via `NBXTensor.empty`. Dual-backend by construction: in compiled / sequential modes the clone interceptor falls back to `input_tensor.clone(memory_format=torch.contiguous_format)` (mirroring the original DAG) and the view interceptor uses `.reshape()` (semantically equivalent to `.view()` on contiguous, but tolerant of upstream non-contiguity). **Validation**: Sana 4Kpx triton compiled now passes `aten.add::86` (the previous regression site) and advances to `aten.convolution::64` — the residual blocker is the post-pixel_shuffle live-watermark from multi-branch retention (24 GB peak with three 8 GiB residual tensors alive), which is a separate chantier (live-set audit). Compiled mode unchanged: PNG coherent in 74 s on V100 32 GB. Sequential mode: PNG coherent in 89 s. The fix is universal: any model decomposing pixel_shuffle through the same NBX expand+clone pattern benefits without further code changes.

- **Opt-in `gc.collect()` retry on cudaMalloc OOM** (`src/neurobrix/kernels/nbx_tensor.py`): when `NBX_GC_ON_OOM=1` and a `cudaMalloc` returns ENOMEM, the allocator triggers a cycle-collecting `gc.collect()` then retries the allocation once. Targets the case where Python frame-args lists or recursive call closures hold NBXTensor refs in cycles (so refcount alone won't free them). Default off; tested on Sana 4Kpx triton — does NOT recover the conv::62 OOM (the 25 GB live watermark is structural, not a Python-cycle leak).

- **Triton diagnostics: deeper OOM forensics — live-block walk + Python-referrer scan** (`src/neurobrix/triton/sequence.py`): the OOM handler at `_run_single_device` (gated by `NBX_LIVE_DUMP_ON_OOM=1`) now also walks `DeviceAllocator._cuda_ptr_size` directly and prints `[LIVE_BLOCKS dev=N] total=…MB big_blocks(>=100MB): CxS …`, exposing the actual cudaMalloc'd block distribution independent of arena membership. Then `[BIG_TENSORS]` walks `gc.get_objects()` for NBXTensor instances whose `_data_ptr` matches the >= 1 GB live blocks and prints each one's shape, ownership, and a summary of `gc.get_referrers` (object types holding a Python ref). On Sana 4Kpx triton this reveals 5 owns=True NBXTensors totaling 25 GB at the conv::62 OOM, in band shapes (1024+halo height) and full-4Kpx shapes — the structural intermediate-tensor pressure of the DC-AE decoder.up.0 block, not a fragmentation issue. Also added `NBX_FORCE_GC=N` (every-N-ops `gc.collect()` to test stale-Python-ref hypothesis; per-op diagnostic only, not for production due to overhead).

- **Triton diagnostics: opt-in env-var instrumentation for live-tensor tracking and OOM forensics** (`src/neurobrix/triton/sequence.py`, `src/neurobrix/triton/flow/iterative_process.py`, `src/neurobrix/kernels/wrappers.py`): three new opt-in env vars for triton-mode memory analysis. `NBX_LIVE_DUMP_EVERY=N` prints `[LIVE_TRACK op_idx=… op_uid=…] live=…MB` every N ops in `_run_single_device` so a log can be scrubbed for the exact op where live memory climbs. `NBX_LIVE_DUMP_ON_OOM=1` dumps the full arena breakdown (weights / inputs / intermediates / deferred queue) at the failing op when a `cudaMalloc` fails, plus `[UNLOAD] <comp>: live X→Y MB (freed Z MB)` lines from `iterative_process._execute_post_loop`. `NBX_DEPTHWISE_DISABLE=1` forces the generic `conv2d_forward_kernel` even when the depthwise signature matches, to bisect leakage suspicions. All gates are no-ops when unset; zero overhead in production.

- **Caching free-list pool in `DeviceAllocator` (opt-in via `NBX_ALLOC_POOL=1`)** (`src/neurobrix/kernels/nbx_tensor.py`): on `free_cuda` the pointer is returned to a per-device free-list pool instead of `cudaFree`, keeping the driver's internal heap intact across the churn of small/medium allocs typical of a triton forward pass. On `malloc_cuda` the pool is checked first (exact-size hit, then smallest-fit ≤ 2× the request); on `cudaMalloc` OOM the pool is flushed back to the driver and the malloc is retried — analog to torch's `CachingAllocator.release_cached_blocks → retry`. Live-byte counters are unchanged for pool returns (the block stays allocated from the driver's POV). Default off until validated across the full model surface; Sana 1024 hot regression intact at 42 s. The OOM error path also now reports a factual VRAM readout (`live_tracked / pool_cached / driver_free / driver_total`) for downstream diagnosis.

### Added

- **`[BIG_TENSORS]` diagnostic now identifies each tensor by graph tensor_id** (`src/neurobrix/triton/sequence.py`): the OOM forensic dump (gated by `NBX_LIVE_DUMP_ON_OOM=1`) builds a reverse arena lookup (`data_ptr → tid` via `_slot_to_tid`) and prints each big tensor's graph tensor_id, or `ORPHAN(not in arena)` if the NBXTensor is alive (held by Python refs) but not tracked by any arena slot. This trivially identifies liveness divergences vs structural pressure: an ORPHAN entry means a tensor's arena slot was killed but its `__del__` hasn't fired, and the leak source must be tracked down via referrer scan. On Sana 4Kpx triton compiled at conv::64 OOM, the diagnostic factually proved that 1 of 3 × 8 GiB live tensors is ORPHAN — case (b) liveness divergence over case (a) structural multi-branch pressure.

- **Op-level in-place residual add fusion in `OpLevelTilingEngine`** (`src/neurobrix/core/module/tiling_engine.py`, `src/neurobrix/kernels/wrappers.py`): a new universal detector scans the DAG of any component where Prism already triggered op-level tiling and finds residual `aten::add` ops where liveness analysis proves at least one input has its last use at this op. For each candidate, an interceptor reuses that input's buffer as the output (analog to in-place `target += other`), skipping a third allocation of `output_size` bytes. Applies wherever the residual-merge pattern appears at high spatial resolution: detection threshold defaults to 1 GiB output (output of an `aten::add` whose inputs share an identical 4D shape and at least one input has a single consumer = this add). On Sana 4Kpx VAE the detector finds 26 candidates (8 above 4 GiB at 4096×4096, 4 at 2048×2048, 14 smaller). New helper `add_inplace_nbx(target, other)` writes the result into target's buffer with the existing `add_forward_kernel`; it auto-handles dtype mismatches by casting the narrower operand to target's dtype, and falls back to standard `add` (non-in-place) when target is the narrower precision so semantics are never lost. The interceptor sets `self_manages_dtype=True` to skip the dtype-engine AMP wrap (consistent with the other op-level tiling interceptors). The fix changes the OOM site on Sana 4Kpx triton from `aten.add::86` (8 GiB allocation refused) to `aten.convolution::64` (later in the DAG); the residual blocker is now the structural multi-branch chain retention at the 4Kpx VAE level (see "Known limitations" below), which requires either multi-GPU NBX pipeline_parallel for the VAE component (P-MULTI-GPU-NBX-ADAPTER backlog) or a NBX caching allocator with splitting/coalescing (deferred). The in-place add fix is universal and benefits any model with multi-branch residual decoders.

### Fixed

- **In-place residual add interceptor no longer crashes compiled / sequential mode** (`src/neurobrix/core/module/tiling_engine.py`): the in-place add interceptor introduced by the multi-branch fusion fix was NBX-only via `add_inplace_nbx`, but `OpLevelTilingEngine` registers its interceptors on BOTH `CompiledSequence` (torch) AND `TritonSequence` (NBX). Compiled mode then crashed at the first registered residual add (`aten.add::66` on Sana 4Kpx) with `'Tensor' object has no attribute '_dtype'`. The interceptor now detects backend at runtime and routes: NBXTensor → `add_inplace_nbx`, `torch.Tensor` → `tensor.add_()` (with `torch.promote_types` fallback to non-in-place `torch.add` if target is the narrower precision so promotion semantics are preserved), unknown → `target + other * alpha`. **Validation**: compiled Sana 4Kpx now PASS in 82 s coherent PNG (was crash in 74 s post-introduction).

- **Sana 4Kpx now produces a coherent 4096×4096 PNG in `--sequential` mode** (`src/neurobrix/core/runtime/graph_executor.py`, `src/neurobrix/kernels/ops/fused_upsample_conv.py`): three orthogonal causes were blocking the non-compiled execution paths from benefiting from Prism's op-level tiling. (1) **R30 op-level tiling parity**: `_op_uid_interceptors` (the per-op_uid hook map populated by `OpLevelTilingEngine.register_into_graph_executor`) was wired only into `CompiledSequence` and `TritonSequence`. `_execute_native_op` (sequential mode) and `_run_triton_sequential` (triton_sequential mode) silently bypassed it — Sana 4Kpx VAE `aten.convolution::54` then arrived raw (36 GiB request) at cuDNN/Triton and OOMed on a V100 32 GB even though Prism had registered a band-streaming variant. The two non-compiled dispatchers now consult `_op_uid_interceptors` with the same priority order as compiled (`op_uid > op_type > native dispatch`). (2) **NBX bias broadcast 8 GiB OOM in band-streamed conv path**: `_fused_upsample_conv2d_nbx` and `_tiled_conv2d_spatial_nbx` were applying the bias add AFTER the full `(N, C, 4096, 4096)` output was materialized via `nbx_add(output, bias.view(1, -1, 1, 1))`. NBX's `_prepare_binary` materializes the bias broadcast as a contiguous tensor — 8 GiB at fp16 — which alone OOMed on V100 32 GB. The bias add now happens INSIDE the band loop on each smaller `(N, C, band_oh, conv_out_w)` slice (~250 MB instead of 8 GiB); mathematically identical because bias is per-channel. (3) **`tiled_rms_norm_spatial` import bug**: the NBXTensor branch tried to import a non-existent `rms_norm_wrapper` symbol, crashing the chain when the rms_norm interceptor fired. Renamed to the actual `rms_norm` function. **Validation**: `neurobrix run --model Sana_1600M_4Kpx_BF16 --sequential --prompt "a red apple" --steps 12` produces a coherent 4096×4096 PNG in 90 s (was FAIL @ conv::54 in 79 s). The artefact is at `validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/etape1_sequential.png`. The `--triton-sequential` mode advances much further with these fixes (now blocks on the residual NBX live-watermark gap vs torch CachingAllocator at the rms_norm boundary — separate chantier).

- **Sana 4Kpx triton VAE decoder no longer crashes on `'FusionUpsampleProxy' object has no attribute '_nbytes'`** (`src/neurobrix/kernels/ops/fused_upsample_conv.py`): the deferred-free accounting in `triton/sequence.py:_run_single_device` reads `_nbytes` from arena-resident objects to size the drain budget. `FusionUpsampleProxy` is the sentinel returned by the upsample-fusion interceptor and holds no GPU allocation of its own (it just references the pre-upsample tensor + scales). The proxy now exposes `_nbytes = 0` so it is a no-op in the deferred-free path; the actual pre-upsample tensor lives in its own arena slot and is freed when that slot is killed independently.

### Added

- **Depthwise convolution specialization in triton mode (453× speedup on the Sana 4Kpx VAE depthwise blocks)** (`src/neurobrix/kernels/ops/depthwise_conv2d.py` new module, `src/neurobrix/kernels/wrappers.py` routing in `conv2d_wrapper`): the generic `conv2d_forward_kernel` (im2col + `tl.dot`) is structurally inefficient for the depthwise pattern (`groups == in_c == out_c`, weight `(C, 1, kh, kw)`) — each group has K = `kh*kw` (often 9), so the matmul inner reduction loop runs once with most of the BLOCK_INF lane idle, yet the kernel still pays full launch + im2col cost for every group. Sana 4Kpx VAE's DC-AE blocks `groups=in=out=11200` triggered this: ~4.8 s per depthwise call vs cuDNN's dedicated path ~2.6 ms — a ~1800× gap. The new dedicated stencil kernel computes a tile of `(BLOCK_HW × BLOCK_C)` output positions per program block, accumulating the unrolled `kh*kw` stencil contributions in fp32 with no im2col buffer and no cross-channel reduction. `conv2d_wrapper` auto-routes when the depthwise signature is detected. **Bench on Sana 4Kpx shape (2, 11200, 128, 128) k=3 g=11200 fp16 (V100)**: cuDNN 2.659 ms, generic im2col ~4800 ms, new stencil **10.65 ms (4× cuDNN, 25% of cuDNN perf, 453× speedup vs the generic path), numerical max \|diff\|=0.0**. Pattern reference (open-source structural reading, no C++ vendoring): MultiPath/DepthwiseConv2d (CUTLASS sm_70 NCHW depthwise iterator), PyTorch PR #22302 (cuDNN dedicated depthwise path on Volta/Turing fp16 since cudnn 7600).

- **Triton conv2d_wrapper now self-tiles spatially when the per-launch output would exceed 4 GiB** (`src/neurobrix/kernels/wrappers.py`): a wrapper-internal threshold check splits the convolution along the H output dimension and streams band-by-band, recursing per band into the same wrapper (which falls through to the single-launch path on the now-smaller H). This is the kernel-level lever of P-SANA-4KPX-RUNTIME — internal to the wrapper, not a Prism op-level interceptor — and applies the same band-streaming pattern that runs the compiled-mode VAE on Sana 4Kpx (`_tiled_conv2d_spatial_nbx`). The threshold is configurable via `NBX_CONV2D_BAND_BYTES`. Sana 1024 stays single-launch (output ≤ 256 MB on every layer); Sana 4Kpx convolutions whose output would otherwise exceed 4 GiB (the 36 GiB OOM trigger included) are now intercepted at the wrapper layer in triton mode without requiring Prism interceptor wiring.

### Fixed

- **Sana 4Kpx 4096×4096 output no longer shows horizontal seams at tile band boundaries** (`src/neurobrix/kernels/ops/fused_upsample_conv.py`): the band-streaming fused upsample+convolution and the standalone tiled convolution now read a real halo of `(kernel-1) * dilation // 2` pixel rows on either side of every internal band frontier instead of letting the convolution's implicit zero-padding fill those rows. Image-edge halos are filled by `F.pad` with the original `pad_h` only at band 0's top and the last band's bottom; convolutions on every band run with `padding=(0, pad_w)` on the H axis so no synthetic zero rows are inserted at internal frontiers. The previous formula sliced the upsampled band to exactly `up_end - up_start` rows and let the convolution's `padding=0` fill the missing kernel-radius rows with zeros, which produced visible seams at every band frontier. Visual artefacts were also amplified by an off-by-one in the standalone tiled conv's `local_offset` calculation that took the halo offset for the conv's first output row. **Validation**: vertical-gradient mean over (W,C) on the new 4096×4096 output of `Sana_1600M_4Kpx_BF16` with prompt `"a red apple"` and `--steps 12` shows median 4.56 with no peak above the 3×median threshold (13.68) anywhere in the image except the top edge (classic image border, not tiling). The two former seam locations dropped from gradient 70 and 51 to 3.66 and 2.72 — a 19× reduction. Render time unchanged at ~73 s on V100 32 GB. The fix is mode-agnostic by construction; both compiled and triton paths use the same band-streaming functions.

### Changed

- **Triton mode now honors per-op_uid interceptors (R30 mirror of compiled mode)** (`src/neurobrix/triton/sequence.py`, `src/neurobrix/core/runtime/graph_executor.py`): the per-op_uid interceptor mechanism added for op-level tiling was wired only on the compiled-mode sequence. Triton and triton-sequential modes use a separate `TritonSequence` that consumed only op_type-wide interceptors, so the same overflow ops would still hit the native Triton path and OOM. The triton sequence now exposes `register_op_uid_interceptor` and `update_op_uid_interceptors` with the same priority (op_uid > op_type > native dispatch), and the graph executor mirrors every `register_op_uid_interceptors` call to the triton sequence as well as the compiled sequence (with a pending dict for the case where the triton sequence is compiled lazily after the call). End result: a single op-level tiling plan emitted by Prism takes effect identically across compiled / triton / triton_sequential modes.

### Added

- **Sana 4Kpx (and similar high-resolution VAE decoders) now produce a 4096×4096 image on a single 32 GB V100** (`src/neurobrix/core/module/tiling_engine.py`, `src/neurobrix/kernels/ops/fused_upsample_conv.py` new module, `src/neurobrix/core/prism/profiler.py`, `src/neurobrix/core/prism/solver.py`, `src/neurobrix/core/prism/memory_estimator.py`, `src/neurobrix/core/runtime/executor.py`, `src/neurobrix/core/runtime/graph_executor.py`, `src/neurobrix/core/runtime/graph/compiled_sequence.py`): the Sana 4Kpx VAE decoder uses fp32 spatial activations whose peak rises to ~28 GB (16 GB upsample output + 8 GB conv output + workspace) and whose individual convolutions request up to 36 GB of cuDNN workspace at full resolution — both unworkable on a single V100 32 GB. The runtime now detects, at planning time, ops whose output + workspace exceeds the per-GPU VRAM budget and intercepts them with band-streaming variants that never materialize the OOM intermediate tensor. Three new tiled implementations cover the patterns observed: (1) a fused upsample→convolution that stores no intermediate upsampled tensor by feeding the upsample band directly into the convolution band, (2) a standalone tiled convolution that bounds the cuDNN workspace per output band, and (3) a tiled rms-norm that exploits the channel-wise normalization to slice along H. Decision flow stays strictly prioritized: every standard placement strategy (single_gpu, component_placement, pipeline_parallel, block_scatter, weight_sharding, lazy_sequential, zero3) is tried first; op-level tiling is only applied to ops Prism flags as overflows under the chosen placement, and only those ops — sibling ops of the same type stay on the native path. Tile factors are computed analytically in O(1) from the per-GPU budget, the op's output bytes, and an mode-aware workspace estimate (cuDNN implicit-gemm bound for compiled mode, zero workspace for triton kernels), then rounded up to a power of two for halo alignment. **Validation**: `neurobrix run --model Sana_1600M_4Kpx_BF16 --prompt "a red apple" --steps 12` produces a coherent 4096×4096 PNG on a 32 GB V100 in ~74 s in compiled mode, where the previous behavior was an immediate `CUDA out of memory: tried to allocate 36 GiB` on `aten.convolution::62`. The artefact is saved at `validation_outputs/p_sana_4kpx_runtime/sana_4kpx/output.png` for inspection. A faint horizontal seam at band boundaries is visible on close inspection (no halo blending in V0); future refinement may add overlap-and-blend. The interception mechanism is generic (per-op_uid match, mode-agnostic interceptor functions) and ready to absorb future overflow cases — high-resolution upscalers, large-resolution video VAE decoders, etc. — without further runtime changes.

### Fixed

- **Runtime: smoke validation hardening for the 9-family dispatch** (`src/neurobrix/config/families/llm.yml`, `src/neurobrix/core/runtime/output_dispatch.py`, `src/neurobrix/cli/commands/run.py`): four small fixes caught during the 12-model smoke validation. (1) `llm.yml` was missing the `output_processing.output_format: "txt"` key, which caused `RuntimeError: 'llm' has no output_processing.output_format` on every LLM run; added. (2) Internal output-token lookup used `outputs.get("output_tokens") or outputs.get("global.output_tokens")`, which raises `Boolean value of Tensor with more than one value is ambiguous` when the first key returns a multi-element tensor; replaced with an explicit `is None` chain. (3) Multimodal image save would fail with `Cannot handle this data type: (1, 1, 384), |u1` because the CHW→HWC permute was skipped when the family YAML declared `layout: "mode_dependent"`; the image saver now treats `"mode_dependent"` like `"CHW"` since it only fires when the dispatcher has already chosen image as the output modality. (4) Multimodal models traced for one generation_type (e.g. Janus image-only) used to silently run the wrong head when the user passed the other `--mode`; the CLI now reads `topology.flow.generation.type`, compares to the requested mode, and errors with `This '<model>' build supports only --mode <X>` instead of writing nonsense output. **Validation**: 10/13 smoke tests pass, 1 produces the expected build/mode error (Janus + `--mode text` on an image-only build), 1 fails on a known unrelated memory issue (Sana 4Kpx 36 GiB convolution), 1 fails on a known unrelated KV-cache wrapper bug (Orpheus GQA). No new failures introduced by the dispatch refactor. Per-model artefacts (output, stats, prompt, verdict) saved under `validation_outputs/runtime_alignment_phase5/` for inspection.

### Changed

- **Runtime: data-driven family dispatch — wrong-modality outputs no longer produced** (`src/neurobrix/core/runtime/output_dispatch.py` new module; `src/neurobrix/cli/commands/run.py`, `src/neurobrix/serving/engine.py`, `src/neurobrix/serving/server.py`, `src/neurobrix/core/prism/solver.py`, `src/neurobrix/core/flow/autoregressive.py`, `src/neurobrix/triton/flow/autoregressive.py` refactored). The legacy 4-family runtime (llm, audio, image, video) was hardcoded across 11 sites; every model whose family was tts, stt, audio_llm, vlm, multimodal, or upscaler fell through to the image-PNG path. Visible to users as `.png` files written when running Whisper transcriptions, Voxtral text answers, Janus text-mode prompts. The new `output_dispatch` module is the single source of truth: it picks the writer (text/audio/image/video) from the family YAML's `output_processing.output_format`, validates inputs against `inputs.required`, and resolves modes via `inputs.optional[].triggers_mode` plus `modes.default`. Multimodal-strict families (currently Janus) require an explicit `--mode {text,image}` per the unified-model pattern; passing the wrong extension on `--output` now errors clearly instead of silently writing wrong content. The same dispatcher is used by both the CLI cold path and the serving warm path so both produce the same outputs. Side fixes: only LLM gets a text warmup prompt at serving load time (other families skip it — `warmup` + max_tokens=1 was meaningless for them); Prism `model_category` is now data-driven from `execution.has_kv_cache` plus topology `flow.generation.type`, so Janus (multimodal + autoregressive_image) correctly classifies as `image_vq` with KV cache instead of falling into the `diffusion` bucket and Voxtral (audio_llm) classifies as `llm` with KV cache; the `family=="image"` check in the autoregressive flow that decided the position_id policy and the image-AR tokenization branch is now `gen_type=="autoregressive_image"` so it covers Janus across the legacy "image" packaging and the current "multimodal" packaging. New CLI flags reserved for the families that need them: `--input-image`, `--mask-image`, `--reference-image`, `--reference-audio`, `--speaker`, `--video`, `--num-frames`, `--fps`, `--system`, `--mode`. Argparse shape only — semantics are validated by the YAML. `audio.yml` legacy is still loadable for any not-yet-migrated path.

### Added

- **Runtime: family configs for the new 9-family taxonomy** (`src/neurobrix/config/families/{vlm,multimodal,tts,stt,audio_llm,upscaler}.yml` created; `llm.yml`, `image.yml`, `video.yml` extended): the runtime now ships configuration for all 9 model families (`llm, vlm, multimodal, tts, stt, audio_llm, image, upscaler, video`). Previously only 4 family configs existed (`audio.yml`, `image.yml`, `llm.yml`, `video.yml`), which meant any model whose manifest declared `family: tts` / `stt` / `audio_llm` / `multimodal` / `vlm` / `upscaler` either crashed at config-lookup time or silently fell through to the legacy image-output path — visible to users as `.png` files being written for Whisper transcriptions, Voxtral text answers, and Janus text-mode prompts. Each YAML now exposes a uniform schema: `output_processing` (output format, channel layout, value range), `execution.has_kv_cache` (true for autoregressive families, false for diffusion / encoder-decoder / static-graph), `inputs.required` and `inputs.optional` (with `requires` and `triggers_mode` per-flag annotations so input validation can become data-driven), `modes.supported` / `modes.default` / `modes.multimodal_strict` (only `multimodal.yml` sets strict=true, mandating an explicit `--mode {text,image}` selection for unified models that expose both heads, following the DeepSeek Janus and Bytedance Doubao pattern), and `output.default_extension_per_mode` (text→.txt, image→.png, audio→.wav, video→.mp4) so the output writer can pick the right format from the family config rather than from a hardcoded if/elif. `audio.yml` is kept temporarily as a transition file because several runtime sites still reference `family=="audio"`; those sites will be migrated to the new family names in a follow-up before `audio.yml` is removed. **Validation**: `get_family_config(f)` succeeds for each of the 9 families and the legacy `audio.yml` still loads. No runtime code changed in this commit — YAMLs only.

### Fixed

- **Runtime: image inference no longer crashes with `ImportError: get_vae_decoder_config`** (`src/neurobrix/core/runtime/executor.py`, `src/neurobrix/core/components/handlers/dc_ae_tiled_decode.py` deleted): every image generation since 2026-04-29 was crashing with this ImportError because a runtime path imported a function that was never added to the module. The path itself was already obsolete — the spatial decomposition it tried to add at inference time is now produced upstream — so it was removed entirely along with its private helper and an orphan handler file (~140 lines total). **Validation**: Sana 1024 image generation now completes in ~23s with a clean 1024×1024 output (vs the previous immediate ImportError on every image model). Sana 4Kpx still fails on a different, unrelated memory issue (one VAE convolution requests 36 GiB on a 32 GB V100); tracked separately. **Anti-regression**: PixArt-XL, PixArt-Sigma image generation unaffected.
- **Runtime: PixArt-XL, PixArt-Sigma and Sana 1024 image generation no longer hit the broken VAE-tiling import** (`src/neurobrix/core/runtime/executor.py`): a defensive guard was added so non-DC-AE VAEs (compression ratio ≠ 32) skip the in-progress VAE-tiling code path that was crashing every image inference. Superseded by the full path removal in the next entry.

- **Layer 8 — Sana 4Kpx: VAE shape-rebind unblocks the full upsample cascade in `--triton`, `--triton-sequential`, and `native`** (`src/neurobrix/triton/promotion.py`, `src/neurobrix/core/runtime/graph/compiled_sequence.py`, `src/neurobrix/kernels/wrappers.py`): Sana 4Kpx previously crashed at the very first VAE attention residual with `Cannot broadcast (1,1024,64,64) and (1,1024,128,128)` (engine ran the decoder at the model's build-time spatial size instead of the runtime 128×128 latent). Same crash in native and both triton paths. Layer 8 adds a spatial-symbol promotion pass to the runtime engine that recognizes height/width-derived scalars in shape args of `aten::view`/`aten::expand`/`aten::reshape`/`aten::ones`/`aten::zeros`/`aten::full`/`aten::new_zeros`/`aten::new_ones`/`aten::empty`, across the multi-stage decoder cascade (scales 1/2/4/8/16/32), in both channels-first `[B,C,H,W]` and channels-last `[B,H,W,C]` 4D layouts, including arithmetic sub-expressions in nested operands. The same pass is shared by `core/runtime/graph/compiled_sequence.py` (native compiled execution) and `triton/promotion.py` (triton sequential dispatcher). Also: `upsample_nearest2d_wrapper` now recomputes the output spatial size from `runtime_input × scale_factor` when the per-call scale args are provided (matches PyTorch native semantic; the wrapper was previously frozen to the build-time output size). **Bit-perfect for build-time-equals-runtime models** (Sana 1024, PixArt 1024, every LLM): the new pass is a no-op when symbols already resolve to themselves. **Validation matrix**: LLM 6/6 PASS (TinyLlama + Qwen3-30B + deepseek-moe × native + triton). Sana 1024 `--triton-sequential`: VAE min=-1.39 max=1.25 (Layer 7 baseline: -1.40, +1.34), no regression. PixArt-Alpha `--triton-sequential`: BIT-IDENTICAL VAE stats vs Layer 7. PixArt-Alpha `--triton` (compiled): ≤LSB diff. PixArt-Sigma both modes: ≤LSB diff. **Sana 4Kpx new state**: pipeline now advances 5 cascading 2× upsamples (64→128→256→512→1024→2048→4096) in all three execution paths — from the prior crash at op ~62 to op ~700 (≥12× further). The new blocker is hardware/memory at the final 4096×4096 conv: triton path requests a 4 GiB output tensor, native path requests 36 GiB of conv workspace, both exceed V100 32 GB. Tracked as Layer 9 follow-up (`docs/follow-ups/layer9-sana-4kpx-vae-memory.md`) — needs a memory-driven spatial-tiling fallback for the VAE decoder (Layer 6.3 disabled the parasitic tiling activation for symbolic-spatial graphs but didn't add a re-entry path for memory-bound runtime cases). Layer 8 closes the runtime-engine shape-rebind regression that masked the memory issue; Layer 9 will add the memory-aware tiling fallback.

- **Layer 7 — math-decomposed attention for non-power-of-2 head_dim eliminates Volta SDPA non-determinism** (`src/neurobrix/kernels/wrappers.py`): PixArt-Alpha and PixArt-Sigma in `--triton`/`--triton-sequential` produced visible banding at `h=126,127` (last patch row) with the diffusion model output otherwise correct. Root cause: the Dao-AILab flash kernel's masked-load path (`EVEN_HEADDIM=False`) is non-deterministic on Volta SIMT for `head_dim < BLOCK_HEADDIM`. Five consecutive calls with bit-identical inputs produce five different outputs (max-diff ~0.03, concentrated at the last few Q-blocks 56-63). Cumulative drift over 28 DiT blocks amplifies the per-block bias into the visible `h=126,127` banding; the per-row L2 of the diff at the final transformer output was 27 vs ~2 elsewhere. The flash kernel docstring already warns about "race conditions on non-64/128 head dimensions". Diagnostic path: cp1 text_encoder cosine 0.99994 (ruled out), per-DiT-block bisect showed block 63 anomaly grew 0.021 → 4.53 in block 1 self-attn, within block 1 self-attn the bias entered at the SDPA output (rank 209) with top-5 Q-blocks 56-63, isolated determinism test showed 5 consecutive calls produced 5 different outputs. **Fix**: when `head_dim` is not a power of 2 (PixArt 72, Sana 112, etc.), route SDPA through math-decomposed attention — `Q @ K^T → softmax → @ V` — using existing `bmm` + `softmax` + `mul` + `add` wrappers. Deterministic by construction: no online-softmax accumulation across K tiles, no MMA reordering, isolated test bit-identical across 5 runs (max_diff = 0). Cosine vs native SDPA: 1.0 (max abs diff 4.6e-04). Pow2 head_dim (LLMs: TinyLlama 64, Qwen3 128, etc.) keeps the original flash kernel — bit-identical to before, zero LLM impact. GQA is supported via `unsqueeze + expand + reshape` (zero-copy view) before the bmm. Memory: scores tensor `[B*H, T_q, T_k]` in fp32 is ~1 GB for PixArt self-attn (CFG batch=2, 16 heads, 4096 tokens), fits within V100 32 GB budget alongside weights and activations. Sana 4Kpx (T=16384) would exceed memory but is blocked by other issues anyway. **Validation matrix** (visual coherence + VAE in normal range): PixArt-Alpha `--triton-sequential` (run1+run2 coherent, pixel diff ≤ 4/255, no `h=126,127` concentration), PixArt-Alpha `--triton` (compiled), PixArt-Sigma `--triton-sequential` (was NaN before), PixArt-Sigma `--triton` (compiled), Sana 1024 `--triton-sequential` (no regression, head_dim=112 also routes to math), Sana 1024 `--triton` (no regression), TinyLlama `--triton` (LLM unchanged, head_dim=64 keeps original flash path). Residual run-to-run pixel diff (~4/255) is from upstream/downstream non-flash kernels (matmul/softmax minor ULP variation), not from the fixed flash path; the pre-fix `h=126,127` cumulative-drift cascade is gone.

- **Layer 6.bis — minimal SDPA fix preserving Volta determinism** (`src/neurobrix/kernels/wrappers.py`): Layer 6 (commit `109676b`) attempted to make SDPA block selection data-driven via runtime YAML lookup + helper function + driver query. Validation revealed an empirical constraint we could not resolve across 6 systematic investigation rounds: on Volta + Triton, ANY Python function call added between `scaled_dot_product_attention_wrapper` argument parsing and the kernel launch causes run-to-run output drift, even when the function returns identical values and the Triton kernel cache is stable across runs. The pre-Layer-6 baseline (commit `06d26c2`) reproduces 4/4 identical "Certainly! Here" on TinyLlama triton (cold + 3 warm cache); Layer 6 plain produces 4 different outputs ("Yes, absolutely! Here", "I'd be happy", "I'm not able", "I'm not able") with 0 new compiled kernels between warm runs (cache verified stable via `find ~/.triton/cache -newer <ref>`). Bisect ruled out: Triton kernel cache invalidation, driver query side-effects (`_fa_max_smem` stub returning constant — still drifted), `_FA_PICKER_CACHE` global state, list literal allocation, try/except blocks, `pick_dtype` property reads, compiled cascade-if via `exec()`, and result-cache lookups (one dict.get per call still drifted). Mechanism remains unexplained; documented for future Layer X investigation in NEW `docs/architecture/data-driven-hardware-contract.md` along with the four hypotheses still in scope (CUDA stream non-determinism, sub-µs Python frame perturbing CUDA driver state, `@triton.heuristics` lambda re-evaluation, Triton/CUDA thread-local state). Layer 6.bis reverts the SDPA wrapper to a strictly minimal modification of the pre-Layer-6 inline cascade table — ONE additional branch added: `elif BLOCK_HEADDIM >= 512: BLOCK_M = 16; BLOCK_N = 16` (PixArt VAE on V100, the original Layer 6 motivation). All other configs (TinyLlama h=64, Qwen3 h=128, Sana 1024 DiT, PixArt DiT h=72) use the EXACT pre-Layer-6 hardcoded values, bit-perfect identical. No helper function, no cache global, no driver query, no list literal, no try/except. Strictly minimal Python diff vs `06d26c2` (24 insertions, 152 deletions in the SDPA wrapper). Sub-fix 6.4 (group_norm 2-pass kernel rewrite) is preserved unchanged — it lives in `kernels/ops/groupnorm.py` and is required for any model with HxW > 1024×1024 group_norm tile (PixArt VAE 1024×1024, Sana 4Kpx VAE) regardless of the SDPA picker. **Vendor/architecture YAMLs** (`src/neurobrix/config/vendors/{nvidia,amd}/*.yml`) were enriched during the investigation with the seqlen_q-aware `sdpa_thresholds` schema, Volta opt-in 96 KB SMEM correction (was 49 KB default), and per-arch SMEM safety factor. They are NOT read at runtime by the SDPA wrapper (runtime reading caused the drift) but serve as documentary source of truth for adding new architecture support: when a dev adds A100/H100/AMD CDNA validation, they consult the YAML and manually align the wrapper's inline cascade. NOTE: `src/neurobrix/config/vendors/` is currently in `.gitignore`, so the YAML enrichments are local-only — promoting them to tracked files is a follow-up out of scope here. **Verified:** V_drift TinyLlama triton 3/3 "Certainly! Here" identical to native baseline; V_pixart PixArt-Alpha triton fp16 SDPA path clears the SMEM exhaustion crash and reaches the Layer 7 blocker (VAE conv fp16 overflow); V_regression Sana 1024 triton zero regression (range ±1.27 mean -0.72) + LLM harness 14 passed, 12 xfailed (all pre-existing), 14 skipped (slow image/video), 0 failed in 1559s. **Open follow-up (Layer X future):** isolate the root cause of the Python-frame drift on Volta + Triton hot path. Diagnostic recommended: dump the kernel binary loaded at each call via `triton.compiler` introspection and compare bit-for-bit between consecutive runs with cached compilation. If binary identical but output diverges → CUDA stream / runtime non-determinism. If binary differs → another mechanism. Resolution would unlock true runtime data-driven SDPA selection per the contract in `docs/architecture/data-driven-hardware-contract.md`.

- **Six-layer fix unblocking diffusion model pipeline in `--triton` mode (Layer 6)** (`src/neurobrix/core/runtime/tensor_compat.py` NEW, `src/neurobrix/core/components/handlers/vae_handler.py`, `src/neurobrix/core/runtime/executor.py`, `src/neurobrix/kernels/wrappers.py`, `src/neurobrix/core/module/tiling_engine.py`, `src/neurobrix/kernels/ops/groupnorm.py`): Layer 6 fixes five independent issues that prevented PixArt-Alpha, PixArt-Sigma, and Sana-4Kpx from progressing in `--triton` mode beyond their respective early crash points. Layer 6 unblocks the pipeline up to two architectural blockers (Layers 7 and 8, documented in `docs/follow-ups/`) that require dedicated work: Prism per-component dtype override for fp32-required VAE convolutions on V100 (Layer 7), and extension of the symbolic shapes contract to runtime-computable buffers (Layer 8). **Sub-fix 6.1 — DRY refactor of tensor type detection helper.** The `_is_tensor()` helper introduced in commit `06d26c2` (Layers 4 and 5) lived in two separate files. Centralized into `core/runtime/tensor_compat.py` with a lazy `NBXTensor` import to avoid a hard triton dependency from core handlers. Zero behavioral change. **Sub-fix 6.2 — Hardware-driven flash attention block sizing.** `kernels/wrappers.py::_pick_attention_blocks` queries shared memory capacity dynamically via `triton.runtime.driver.active.utils.get_device_properties()` and selects `(BLOCK_M, BLOCK_N)` to fit within the SMEM budget on any hardware: V100 (98 KB), A100 (164 KB), H100 (228 KB), AMD CDNA, all handled by the same code path. Replaces the previous hardcoded table that targeted V100 only and missed `head_dim ≥ 512` (PixArt VAE DC-AE) — which produced the SMEM exhaustion error `Required: 131072, Hardware limit: 98304` at `aten._scaled_dot_product_efficient_attention::0`. Includes an automatic fp32→fp16 fallback when the (head_dim, dtype) combination cannot fit any `(BLOCK_M, BLOCK_N) ≥ (16, 16)` pair: PixArt VAE attention runs fp32 inputs at head_dim=512 → kernel cannot fit on V100 → wrapper downcasts Q/K/V to fp16, output cast back to fp32. The kernel always accumulates softmax in fp32 internally, so the only precision loss is the input quantization. Verified on synthetic VAE-shape input ([1, 1, 16384, 512] with values up to ±525): cosine 0.999721 vs `torch.nn.functional.scaled_dot_product_attention` fp32 reference. LLM cases (TinyLlama h=64 prefill+decode, Qwen3 h=128 prefill, Sana DiT h=112, PixArt DiT h=72) pick identical or better block sizes than the previous hardcoded table, no regression. **Sub-fix 6.3 — Symbolic-shape-aware tiling engine.** `core/module/tiling_engine.py::from_component_config` was reading `tensor["shape"]` (concrete build-time value) and ignoring `tensor["symbolic_shape"]["dims"]`. For models with build-time captured size < runtime size on spatial dimensions (Sana 4Kpx VAE captured 64×64 latent, runtime 128×128 latent), TilingEngine activated incorrectly and fed `NBXTensor` into the torch-only `torch.zeros(device=NBXTensor)` accumulator path → `TypeError`. NeuroBrix's NBX format defines symbolic shapes as a **master contract** (formalized in NEW `docs/architecture/symbolic-shapes-contract.md`): graphs declaring spatial dims as symbolic rebind to runtime via `CompiledSequence`'s symbol binding pass and do NOT need TilingEngine. Fixed: `from_component_config` now refuses instantiation when any spatial dim of the input tensor is symbolic. TilingEngine remains active for genuinely tile-only models (Swin2SR upscalers etc.) with concrete spatial shapes. This sub-fix also fixes a latent bug for any user attempting `--width`/`--height` beyond the build-time captured size on Sana 1024 or PixArt 1024 — they would have hit the same parasitic tiling. Layer 6 makes these models true spatial-adaptive in `--triton` mode. **Sub-fix 6.4 (bonus) — GroupNorm 2-pass rewrite.** `kernels/ops/groupnorm.py` rewrites the kernel as a 2-pass loop (sum + sum-of-squares first, then normalize-and-store) over `hidden = group_size × HxW` in `BLOCK_SIZE` chunks, eliminating the `numel exceeds 1048576` Triton crash on large spatial inputs. The previous single-tile design allocated `[BLOCK_GROUP_SIZE × BLOCK_HW_SIZE]` registers in one shot — for PixArt VAE 1024×1024 that's 4194304 elements, way past Triton's 2^20 numel ceiling. The new kernel chunks the hidden dim with `BLOCK_SIZE = min(16384, next_pow2(min(hidden, 16384)))` so the per-program tile stays bounded regardless of input HW. Cosine bit-perfect (1.0) vs `torch.nn.functional.group_norm` reference on three scales: PixArt VAE early (128×128), mid (256×256), large (1024×1024), max_abs_diff < 3e-6. Required to reach the VAE decode path on Sana 4Kpx and PixArt 1024 in triton. **Validation matrix (V1 synthetic + V3 regression):** SDPA picker tested across `head_dim ∈ {32, 64, 72, 80, 112, 128, 160, 256, 512}` × `dtype ∈ {fp16, bf16, fp32}` × `seqlen_q ∈ {1, 4, 16, 64, 1024}` (90 combos), all `head_dim=512 fp16` cases pass cosine ≥ 0.999 (the critical PixArt VAE shape); LLM regression smoke-test passes (TinyLlama --triton 5 tokens coherent: "I am not able to"); Sana 1024 --triton zero regression (range ±1.09 mean -0.72, no NaN). **Sub-fix 6.5 (conv2d `out_dtype=tl.float32`) was tested empirically on the actual PixArt VAE conv [1, 512, 512, 512] @ [512, 512, 3, 3] with values matching the run-time observed range (X up to ±2850, W up to ±2.67) and produced identical results with and without the change (15526 inf, finite_abs_max=65504) — Cas Y in the validation matrix. The reason: `accum += tl.dot(...)` already forces fp32 accumulation because `accum` is fp32-allocated; the truncation happens at the fp16 output buffer store, which is a wrapper-level decision out of scope for Layer 6.** Sub-fix 6.5 was therefore retired from the commit; the same change will become useful as part of Layer 7 (Prism per-component dtype override) once VAE buffers are allocated fp32. **Documentation:** NEW `docs/architecture/symbolic-shapes-contract.md` formalizes the NBX symbolic shapes master contract (build-time concrete vs runtime symbolic dims, audit pattern, sites that already honor it, sites still to audit). NEW `docs/follow-ups/layer7-prism-dtype-override.md` documents the PixArt VAE conv fp16 overflow blocker requiring architectural Prism work (~200+ lines + plumbing). NEW `docs/follow-ups/layer8-computable-buffers-extension.md` documents the Sana 4Kpx aten.add broadcast blocker requiring architectural build-time + runtime work (~200+ lines). The existing `docs/follow-ups/pixart_triton_arena_inter_run_bug.md` is archived with a status update pointing to Layer 7 as the active blocker. **Mission "3 models green" not yet complete** — Layer 6 unblocks the pipeline measurably (PixArt advances to VAE conv, Sana 4Kpx advances to transformer add-broadcast) and exposes the precise architectural blockers (N1 fp16 storage overflow, N2 missing computable buffers rebind) for Layers 7 and 8. Each layer is a complete fix in its scope; Layers 7 and 8 will get their own commits.

- **Sana_1600M_1024px_MultiLing `--triton` produces visually coherent images: VAE output cosine -0.825 → +0.9993 vs native, end-to-end five-layer fix** (`src/neurobrix/kernels/ops/flash_attention.py`, `src/neurobrix/kernels/wrappers.py`, `src/neurobrix/kernels/nbx_tensor.py`, `src/neurobrix/core/components/handlers/text_encoder_handler.py`, `src/neurobrix/core/components/handlers/vae_handler.py`, `src/neurobrix/core/runtime/executor.py`): five independent bugs compounded — each masked by the layer downstream of it — to produce visually-incoherent Sana-1024 output in triton mode while native rendered correctly. The investigation traversed flash attention kernel selection, NBXTensor's silent rejection of fancy indexing, and two distinct sites where `isinstance(x, torch.Tensor)` returned False for `NBXTensor` and silently no-op'd critical input transforms. **Layer 1 (`flash_attention.py`)**: the SDPA kernel had three `BIAS_TYPE` configurations (`none`/`vector`/`matrix`); the `none` path materialized zero bias via `tl.zeros` in registers, while `vector`/`matrix` paths loaded it from memory. Triton's IR optimization passes propagate the constexpr-in-register vs memory-load distinction down to MMA selection in `tl.dot`, producing non-bit-equivalent results across configurations on fp32 Q/K/V (synthetic isolated benchmark: cosine 0.937 vs 0.9999+ at hd=256). Same pattern at `IS_CAUSAL` via `tl.where(causal, 0, -inf)`. The kernel is now strictly bias-driven (`BIAS_TYPE ∈ {vector, matrix}` only, no `IS_CAUSAL` constexpr); hardware-universal by construction (V100 SIMT, A100/H100 SIMT/TF32, AMD CDNA matrix cores). **Layer 2 (`wrappers.py`)**: the Python wrapper now always provides bias as a memory-resident `NBXTensor` — user-provided `attn_mask`, or a cached zero buffer (no mask, no causal), or a cached causal additive mask (no mask, causal). Memory cost ~4–16 MB per session (negligible) for the bias caches; SDPA cosine on Sana DiT rank-101 attention 0.969 → 0.999+. **Layer 3 (`nbx_tensor.py` + `text_encoder_handler.py`)**: Sana CHI handling sliced the text encoder's hidden_state to keep BOS + last 299 tokens via PyTorch fancy indexing `hidden_state[:, [0, -299, -298, ..., -1]]`. `NBXTensor.__getitem__` had no Python-list-key branch and silently no-op'd, returning the original 506-token tensor; the transformer DiT then received `[2, 506, 2304]` instead of `[2, 300, 2304]`, every cross-attention K/V operated on wrong-shaped inputs, and the resulting 36% L2 amplitude divergence compounded through 20 DiT layers × 10 diffusion steps (text_encoder.output_0 cosine 0.888 → 0.99+ post-fix). The handler is rewritten as `narrow + cat` (compatible with both `torch.Tensor` and `NBXTensor`); `NBXTensor.__getitem__` now raises an explicit `ZERO FALLBACK` error on Python list keys to prevent any future silent recurrence. **Layer 4 (`vae_handler.py`)**: three `isinstance(x, torch.Tensor)` gates in `transform_inputs` and `_find_latent_key` silently rejected `NBXTensor` in triton mode, causing the VAE handler to short-circuit and skip the DC-AE `scaling_factor` division (Sana: ÷0.41407 = ×2.415). Without the scaling, the latent reaching the VAE decoder had wrong amplitude (triton std 1.30 vs native 2.78). The fix adds a module-level `_is_tensor()` helper recognizing both tensor types, applied at all three sites. The per-channel denormalization tensor creation (`mean_t`, `std_inv`) is dual-typed so the handler also works for VAEs that train with a normalized latent space (LTX2Video etc.) on both native and triton paths. **Layer 5 (`executor.py`)**: same `isinstance(x, torch.Tensor)` pattern at two further sites in the runtime executor. (a) `_replace_with_state_variable`: in triton mode every `value` in `comp_inputs` was an `NBXTensor`, the isinstance check returned False on every iteration, and the function kept the wrong tensor (`transformer.output_0` = post-CFG `noise_pred`, cosine ~0.93 vs native) instead of replacing with `final_latents` (`state_variable`, cosine 0.9953 with native). The VAE consumed the wrong source; combined with the missing scaling from Layer 4 this produced VAE input cosine -0.798 (negative). (b) `_find_spatial_input`: silently disabled tiling for any 4D input in triton mode — latent bug that hadn't surfaced on Sana-1024 (no tiling activated at trace size) but would corrupt PixArt and Sana 4Kpx in any triton tiling path. Same `_is_tensor()` helper as Layer 4, applied at both sites. Tiling now activates correctly in triton for components that need it. **Verification**: VAE input cosine -0.798 → +0.998, VAE output cosine -0.825 → +0.9993 vs native (both at the same prompt/seed), state_variable cosine preserved at 0.9953 throughout the 10-step trajectory; image visually coherent with prompt. Regression harness: `14 passed + 14 skipped + 12 xfailed + 0 failed`, unchanged. **Follow-up TODO (out of scope for this commit)**: `output_extractor.py` has 6 additional `isinstance(*, torch.Tensor)` sites (`extract_hidden_states`/`logits`/`embedding`/`image`) that affect autoregressive_generation, audio_llm, dual_ar, and tts_llm flows in triton mode — not affecting `iterative_process` (Sana, PixArt) so out of scope here. Audit-and-fix in a dedicated commit when those flows are exercised in triton.

- **Triton CFG engine persists the post-guidance `noise_pred` into `variable_resolver.resolved[f"{comp_name}.output_0"]`, matching the state variable semantics and stopping CFG-batched shapes from leaking into post-loop consumers** (`src/neurobrix/triton/cfg/engine.py::_execute_batched_cfg`, `_execute_sequential_cfg`): every triton CFG engine call invoked `RuntimeExecutor._execute_component(...)` to run the transformer, which stored the *raw* forward-pass output in `variable_resolver.resolved[f"{comp_name}.output_N"]` via `OutputExtractor.store_component_outputs` (`src/neurobrix/core/runtime/resolution/output_extractor.py:105-123`). For batched CFG this raw output was `(2*B, C, H, W)` (uncond + cond concatenated); for sequential CFG the last pass wrote `noise_pred_cond` (the unguided conditional output). The CFG engine then computed `noise_pred = uncond + scale * (cond - uncond)` in `batch=B` and returned it to the flow handler, but it never updated `variable_resolver.resolved`. Post-loop consumers that resolve the default DiT topology connection `transformer.output_0 → vae.z` (identical for PixArt-XL / PixArt-Sigma / Sana_1600M_1024px_MultiLing — confirmed by grep over `topology.json`) therefore received the stale pre-CFG tensor, not the post-guidance result. Consequences: Sana 1024 triton fed `(2, 32, 32, 32)` into the DC-AE decoder and produced an incoherent `(2, 3, 1024, 1024)` image (diagnostic probe `tests/scratch/sana_batch_propagation_diag/probe.py` showed `state=(1, 32, 32, 32)` at `post_loop_entry` but `ge.run_input[z]=(2, 32, 32, 32)` at VAE dispatch, with `resolve_source_chain_sources=['transformer.output_0']` resolving to shape `(2, 32, 32, 32)`). PixArt-Alpha / PixArt-Sigma triton had the same bug but it was mostly silent because uncond and cond converge near the end of denoising, so decoding both independently yields two visually-similar images that blur into a "coarse cat" when rendered as the first batch slice. **Fix**: in both `_execute_batched_cfg` (just before `return {"output_0": noise_pred}`) and `_execute_sequential_cfg` (same site), assign `self._ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = noise_pred`. Pure-dict assignment, zero torch, scope-limited to `triton/cfg/engine.py`. The raw pre-CFG output is no longer relied on by any downstream consumer — the flow handler's state update uses the returned dict directly, and post-loop consumers get the guided result via the `resolved` dict. Native CFG engine (`core/cfg/engine.py`) lives in a separate file, is reached on a different execution path, and was not touched (Rouge 1 separation maintained). Zero impact on LLMs: neither TinyLlama, Qwen3, DeepSeek-MoE nor Janus invoke the CFG engine. Harness `14 passed + 14 skipped + 12 xfailed + 0 failed`, unchanged. PixArt-Alpha / Sigma / Sana triton now produce `(1, 3, 1024, 1024)` images in the compiled path.

- **Triton sequential dispatcher now evicts dead intermediates from its tensor store via a `_compute_liveness` mirror** (`core/runtime/graph_executor.py::_run_triton_sequential`): the sequential path maintained an unbounded `store: Dict[str, NBXTensor]` (graph_executor.py:1545) throughout the full component forward pass — every op's output tensor was inserted and never removed (grep over the file showed zero `store.pop` / `del store[` / `store.clear` references before this commit). Invisible on small LLMs (TinyLlama 4-D intermediates, ≤ 5 GB peak) but fatal for diffusion VAEs: Sana DC-AE 32× upsamples to `(1, C, 1024, 1024)` tensors during decode, and accumulating every intermediate OOM'd at the `_wrap_lower_precision` dtype cast (`triton/dtype.py:157`) with a 1 GB malloc failure. **Fix**: port `TritonSequence._compute_liveness` (triton/sequence.py:1394-1421) to the sequential path. Walk `attrs.args` / `attrs.kwargs` for every op in `execution_order` (recursing into `tensor`, `tensor_ref`, `tensor_tuple`, and nested `list` arg types — same coverage as the compiled-path extractor), track each tid's last consumer `op_idx`, build `dead_at_op[op_idx] -> [tid, ...]`, and after each dispatch `store.pop(dead_tid, None)`. Protected set mirrors the compiled path: `param::*`, `buffer::*`, `input::*`, and the DAG's declared `output_tensor_ids`. Unlike the compiled path, no `_deferred` batching or `sync_device()` is needed — the sequential dispatcher is synchronous from Python's perspective, so refcount-0 → `NBXTensor.__del__` → `DeviceAllocator.free_cuda` is safe immediately. Measured on TinyLlama-1.1B-Chat-v1.0 `--triton-sequential` 15 tokens: peak `DeviceAllocator.memory_allocated` 2.458 GB → **2.238 GB** (9 % reduction, ~220 MB freed by the eviction of transient residual-stream intermediates between layers). On Sana 1024 × 1024 `--triton-sequential`, the 1 GB OOM is gone and the pipeline now completes end-to-end (400 s, 20 steps) — two orthogonal bugs surface downstream (a `batch=2` state propagation from transformer to VAE, and a numerical divergence vs the compiled path) which are tracked as follow-up sub-chantiers. Harness `14 passed + 14 skipped + 12 xfailed + 0 failed`, unchanged.

- **Triton `strided_copy_kernel` / `strided_scatter_kernel` are now scalable in `NDIM`, replacing a silent dim-drop bug on > 5-D tensors** (`kernels/ops/strided_copy.py`, `kernels/nbx_tensor.py`): prior kernels had five positional dim and stride parameters (`d0..d4`, `s0..s4`). The Python wrapper padded to exactly 5 via `list(x._shape) + [1] * (5 - x.ndim)` — for `x.ndim > 5` the multiplier was negative, Python returned an empty list, and the wrapper passed the first 5 values unchanged while dropping the rest. The kernel then computed a flat-to-multidim decomposition assuming `n_elements = d0*d1*d2*d3*d4`, but was actually launched with `n_elements = src._numel` (product of all dims including the dropped ones). Half the threads computed indices with `i0` beyond the source's dim-0 extent, accumulated src offsets that walked past the source allocation, and either (a) read adjacent mapped memory and corrupted the output silently — PixArt-Alpha `aten.clone::28` on the 6-D `[2, 8, 64, 2, 64, 2]` patchify shape, or (b) hit an unmapped page and raised `cudaErrorIllegalAddress` — PixArt-Sigma same op, surfacing at the next CUDA call (the CFG `cond - uncond` subtraction, which was a red herring in the initial diagnosis). 8-D SANA-Video VAE clone shapes (e.g. `[1, 128, 4, 2, 8, 2, 8, 2]`) never reached production because of this path. **Fix**: the kernels are now parameterised by `NDIM: tl.constexpr`, and shape / strides travel as GPU-resident int64 scratch buffers loaded inside the kernel body (`shape_ptr`, `stride_ptr`). The per-dim decomposition loop is unrolled at compile time via `tl.static_range(NDIM)`. Triton specialises and caches one compiled kernel per distinct ndim encountered at runtime — first pass at a new ndim compiles (1-175 ms measured for ndim ∈ {3..16} on V100), subsequent calls hit the kernel cache at ~µs overhead. The wrapper `_strided_copy` / `_strided_scatter` allocates two tiny int64 scratch buffers (`_upload_int64_array`) via the zero-torch `NBXTensor.from_numpy` path and passes them to the kernel. An assert caps ndim at 25 (PyTorch's `TensorIterator::MAX_DIMS` hard limit) so accidental future ndim > 25 fails loudly instead of wrapping around. The fix is hardware-agnostic (no arch-specific branches, works on CUDA + HIP via `_GPU_BACKENDS`), model-agnostic (scales to whatever ndim a traced graph emits), and stays fully within the zero-torch triton contract. Sigma's `aten.clone::28` `cudaErrorIllegalAddress` at transformer completion disappears; Alpha's previously-silent pixel corruption in the same op is gone. Standalone validation in `tests/scratch/strided_copy_scalable_poc.py` exercises ndim ∈ {3, 4, 5, 6, 8, 10, 12, 16} and the exact 8-D SANA-Video VAE clone shape — 9/9 match. The 6-D `tests/scratch/strided_copy_6d_repro.py` that flagged the original bug now reports `match: True`. Regression harness unchanged. Both PixArt models still crash downstream on an unrelated V100 Volta shared-memory limit in the VAE triton SDPA kernel (`aten._scaled_dot_product_efficient_attention::0`, requires 128 KB shared memory vs V100's 96 KB per SM) — not in scope, tracked separately.

- **Triton arena is now cleared on `GraphExecutor.unload_weights` and `cleanup` — native parity for diffusion pre_loop components** (`core/runtime/graph_executor.py`, `triton/arena.py`): prior to this commit, both methods cleared `_compiled_seq._arena` (native torch tensors) but not `_triton_seq._arena` (NBXTensor slots). `_triton_seq._arena` was in fact never cleared anywhere in the codebase (verified by grep over `src/`), so calling `_unload_component(...)` on a pre_loop triton component (PixArt's T5 text_encoder ~9.5 GB fp16) would free its weights dict but leave the NBXTensor references alive in the compiled sequence's arena slots — their `data_ptr()`s kept the underlying cudaMalloc blocks alive and the memory never returned to the driver. When the transformer DiT subsequently started, 9+ GB of T5 state was still resident, halving the free headroom on V100 32 GB and fragmenting the arena ahead of the heavy feed-forward path. **Fix**: `Arena` (triton) gains `clear_all` and `clear_inputs` in strict parallel with the native `TensorArena` (three-line loops, zero torch). `unload_weights` and `cleanup` in `GraphExecutor` now additionally clear `_triton_seq._arena` when that attribute exists, guarded by `hasattr` because `_triton_seq` is created lazily in `_ensure_triton_compiled`. The triton path calls `DeviceAllocator.sync_device()` *before* `clear_all()` — dropping the slot's Python ref triggers `NBXTensor.__del__` → `DeviceAllocator.free_cuda` synchronously, which would UAF-race an async kernel still reading the memory; the sync drains outstanding kernels first. `unload_weights` sets `_triton_seq = None` to release the sequence object too; `cleanup` keeps it alive (same pattern as native's `_compiled_seq`) so subsequent requests can rebind fresh weights into the same slot mapping without recompiling. Verified via `NBX_UNLOAD_DIAG=1` on PixArt-Alpha triton: post-`_unload_component[text_encoder]`, `used` drops from 9.85 GB → 0.34 GB, matching native. Universal for any triton model with a pre_loop-loaded encoder whose weights are released before the main loop: PixArt Sigma/Alpha, Sana, Flex.1, SANA-Video, VibeVoice, Voxtral, Canary-Qwen, OpenAudio.

- **Triton `skip_kills` is now an explicit caller-controlled flag (removes shape-based `is_decode` heuristic that mis-fired on diffusion runs)** (`core/runtime/graph_executor.py::_run_triton_compiled`, `triton/session.py`): prior behaviour read `any(t.shape[1] == 1 for t in inputs)` on every triton compiled run and set `skip_kills=True` when any 2D input had `shape[1] == 1`. Intended to detect LLM decode steps (`input_ids` shape `(B, 1)`, `seq_len == 1`). **Shape-indistinguishable** from a PixArt `aspect_ratio` micro-conditioning input (`(B, 1)`, a per-item scalar synthesised by `InputSynthesizer.compute_ratio`): both are 2D with `shape[1] == 1`. The heuristic therefore fired `skip_kills=True` on every PixArt batched-CFG transformer call, which disabled the `kill_slots` branch in `TritonSequence._run_single_device` — every intermediate stayed alive in its arena slot for the full 2495-op forward pass → peak 31.02 GB / 953 live allocs on V100 32 GB and OOM at `aten.native_layer_norm::49` (Alpha) / `aten._scaled_dot_product_efficient_attention::49` (Sigma). Confirmed via a liveness-vs-malloc-trace diff in `tests/scratch/pixart_liveness_probe/`: the DAG's `_compute_liveness` is structurally correct (412 binary-op outputs, 412/412 with a consumer, 79 % age ≤ 5 ops, p99 age 37 ops = 1 transformer block) but 361/1228 `_prepare_binary` allocations were never freed during the run, all of them the function's *output* tensors (the ones that land in arena slots), while the intermediate upcast/contiguous temporaries (lines 396/398/407/408 — function-scope Python locals) died within 2-4 events. Exactly the `skip_kills` signature. Fix: thread an explicit `skip_kills: Optional[bool] = None` through `GraphExecutor.execute` → `.run` → `._run_triton` → `._run_triton_compiled`/`._run_triton_sequential`, default `False` at the bottom, remove the shape sniffing entirely. Two legitimate caller sites pass `skip_kills=True` from `triton/session.py::decode_step` (KV-cache fast path) and `._decode_step_full_context` (O(n) fallback) — these are the only places where the "same arena slots overwritten every step before being read" invariant holds. `session.prefill` now passes `skip_kills=False` explicitly (distinct per-op output slots for the whole prompt, kill_slots must fire). Diffusion flow handlers (`triton/flow/iterative_process.py`, the CFG engines, the stage handlers) don't pass the flag at all, inheriting the safe default. Post-fix peak: **PixArt-Alpha 10.48 GB / 267 live allocs, PixArt-Sigma 10.83 GB / 125 live allocs** — a 3× drop, comfortably under the V100 32 GB limit. TinyLlama-1.1B-Chat-v1.0 `--triton` decode speed unchanged (60 tokens / 12.29 s = 4.88 tok/s, identical to pre-fix because the session path explicitly passes `skip_kills=True` for decodes). Regression harness: `14 passed + 14 skipped + 12 xfailed + 0 failed`, unchanged. Two orthogonal latent crashes now surface because transformer completes end-to-end — Alpha VAE `aten._scaled_dot_product_efficient_attention::0` hits a V100 shared-memory limit (131 KB required, 98 KB available on Volta) in the VAE triton SDPA kernel, and Sigma CFG-batched subtraction returns `cudaErrorIllegalAddress` (error 700) — both are tracked as separate follow-ups and do not gate this commit.

- **Triton input binding for synthesised nested-dict inputs (`added_cond_kwargs.resolution`, `aspect_ratio`) + cross-device placement of synthesised tensors** (`core/runtime/graph_executor.py::_run_triton`, `core/runtime/resolution/input_synthesizer.py`): two orthogonal universal bugs in the triton input path, confirmed by diff against the native equivalents. **(1) Nested-dict lookup**: `InputSynthesizer.synthesize_missing_inputs` writes dotted inputs as *nested* dicts via `_set_nested` (`input_synthesizer.py:130-137`) — e.g. `inputs["added_cond_kwargs"]["resolution"] = tensor`. Native reads them back via a two-strategy cascade in `graph/tensor_resolver.py:82-101` (direct lookup, then dotted-path navigation). Triton's `_run_triton` only did a flat `inputs.get(input_name)` and silently skipped the binding when the name contained a dot. Downstream the graph op whose input was left unset received an undefined value; NOP propagation (`args[0] is None`) then cascaded through the micro-conditioning chain and crashed at `aten.cat::2` with `'NoneType' object has no attribute 'ndim'` on PixArt-Alpha, and at a later op on PixArt-Sigma. Fix: add the same dotted-path walk in the triton input-map builder — on a missed flat lookup, split on `"."` and navigate the nested dict. Universal across any model with synthesis rules (PixArt Sigma/Alpha, Sana, Flex.1, SANA-Video, any future diffusion model declaring `topology.synthesis.<comp>.<name.with.dots>`). **(2) Cross-device synthesis placement**: `InputSynthesizer` falls back to `getattr(self._plan, 'primary_device_index', 0)` (`input_synthesizer.py:122`) when it cannot read a device off an existing torch tensor in the current inputs. But `Plan` (`core/prism/solver.py`) exposes `primary_device` as a string ("cuda:2") — it does *not* expose `primary_device_index`. The `getattr` default silently returned `0` on every call, placing synthesised tensors on cuda:0 regardless of where the component actually runs. On a multi-GPU profile where the executor targets cuda:N (N ≠ 0), Triton then rejected the cross-device pointer with the same "cpu tensor?" error as genuine CPU pointers. Fix: prefer the component's own allocation (`plan.components[comp_name].device`), fall back to `plan.primary_device`, parse the device string ("cuda:2" → index 2); the old vendor-prefix/arch-based derivation stays as a last-resort for plans with no device strings. Triton path separately moves any mismatched torch input to the executor's device before wrapping (`value.to(f"cuda:{device_idx}")`), mirroring native's `.to(self.device)` at `_execute_compiled_graph` entry. **Impact**: both fixes unblock the triton input-binding stage for every model declaring dotted synthesis inputs on multi-GPU hardware. PixArt-Alpha/Sigma triton now advance past the first transformer op (previously died at `cat::2` / early `addmm`) into the transformer body on a V100 32 GB, where they still hit OOM at the peak-concurrency layer (tracked separately — the drain-threshold tuning / caching-pool work in `docs/follow-ups/pixart_triton_arena_inter_run_bug.md`). Post-fix regression harness: `14 passed + 14 skipped + 12 xfailed + 0 failed`, unchanged from `bd3ec92`. TinyLlama native+triton, Janus-Pro-7B triton, PixArt-Sigma/Alpha native 1024×1024 all GREEN.

### Added
- **Periodic `_deferred` drain in `TritonSequence` hot loop ("Route A") — bounds triton peak VRAM to a configurable window instead of run-total** (`triton/sequence.py`, `kernels/nbx_tensor.py`, docs `docs/follow-ups/pixart_triton_arena_inter_run_bug.md`): before this commit both `_run_single_device` and `_run_multi_device` pushed every arena-slot eviction (output overwrite + kill_slot) into a local `_deferred` list and drained it exactly once per run, after a final `DeviceAllocator.sync_device()`. On a DiT with ~5600 ops per forward pass (PixArt-Sigma 28 layers × ~200 ops/layer, CFG batch=2), `_deferred` grew to ~947 live tensors × ~40 MB avg = 30.97 GB at peak — OOM on a 32 GB V100 inside the transformer at `aten._scaled_dot_product_efficient_attention::49` for a 37 MB alloc. Not a leak: 88/89 GB of total allocation volume was freed during the run; what pinned the peak was the *retention of overwritten intermediates inside `_deferred` until end-of-run*, confirmed by walking the allocation log (`NBX_MALLOC_TRACE`) and aggregating the live set at peak — median age 1472 events, max 2942, i.e. outputs from the far past were still alive. New behaviour: both hot loops now drain when `_deferred` crosses OR of `bytes ≥ NBX_DEFERRED_DRAIN_BYTES` (default `2_000_000_000`) and `count ≥ NBX_DEFERRED_DRAIN_COUNT` (default `512`); env vars read once at run entry, not in the hot loop. Each drain is the same `sync_device()` + `_deferred.clear()` pair the end-of-run path already does — correctness identical, just sooner. `NBX_DEFERRED_DRAIN_DIAG=1` prints per-drain count/bytes/trigger and a per-run totals line. Empirical: TinyLlama-1.1B-Chat-v1.0 `--triton` with `NBX_DEFERRED_DRAIN_COUNT=8` fired 198 aggressive drains across 15 decode tokens with coherent output ("I'd be happy to help you..."), confirming drain is UAF-safe in isolation. PixArt-Sigma `--triton` defaults fired 36 drains across 4 transformer calls (9 per call), transformer completed 4× with no OOM, peak VRAM stayed ≤ ~18 GB. **Route A is necessary but not sufficient for PixArt-Sigma/Alpha triton**: beyond Route A's bounded peak, PixArt crashes on a distinct arena inter-run corruption (Sigma: `aten.clone::28` error 700 on the third transformer run; Alpha: `aten.cat::2` reads a None arena slot). That bug was latent before Route A (masked by OOM-before-third-run) and is tracked as `docs/follow-ups/pixart_triton_arena_inter_run_bug.md` — universal across any triton model that calls `run()` multiple times per request.
- **`NBX_MALLOC_TRACE=<path>` env-gated allocation logger in `DeviceAllocator`** (`kernels/nbx_tensor.py`): every `malloc_cuda` / `free_cuda` writes a tab-separated row `(event_id, M|F, ptr, nbytes, file:line func)` to an in-memory buffer flushed to `<path>` at interpreter exit via `atexit`. The caller frame is resolved with a stack walk that skips `nbx_tensor.py` and returns the nearest `neurobrix/` frame — so rows attribute allocations to `wrappers.py:1410 addmm`, `wrappers.py:414 _prepare_binary`, `dispatch.py:50 _meta_view`, etc., not to the allocator itself. Zero cost when unset: a single `_MALLOC_TRACE_FILE is None` check in the malloc hot path. Used recipes are in the module-top docstring: `awk` aggregate of total volume per site; walk-events-track-live-set to extract the set composition at peak, which discriminates "leak" vs "peak concurrency" vs "fragmentation" — in April 2026 this killed a wrong hypothesis in ~30 min on PixArt triton (the expected culprit was workspace buffers created by `_prepare_binary` / SDPA setup; the actual culprit was `_deferred` accumulation of arena-output slots, landed above as Route A). Kept permanent because the same discrimination work will be needed for Qwen3 long-context KV growth, Sana 4Kpx, video upscalers, any future VRAM investigation.

### Fixed
- **PixArt-Sigma-XL-2-1024-MS and PixArt-XL-2-1024-MS native 1024×1024 green — four orthogonal universal blockers closed** (`core/runtime/graph_executor.py`, `triton/weight_loader.py`, `kernels/nbx_tensor.py`, `core/flow/iterative_process.py`, `triton/flow/iterative_process.py`): **(1)** `_compute_computable_buffers()` previously ran only inside `_load_weights_triton`; the native path never populated `sincos_2d_pos_embed` → `aten.add::0` crashed with an undefined input for PixArt/Sana/any DiT with sincos pos embed. Hoisted into shared `load_weights()` after `_load_constants_from_graph` so both modes compute runtime-resolution buffers. Legacy component-handler `prepare_weights` fallback kept for learned-pos-embed models. **(2)** `_target_nbytes` in the triton weight loader ignored the fp32 → half downcast that the native `WeightLoader(torch_dtype=fp16)` applies implicitly. PixArt's T5 text_encoder ships fp32 on disk but compute is fp16; the triton path was sizing the arena for 19 GB of fp32 while only writing 9.5 GB of fp16, OOMing a 32 GB V100 before the transformer even started. Arena sizing now mirrors the load-loop remap logic exactly (`fp32 + compute ∈ {fp16,bf16} → compute`); matching elif added in the load body handles `fp32 → fp16` and `fp32 → bf16` numpy conversions at load time. **(3)** `NBXTensor.select` did not normalize negative indices — `select(0, -1)` on a `(300,)` contiguous tensor produced `new_off = -1`, `data_ptr() = base − 4`, and Triton's `cuPointerGetAttribute` rejected the pointer as CPU memory with "Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)". Hit in PixArt's T5 `position_ids` build chain (`aten.arange → aten.select → aten.add::1`). Fix: `if index < 0: index = self._shape[dim] + index` before the offset computation. **(4)** Pre-loop components (e.g. PixArt's T5 encoder, ~9.5 GB fp16) run exactly once per request but `loading_mode=eager` kept them resident through the entire transformer loop, fragmenting the triton raw-cudaMalloc allocator into small-alloc failures partway through the DiT. `_execute_pre_loop` now calls `_unload_component(comp_name, force=True)` in both native and triton paths for parity; `persistent_mode` (serve mode) still short-circuits at the top of `_unload_component` so serve-mode latency is unchanged. Native worked anyway via PyTorch's caching allocator, the fix is universal. Post-fix baselines: TinyLlama native + triton GREEN, Janus-Pro-7B triton GREEN (unchanged from `4f6c78e`), PixArt-Sigma/Alpha native 1024×1024 GREEN (coherent with prompt, 15–16 s end-to-end, raw VAE range inside expected `[-1.24, 1.23]`). PixArt triton still OOMs mid-transformer on a separate intermediate-tensor leak (monotonic VRAM climb 6.8 → 32.3 GB at `aten._scaled_dot_product_efficient_attention::49`) tracked as follow-up allocator work.

- **Janus-Pro-7B `--triton` numerically validated: image matches prompt, step-0 logit cosine = 1.000000 vs native** (`core/runtime/graph_executor.py`, `triton/session.py`, closes follow-up `docs/follow-ups/janus_triton_anticorrelation.md`): closes the "step-0 numerical parity still open" note on the previous Janus commit. Prior state at `bc8b9b8` had the pipeline running end-to-end but producing an incoherent image (blue sky for "a cat"); `cos(native cond, triton cond)` on full-vocab pre-CFG logits was -0.986 with triton L2 magnitude ~120× native's, AND triton `cond ≈ uncond` (0.9993) vs native baseline 0.9944. **Root cause**: `TritonLMSession._extract_hidden` shape-guessed for `tensor.shape[-1] == hidden_dim` and fell back to "first graph output" when that didn't match. Janus's `language_model` graph includes the text-vocab `lm_head` (`aten.mm::210 → aten._unsafe_view::301`), so its declared output is text-vocab logits `(2, S, 102400)`, NOT hidden states — the fallback returned these logits to the session, and `gen_head`'s first `aten::view(s0, 4096)` silently reinterpreted `(1, 1, 102400)` as `(25, 4096)` before running the two addmms + gelu, producing the observed ~-1 cosine + 120× magnitude + batch-alias signature exactly. Native's `core/flow/autoregressive.py::GraphLMSession.prefill` handles this via `executor.enable_hidden_states_capture()` which marks the first input of the last `aten::mm` (pre-`lm_head` hidden state) as persistent, then `executor.get_hidden_states()` retrieves it from `ExecutionContext.tensor_store`; triton never called this path. **Fix**: (1) `_ensure_triton_compiled` now pre-processes the DAG to append `self._persistent_tensor_ids` to the DAG's declared `output_tensor_ids` before the `TritonSequence` is constructed — `TritonSequence`'s existing liveness analysis already protects tids in `dag.output_tensor_ids`, so this routes the native capture mechanism into triton without touching the triton sequence; (2) `get_hidden_states` gains a triton-mode branch (`_get_hidden_states_triton`) that reads from `self._triton_seq.gather_outputs(...)` instead of `ExecutionContext.tensor_store`, mirroring the native 2-strategy cascade (graph output is hidden-dim OR last-`aten::mm` first input); (3) `TritonLMSession.prefill` / `decode_step` now call `executor.enable_hidden_states_capture()` before first run and prefer `executor.get_hidden_states(...)` over the old shape-guessing `_extract_hidden` fallback (fallback kept for DeepSeek-MoE-style graphs where output IS hidden states). Post-fix measurements (same prompt "a cat", same hardware): triton image is a photorealistic cat indistinguishable by eye from native (`/tmp/janus_parity/triton_v2.png` vs `native.png`); `cos(native cond, triton cond) = 1.000000`, `cos(native uncond, triton uncond) = 1.000000`, `cos(triton cond, triton uncond) = 0.994395` (matches native baseline 0.994400 within 5×10⁻⁶); argmax matches exactly on both cond (2122) and uncond (3736); L2 magnitudes match to 4 decimal places. Regression harness: 14 passed + 14 skipped + 12 xfailed + 0 failed, unchanged from `bc8b9b8`. Decode time triton 158 s vs native 192 s on cuda:2 (triton faster by ~17 %, unchanged from pre-fix).

## [0.1.6] - 2026-04-20

### Fixed
- **Regression harness: 4 `::native` audio fails eliminated (3 passes recovered + 1 xfail with sourced cause)** (`tests/regression/conftest.py`, `tests/regression/conftest.py::KNOWN_FAILURES`, `docs/follow-ups/kokoro_cudnn_batch_norm_regression.md`, `.gitignore`): prior harness runs at HEAD reported `4 failed / 11 passed / 11 xfailed / 14 skipped`. Three failures (`Voxtral-Mini-3B-2507::native`, `whisper-large::native`, `whisper-large-v3-turbo::native`) were rooted in interpreter selection, NOT in the dependency pin they appeared to complain about — `pytest` at `~/.local/bin/pytest` uses a `#!/usr/bin/python3` shebang, so `sys.executable` inside the pytest subprocess is `/usr/bin/python3` whose user-site has a stale `transformers` with a `tokenizers<0.20` pin and no `mistral_common` installed, while the working venv (`$VIRTUAL_ENV`) has `transformers 5.2.0`, `tokenizers 0.22.2`, and `mistral_common 1.9.1` — all imports clean. The fourth failure (`Kokoro-82M::native`) was a neurobrix-side regression: `aten::cudnn_batch_norm` in the compiled sequence was fed an undefined tensor, failing with `sym_strides() called on an undefined Tensor` at `core/runtime/graph/compiled_sequence.py:3078`. Was green at `a64aa4b` (v0.1.5), red by `be5c7b8` (zero3 ratchet pipelining). Fix applied in this commit: (a) new `pytest_configure` hook in `tests/regression/conftest.py` auto-detects `$VIRTUAL_ENV/bin/python` when `NEUROBRIX_PYTHON` is not set, probes that it can import `neurobrix` and `transformers`, and if so exports `NEUROBRIX_PYTHON` pointing at the venv — single ~500 ms probe at session start, user's explicit override always wins. (b) `KNOWN_FAILURES` gains a `("Kokoro-82M", "native", …)` entry with the exact stderr and commit window sourced to the follow-up at `docs/follow-ups/kokoro_cudnn_batch_norm_regression.md`. Post-fix harness: `14 passed + 12 xfailed + 14 skipped + 0 failed` — verified end-to-end. Alongside: the regression harness itself (`tests/regression/`) is now tracked in git via a focused `.gitignore` exception (`tests/*` still ignores scratch, `__pycache__`, and the rest of `tests/`; only `tests/regression/` is re-included). Prior versions of the repo carried the harness as gitignored, so CHANGELOG claims about "harness: N passed" could not be independently reproduced from the committed tree — that is now fixed. `tests/scratch/` remains ignored (forensic session reports, not part of the shipped harness).

- **Item 3 closed — in-kernel fp16→fp32 weight promotion for `mm` / `bmm` / `addmm` on pre-Ampere hardware (Path A')** (`kernels/ops/matmul.py`, `kernels/wrappers.py`, `core/runtime/graph_executor.py`): `matmul_kernel` and `addmm_kernel` gain a `PROMOTE_B: tl.constexpr = False` flag; when True, the `b` tile is cast to `a.dtype` after `tl.load` and before `tl.dot`, fused with the load, register-scoped, zero heap allocation. The cast is bit-exact because every fp16 value is representable in fp32 without rounding; step-2 activation upcast, fp32 accumulator, and `IEEE_PRECISION` are all unchanged. Phase 1 probe established that Triton 3.6.0 rejects naive mixed-dtype `tl.dot` at compile time (`CompilationError: Both operands must be same dtype. Got fp32 and fp16`, 8 / 8 configurations FAIL in `probe_via_nbx.py`), killing Path A; the in-kernel tile cast probe (`probe_inkernel_upcast.py`) compiles and produces fp32-accurate output in 8 / 8 configurations with max-abs diff 1.2–2.2 × 10⁻⁵. In the wrappers, `mm` / `bmm` / `addmm` detect `fp32 activation × fp16 weight × pre-Ampere` after the existing step-2 activation upcast, set `promote_b = True`, skip the step-3 full-weight widening, and thread `PROMOTE_B=promote_b` to the kernel. All other mismatches (bf16 weight × fp32 activation, same-dtype, bf16-capable hw) fall through to the unchanged widening path. The `mv_kernel` / `addmv_kernel` already upcast both operands to fp32 inside their inner loops (`ops/mv_op.py:39-40`, `ops/addmv_op.py:44-45`), so the M ≤ 4 decode fast path handles mixed dtypes natively and needed no change. Measured on hardware where the bind-time upcast was active pre-fix (fp16 models fitting the fp32 loader budget): TinyLlama-1.1B `v100-16g --triton` peak VRAM 4.74 → 2.54 GB (-46 %) and decode 9.94 → 5.04 s on 8 tokens (-49 %), character-identical output across 30-token greedy decode. On Qwen3-30B-A3B-Thinking-2507 `v100-16g --triton` zero3, bind-time upcast was already off at HEAD (60 GB fp32 > 8 GB loader budget → `weight_loader.py:180-188` rejects the upcast pre-fix), so the change only converts the per-call weight widen into an in-kernel tile cast: peak VRAM 10 861 → 10 797 MB (-64 MB, -0.6 %) and decode 411.9 → 391.2 s (-5 %) on 4-token generation, output identical (`"Okay, the user"`). Regression harness (`pytest tests/regression/ -v`): 11 passed + 11 xfailed unchanged vs `8854dee` baseline; the 4 `::native` audio fails observed at the time of this commit were root-caused and resolved in `e971900` — see the "Regression harness" entry above for the actual diagnosis (interpreter selection + one Kokoro regression), which is NOT the tokenizers/transformers env-drift that prior sessions had attributed them to without a sourced stderr.

- **Multi-GPU Triton latently broken at 163 / 211 kernel-launch sites — systemic `_set_device` hardening** (`kernels/wrappers.py`, `kernels/dispatch.py`, `kernels/nbx_tensor.py`, `kernels/utils/shape_utils.py`, `triton/weight_loader.py`): `kernels/wrappers.py:_set_device(t)` (docstring: "Called before every Triton kernel in wrappers. ALWAYS calls ensure_triton_device — Triton's internal state needs the device context set before each kernel launch") is the contract that synchronises Triton's internal driver state with the tensor's physical device. A full audit across the 5 production files that launch Triton kernels found **163 launch sites missing the guard** (against 48 honouring it). In single-GPU runs the contract is vacuous (active device always matches tensor device), so the bug was invisible. In any multi-GPU Prism plan that placed a component on a device other than the last-active one, the first kernel on the "wrong" device raised `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)` — the generic Triton inaccessible-pointer error, not an actual CPU tensor. Most visible failure: `argmax_wrapper` at the autoregressive sampler under `component_placement` / `pipeline_parallel` with `lm_head → cuda:1`, but `bmm`, `addmm`, `softmax`, `rms_norm`, `layer_norm`, every unary/elementwise op, conv, pooling, MoE scatter, gather, and every RNG wrapper had the same bug. Fix promotes `_set_device` to a single source of truth in `kernels/nbx_tensor.py` (next to `DeviceAllocator.ensure_triton_device`), removes the local duplicate in `wrappers.py`, and inserts the guard at every one of the 163 red sites plus one shared guard before the `scatter_reduce_wrapper` branch. Re-audit: 211 / 211 launch sites 🟢. Validated end-to-end: the 4 multi-GPU Prism strategies that engage on tight-memory profiles (`component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding`) now pass 8 / 8 cells on the Phase 2 extension matrix, in both native and triton modes. Previously `component_placement · triton` and `pipeline_parallel · triton` crashed at the sampler; `block_scatter · triton` and `weight_sharding · triton` only passed by accident because `lm_head` happened to land on cuda:0 under the "most free space" placement heuristic — they are now robust under any placement.

### Added
- **`v100-16g-x2-tight-2200.yml` and `v100-16g-x2-tight-2000.yml` matrix-test hardware profiles** (`config/hardware/`): permanent 2 × V100 16 GB NVLink profiles that report reduced per-GPU memory (2200 MB and 2000 MB respectively) while leaving the real 16 GB available to the runtime. The reported capacity is sized so TinyLlama-1.1B crosses `_try_component_placement` / `_try_pipeline_parallel` / `_try_block_scatter`'s 80 %-of-largest gate (tight-2200) and `_try_weight_sharding`'s full-capacity gate (tight-2000) without OOMing at execution time. Used by the Phase 2 extension matrix (`tests/scratch/prism_matrix_validation/matrix_ext_driver.py`) to exercise all four engagement-gated multi-GPU strategies on a small reference model without renting large hardware. Use cuda:0 + cuda:1 so they never contend with the regression harness on cuda:2 / cuda:3.
- **`NBX_FORCE_STRATEGY` env var for deterministic Prism strategy selection** (`core/prism/solver.py`): setting `NBX_FORCE_STRATEGY=<strategy>` short-circuits the score cascade to the named strategy. If the strategy is unknown → `RuntimeError` listing the 9 valid values (`single_gpu`, `single_gpu_lifecycle`, `component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding`, `component_placement_lazy`, `lazy_sequential`, `zero3`). If the strategy is valid but unavailable for the given device count (multi-GPU strategy on a single-GPU profile) → `RuntimeError` distinguishing the two cases. If the strategy is valid and in-set but its `_try_*` method cannot fit the model → `RuntimeError: ZERO FALLBACK: NBX_FORCE_STRATEGY={strategy} cannot fit ...` — no silent fallback to an alternative, because the operator asked for X specifically and picking Y without notice would hide bugs. Without the env var, behaviour is unchanged. Unit tests (`tests/scratch/prism_matrix_validation/test_force_strategy.py`) exercise all 5 code paths against cached TinyLlama-1.1B + Qwen3-30B fixtures. Unblocks the Phase 2 strategy matrix (`tests/scratch/prism_matrix_validation/matrix_driver.py` + `MATRIX_REPORT.md`) and any future per-strategy regression harness.
- **`NBXTensor.is_expanded()`** (`kernels/nbx_tensor.py`): pure metadata predicate — `any(stride == 0 and shape > 1 ...)`. One canonical helper per R2; used by `to_cuda` / `to_cuda_async` and by `TritonSequence._transfer_tensor` to materialise expand views before H2D.
- **`NBXTensor._contiguous_cpu()`** (`kernels/nbx_tensor.py`): CPU-side strided-copy helper. Wraps the view's byte window via `np.ctypeslib.as_array` + `np.lib.stride_tricks.as_strided` with esz-byte granularity, runs `np.ascontiguousarray` in C, and `ctypes.memmove`s into a fresh `NBXTensor.empty_cpu` preserving pinned/unpinned backing. Handles expand views automatically (numpy replays along stride-0 axes). Zero triton, zero torch.
- **Heterogeneous 2 × V100 16 GB NVLink hardware profile** (`config/hardware/v100-16g-x2-01.yml`): describes cuda:0 + cuda:1 on the Dell C4140 host, leaving cuda:2 / cuda:3 free for parallel workloads (regression harness). Used by the Phase 2 multi-GPU strategy matrix. Header cleaned in the systemic `_set_device` hardening commit to drop the "REMOVE BEFORE COMMIT" placeholder (the profile was always intended to be permanent; the placeholder was a pre-commit gate that got left in).
- **Phase 2 strategy matrix validation artefacts** (`tests/scratch/prism_matrix_validation/`): `matrix_driver.py` (spawns `neurobrix run` per strategy × mode, classifies outcomes, emits JSON + Markdown), `test_force_strategy.py` (5 unit tests over the env var), `MATRIX_REPORT.md` (8 / 16 🟢; the 4 reds are documented as expected "strategy-not-applicable" behaviour for TinyLlama — each red is sourced to the `_try_*` method + line that correctly returned None).
- **Item 1 + Item 2 scratch artefacts** (`tests/scratch/dette_technique_session/`): `test_item1_contiguous_cpu.py` (5 / 5 byte-correct vs torch), `test_item2_expand_h2d.py` (5 / 5 byte-identical to torch reference), `SESSION_REPORT.md` (prior-session context for the continuation).
- **Zero3 block-wise ratchet pipelining (native + triton)** — zero3 is no longer a per-op CPU→GPU transfer "slow path". `Zero3Strategy` now drives a sliding 2-block window (current + prefetched) on the compiled op sequence. At each op-to-block transition it (a) ensures block N+1's weights are resident via an async H2D on a dedicated transfer stream, (b) binds the arena to block N's GPU copy, (c) evicts block N-1 after materializing any dependent intermediates, returning VRAM to the driver immediately. Multi-pass autoregressive decode resets the ratchet via `_post_run_hook` so VRAM stays flat across tokens. Polymorphic on tensor flavor — same strategy code handles both `torch.Tensor` + `CompiledSequence` and `NBXTensor` + `TritonSequence` via the parity APIs. Closes Qwen3-30B-A3B-Thinking-2507 Test A on 16 GB V100: peak VRAM bounded to two blocks' weights + KV cache + activations (~6.7 GB target), prefill dramatically faster than the per-op slow path. The legacy per-op slow path survives only for non-pipelined weights (embeddings, final norm, lm_head — sub-percent of total weight mass).
- **MoE fusion — Pass 2 output-side dead-op sweep** (`core/runtime/graph/moe_fusion.py`): the existing post-fusion dead-op elim only removed ops whose INPUTS were in `removed_producers`. It missed ops whose only consumer (typically `aten::mm` for expert projections) was collapsed into `custom::moe_fused` — notably the `aten::t(expert_weight)` that precedes each expert matmul. Left in the compiled sequence these ran at execution, stored a `.t()` view in the arena, and the view's `_base` pinned the CPU-offloaded expert weight to GPU memory under zero3 (~1.15 GB leaked per block on Qwen3-30B). New Pass 2 iterates to fixed point removing any op in the surviving execution order whose outputs have no active consumer and are not DAG-level outputs. Shared-expert paths and the fused op itself are protected. Empirical: Qwen3-30B-A3B sees 25,009 orphaned ops removed (18,432 are `aten::t` on expert weights — exact 128 × 3 × 48 match; remaining 6,577 are cascaded routing orphans: `aten::unsqueeze` on topk outputs, `aten::zeros` allocator buffers, one `aten::arange`). TinyLlama and other dense LLMs: 0 (no MoE fusion runs). Log controlled by `NBX_MOE_FUSION_LOG=1`.
- **Async stream + event primitives in `DeviceAllocator`** (`kernels/nbx_tensor.py`): `create_stream` / `destroy_stream` / `stream_synchronize`, `create_event` / `destroy_event`, `record_event(event, stream)`, `stream_wait_event(stream, event)`, `memcpy_async(dst, src, nbytes, kind, stream)` — thin ctypes wrappers over `cudaStreamCreate` / `cudaEventCreate` / `cudaMemcpyAsync` (HIP equivalents on ROCm). Backend mapping extended in `_GPU_BACKENDS`. Zero torch; zero allocation overhead for stream handles (opaque ints). Enables the zero3 ratchet to overlap H2D(N+1) with compute(N) on the default stream.
- **`NBXTensor.to_cuda_async(device_idx, stream)`** (`kernels/nbx_tensor.py`): non-blocking variant of `to_cuda` that enqueues the H2D on a user-supplied stream. Caller is responsible for sequencing via `stream_wait_event` or `stream_synchronize`. Documented that the source tensor MUST be pinned for the driver to actually overlap on a non-default stream.
- **`materialize_slots_depending_on(weight_slot_ids)`** — parity API on both `CompiledSequence` (`core/runtime/graph/compiled_sequence.py`) and `TritonSequence` (`triton/sequence.py`). Copies every arena slot whose tensor aliases a weight slot (via PyTorch storage identity / NBXTensor `_base`) into fresh storage via `.contiguous()` so the weight can be evicted without leaving dangling pointers. Under the Pass-2-fixed MoE graph this typically returns 0 — dead views never land in the arena to begin with — but remains as the correct defensive primitive for legitimate future cases (non-MoE patterns, computed-view aliases).
- **MoE `_ptr_cache` fingerprint + LRU** (`triton/moe.py`): the pointer-table cache key was previously `f"moe_{gate_weights[0].data_ptr()}_{num_experts}"`. Under zero3 pipelining weights are freed and reallocated between blocks, frequently at identical virtual addresses for same-sized buffers, so this static key would resolve to a stale `PtrTables` and cause silent garbage or illegal memory accesses. New `_ptr_cache_fingerprint(gate, up, down, num_experts)` hashes ALL 3 × `num_experts` expert `data_ptr()`s so any swap invalidates. `_ptr_cache_get` / `_ptr_cache_put` implement an LRU bounded to `_PTR_CACHE_MAXSIZE = 256` entries (OrderedDict with `move_to_end` on hit, `popitem(last=False)` on overflow) — keeps cache memory flat across hundreds of blocks × devices during long pipelined runs.
- **`TensorArena.__len__` / `Arena.__len__`**: both the native (`core/runtime/graph/compiled_sequence.py`) and triton (`triton/arena.py`) arenas now expose `__len__`, needed by `materialize_slots_depending_on` for slot iteration and by diagnostic scripts that enumerate the arena.
- **Zero3 universality in triton mode** — Qwen3-30B-A3B-Thinking-2507 now runs end-to-end with `--triton --hardware v100-16g` (zero3 cascade, 30 B weights on 16 GB GPU). Previously the feature was non-functional in triton: weight load failed with a 60 GB `cudaMalloc` attempt because the `shard_map[*]="cpu"` contract was silently ignored, and the TritonSequence hot loop had no parity with the correctness APIs added for native zero3. This release lands the full infrastructure + the specific memory-footprint fix that closes Test A.
- **`NBXTensor` CPU form** (`kernels/nbx_tensor.py`): `NBXTensor.empty_cpu(shape, dtype, pinned=False)` / `to_cuda(device_idx)` / `to_cpu(pinned=False)` / `pin_host()` — numpy-backed by default, `cudaMallocHost`-backed when `pinned=True` for fast non-blocking H2D DMA. `__del__` dispatches to `free_host_pinned` or lets numpy GC handle unpinned backing. New accessors: `is_cpu`, `is_pinned`. Zero torch dependency.
- **`DeviceAllocator` accounting + pinned host** (`kernels/nbx_tensor.py`): `malloc_host_pinned` / `free_host_pinned` wrap `cudaMallocHost` / `cudaFreeHost`. Running byte counters: `memory_allocated(device_idx)`, `peak_memory_allocated`, `reset_peak_memory`, `host_pinned_allocated`, `host_pinned_peak`. Enables per-NeuroBrix memory diagnostics without round-tripping through `cudaMemGetInfo` (which reports device-wide usage including other processes).
- **`DeviceAllocator._cuda_ptr_device` per-allocation device tracking** (`kernels/nbx_tensor.py`): every `malloc_cuda` records the allocating device so that `free_cuda`'s `_cuda_live_bytes` decrement always targets the owning device, regardless of which device is current at the GC site. Closes a reporting-only accounting drift that surfaced while instrumenting the zero3 leak investigation. (A stream-ordered `cudaMallocAsync`/`cudaFreeAsync` path was prototyped in the same session and reverted — the cross-device pool semantics broke `pipeline_parallel` multi-GPU on Qwen3-30B; the `_find_cuda_arg` fix below was what actually closed the arena retention leak, not the allocator swap.)
- **Triton weight_loader CPU partition** (`triton/weight_loader.py`): detects Prism's `shard_map = {shard_path: "cpu"}` convention used by zero3 and partitions weights via the `_BLOCK_RE` regex. Block weights (`block.N.*` / `blocks.N.*` / `layers.N.*`) land in pinned host memory as CPU-backed `NBXTensor` via `_load_to_pinned_cpu`; non-block weights (embeddings, final norm, lm_head) stay on the GPU arena because they are accessed directly by flow handlers (notably `GraphLMSession.prefill` → `w.embedding`) that bypass the compiled sequence and expect GPU pointers.
- **`TritonSequence` zero3 parity API** (`triton/sequence.py`) — mirrors commit ea90d66's CompiledSequence additions in triton, same signatures and semantics:
  - `rebind_partial(partial_map) → List[int]` — swap a subset of weight slots, honours `_pretranspose_weights`.
  - `recompute_op_devices_for_slots(modified_slots)` — patch `op.device_idx` / `op.needs_transfer` for only the ops whose weight inputs intersect the modified set. Treats CPU-backed tensors correctly (no conflation with cuda:0 via shared `_device_idx=0`).
  - `get_op_blocks()` — group ops by transformer block index using the shared `_BLOCK_RE`. Cached post-compile.
  - `override_weightless_op_devices(device_idx)` — force tensor-creation ops (arange, scalar_tensor, full) to allocate on the execution device instead of inheriting CPU from the activation-device chain.
  - `mark_cpu_weighted_ops_for_transfer(exec_device_idx)` — flag every CPU-weighted op for the slow path.
  - `run(pre_op_callback=…)` / `_run_multi_device(pre_op_callback=…)` — optional hook fires BEFORE each op's args are resolved, used by zero3 priming and reserved for future pipelining.
- **`GraphExecutor` triton callback plumbing** (`core/runtime/graph_executor.py`): `_run_triton_compiled` now threads `self._pre_op_callback or self._persistent_pre_op_callback` into `TritonSequence.run(...)`. `_ensure_weights_loaded` installs zero3 hooks on both native (`_compiled_seq`) and triton (`_triton_seq`) executors, so flow handlers that bypass `strategy.execute_component` (autoregressive LLM prefill) get the fix transparently in either mode.
- **`Zero3Strategy` universal pin + install** (`core/strategies/zero3.py`): `_pin_cpu_weights` branches polymorphically on `torch.Tensor` vs `NBXTensor` — native uses `.contiguous().pin_memory()`, triton uses `NBXTensor.pin_host()` (no torch leak). `_install` callback resolves either `_compiled_seq` or `_triton_seq` on the executor and calls the matching parity API.
- **`ExecutionStrategy.transfer_tensor` polymorphism** (`core/strategies/base.py`): duck-types on `hasattr(tensor, 'to_cuda') and hasattr(tensor, '_device')` to route NBXTensor through the zero-torch path (`to_cuda(dev_idx)` / `to_cpu()`), and torch.Tensor through `.to(device)`. Single public API, no mode-specific method fork. Zero torch in the NBXTensor branch.
- **MoE fused CPU promotion** (`triton/moe.py`): `execute_moe_fused` detects CPU-backed expert weights and promotes them to the activation device via `NBXTensor.to_cuda(act_dev)` BEFORE `_build_ptr_tables` bakes raw pointers into the GPU int64 table. Under zero3, `_ptr_cache` is bypassed per-call because the promoted tensor addresses are fresh; under normal multi-GPU paths the cache is unchanged. Explicit `del` + `gc.collect` of the promoted lists + `tables` at function exit ensures the 768 MB of expert weights released before the next MoE call.
- **`ComponentArena.__del__` safety** (`triton/memory_pool.py`): `_base_ptr` is initialised to `0` before the `malloc_cuda` call so `__del__` is a safe no-op when `__init__` raises on a failed allocation (secondary issue surfaced during the zero3 leak investigation — previously a failed 60 GB cudaMalloc in `ComponentArena(...)` triggered a second exception from `__del__`).
- **Zero3 correctness path + block-pipelining groundwork**: `Zero3Strategy` now installs a per-executor priming hook that fires on the first op of the first run and flips `op.device` / `op.needs_transfer` on every CPU-weighted and weightless op so the multi-device slow path transfers weights on-the-fly (working set = one op). Previous implementation had four bugs (wrong prefetch target when non-block weights existed, missing wait on the async second prefetch, no per-block loop driving the ratchet, arena never rebound so the prefetched GPU tensors were never consumed) that left Qwen3-30B in zero3 crashing at `aten.mm::0` with `mat2 on cpu`. Hooks install at weight-load time via `RuntimeExecutor._ensure_weights_loaded`, so flow handlers that bypass `strategy.execute_component` (notably `GraphLMSession.prefill` for autoregressive LLMs, which calls `executor.run` directly) get the fix transparently. Verified end-to-end: Qwen3-30B-A3B-Thinking-2507 on a single 16 GB V100 (forced via `--hardware v100-16g`) generates coherent tokens where it previously crashed on the first matmul.
- **`CompiledSequence.rebind_partial(partial_map) → List[int]`**: replace a subset of weight slots on the arena without touching the rest. Honours the same `_pretranspose_weights` contract as `bind_weights`. Returns the list of modified slot indices.
- **`CompiledSequence.recompute_op_devices_for_slots(modified_slots)`**: patch per-op `op.device` + `op.needs_transfer` for exactly the ops that read the modified slots. Complements `compute_op_devices()` for post-bind changes without a full rescan.
- **`CompiledSequence.get_op_blocks() → Dict[int, Dict]`**: introspect the compiled op list and group ops by transformer block index (`block.N.` / `blocks.N.` / `layers.N.` / `model.layers.N.` / `encoder.layers.N.` / `decoder.layers.N.`). Non-block weights → `-1`; weightless ops inherit predecessor's block. Result cached on the sequence (immutable post-compile).
- **`CompiledSequence.override_weightless_op_devices(device)`**: zero3 helper that forces tensor-creation ops (arange, scalar_tensor, full, attn-mask casts) to allocate on the execution GPU instead of inheriting device from the (CPU-correct-for-FGP-but-wrong-for-zero3) activation-device chain built by `compute_op_devices()`.
- **`CompiledSequence.mark_cpu_weighted_ops_for_transfer(exec_device)`**: flip `needs_transfer=True` for every weighted op whose weight is currently on CPU, so the multi-device slow path handles the per-op transfer. Returns the count of flips for diagnostic.
- **Optional `pre_op_callback` on `CompiledSequence.run` / `_run_inner` / `_run_inner_multi_device`**, plumbed through `GraphExecutor.run` via new `_persistent_pre_op_callback` and `_post_run_hook` attributes on GraphExecutor. Explicit per-call callback wins over the persistent one. The multi-device hot loop invokes the callback with `(op_idx, op)` before arg resolution; fast-path single-device ignores it to preserve zero overhead when unused.
- **Hardware-gated fp16 overflow protection (WIP, architectural surface only)**: `PrismProfile.has_native_bf16` property data-driven from `devices_support_dtype("bfloat16")` (covers all vendors). `kernels/wrappers.set_hardware_profile()` propagates the flag into a module-level `_NBX_HAS_NATIVE_BF16` gate; on pre-Ampere hardware (no native bf16), `mm`/`bmm`/`addmm` upcast fp16 inputs and land output in fp32. Triton `matmul_kernel`/`addmm_kernel` gain `IEEE_PRECISION` constexpr to force `tl.dot(input_precision="ieee")` when inputs were promoted to fp32. **Known incomplete**: openaudio DualAR still crashes (upstream `_to_copy(fp32→fp16)` clamps to Inf before mm); perf of per-call weight upcast not yet measured; Ampere+ no-op path not yet mock-verified.
- Flow-aware CLI dispatch in regression harness: STT models now auto-dispatch `--audio`, TTS-with-reference models auto-dispatch reference audio. Unblocks whisper, parakeet, canary-qwen, Voxtral, granite-speech, Kokoro native in automated testing.
- New kernel wrappers in Triton dispatch: `linear`, `isin`, `is_nonzero`, `layer_norm` alias. Enables chatterbox Triton LM stage and openaudio DualAR entry.
- NBXTensor→numpy D2H helper (`_to_numpy`) for flow handlers that need host-side arrays without going through torch.

### Fixed
- **NBXTensor.contiguous() on CPU tensors silently promoted to GPU** (Item 1, `kernels/nbx_tensor.py:1247`): the method hard-coded `NBXTensor.empty(self._shape, self._dtype, f"cuda:{self._device_idx}")` for the destination, then dispatched `_strided_copy` — a Triton kernel — against the source's pointer. On a CPU-backed `NBXTensor` (e.g. zero3-offloaded block weight, or any host-allocated buffer via `empty_cpu`) the destination was silently allocated on GPU and the Triton kernel read host addresses as device pointers, producing undefined behaviour. Fix branches on `self._device`: CPU tensors now go through a new `_contiguous_cpu()` helper that materialises the view in numpy (`as_strided` over the byte window, `ascontiguousarray`, `ctypes.memmove` into a fresh `empty_cpu(..., pinned=self._pinned)`) — CPU stays on CPU, pinnability preserved. GPU path unchanged. Validated in `tests/scratch/dette_technique_session/test_item1_contiguous_cpu.py` (5 / 5 byte-correct vs `torch.Tensor.contiguous().numpy()` across transpose, expand, narrow + permute, pinned-preservation, and the already-contiguous short-circuit).
- **NBXTensor expand views (`stride == 0` axes) over-read backing storage during H2D / D2D** (Item 2, `kernels/nbx_tensor.py` + `triton/sequence.py`): `to_cuda`, `to_cuda_async`, and `TritonSequence._transfer_tensor` all memcpy'd `tensor._nbytes = numel × element_size` bytes from `data_ptr()` into a same-sized GPU allocation. For an expand view — e.g. `(1, 768).expand(2, 512, 768)`, stride `(0, 0, 1)` — numel counts the expanded shape while the source allocation holds only the unbroadcast elements; the memcpy ran past the real buffer and stamped garbage into the GPU tensor. Fix: new `NBXTensor.is_expanded()` predicate (`any(st == 0 and sh > 1 ...)`) gates a `contiguous()` materialisation before the memcpy in all three call sites. Pure transposes (non-zero strides, `nbytes == backing_bytes`) are untouched so the zero3 `.t()` pre-transpose contract survives. Validated in `tests/scratch/dette_technique_session/test_item2_expand_h2d.py` (5 / 5 byte-identical to torch reference across bias broadcast, scalar broadcast, multi-dim expand, async variant, and a non-expand transpose negative control). Supersedes the "Deferred — NBXTensor expand views on CPU" note that was in this file in the earlier revision.
- **`--triton-sequential` did not thread the zero3 pre-op callback** (Item 4, `core/runtime/graph_executor.py:~1481`): `_run_triton_sequential` dispatched ops directly against the store with no callback hook, while `_run_triton_compiled` passes `pre_op_callback=cb` into `TritonSequence.run`. Any strategy that installed a persistent hook — notably zero3 — therefore never fired on the sequential path and zero3-weighted ops ran against CPU pointers. Fix resolves `self._pre_op_callback or self._persistent_pre_op_callback` once outside the op loop and invokes it with `(op_idx, op_data)` before every op, wrapped in a try / except to match the `TritonSequence.run` contract where callback failures are isolated from the compute loop. Smoke-verified with TinyLlama-1.1B-Chat-v1.0 `--triton-sequential --hardware v100-16g-x2-01` — coherent tokens ("Certainly! Here"). Full zero3 end-to-end under `--triton-sequential` on Qwen3-30B is deferred to the Phase 2 matrix extension (the installed branch exists and is exercised on a fitting model; a zero3-forcing spec belongs in that matrix when `NBX_FORCE_STRATEGY=zero3 --triton-sequential` is added there).
- **`_to_copy(fp32 → fp16)` saturated to ±Inf for values outside fp16 range** (Item 5, `core/dtype/engine.py` in both branches of `_make_to_copy`): when the graph emits an explicit fp16-targeted `aten::_to_copy` and the input contains fp32 / fp64 / bf16 values beyond ±65504 (observed upstream of the OpenAudio DualAR pre-projection, flagged in the v0.1.5 known-incomplete list), `inp.to(torch.float16)` produced ±Inf and the next `mm` propagated NaN through the rest of the graph. Fix clamps to `[-65504, 65504]` before the narrowing cast, matching the pattern already established in `core/dtype/converter.py::safe_dtype_convert` line 45. Clamp is identity for in-range values (no numerical change for models that don't overflow); only the saturating path is modified, and it now produces ±65504 instead of ±Inf, which the next `mm` handles correctly. TinyLlama native decode unchanged (3.59 s, "Certainly! Here").
- **Triton zero3 numerical divergence — every pretransposed CPU weight silently untransposed**: `TritonSequence._transfer_tensor` built the destination GPU NBXTensor via `NBXTensor.empty(tensor._shape, tensor._dtype, f"cuda:{target_dev}")`, which hard-sets `_contiguous_strides(shape)` on the destination regardless of the source's actual strides. The memcpy then copied the source's raw backing bytes (which describe the *original*, pre-transpose row-major layout) to the destination, and the destination's contiguous strides re-indexed those bytes as if they were the transposed layout. Effect: every linear weight that was pre-transposed at bind time by `_eliminate_weight_transpose_ops` (i.e. all of them, across all 48 blocks and 128 MoE experts for Qwen3-30B) arrived on the GPU with dims 0 and 1 swapped back to original; every `mm` then computed `act @ W` instead of `act @ W.t()`. Test A (`Qwen3-30B-A3B-Thinking-2507 --triton --hardware v100-16g --prompt "2+2="`) produced `"OTTéraquate RED"` with a flat, ~half-magnitude logit distribution (top1 +17.42 vs native +37.31, disjoint top-10) — the model was running forward with structurally wrong weights. Fix: build the destination directly via `NBXTensor(dst_ptr, tensor._shape, tensor._strides, tensor._dtype, 'cuda', owns_data=True, device_idx=target_dev, offset=0)` so the stride semantics of the view survive the transfer; downstream `.contiguous()` inside `wrappers.mm` then correctly materialises via `_strided_copy`, matching how `torch.Tensor.to(device)` preserves strides on the native side. Validation: Test A triton now produces `"Okay, the user"` (identical to native), cosine similarity of step-0 logits = 0.999999, argmax match, top-10 same order with max per-token delta of 0.13 (fp16 rounding). Bonus: deepseek-moe-16b-chat `--triton` previously returned gibberish (`"eses самоу思acular"`) — root cause was the same bug via MoE expert-weight promotion through `_transfer_tensor`; it now matches native (`" 4"`). Investigation log in `tests/scratch/divergence_inv/DIVERGENCE_REPORT.md`; regression coverage in `tests/scratch/zero3_triton_impl/test_triton_seq_parity.py::test_transfer_tensor_preserves_transposed_view` (+ contiguous-source sanity + GPU-to-GPU variants). Pipeline_parallel triton (all-GPU weights, historical caller of `_transfer_tensor` on contig activations only) unchanged and confirmed non-regressed.
- **Triton zero3 arena retention leak (+1.28 GB per transformer block → OOM at block 7/9 on V100 16 GB)**: the slow path in `TritonSequence._run_multi_device` was unconditionally promoting CPU-backed weight inputs to the execution GPU, even for metadata ops (`aten::t`, `aten::view`, `aten::reshape`, `aten::permute`, `aten::unsqueeze`) that only manipulate strides and produce VIEWS. Each metadata op on a CPU-resident zero3 weight therefore allocated a fresh 3 MB GPU temp, then returned a view whose `NBXTensor._base` held that temp alive for the lifetime of the arena slot — ~389 unfreed allocations accumulated per transformer block, exactly the block's MoE expert weight size. Native CompiledSequence's slow path scans args for a CUDA tensor first and only promotes when compute on GPU is actually required; ported to triton via `_transfer_args` + a new `_find_cuda_arg(args)` helper. Investigation methodology in `tests/scratch/zero3_triton_impl/LEAK_PINPOINT_REPORT.md`; arena slot histogram confirming the fix in `analyze_histograms_v2.py` (triton `aten::t` total: 2415 MB of unique storages before fix, 0 MB after — matching native). Per-block growth post-fix: ~67 MB (identical to native, matches KV-cache accumulation rate).
- `MemoryManager.unload_weights` silent use-after-free: `device_sync()` was called AFTER `weights_dict.clear()` (too late — `clear()` already triggered `ComponentArena`/`NBXTensor` finalizers that call `cudaFree`) and with no device argument (no-op on multi-GPU — `device_utils.py:27` returns early on `None`). If a kernel was still in-flight on the stream when its buffer was freed, the CUDA context was silently corrupted and the next `cudaMalloc` failed with `cudaErrorIllegalAddress` (err 700), which the allocator wrapper misreported as "GPU malloc failed". The cache flush at the end of the same function had the exact same bug (`device_empty_cache()` with no argument also returns early — `device_utils.py:43`). Fix enumerates every device in the dict (`_arenas`, `NBXTensor`, `torch.Tensor`) once, syncs each BEFORE clearing refs, then flushes each device's cache AFTER gc. Exposed by lifecycle / lazy strategies that actually unload between phases.
- Pre-Ampere LLM decode regression introduced by the wip fp16 overflow protection: weights now upcast to fp32 once at bind time (when VRAM permits) instead of on every matmul call. TinyLlama 1.1B `--triton` decode on V100 returns to the v0.1.5 baseline (matmul ~28 ms/step vs ~285 ms/step in the regressed wip). Models too large for fp32 weights (e.g. Qwen3-30B) fall back silently to per-call upcast.
- Janus-Pro-7B Triton: autoregressive flow now family-aware, no longer tries to apply `chat_template` on image-generation models.
- Zero-torch contract in `triton/flow/audio.py`: `_get_compute_dtype` now returns a string; torch conversion pushed to stage handlers (`core/flow/stages/`) where torch is accepted as boundary.

### Changed
- **NBXTensor weight loader no longer bind-time upcasts fp16 weights to fp32 on pre-Ampere hardware** (`core/runtime/graph_executor.py`, `triton/weight_loader.py`): `GraphExecutor._ensure_weights_loaded` previously passed `upcast_fp16_to_fp32 = not _w.has_native_bf16()` into `load_component_weights`, which on V100 / Turing would double fp16 weights to fp32 at load time when the fp32 footprint fit a per-device 50 %-VRAM budget — a workaround for the per-call widen in `wrappers.mm` step-3. With the in-kernel `PROMOTE_B` path (above) that per-call widen is gone at the site where it actually mattered, so bind-time upcast is no longer load-bearing: `upcast` is now hardcoded to `False` on all paths, and the dead `per_device_budget` computation it fed is removed. The `load_component_weights` call site drops both `upcast_fp16_to_fp32=` and `per_device_vram_budget=` kwargs (they still default to `False` / `None` on the loader signature — no loader-side change). Net effect for fp16 models that previously bind-time-upcast: half the weight VRAM footprint, no per-call weight copy, and measurable speedup from the skipped load-time work (TinyLlama-1.1B `v100-16g --triton`: 2.2 GB less peak, 50 % faster decode). The fall-through widening path for non-promoted mismatches (bf16 weight × force-fp32 activation, etc.) is untouched.

- **NBXTensor metadata ops propagate `pinned=self._pinned`** (`kernels/nbx_tensor.py`, 10 sites — `view`, `unsqueeze`, `squeeze`, `permute`, `transpose`, `expand`, `narrow`, `select`, `unfold`, `as_strided`): view constructors previously dropped the pinned flag, so a transposed / reshaped view of host memory allocated via `cudaMallocHost` lost its `_pinned` marker. `_contiguous_cpu()` (Item 1) reads this flag to decide whether the materialised contiguous copy should also be pinned, so correct propagation is a prerequisite for pinned-roundtrip byte-correctness. All 10 sites changed by a single `replace_all` — pattern was identical. Validated by `test_item1_contiguous_cpu.py::test_pinned_preservation`.
- **Zero3 execution model — from per-op slow path to block-wise pipelining**: prefill and decode under the zero3 cascade are substantially faster and use bounded VRAM (two blocks' weights + KV cache + activations) regardless of model size. The per-op transfer path is retained only for the small non-block weights (embeddings, final norm, lm_head) which stay on CPU through the whole run.
- Stage handlers (`core/flow/stages/kokoro.py`, `vibevoice.py`): added `_coerce_torch_dtype` helper to accept both string (from Triton engine) and `torch.dtype` (from native engine).

### Removed
- **Dead `per_device_budget` block in `graph_executor.py::_ensure_weights_loaded`**: computed per-device 50 %-VRAM budgets for the (now-unreachable) bind-time fp16→fp32 upcast path. Since `upcast = False` always, the budget dict was dead code. Removed along with the `hw_profile` variable that was only used to build it. ~10 lines freed.
- **Session-local A/B benchmark env gates `NBX_FP16_BIND_UPCAST` and `NBX_SKIP_PROMOTE_B`**: used during the Item 3 Phase 1 probe and Phase 3 baseline reproduction to force-enable the old bind-time upcast path and to bypass the new `promote_b` branch respectively. Both served a single investigation and have no place in production. Verified absent from `src/` via grep.

- **Unused locals in `core/prism/solver.py::solve`**: `total_mem` (line 473), `total_vram` (line 480), and `scale_factor` (line 923 inside `_apply_attention_correction`) — computed but never referenced in the method bodies. Kept the computation pipeline around them (`sum(component_memory.items())`, device preparation) because those remain load-bearing. ~3 lines freed.
- **Duplicate local `import` sites consolidated to top-level** in `core/prism/solver.py`: `import logging` (was re-imported inside the serve-mode fallback branch), and two redeclarations of `from neurobrix.core.prism.structure import AllocationStrategy` inside `_score_strategy` and `_build_plan`. The top-of-file imports (`import logging`, `from neurobrix.core.prism.structure import AllocationStrategy, ...`) now cover these use sites once. ~4 lines freed.
- **Unused `except RuntimeError as e:`** in `core/prism/solver.py::solve` KV-cache fit-check (the exception object was never referenced in the except body). Replaced by bare `except RuntimeError:` — same behaviour, 1 character saved and Pyright no longer flags the unused binding.
- **Dead zero3 pipelining scaffolding** that never worked: `_prefetch_block`, `_wait_prefetch`, `_evict_block_from_gpu`, `_gpu_weight_cache`, `_block_groups`, `_group_weights_by_block`, and the module-level `_BLOCK_RE` regex (regex moved to `compiled_sequence.py` where it serves the general-purpose `get_op_blocks` API). These methods were never reached correctly at runtime — the prefetched GPU tensors ended up in a dict that the compiled sequence never consulted, so the existing CPU→GPU slow path was carrying all the work anyway.

### Deferred
- **Qwen3-30B `v100-16g --triton` peak VRAM ≤ 6.0 GB target**: the Item 3 brief (preceding session) targeted ≤ 6.0 GB on Qwen3 down from an "8.3 GB actuel" baseline. Phase 3 measurement at HEAD `8854dee` shows the actual current peak is ~10.8 GB, not 8.3 GB — the brief's baseline does not reproduce at HEAD. For Qwen3 on V100, bind-time fp16→fp32 weight upcast was already off pre-fix (60 GB fp32 > 8 GB loader budget → `weight_loader.py:180-188` rejects), so the only thing this commit changes for Qwen3 is where the activation×weight dtype alignment happens (heap per-call → tile inside kernel). Measured gain on Qwen3: -64 MB peak, -5 % decode. The remaining ~10.7 GB is dominated by zero3 working set (two block-windows of expert weights under ratchet pipelining), lm_head + embeddings non-block residents on cuda:0, MoE expert promotion buffers (`triton/moe.py::execute_moe_fused` copies CPU-backed expert weights onto the activation device per MoE op call), prefill activation working-set, and KV cache. Reaching ≤ 6.0 GB requires orthogonal work on zero3 window sizing, MoE expert residency policy, or KV-cache compression. Scoped in `docs/follow-ups/qwen3_vram_investigation.md` as a follow-up chantier.

- **Item 3 — fp16 bind-time upcast safety gate** (session SESSION_REPORT.md): the naive implementation (gate the bind-time upcast on `_scan_bf16_fp16_safety`) was prototyped, validated byte-correct on TinyLlama but regressed per-token decode ~85 % (0.246 ms/token → 0.452 ms/token). Root cause: `wrappers.mm` at `_NBX_HAS_NATIVE_BF16 == False` upcasts activations to fp32 per call, and the "dtype alignment" block (`wrappers.py` lines 1053–1059) then widens the fp16 weight to fp32 per call too when bind-time upcast hasn't happened. Skipping bind-time upcast shifts the fp32 weight copy cost from load-once to per-call-many. Fix requires kernel-level work (teach `matmul_kernel` to accept mixed fp32_act × fp16_weight natively OR switch to op-level `AMP_FP32_OPS` classification on the triton side instead of the current blanket pre-Ampere gate). **Closed in this commit** via Path A' in-kernel tile promotion; see Fixed section above.
- **Phase 2 red verdicts for multi-GPU strategies** (`tests/scratch/prism_matrix_validation/MATRIX_REPORT.md`, 4 / 8 strategies red in both modes): `component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding` all return None from their `_try_*` methods when TinyLlama-1.1B is forced onto `v100-16g-x2-01` because the model is small enough to fit a single GPU, which the strategy-specific fit heuristics correctly reject. Sourced per-strategy in the report. These are NOT Prism regressions — reds are produced by the new `ZERO FALLBACK: NBX_FORCE_STRATEGY=X cannot fit` path (added in this session). Demonstrating these strategies' green behaviour requires a larger (model, hardware) pairing (Qwen3-30B on 4 × 16 GB or DeepSeek-MoE on 2 × 32 GB). Matrix extension belongs in a dedicated session so it can run Qwen3-30B × 10 multi-GPU runs without harness-cuda:2 contention.

## [0.1.5] - 2026-04-15

### Added
- Regression harness (`tests/regression/`) — automated model×mode matrix, golden output comparison, pytest-based with `--runslow` flag for heavy models.
- Three graph-level fusion passes for Triton decode optimization:
  - Dead causal mask elimination: removes ~132 ops/step (ones→tril→logical_not→where chain feeding SDPA attn_mask, replaced by kernel-native IS_CAUSAL).
  - SwiGLU fusion: collapses silu+mul into single `custom::swiglu_fused` kernel (~22 ops/step).
  - RoPE fusion: replaces 18-op rotate_half chain per layer (slice×4, neg×2, cat×2, mul×4, add×2) with single `custom::rope_fused` kernel backed by Liger-Kernel's `rope_forward_kernel` (~396 ops dropped for 22-layer models).
- Cumulative Triton decode performance (TinyLlama V100 fp16): step time 460 ms → 94 ms (4.9× faster), element-wise ops 684 → 376 (−45%).

### Fixed
- Sana diffusion transformer NaN in Triton mode: `bmm` attention scores overflowed fp16 on V100. `bmm` now always outputs fp32 for half-precision inputs (attention intermediates are temporary, no OOM impact). SDPA wrapper aligns Q/K/V dtypes before kernel launch.
- Native CFG engine crash on diffusion models (Sana, PixArt-Sigma): string dtype from Prism's allocation was passed to `torch.Tensor.to()` which interpreted it as device name. Added `_resolve_torch_dtype` helper.
- Kokoro Triton startup crash on 1-D constant tensors in models without a `seq_len` symbol.

### Changed
- Weight transpose elimination (`_eliminate_weight_transpose_ops`) ported from native to Triton — 154 fewer ops/step, structural parity with native CompiledSequence.
- Orphan `rope_wrapper` removed (incompatible with kernel, zero call sites in any model graph). Replaced by `rope_fused_wrapper` with correct Liger kernel signature.

### Documented
- WARNING blocks added to `stages/kokoro.py` and `stages/vibevoice.py` flagging runtime dependency violations (phonemizer/espeak-ng imports, PyTorch native bypass of TensorDAG).
- `KNOWN_FAILURES` in regression harness `conftest.py` with exact reasons for each xfail.

## [0.1.4] - 2026-04-14

### Changed
- DeepSeek benchmark script now requires `HF_TOKEN` to be provided via the shell environment or a gitignored `.env` file; no token is ever hardcoded. This replaces the previous version of the same file, which shipped with a hardcoded token — users who pulled `0.1.3` from the sdist should upgrade.
- GitLab CI `publish-pypi` stage switched to `when: manual`. New version tags no longer trigger an automatic PyPI upload; an operator now reviews the build artefacts on GitLab and clicks the job explicitly.

### Security
- Rotate the credential that was shipped in the `neurobrix-0.1.3.tar.gz` sdist on PyPI (hardcoded HuggingFace access token in `benchmarks/profile_hf_deepseek.py`). The sdist has been yanked; `pip install neurobrix` now resolves to `0.1.4` by default. Users who installed `0.1.3` from the sdist (not the wheel — the wheel does not include benchmarks) should upgrade to `0.1.4`.

## [0.1.3] - 2026-04-14

### Added
- `--triton` mode: DeepSeek-MoE-16B now supported end-to-end (greedy output `" Hello! How can I help you today?"` on `"Hello"`, matching the native path semantically). Joins TinyLlama-1.1B and Qwen3-30B-A3B as fully working LLMs in the Triton runtime.
- `--triton` decode speedups for LLMs across the board (V100 numbers, fp16):
  - TinyLlama-1.1B: full decode step 443 → 160 ms (2.8× faster).
  - Per-matmul on the decode hot path: 2.02 → 0.20 ms (10–18× depending on shape).
  - Per-SDPA on decode with GQA: 1.58 → 0.94 ms (1.7×).
- Decode-aware output precision for matrix multiplication: when running one token at a time, accumulation now lands in fp32, preventing silent overflow on very deep MoE stacks (Qwen3-30B observed crash → now stable). Prefill and image/video spatial matmuls keep their fp16 output — no memory regression on diffusion.
- Decode-aware attention: short-query attention now uses a compact block size (no more 99% wasted compute when generating one token at a time). GQA models compute in place — K/V are no longer expanded to the Q head count in front of every attention call.
- Triton profiling harness (opt-in, off by default):
  - `NBX_TRITON_PROF=1` — per-category ms/op breakdown (matmul / sdpa / elem / meta / embed / other) for every run.
  - `NBX_DUMP_TIDS=<path>` + `NBX_DUMP_TIDS_FILTER=<substrings>` — dump any op output as JSON for side-by-side native vs Triton numerical diff.
  - `NBX_MOE_DIAG=1` — dump MoE routing intermediates on the first forward pass.
- New benchmark: `benchmarks/profile_hf_deepseek.py` — reference timings against the HuggingFace + Accelerate device_map=auto baseline on the same hardware.

### Changed
- Triton dtype policy simplified: only `div` still forces an fp32 input upcast. Matmul ops (`mm`, `bmm`, `addmm`) now cooperate with the new decode-aware output precision instead of forcing every input to fp32 per call — removes a ~3.5 GB per-decode-step weight-copy cost that was silently capping throughput.
- Triton graph-load pipeline is more permissive about trace-shaped vs declared-shaped buffers: models that ship position-indexed lookup tables sized from the trace sample (DeepSeek's per-block rotary cache is the reference case) now load instead of crashing on a shape mismatch.
- Embedding wrapper accepts any scalar index dtype and casts internally. Fixes diffusion timestep → embedding paths (PixArt-Sigma and similar DiTs) that used to crash with a pointer/float type error.

### Fixed
- DeepSeek-MoE-16B `--triton`: previously produced gibberish at decode (`"роко"` / `"!!!!!"`). The three root causes were all addressed in this release:
  1. The model's MoE routing normalisation flag was silently ignored in Triton mode (defaulted to the Qwen3 convention), collapsing routed-expert magnitudes ~20×.
  2. Top-k selection over a softmax with non-power-of-two k returned a corrupted tail — the fix skips a redundant sort stage when the input fits in a single chunk. Side effect: one fewer kernel launch per MoE layer for every model whose expert count fits in that chunk.
  3. RoPE position indexing collapsed after the first decode step when cos/sin were recomputed per forward (DeepSeek's pattern). The runtime now pins the RoPE chain at its traced size so subsequent decode positions stay in-bounds.
- Qwen3-30B-A3B `--triton` now runs noticeably faster at decode from the new attention block-size heuristic and the GQA-in-place kernel path.
- TinyLlama-1.1B `--triton` decode is faster end-to-end (2.8× step time) and keeps the same output as before on greedy runs.
- PixArt-Sigma `--triton` no longer crashes on the embedding kernel (timestep dtype) or on the first `aten::add` after the timestep path (computable buffers now enter the Triton runtime in the expected tensor type). Further progress is blocked by an SDPA VRAM allocation failure partway through the transformer on 16 GB V100s — tracked as a separate issue; the native path has an unrelated config bug on the same setup.

### Removed
- Per-attention-call GQA materialization (`unsqueeze → expand → reshape → contiguous` of K and V). Replaced by kernel-native stride indirection, active only when the model has GQA; non-GQA models are bit-identical to before.

## [0.1.2] - 2026-04-03

### Added
- `--triton` mode — compiled Triton kernel inference (136 kernel files, 128 dispatch entries)
- `--triton-sequential` mode — sequential Triton execution for kernel debugging
- `TritonSequentialDispatcher` — extends NativeATenDispatcher, routes compute ops to Triton kernels
- 136 pure `@triton.jit` kernel files extracted from FlagGems, attorch, Liger-Kernel, Flash-Attention (Dao-AILab)
- NBXTensor — lightweight tensor descriptor for zero-PyTorch metadata ops
- Universal launch layer: `_prepare_binary`, `_prepare_comparison` — broadcasting, scalar handling, device context for all ops
- `_cuda_guard` in dispatch — handles multi-GPU + Zero3 CPU offloading transparently
- Metal GPU detection: `--triton` on Apple Silicon shows "not compatible" message
- CPU Triton backend: auto-enables `TRITON_CPU_BACKEND=1` on CPU-only machines
- Symbolic shape patching for sequential mode: `_patch_seq_len_in_ops` resolves trace-time seq_len in creation ops
- Pure Triton inference mode for LLM autoregressive generation (`--triton` flag)
- Zero-torch flow handler: autoregressive.py, samplers.py, generator.py, session.py
- Triton sequential debug mode (`--triton-sequential` flag)
- KV cache with GQA support for Triton decode (O(1) per token)
- Strided scatter kernel for non-contiguous KV cache writes
- NBXTensor boundary functions: nbx_to_torch(), nbx_dtype_to_torch()

### Changed
- DtypeEngine: merged `FP16_PRECISION_OPS` into `_FP16_NEED_FP32` (subset of `AMP_FP16_OPS`), eliminated duplicate sets
- DtypeEngine: `amp_cast_inputs()` now handles `_FP16_NEED_FP32` (was only in `compile_op`)
- Conv2d kernel: replaced FlagGems with attorch (V100 `num_stages` compatibility)
- Conv1d: routes through conv2d via unsqueeze (V100 safe)
- `pow` kernel: uses `libdevice.pow` for negative base handling (was `exp(e*log(x))` → NaN)
- `compiled_ops.py` enforces: missing Triton kernel for compute op = crash with descriptive error
- Remove @triton.autotune from 100+ element-wise kernels (fixed BLOCK_SIZE=1024)
- Remove @triton.autotune from 36 compute-bound kernels (fixed conservative configs)
- Cold start reduced from 8+ minutes to ~5 seconds

### Fixed
- NBXTensor.cat() called is_contiguous as attribute instead of method — corrupted RoPE
- Symbolic promotion skipped when multiple symbols share trace_value (s1/s3 ambiguity)
- SDPA double-masking when graph passes explicit causal mask with is_causal=False
- NBXTensor.__setitem__ used flat copy_kernel on non-contiguous narrow view — corrupted KV cache
- NBXTensor.contiguous() used memcpy instead of strided copy for non-contiguous views

### Removed
- `kernels/adapter.py` (1181 lines) — replaced by `dispatch.py` + `wrappers.py`
- `kernels/mapping.py` (155 lines), `kernels/resolver.py` (316 lines), `kernels/registry.py` (68 lines), `kernels/exceptions.py` (15 lines)
- `kernels/ops_legacy/` directory, `kernels/arch/` directory, `kernels/spec.py`
- `_execute_triton_op`, `_precompile_dispatch_table`, `_exec_type_map` from graph_executor.py
- Apple Silicon (MPS) support — M1 through M5 Ultra, unified memory, auto-detection
- `DeviceBrand.APPLE` with `"mps"` device prefix in Prism hardware abstraction
- Apple Silicon chip database (20 variants: M1-M5 base/Pro/Max/Ultra with GPU cores, bandwidth, memory)
- `device_utils.py` — unified device abstraction (`device_sync`, `device_empty_cache`, `device_seed`, `device_memory_stats`, `device_multinomial`)
- No-silent-fallback guardrail hook (blocks `PYTORCH_ENABLE_MPS_FALLBACK` and try/except device swallowing)
- Single-GPU strategy shortcut in Prism solver (skips multi-GPU cascade for 1-device hardware)
- `neurobrix doctor` command with OS-specific PATH fix instructions
- GitLab CI/CD pipeline for PyPI publishing (OIDC trusted publisher + API token fallback)

### Changed
- All `torch.cuda.empty_cache()` calls replaced with device-agnostic `device_empty_cache()` (26 call sites across flow handlers, strategies, graph executor, serving engine)
- All `torch.cuda.synchronize()` for timing replaced with `device_sync()` (serving engine, strategy base)
- All `torch.cuda.manual_seed_all()` replaced with `device_seed()` (serving engine)
- VRAM reporting in serving engine uses `device_memory_stats()` (supports CUDA + MPS)
- `torch.multinomial` replaced with `device_multinomial()` — CPU round-trip on MPS (9 call sites)
- Removed hardcoded `"cuda:0"` defaults from loaders and strategies — crash explicitly if Prism provides no device
- All repository URLs migrated from GitHub to GitLab (`gitlab.com/neurobrix/Neurobrix`)
- Dependencies updated: added `pydantic`, `packaging`, `torchaudio`, `snac`, `phonemizer`, `imageio-ffmpeg`, `transformers`, `mistral-common`, `tiktoken` — all families work out of the box
- bf16 dtype support gated by Apple chip generation (M2+ with macOS 14+)

### Removed
- `licenses.py` — hardcoded license classifications deleted. Hub is the single source of truth.

### Fixed
- License gating desync between CLI and hub — CLI now reads `gated`/`licenseName`/`licenseUrl` from hub API
- Serving engine crash on `ExecutionPlan.allocations` — use `primary_device` property
- Prism profile loader mapped unknown brands to NVIDIA silently — Apple got `cuda:0` instead of `mps:0`. Now crashes on unknown brand.
- Weight loader only transferred weights to CUDA GPUs — MPS weights stayed on CPU, triggering multi-device path. Now transfers to any GPU device.
- macOS daemon used `os.fork()` + `os.setsid()` which breaks Metal GPU access (MTLCompilerService is per-session). Now uses `subprocess.Popen` like Windows.
- False `avx2` ISA warning on Apple Silicon — ARM chips use NEON, not x86 ISA. Skip check for arm64.
- Apple M2+ now prefers bf16 (not fp16) — bf16 has fp32 exponent range, prevents overflow in matmul/conv accumulation that caused blurry image output
- MPS dtype flow: AMP stays ON (same rules as CUDA). fp32 precision chain flows through single-input ops (pow, mean, rsqrt) safely. Multi-input ops (mm, addmm) cast inputs to compute_dtype via AMP FP16 wrappers. No mixed dtype at multi-input op boundaries.
- SNAC audio decoder had silent `except ImportError` fallback returning zeros — now crashes explicitly
- `python -m neurobrix` shows PATH hint when CLI not on PATH

## [0.1.0] - 2026-03-26

First stable release of NeuroBrix — universal deep learning inference engine.

### Added
- NBX container format with TensorDAG, topology, manifest
- Prism hardware solver with multi-GPU allocation (11 strategies: single_gpu through zero3)
- CompiledSequence zero-overhead execution engine (eliminates all Python dict lookups)
- DtypeEngine with automatic mixed precision (standard PyTorch AMP rules)
- 4 model families: image (diffusion + VQ), LLM, audio, video
- CLI commands: `run`, `serve`, `chat`, `stop`, `hub`, `import`, `list`, `remove`, `clean`, `inspect`, `validate`, `info`, `doctor`
- MoE (Mixture of Experts) fused dispatch with NOP propagation
- KV cache with data-driven sizing and on-demand growth
- Triton GPU kernel framework
- NeuroBrix model registry at neurobrix.es
- Support for 34 models across 4 families (LLM, image, audio, video)
- Audio family: all 11 models working — Whisper, Whisper V3 Turbo, Parakeet, Orpheus, Canary-Qwen, Kokoro-82M, VibeVoice-1.5B, Voxtral, OpenAudio-S1, Granite Speech, Chatterbox
- Audio flow handlers: encoder_decoder, audio_llm, dual_ar, rnnt, tts_llm
- Universal AudioEngine with data-driven flow routing
- SANA-Video 720p support (video generation)
- Persistent model serving: `neurobrix serve`, `neurobrix chat`, `neurobrix stop`
- Multi-turn conversation with context management and automatic summarization
- Universal hardware auto-detection — `--hardware` flag is optional
- Cross-platform support: Windows, macOS, and Linux
- Platform-adaptive IPC: AF_UNIX on Unix/macOS, TCP localhost on Windows
- Universal TilingEngine — data-driven per-component tiling with accumulate-and-divide blending
- Symbolic spatial dims in compiled graphs — view/reshape ops use expression trees for multi-resolution
- ExprArg in CompiledSequence — runtime resolves symbolic expressions
- 12+ GPU hardware profiles (RTX 20/30/40 series, A10, A100, H100, L40S, T4, V100)
- Pipeline parallel, block scatter, weight sharding allocation strategies
- Prism hot/cold budget split for serve vs run mode
- Zero3 layer-wise pipelining with dual CUDA streams
- License system for model distribution with acceptance flow in `neurobrix import`
- Enterprise-grade documentation system (MkDocs Material)
- `neurobrix doctor` command for diagnosing PATH and installation issues

### Security
- Enforced `weights_only=True` for torch.load (prevents pickle RCE)
- Zip-slip path traversal validation in registry import
- Safe arithmetic parser replacing `eval()` in shape resolver

[Unreleased]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.6...main
[0.1.6]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.5...v0.1.6
[0.1.5]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.4...v0.1.5
[0.1.4]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.3...v0.1.4
[0.1.3]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.2...v0.1.3
[0.1.2]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.0...v0.1.2
[0.1.0]: https://gitlab.com/neurobrix/Neurobrix/-/releases/v0.1.0
