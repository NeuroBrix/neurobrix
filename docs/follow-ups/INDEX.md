# Follow-ups index (D10 backlog)

Named, scoped technical-debt items that are documented but not
fixed in the chantier where they were surfaced. Each entry:
**severity** (P0 blocks shipping / P1 user-visible / P2 hygiene),
**scope** (one-liner), **site** (file:line if pinpointable),
**repro** (minimum command).

When a follow-up is fixed, the entry moves to the matching
verdict in `docs/verdicts/<chantier>/` and is **removed from
this index**; the orphan-tracking belongs here, the resolution
belongs there.

---

## Dtype engine

### P-DTYPE-MOE-ROUTER-FP32-TRITON-MIRROR — P2
**Scope**: the torch-mode MoE router fp32 upcast now lives in the dtype
engine (`dtype.engine.routing_upcast_fp32`, both torch sites routed through
it — done, see verdict). The symmetric move for the triton path is open: the
NBXTensor routing upcast at `triton/moe.py:375` (`gate_scores.to(float32)`)
should route through a `TritonDtypeEngine` seam so the triton engine is
likewise the single authority for its routing-upcast policy (R30 symmetry of
authority, not of code — R33 keeps the two engines sealed). Deferred here
because it pulls in triton-MoE re-validation (P-TRITON-MOE-DETERMINISM
territory) the torch chantier did not scope.
**Site**: `src/neurobrix/triton/moe.py:375`; new seam in `src/neurobrix/triton/dtype.py`.
**Repro**: n/a (architecture symmetry, no defect).
**Surfaced**: P-DTYPE-MOE-ROUTER-FP32 closure (2026-05-26).

### P-SEQUENTIAL-RUN-BACKTOBACK-HANG — P2
**Scope**: two `neurobrix run` invocations chained in the same shell — the
**second** hangs indefinitely (observed >30 min, SIGTERM'd) while the first
completes normally (~27 s). Reproduced 2/2 on back-to-back sequential runs;
0/2 on standalone single runs (fresh shell each). Suspect: GPU
memory/CUDA-context not reclaimed fast enough between consecutive processes
on the same device (cuda:2, ~30 GB DeepSeek-MoE) — the second process's
allocation stalls. Bites any harness that chains model runs in one shell
(the regression suite is a candidate); the workaround is one run per shell.
Not a runtime-correctness bug (outputs are deterministic when runs complete).
**Site**: TBD — process-teardown / driver context-reclaim boundary. Likely
a CUDA driver lazy-release on process exit (outside NeuroBrix code) rather
than `core/memory/manager.py` cleanup; confirmed empirically that the hang
needs back-to-back processes within one shell — separate invocations (with
the few-second gap of a fresh shell) reclaim fine.
**Repro**: `python3 -m neurobrix run --model deepseek-moe-16b-chat --prompt Hello --max-tokens 8 --sequential; python3 -m neurobrix run --model deepseek-moe-16b-chat --prompt Hello --max-tokens 8 --sequential` (2nd hangs).
**Surfaced**: P-DTYPE-MOE-ROUTER-FP32 proof runs (2026-05-26).

---

## Triton kernel-op coverage gaps

### P-TRITON-NBXTENSOR-REPEAT-MISSING — P1
**Scope**: `NBXTensor` has no `.repeat()` method; `aten::repeat`
hits the missing-meta path at runtime.
**Site**: `src/neurobrix/kernels/dispatch.py:161` (`_meta_repeat`).
**Repro**: `nbx run --model canary-qwen-2.5b --audio test_speech_ref.wav --prompt "..." --triton` (encoder hits `aten::repeat`).
**Surfaced**: Ch5 audio_llm port verdict.

### P-TRITON-SAFE-SOFTMAX-MISSING — P1
**Scope**: `aten::_safe_softmax` has no triton kernel/handler.
**Site**: `src/neurobrix/triton/sequence.py:1526` (`_compile_op`).
**Repro**: `nbx run --model granite-speech-3.3-8b --audio … --triton` (encoder compile aborts).
**Surfaced**: Ch5.

### P-TRITON-IM2COL-KERNEL — P1
**Scope**: HAT OCAB unfold is not covered by the triton upscaler
path; HAT-S/L triton fails. Predates Ch7/Ch8.
**Site**: triton upscaler dispatch (no specific line — kernel gap).
**Repro**: `pytest 'tests/regression/test_all_models.py::test_model_runs[hat-l-x4::triton]'` → exit 1.
**Existing file**: `docs/follow-ups/p-triton-im2col-kernel.md`.
**Surfaced**: P-NEUROBRIX-UPSCALERS-V1 closure.

### P-TRITON-SORT-TRITON36 — P1
**Scope**: `samplers.py:142` and `dispatch.py:589` import a sort_op
helper from a stale FlagGems vendor that breaks on Triton 3.6.
**Site**: `src/neurobrix/triton/samplers.py:142`, `kernels/dispatch.py:589`.
**Repro**: any triton LLM run that exercises sort (TBD specific model).
**Surfaced**: Ch4 P-TRITON-MOE-DETERMINISM.

### P-FLAGGEMS-VENDORING-REFRESH — P2
**Scope**: Vendored FlagGems is from the stale FlagOpen org;
the live upstream is `flagos-ai/FlagGems` (with a Triton 3.3.0 pin).
**Site**: `vendors/flaggems/` (whole vendor tree).
**Repro**: `git log` on `vendors/flaggems/` vs `flagos-ai/FlagGems`.
**Surfaced**: Ch4 P-TRITON-MOE-DETERMINISM.

### P-INDEX-PUT-ADVANCED-GENERAL — P2
**Scope**: triton `aten::index_put` only supports simple integer
indices, not advanced indexing (boolean masks, broadcasting).
**Site**: `src/neurobrix/kernels/ops/index_put.py`.
**Repro**: any model with `tensor[mask] = value` in the graph.
**Surfaced**: P-CORRECTNESS-SILENT-FAILURES.

---

## Symbolic / build-side gaps

### P-SWIN-WINDOW-VIEW-SYMBOLIC-DIMS — P1
**Scope**: Swin window-partition `aten::view` shape arg is frozen
to build-time ints; `[1, 8, 8, 8, 8, 180]` for a model built at
64 px breaks at arbitrary runtime resolutions (BL-1 upscaler
arbitrary-size blocker).
**Site**: build-side symbolic shape-propagation rule for
`aten::view` (factor-partition heuristic; window-grid dims
`H//ws, W//ws` not kept symbolic). Resolution lives in the
private build subtree.
**Repro**: `nbx upscale --model swin2SR-classical-sr-x2-64 --input <1234x789.png>` →
`aten.view::2 (aten::view): shape '[1, 8, 8, 8, 8, 180]' is invalid…`.
**Surfaced**: Ch6 P-SYMBOLIC-DIMS-FAMILY-AWARE.

### P-CONTAINER-EMBED-ORPHAN-SCALARS — P2
**Scope**: build-side handling of orphan scalar constants
referenced from in-forward `mask[slice]=cnt` patterns.
**Existing file**: `docs/follow-ups/p-container-embed-orphan-scalars.md`.
**Surfaced**: P-NEUROBRIX-UPSCALERS-V1.

### P-BUILD-KOKORO-DYNAMIC-FRAMES — P1 (was P0; phoneme seq_len + predictor RESOLVED 2026-05-28)
**Resolved half**: the phoneme `seq_len` is now symbolic and the
text_encoder + predictor run **fully variable-length** in-graph (verified
`asr (1,512,360)→(1,512,3648)`, `F0_curve 720→7296`, scaling 10x with
phoneme count for 15 vs 152 phonemes). This removes the 23-phoneme input
truncation and runs the predictor (incl. its LSTM) in-graph instead of the
hand-rolled native handler. Landed via: the cuDNN-RNN→`aten::lstm`
reassembly with the full pack-ecosystem fold + the predictor broadcast-eq
length regulator (build-side); opt-in seq-length symbolization for the
component inputs the locked dim-naming convention couldn't name
(`x/m/texts/t_en`); and the runtime length-symbol promotion extended to the
wrapped-list `zeros/full` shape form + single-sequence symbol collapse.
**Remaining → [[P-KOKORO-DECODER-ISTFT-FLEX]]**: the istftnet decoder still
freezes its OUTPUT length. The signal path is dynamic at runtime (the 1D
resample, the framing view, and the overlap-add reconstruction all recompute
their length from the live input), but the **window-normalization branch**
broadcasts `window^2` to a baked frame count with NO live tensor to recompute
from (unlike the signal path, which derives its frame count from the inverse
FFT output). The decoder's frame counts live at four rate-scales
(asr=128, F0=256, samples=76800, windows=15361) so a single promotion symbol
can't capture them, and 128/256 collide with weight dims. This is the
documented VibeVoice class (STFT/causal-conv `math.ceil` length arithmetic
resists a symbolic length — the VibeVoice decoder is traced at concrete
length for the same reason). Until it lands, the registry keeps the Kokoro
predictor/text_encoder on the native handler (the predictor variable-length
capability is committed but not wired for Kokoro) so Kokoro runs via the
existing fixed-length path; marked debt, not a solution.
**Decision (2026-05-28, Hocine)**: predictor variable-length is the milestone
(it fixes the real "phoneme stretching" babbling — the native handler scaled
every utterance to a fixed ~128 frames). The decoder iSTFT frame-flexibility
is the named follow-on: a VibeVoice-pattern handler that decodes at the
runtime length, or threads the runtime frame count to the norm branch.
**Site**: build-side audio seq-length symbolization (done for the phoneme
axis); remaining runtime consumer is the istftnet window-norm reconstruction.
**Repro**: `neurobrix run --model Kokoro-82M --prompt "<long sentence >23 phonemes>"`
→ predictor now scales (asr frames track phonemes), decoder output still
fixed-length (band-aid path active until the follow-on).
**Surfaced**: P-AUDIO-P0a (2026-05-22); predictor half resolved 2026-05-28.

### P-SYMBOLIC-ITEM-TRACKING (was P-SYMBOLIC-ARANGE-SUM-FROM-ITEM) — P1, PARKED capability chantier
**Scope**: chatterbox vocoder (s3gen) crashes building its pad mask. The mask
sequence length is `arange(prompt_token_len + generated_token_len)`; the
`.item()` on that token-count sum severs the symbolic link, so the mask dim
freezes at the build-time value while the token embedding beside it stays
symbolic → mismatch at the first generated token whose total length differs
from the build-time one.
**Vendor source (s3gen flow.inference + utils/mask.py:186)**:
`token = concat([prompt_token, token])` has symbolic shape [1, s3+s1], but
`token_len = prompt_token_len + token_len` is a VALUE sum of the two `*_len`
INPUT tensors, then `make_pad_mask(token_len)` → `arange(token_len.max().item())`.
The arange end is value-derived, NOT shape-derived, so shape propagation cannot
symbolize it. Confirmed distinct from the orpheus arange (shape-derived →
symbolized → fixed by a promotion-branch change): chatterbox's end is a frozen
scalar 180 = prompt(157)+generated(23), a SUM of two seq symbols (s3+s1), and
the symbolic context has `expressions: {}` (no sum registered).
**Why a clean source fix is a new capability, not a bounded change**: the
build-side symbolic engine would have to (a) recognise that a `_len` input
tensor's VALUE corresponds to another tensor's SHAPE symbol, (b) propagate it
through the `+`, (c) carry it through `.item()` — i.e. value-symbolic tracking,
matrix-wide. The two shortcuts are both rejected on principle: re-enabling the
deprecated expression-value match (disabled after a spatial-dim false-match
incident) is value-coincidence bricolage; a runtime patch that re-derives the
mask length treats the symptom and fragilises the general mechanism.
**Decision (2026-05-24)**: PARK as a dedicated capability chantier, exactly like
[[P-CEIL-PAD-WINDOW]] (granite) and P-VIBEVOICE-NEXT-TOKEN-DIFFUSION-FLOW — a
missing build-side capability, not a finishable bug. chatterbox conditioning is
RESOLVED (audio-family section); only this vocoder mask-length symbolization
remains.
**Repro**: `neurobrix run --model chatterbox --prompt "Hello world."` →
`Failed at op aten.mul::0: size of tensor a (178) must match b (180) at dim 1`.

### P-CEIL-PAD-WINDOW (granite Q-Former windowing) — RESOLVED (2026-05-24)
**Resolution**: fixed in the runtime, NOT the build side as first framed. A
windowed-attention projector pads its sequence to a multiple of window W and
reshapes into `ceil(seq/W)` windows; the trace baked num_windows as
`floordiv(seq + trace_pad, W)` (floor of the trace pad). Two coupled changes in
`compiled_sequence.py` (commit `a66e135`): the pad-output expression
`add(input_sym, pad_total)` is rewritten to `mul(ceil(input_sym/W), W)`
downstream of the pad (trace-value preserving, BFS-scoped); and the cross-branch
injection runs to a fixpoint persisting dim-merge products so chained flattens
propagate the windowed count. R23 proven (TinyLlama / Swin2SR / Sana 1024).
Granite projector now runs end-to-end (`audio_embeds [1,42,4096]`). Resolution
recorded in CHANGELOG [Unreleased].

### P-GRANITE-CONFORMER-VARIABLE-FRAMES — RESOLVED (2026-05-26)
**Resolution**: granite is GROUNDED and frame-flexible (STT correct at 1/4/7
blocks). Both prior diagnoses were wrong — corrected by reading the vendor
source (`transformers/models/granite_speech/`):
- The `[200,200,128]` was NOT "frame²" — it is `[context_size, context_size,
  dim_head]`, a CONSTANT Shaw relative-position matrix per block
  (`context_size=200`). The variable dim is `num_blocks = ceil(seq/context_size)`
  (modeling l.155). Not a missing capability — the ceil-window pattern already
  tooled by a66e135.
- The 2× frame rate was NOT a "20 ms shift" — the vendor mel uses `hop_length=160`
  (10 ms) THEN stacks adjacent frame pairs (`reshape(-1, 2·n_mels)`) → 160-dim,
  half the frames. NeuroBrix used kaldi fbank + first-difference deltas (right
  dim, wrong values + no halving).
Three coupled fixes: (1) the conformer feature extractor reproduces the vendor
recipe — log10-normalised mel + adjacent-frame stacking, not kaldi fbank +
deltas (`2443f89`, §5.8 bit-match vs the vendor extractor); (2) the model is
re-traced so its graph exercises more than one attention block, leaving the
padded sequence length symbolic rather than equal to the fixed block size;
(3) `num_blocks` made symbolic `ceil(seq/W)` across the conformer block reshapes
(`5acab8f`, runtime extension of the a66e135 pad→view rewrite). Verdict +
multi-duration R29 artefacts: `validation_outputs/p_granite_grounded/`. Reusable
brick: `num_blocks=ceil(seq/W)` flexible windowing.

### P-VIBEVOICE-VAE-ARCH — P1 (VibeVoice TTS; generation flow done, VAE blocked)
**Status**: the `next_token_diffusion` generation flow is implemented
(`core/flow/next_token_diffusion.py`, held uncommitted) and STRUCTURALLY VERIFIED
— the LM control loop + DDPM diffusion-head sampling + connector feedback run
correctly (emits the expected speech-diffusion token stream → speech_end → eos,
natural stop). End-to-end is blocked on the acoustic/semantic VAE tokenizers.
**Blocker**: the acoustic/semantic VAE tokenizer components don't match the
model's actual weights — they were constructed for one VibeVoice tokenizer
variant, but the shipped weights are a different variant (different module
hierarchy + a depthwise stem). The graph expects `decoder.stem.conv [2048,64,7]`
(standard conv) while the weights are `decoder.stages.*.mixer.conv [2048,1,7]`
(depthwise); a non-strict load masked the mismatch, so the decoder fails at
`aten.convolution::0` (channel clash). Plus a structural issue: the decoder is
causal-streaming with a fixed-window (T=64) cache and the semantic encoder a
fixed 24000-sample window, neither of which the stateless runtime can replicate
per latent.
**Resolution (build-side, substantial)**: reconstruct the VAE tokenizer
components to match the model's actual tokenizer architecture and emit a FULL
non-streaming decode (variable latent count, no causal cache) + non-windowed
encode. If the decoder is genuinely stateful (cannot run non-streaming), that
piece is the structurally-non-traceable park. The generation flow is done and
waits on it.
**Repro**: `neurobrix run --model VibeVoice-1.5B --prompt "Hello, this is a test." --output out.wav` → LM loop + diffusion run; acoustic decode crashes at conv::0.
**R29**: `validation_outputs/p-vibevoice-next-token-diffusion-v1/`.
**Surfaced**: 2026-05-24.

---

## Prism / runtime allocator gaps

### P-PRISM-VIDEO-5D-UNPACK — P1
**Scope**: video allocator hardcodes a 4-tuple unpack on
shape; 5D `[B,C,T,H,W]` tensors fail with
`too many values to unpack (expected 4)`. Reproduces with
`NBX_DISABLE_AUTO_FP32=1` → not Ch8-introduced.
**Site**: `src/neurobrix/core/prism/solver.py` (or callee — exact
line TBD on next investigation).
**Repro**: `nbx run --model SANA-Video_2B_720p_diffusers --prompt "…" --output /tmp/x.mp4` (both modes) → exit 1.
**Surfaced**: Ch8 verdict harness triage.

### P-FLEX1-VAE-FP32-GATE — P2
**Scope**: Flex.1-alpha VAE does not satisfy the Ch8 conv-cascade
auto-fp32 gate (`conv2d ≥ 20 AND conv2d ≥ 10·sdpa`). Triton
path was already failing pre-Ch8; diagnose VAE structure vs
gate thresholds — may need a different family of conv-dominance
criteria, or the failure may be unrelated to fp16.
**Site**: `src/neurobrix/config/families/image.yml dtype_policy`
+ `src/neurobrix/core/prism/solver.py:_auto_fp32_components`.
**Repro**: `pytest 'test_model_runs[Flex.1-alpha::triton]'` → exit 1.
**Surfaced**: Ch8.

### P-TRITON-LIVE-WATERMARK-AUDIT — P1
**Scope**: triton allocator live-tensor watermark gap vs compiled
on Sana 4Kpx VAE — 26 GB live at conv::62 vs compiled completes
in 5 GB driver-free. Suspect: kill_slots laxness, deferred-free
queue retention.
**Site**: `src/neurobrix/triton/sequence.py` (kill_slots metadata).
**Repro**: Sana 4Kpx triton SIGTERM at >3h budget.
**Surfaced**: P-SANA-4KPX-RUNTIME.

---

## Volta / SDPA non-determinism residue

### P-TRITON-VOLTA-RESIDUAL-NONDETERMINISM — P2
**Scope**: V100+Triton has a known residual ~4/255 pixel
run-to-run non-determinism on PixArt-class diffusion
(matmul/softmax/RMSNorm ULP variation accumulating across 28
DiT blocks). Layer 7 fixed the headdim≠pow2 flash kernel
non-determinism; this is the residual from non-flash kernels.
**Site**: see Layer 6.bis CHANGELOG line 475 for the four
remaining hypotheses (CUDA stream, sub-µs Python frame, Triton
heuristics lambda, thread-local state).
**Repro**: PixArt-Sigma `--triton --steps 12` 5× — pixel diff
in `[-4, 4]` range across runs.
**Surfaced**: Layer 6.bis (commit `06d26c2` era).

### P-PIXART-XL-VOLTA-WHITE-BAND — P2
**Scope**: rare top-row white saturation on PixArt-XL `--triton`
at specific scheduler-step sequences. Manifestation of the
P-TRITON-VOLTA-RESIDUAL-NONDETERMINISM at certain seeds.
**Site**: not pinpointable without a deterministic repro (1/5
attempts produced it; 4/5 clean).
**Repro**: run PixArt-XL `--triton --steps 12 --prompt "<landscape>"`
many times, occasionally produces 70 rows of pure white at top.
**Surfaced**: Ch8 R29 + Dette D differential.

---

## Audio-family quality (full audio chantier scope)

### P-AUDIO-OPENAUDIO-CARRIER-TONE — P0
**Scope**: openaudio-s1-mini produces a quiet ~689 Hz carrier tone
(RMS 0.006, peak 0.01, ZCR 0.03) instead of speech.
**Root cause (2026-05-22, root-caused — NOT the codec kernel)**: the DualAR
flow runs only the **slow** backbone (semantic tokens) and feeds the model's
**text** `embed.weight` embeddings to `codec.decoder`, which expects acoustic
VQ feature frames `[1, 1024, T]`. The **fast/depth transformer** (`fast_block.*`,
`codebook_embeddings` — present in the `.nbx`) that generates the N acoustic
codebook tokens, and `codec.quantizer`, are **never run**. Codec decoder gets
text embeddings → carrier tone (embed dim 1024 coincidentally matches, so no
crash — silent semantic failure). Secondary: semantic loop never emits EOS
(2048-token max → 23.8 s every time).
**Confirmed cause = INCOMPLETE BUILD TRACE (not missing arch, not key-mapping)**:
the fast-transformer weights ARE present + correctly normalized, but in the
`model` graph (3116 ops) every fast weight has `consumers=0` — the build trace
captured only the slow backbone inference forward; the dual-AR generation
forward (which runs the fast transformer) was bypassed. `codebook_embeddings` IS
consumed 10× on the slow input side, proving the key normalization works.
**Site**: build-side trace stimulus for `dual_ar` model (slow-only forward).
**Fix = build-side re-trace** → [[P-BUILD-OPENAUDIO-DUALAR-TRACE]]. Runtime
hand-roll REJECTED (would re-implement vendored, traceable compute). Hocine
confirmed the build route.
**Repro**: `neurobrix run --model openaudio-s1-mini --prompt "Hello world." --audio test_speech_ref.wav`.
**R29**: `validation_outputs/p_audio_p0b_openaudio/` + diagnosis
`docs/audits/audio_p0b_openaudio_carrier_tone_diagnosis.md`.
**Surfaced**: Dette E; root-caused + build-localised by P0b (2026-05-22).

### P-BUILD-OPENAUDIO-DUALAR-TRACE — P0
**Scope**: re-trace the openaudio-s1-mini `model` component so the **fast/depth
transformer** forward is captured into the graph (currently only the slow
backbone inference forward is traced; fast weights are vendored but have zero
graph consumers → 689 Hz carrier tone, see [[P-AUDIO-OPENAUDIO-CARRIER-TONE]]).
The re-trace must expose the dual-AR generation path (slow hidden → fast
transformer → 10 codebook logits) so the runtime can drive the fast-AR loop and
feed real acoustic codes to the codec decoder.
**Site**: private build subtree (the `dual_ar` model trace stimulus currently
redirects to the slow-backbone forward because the generation forward is gated
"training-only" on this arch).
**Surfaced**: P0b (2026-05-22). The fast transformer is 4 standard transformer
layers — fully traceable, so this is the correct fix (not a runtime hand-roll).

### P-KOKORO-NATIVE-PORT-FIDELITY (was P-AUDIO-KOKORO-PHONEMES) — closed
Kokoro-82M babbling RESOLVED (2026-05-22): two native-port divergences vs the
vendor oracle — inverted `text_mask` convention (True=valid vs vendor
True=padding, so the model ran on padding) and a wrong text_encoder CNN block
(op order + LeakyReLU slope + missing per-block mask). "Hello world." now
transcribes 1.00. Resolution + §5.8 evidence:
`docs/verdicts/p_kokoro_native_port_fidelity/verdict.md`. Remaining long-prompt
shortfall is the 23-phoneme input truncation → [[P-BUILD-KOKORO-DYNAMIC-FRAMES]].

### P-CHATTERBOX-CONDITIONING-REARCH (was P-AUDIO-CHATTERBOX-LOOP) — RESOLVED (conditioning)
The over-generation (1052+ speech tokens → ~43 s garbage, whisper empty) was a
symptom of WRONG conditioning, not a stop-token bug. The tts_llm flow hand-rolled
T3CondEnc and produced `cond_emb [1,2,1024]` (speaker(zero)+emotion), **skipping
the Perceiver resampler**; the vendor produces `[1,34,1024]` (1 speaker + 32
perceiver + 1 emotion → 0.96 s "Hello world.").
**RESOLVED 2026-05-23**: conditioning is now produced by running the `cond_enc`
component (full speaker + Perceiver + emotion) embedded in the container, fed from
the embedded default-voice conditioning; the flow hand-roll is removed
(`core/flow/tts_llm.py`). cond_emb matches the vendor `[1,34,1024]`, and with
correct conditioning the speech LM stops over-generating (1052+ → ~24-48 tokens,
~1 s). The vocoder ref-dict is fed from the same embedded conditioning;
`--reference-audio` is wired through the CLI.
**Remaining**: coherent vocoder audio is blocked on a build-side sequence-length
symbolic limitation → [[P-SYMBOLIC-ARANGE-SUM-FROM-ITEM]]. The voice-clone path
(`--reference-audio` with a custom voice) is a further step — it needs the
reference tokenizer + speaker encoder captured, plus that same dynamic-length
symbolic capability.
**Repro (was)**: `neurobrix run --model chatterbox --prompt "Hello world."` → ~43 s garbage.
**Surfaced**: Dette E; diagnosed + conditioning fixed 2026-05-23.

### P-ORPHEUS-DECODE — RESOLVED (2026-05-24)
orpheus-3b-0.1-ft TTS is functional and STT-validated (whisper-large
transcribes "Hello World!", 1.88 s, 32.75 s runtime). The earlier
"multi-seq-symbol" hypothesis was WRONG — the graph has only s0/s1. Real chain
(all fixed): (1) lm_config emission; (2) two-arg `arange(0,seq_len)` decode
shift; (3) `expand` size promotion of the wrapped-list form (the RoPE position
expand kept its trace seq_len literal → broadcast the decode position to 23);
(4) SNAC 7-per-frame hierarchical de-interleaving (was contiguous → noise);
(5) speech-end EOS 128258 (was rambling to max_tokens). Verdict:
`validation_outputs/audio_family/orpheus/verdict.md`.

### P-VOXTRAL-HALLUCINATION — P2
**Scope**: Voxtral audio_llm answers conversationally instead
of transcribing (responds to the audio content as a chat
partner, not an STT system). IDENTICAL in compiled and triton
modes — model-side processor / system-prompt issue, not runtime.
**Site**: build-side multimodal-processor wiring for Voxtral
(equivalent to Qwen-VL processor system-prompt).
**Repro**: `nbx run --model Voxtral-Mini-3B-2507 --audio test_speech_ref.wav --prompt "transcribe the audio" --triton` → response is conversational text, not transcription.
**Surfaced**: Ch5 audio_llm port verdict (D10 already-known).

### P-TRITON-PERF-AUDIO-LLM — P2
**Scope**: triton LM decode is ~50× slower than compiled on
Voxtral (~12 min vs ~14 s). No prior triton baseline existed
for audio_llm (was unported/xfail'd pre-Ch5) so this is not a
regression. Performance optimisation target.
**Site**: triton autoregressive decode loop.
**Repro**: time `nbx run --model Voxtral-Mini-3B-2507 --triton` vs `--compiled`.
**Surfaced**: Ch5.

---

## Harness / CLI hygiene

### P-HARNESS-AUDIO-LLM-OUTPUT-DISPATCH — P2
**Scope**: harness `--output *.txt` is rejected for canary-qwen
/ granite-speech because their family=audio routes the strict-
extension check to .wav. The text transcription is produced and
printed; only the file dispatch is mismatched.
**Site**: `src/neurobrix/core/runtime/output_dispatch.py:resolve_output_path`
strict-extension check.
**Repro**: `nbx run --model canary-qwen-2.5b --output /tmp/x.txt …` → strict-mismatch error.
**Surfaced**: Ch5.

### P-HARNESS-STT-WAV-EXTENSION — P2
**Scope**: STT models (whisper-large, parakeet-tdt) write
malformed `.wav` files (small/truncated; not RIFF). Same root
cause as P-HARNESS-AUDIO-LLM-OUTPUT-DISPATCH — family=audio
dispatches to `.wav` even for transcription text outputs.
**Site**: same as above.
**Repro**: `pytest test_model_runs[whisper-large::native]` → wav 243 bytes, not RIFF.
**Surfaced**: Dette E.

### P-CLI-DEAD-REFERENCE-AUDIO-FLAG — P2
**Scope**: `--reference-audio` argparse flag in `nbx run` is
parsed but never consumed in `cmd_run`. Only `--audio` is
routed to the runtime. Either alias `--reference-audio` to
`--audio` for TTS-with-ref families, or remove the dead flag.
**Site**: `src/neurobrix/cli/__init__.py:125` (argparse add),
`src/neurobrix/cli/commands/run.py` (no use of args.reference_audio).
**Repro**: `nbx run --reference-audio test_speech_ref.wav … --triton` — flag accepted, audio not used.
**Surfaced**: Dette E.

### P-MULTIMODAL-TIMEOUT-MISSING-FROM-CONFTEST — closed
**Scope**: `FAMILY_TIMEOUT_S` lacked `multimodal` (and audio_llm,
stt, tts, vlm) — fell to the unknown-family default 300 s.
**Status**: **FIXED in Dette B** (added all 5 R26 sub-families
to `FAMILY_TIMEOUT_S` in `tests/regression/conftest.py`). Entry
kept here for trail; will be removed in next sweep.

### P-DOC-ADAPTER-PY-OBSOLETE — P3
**Scope**: `src/neurobrix/CLAUDE.md §5` references
`kernels/adapter.py` which no longer exists; the section
describes the adapter architecture using the obsolete path.
**Site**: `src/neurobrix/CLAUDE.md §5` "Adapter Architecture".
**Repro**: `ls src/neurobrix/kernels/adapter.py` → file not found.
**Surfaced**: P-VERDICTS-HYGIENE meta-audit.

---

## Model-side quality (not runtime)

### P-VOXTRAL-PROCESSOR-MULTIMODAL — P2
**Scope**: Voxtral processor wires audio differently from Qwen-VL
template; result is conversational reply to the audio question
rather than transcription. Build-side issue.
**Surfaced**: Ch5.

### P-CHATTERBOX-DECODING-CHARABIA — P2
**Scope**: Chatterbox occasionally produces unintelligible
output even when not looping (separate from the 82s loop bug).
Build-side / model-state issue.
**Surfaced**: Ch5 era.

### P-JANUS-COLOR-FIDELITY — P2
**Scope**: Janus-Pro-7B image generation has reduced color
fidelity vs vendor reference (autoregressive image tokens vs
diffusion VAE-decoded image).
**Site**: build-side vendor reference comparison.
**Surfaced**: earlier upscaler/multimodal audit.

---

## Resolved / SUPERSEDED tags

### P-SANA-4KPX-16GIB-POINT9 (SUPERSEDED)
The POINT 9 closure on Sana 4Kpx with 16 GiB budget was
SUPERSEDED by the P-SANA-4KPX-RUNTIME v2 closure (2026-05-13);
the current state is documented in
`docs/verdicts/p_sana_4kpx_runtime/`. Marked here so a future
reader does not chase the orphan POINT 9 reference.

### P-GQA-WRAPPER-LATENT — P2
**Scope**: GQA path in `kv_cache_wrapper.py:461` has a latent
bug under certain head-grouping configurations. Tracked but
not currently triggered by any cached model.
**Surfaced**: earlier triton chantier.

### Gap B — P-OP-LEVEL-CROSS-DEVICE-SPLIT — P2
**Scope**: op-level cross-device split (multi-GPU per-op
dispatch) needs Prism integration for Sana 4Kpx multi-GPU
fallback.
**Surfaced**: P-SANA-4KPX-RUNTIME residual.

### Layer X — Volta SDPA non-determinism root cause — P2
**Scope**: Layer 6.bis documented four remaining hypotheses for
the V100+Triton SDPA non-determinism. Layer X is the future
investigation that picks one.
**Existing file**: see CHANGELOG.md Layer 6.bis (commit `06d26c2`
era) for the four hypotheses.

### P-CEIL-PAD-WINDOW — P1
**Scope**: granite-speech-3.3-8b STT is non-functional — its
window-partition projector (Q-Former) computes the window count
as `ceil(seq/W)` with a dynamic pad-to-multiple, but a
trace-time-constant pad (7, the value at the trace sequence) is
baked, so the window count is wrong for any sequence length other
than the trace one.
**Site (runtime side)**: the pad-to-multiple windowing handling at
`src/neurobrix/core/runtime/graph/compiled_sequence.py:1532-1639`
does not fire for this projector and must be extended; the
shape-capability side is parked in a dedicated build chantier.
**Shared primitive** — affects every pad/reshape/window-partition
model (LLM masks, Swin upscalers, image); R23 byte-identical
re-build of all such models mandatory before fixing.
**Repro**: `neurobrix run --model granite-speech-3.3-8b --audio test_speech_ref.wav --prompt "Transcribe this audio."`
→ `Failed at op aten.bmm::2: Expected [208,64] got [224,64]`.
**Status**: PARKED — dedicated chantier, after the audio loop.

### P-VIBEVOICE-NEXT-TOKEN-DIFFUSION-FLOW — P1
**Scope**: VibeVoice-1.5B TTS needs a complete next-token-diffusion generation
flow (LatentLM-style), which does not exist yet. The model runs end-to-end at
runtime (OOM fixed, commit c0517a1) but emits SILENCE because the flow does a
single LLM `forward` + a single diffusion pass on a fixed `[1,64,64]` latent
instead of the autoregressive generation loop. This is a missing CAPABILITY (a
new flow), not a bug — same nature as P-CEIL-PAD-WINDOW.
**Required flow (per step, autoregressive)**:
  1. LLM forward with KV cache on the running context → hidden state;
  2. diffusion head (DDPM, ~10-20 steps, conditioned on that hidden) → next
     acoustic VAE latent;
  3. the acoustic connector (`SpeechConnector`) reprojects the latent to the LLM
     hidden dim and it is appended as the next LLM input;
  4. repeat until the stop condition; then the acoustic tokenizer (VAE decoder,
     7.5 Hz frame rate) decodes the accumulated latent sequence → waveform.
**Prerequisite (BLOCKER)**: the local vendor only ships the VibeVoice COMPONENTS
(acoustic tokenizer, asr), NOT the full model with the generation loop (it is an
unmerged transformers PR #40546; the model is assembled component-by-component).
The complete vendor `modeling_vibevoice` streaming-inference loop (stop condition
+ speech-start protocol) must be retrieved (microsoft/VibeVoice source repo or
the PR) as an executable oracle BEFORE building — op-by-op validation against an
executable oracle is what made openaudio's dual_ar build converge; without it we
would diverge blind.
**Scope reference**: comparable to the dual_ar flow built for openaudio.
**Site (runtime side)**: `core/flow/` (new flow handler) +
`core/flow/stages/vibevoice.py`. **Repro**:
`neurobrix run --model VibeVoice-1.5B --prompt "Hello world."` → silent .wav
(RMS ~2e-5). **Status**: PARKED — dedicated chantier, after chatterbox + orpheus.

---

## Per-chantier follow-up archive

When a follow-up is resolved, the entry moves to the relevant
verdict in `docs/verdicts/<chantier>/`. Historical archive files
under `docs/follow-ups/archive/` preserve the original
investigation context (e.g., `kokoro_cudnn_batch_norm_regression.md`
for the Ch3-era Kokoro crash, now closed).
