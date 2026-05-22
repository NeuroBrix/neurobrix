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

### P-BUILD-KOKORO-DYNAMIC-FRAMES — P0
**Scope**: re-trace the Kokoro decoder with a symbolic `asr_frames`
dim AND a symbolic phoneme `seq_len` (same paradigm as Ch6 symbolic
dims / LLM seq_len applied to audio: `asr_frames` should be symbolic
like `n_codec_frames`, not a build-time-frozen 128). Removes three
runtime band-aids: the fixed 128-frame decoder window, the 23-phoneme
input truncation (vendor handles 48), and the P0a post-decode crop.
The hand-rolled native predictor in `core/flow/stages/kokoro.py` exists
only because the LSTM/pack_padded path can't trace; a clean re-trace
with dynamic frames would let the predictor run in-graph, remove the
native handler entirely, and thereby also fix Kokoro `--triton` (the
native handler currently can't consume NBXTensor weights:
`embedding(): weight must be Tensor, not NBXTensor`). The compute
fidelity itself is already fixed in compiled mode
([[P-KOKORO-NATIVE-PORT-FIDELITY]] closed); this lifts the remaining
fixed-shape and triton limits.
**Site**: private build subtree (audio symbolic-dim conventions);
runtime consumer is `core/flow/stages/kokoro.py:_scale_kokoro_durations`
+ `_get_kokoro_decoder_shapes`.
**Repro**: `neurobrix run --model Kokoro-82M --prompt "The quick brown fox jumps over the lazy dog."` → only first ~23 phonemes survive.
**Surfaced**: P-AUDIO-P0a (2026-05-22). Option B in the P0a diagnosis doc.

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
**Scope**: openaudio-s1-mini codec decode produces a quiet ~689 Hz
tone instead of speech (peak amplitude 1%, ZCR 0.029).
**Site**: codec.encoder → codec.quantizer → codec.decoder
pipeline; suspected codec.decoder mel→waveform vocoder kernel.
**Repro**: `nbx run --model openaudio-s1-mini --prompt "Hello world" --audio test_speech_ref.wav --output /tmp/x.wav` → listen.
**R29**: `validation_outputs/p_dette_e_tts_audio_diagnosis/openaudio_carrier_tone.wav`.
**Surfaced**: Dette E.

### P-KOKORO-NATIVE-PORT-FIDELITY (was P-AUDIO-KOKORO-PHONEMES) — closed
Kokoro-82M babbling RESOLVED (2026-05-22): two native-port divergences vs the
vendor oracle — inverted `text_mask` convention (True=valid vs vendor
True=padding, so the model ran on padding) and a wrong text_encoder CNN block
(op order + LeakyReLU slope + missing per-block mask). "Hello world." now
transcribes 1.00. Resolution + §5.8 evidence:
`docs/verdicts/p_kokoro_native_port_fidelity/verdict.md`. Remaining long-prompt
shortfall is the 23-phoneme input truncation → [[P-BUILD-KOKORO-DYNAMIC-FRAMES]].

### P-AUDIO-CHATTERBOX-LOOP — P1
**Scope**: chatterbox produces 82 s of saturated audio for short
prompts (expected ~1 s); peak hits full scale (0.99). Length
unconstrained, possibly looping.
**Site**: `src/neurobrix/core/flow/tts_llm.py` stop-token /
max-length logic; decoder length cap.
**Repro**: `nbx run --model chatterbox --prompt "Hello world" --audio test_speech_ref.wav --output /tmp/x.wav` → wav 82 s.
**R29**: `validation_outputs/p_dette_e_tts_audio_diagnosis/chatterbox_82s_loop.wav`.
**Surfaced**: Dette E.

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

---

## Per-chantier follow-up archive

When a follow-up is resolved, the entry moves to the relevant
verdict in `docs/verdicts/<chantier>/`. Historical archive files
under `docs/follow-ups/archive/` preserve the original
investigation context (e.g., `kokoro_cudnn_batch_norm_regression.md`
for the Ch3-era Kokoro crash, now closed).
