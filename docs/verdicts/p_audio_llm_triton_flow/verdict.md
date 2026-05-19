# P-AUDIO-LLM-TRITON-FLOW — verdict (2026-05-19)

Branch `p-audio-llm-triton-flow` from `edc8a27` (Ch4bis HEAD).

## Section 1 — Goal & debt status

`triton/flow/audio_llm.py` was missing; the audio_llm flow dispatch
returned the torch-coupled `core/flow/audio_llm.py AudioLLMEngine`
for ALL modes (the only flow with no triton branch in
`executor.py`). Voxtral / canary-qwen / granite-speech were xfail'd
in `--triton` for this reason. Pre-prompt-reading confirmed the debt
**LIVE** (Ch4/Ch4bis touched kernels, not flow handlers — not a
P-KOKORO-style dead-debt hygiene pivot).

## Section 2 — Implementation (R33-pure port)

New `src/neurobrix/triton/flow/audio_llm.py` =
`TritonAudioLLMEngine`, a faithful NBXTensor mirror of
`AudioLLMEngine`. All compute is NBXTensor + kernel wrappers
(`embedding`, `NBXTensor.cat`, `matmul_wrapper`, `argmax_wrapper`,
`softmax`, `multinomial_wrapper`, `.to(NBXDtype)`); position_ids via
`np.arange → NBXTensor.from_numpy`. Zero torch on the compute path.
Two sanctioned BOUNDARY helpers imported from
`core.flow.audio_utils` (`preprocess_audio_input`,
`postprocess_text_output`) — identical pattern to the existing
triton handlers `encoder_decoder.py`/`dual_ar.py` (audio I/O
torch-boundary, output bound as NBXTensor). NBXTensor→host reads use
an R33-pure D2H copy (`_nbx_f32_to_numpy`, mirror of
`autoregressive.py::_read_ids_to_list`), not `nbx_to_torch`.
Compute-dtype is string→NBXDtype (no torch.dtype), unlike the core
handler's `get_compute_dtype`.

`core/runtime/executor.py` audio_llm branch: added the
`if ctx.mode in ("triton","triton_sequential"):
TritonAudioLLMEngine(...)` dispatch, mirroring audio/rnnt/
encoder_decoder/dual_ar.

## Section 3 — R30 symmetric audit

Method parity core↔triton confirmed (7/7): `execute`,
`_get_component_output`, `_get_last_forward_output`,
`_get_embed_weight`, `_compute_logits`, `_store_output`,
`_reshape_output_for_connections`. R33 grep audit: zero `import
torch`/`torch.`/`F.` on the compute path (the only "torch" token is
a docstring referencing `F.embedding` as the semantic equivalent).
Dispatch consistency: `_create_flow_handler` returns the triton
handler directly for all these flows (FLOW_REGISTRY bypassed by
design for every flow, unchanged); core `register_flow("audio_llm")`
still registers `AudioLLMEngine` for the non-triton path.

## Section 4 — Validation (compiled oracle ↔ triton)

`test_speech_ref.wav`, prompt `"transcribe the audio"`, greedy
`temperature=0`.

- **Voxtral-Mini-3B-2507 — PASS**: compiled and triton produced the
  **string-identical** 112-byte transcription
  (`I'm sorry, I didn't quite catch that. Could you please repeat or
  clarify what you meant by "more than 20 times"?`), `cmp` match,
  both `errs=0`. The R33-pure port is proven faithful to the
  compiled oracle end-to-end.
- **canary-qwen-2.5b — TRITON BLOCKED (not the port)**: compiled ran
  the flow (output `<think>`); triton reaches the new handler then
  crashes in the **pre-existing triton kernel-op layer**:
  `aten::repeat` → `'NBXTensor' object has no attribute 'repeat'`
  (`kernels/dispatch.py:161 _meta_repeat`), in the encoder stage.
- **granite-speech-3.3-8b — TRITON BLOCKED (not the port)**:
  compiled ran the flow (output `erer `); triton encoder-graph
  compile aborts on `[triton] Missing op: aten::_safe_softmax`
  (`triton/sequence.py:1526 _compile_op`).

R29 artefacts:
`validation_outputs/p_audio_llm_triton_flow/<model>/...` (+ INDEX).

## Section 5 — xfail disposition

`tests/regression/conftest.py`:
- **Voxtral-Mini-3B-2507**: xfail **removed** (passes
  compiled↔triton byte-identical).
- **canary-qwen-2.5b** / **granite-speech-3.3-8b**: xfail **kept**,
  reason updated — the flow port is done and reached; the residual
  blockers are pre-existing triton kernel-op gaps, not the R33 flow
  port (strict Ch5 scope: document, do not fix kernel ops).

## Section 6 — Latent observations / named follow-ups

- **P-TRITON-NBXTENSOR-REPEAT-MISSING**: `kernels/dispatch.py:161`
  `_meta_repeat` calls `.repeat()` on an `NBXTensor` (no such
  method). Blocks canary-qwen-2.5b::triton encoder.
- **P-TRITON-SAFE-SOFTMAX-MISSING**: `aten::_safe_softmax` has no
  triton op/handler (`triton/sequence.py:1526`). Blocks
  granite-speech-3.3-8b::triton encoder compile (+ prior CFormer
  projector native-stage note).
- D10 already-known (out of Ch5 scope, do-not-fix): Voxtral answers
  conversationally instead of transcribing — IDENTICAL in compiled
  and triton, so the port is faithful; this is the Voxtral
  multimodal-processor / hallucination latent bug. canary-qwen
  `<think>` and granite `erer ` compiled outputs likewise point at
  the same family of model-side processor bugs (D10).
- Harness note: canary-qwen / granite-speech classify under family
  `audio` for output dispatch → `--output *.txt` is rejected
  (expects `.wav`); transcription still produced and captured from
  the run log. Not a code defect; flag for the output-dispatch
  family review (Ch10).
- Triton LM decode is ~50× slower than compiled on Voxtral
  (~12 min vs ~14 s). No prior triton baseline existed (audio_llm
  was unported/xfail'd) so this is not a Ch5 regression; perf is a
  separate concern.

## Section 7

Commits: see closure report. Hocine validation: TODO (Voxtral text
artefacts present for sign-off).
