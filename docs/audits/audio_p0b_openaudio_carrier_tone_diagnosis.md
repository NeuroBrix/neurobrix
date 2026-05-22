# Audio P0b — openaudio-s1-mini carrier-tone diagnosis (2026-05-22)

Root-caused the Dette E "tonalité de mire / 689 Hz carrier tone" symptom on
`openaudio-s1-mini`. Unlike Kokoro P0a (a port-fidelity divergence), this is a
**missing-architecture** bug: the second autoregressive stage of the DualAR
model is never executed.

## Section 1 — Empirical signature (reproduced 2026-05-22)

`neurobrix run --model openaudio-s1-mini --prompt "Hello world." --audio test_speech_ref.wav`:

```
[model] Generated 2048 semantic tokens in 154712ms   (hit max_tokens, no EOS)
[model] Token range: 0..151658, embed vocab: 155776  (TEXT vocab)
[model] Embedded → [1, 1024, 2048] for codec.decoder
[codec.decoder] Waveform: [1, 1, 1048576]            (= 23.78 s @ 44.1 kHz)
```

Audio stats: 23.78 s, RMS 0.0057, peak 0.010, ZCR 0.030 — near-silent
near-pure sinusoid. Matches the Dette E `openaudio_carrier_tone.wav` signature.
R29: `validation_outputs/p_audio_p0b_openaudio/`.

## Section 2 — Architecture vs implementation

`openaudio-s1-mini` is a Fish-Speech-style **DualAR** model:

- **Slow backbone** (`model`): processes the `[B, N+1, T]` token grid, predicts
  the next *semantic* token + a hidden state per position.
- **Fast / depth transformer** (`fast_block.0..N`, `codebook_embeddings.weight`
  — present in the `.nbx`, 32 weight keys): per position, autoregressively
  generates the **N acoustic codebook tokens** from the slow hidden state.
- **codec.quantizer**: maps codebook indices → acoustic feature embeddings.
- **codec.decoder**: expects `x = [1, 1024, T]` acoustic feature frames →
  waveform.

`core/flow/dual_ar.py` implements only the slow backbone:

1. Generates *semantic* tokens (`generated_semantic`, `dual_ar.py:106-149`).
2. Embeds them through the model's **text** `embed.weight`
   (`dual_ar.py:154-180`).
3. Feeds the text embeddings `[1, 1024, T]` directly to `codec.decoder`.

The `fast_block` transformer and `codec.quantizer` are **never run**. The codec
decoder receives text-token embeddings where it expects acoustic VQ features →
it decodes meaningless input into a near-constant carrier tone. The embed dim
(1024) coincidentally matches the decoder's expected channel count, so the
shape is valid and nothing crashes — a silent semantic failure.

Secondary: the semantic loop never emits EOS (runs the full 2048-token budget),
so the output is always max length (23.8 s). Whether the semantic stream itself
is also degenerate is moot until the fast-AR path exists.

## Section 3 — Fix scope (NOT a port-fidelity diff)

The fix is to **implement the missing fast-AR**: for each slow-backbone step,
run the `fast_block` transformer to generate the N codebook tokens, map them
through `codec.quantizer` (`codebook_embeddings`) to `[1, 1024, T]` acoustic
features, and feed those to `codec.decoder`. Two routes:

- **Route 1 — hand-rolled native fast-AR** in `core/flow/dual_ar.py` using the
  extracted `fast_block` weights (torch native, same documented R33-violation
  pattern as the Kokoro predictor stage). Self-contained, no build dependency,
  but substantial (a full second AR loop + KV cache + codebook sampling) and
  needs an R16 study of Fish-Speech's exact fast-AR conditioning/ordering.
- **Route 2 — build-side re-trace** capturing the full DualAR (slow + fast) so
  the runtime drives both heads in-graph. Cleaner long-term, build-subtree work.

This is an architectural decision (escalated to Hocine), not a divergence to
diff. Tracked: `P-AUDIO-OPENAUDIO-CARRIER-TONE` (follow-ups INDEX).

## Section 4 — Method note

R16 reference: Fish-Speech / OpenAudio inference (slow + fast dual transformer,
firefly/codec decoder). The `.nbx` contains the fast-transformer weights, so the
runtime, not the build, is missing the second AR loop — the same class of gap as
a missing flow handler, scoped larger.
