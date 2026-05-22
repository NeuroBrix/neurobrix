# Audio P0b — openaudio-s1-mini carrier-tone diagnosis (2026-05-22)

Root-caused the Dette E "tonalité de mire / 689 Hz carrier tone" symptom on
`openaudio-s1-mini`. The cause is an **incomplete build-side trace**: the second
("fast") autoregressive transformer of the DualAR model is present in the `.nbx`
but was never captured into the execution graph, so the runtime cannot run it.

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

## Section 2 — Architecture vs what the runtime executes

`openaudio-s1-mini` is a Fish-Speech-style **DualAR** model:

- **Slow backbone** (`model`): processes the `[B, N+1, T]` token grid, predicts
  the next *semantic* token + a hidden state per position.
- **Fast / depth transformer** (`fast_block.0..3`, `fast_output`, `fast_norm`,
  `fast_embeddings`, `fast_freqs_cis`, `codebook_embeddings`): per position,
  autoregressively generates the **N=10 acoustic codebook tokens** from the slow
  hidden state.
- **codec.quantizer / codec.decoder**: `codec.decoder` expects `x = [1, 1024, T]`
  acoustic feature frames → waveform.

`core/flow/dual_ar.py` runs only the slow backbone: it generates *semantic*
tokens (`dual_ar.py:106-149`), embeds them through the model's **text**
`embed.weight` (`dual_ar.py:154-180`), and feeds those text embeddings
`[1, 1024, T]` straight to `codec.decoder`. The codec decoder receives text-token
embeddings where it expects acoustic VQ features → it decodes meaningless input
into a near-constant carrier tone. The embed dim (1024) coincidentally matches
the decoder's expected channel count, so the shape is valid and nothing crashes —
a silent semantic failure.

## Section 3 — Root cause: incomplete trace (NOT missing architecture, NOT a key-mapping break)

The fast-transformer weights ARE in the `.nbx` `model` component and are
correctly normalized to the standard weight-key scheme (`attn.wqkv`,
`ffn.gate/up/down`, `attention_norm`, `ffn_norm`, …). The key normalization is
not the problem — proof: `codebook_embeddings.weight` is consumed by **10** graph
ops (it embeds the 10 codebook rows of the input grid on the slow path).

But in the `model` component `graph.json` (3116 ops):

- the only graph output is `logits [1, 23, 155776]` (the slow semantic head);
- **every** fast-transformer weight (`fast_block.0..3.*`, `fast_output.weight`,
  `fast_norm.weight`, `fast_embeddings.weight`, `fast_freqs_cis`) has
  `consumer_op_uids = []` — **zero ops consume them**.

So the weights were vendored into the container but the fast forward was never
recorded into the graph. The build-side trace exercised only the slow backbone
inference forward (the dual-AR generation forward is gated as "training-only" on
this architecture and was deliberately bypassed during the trace). The
per-codebook fast loop simply never ran while the graph was being captured.

This is a build-side **incomplete-trace** condition, distinct from:
- *missing architecture* (the weights ARE present), and
- *key-mapping break* (the normalization works — `codebook_embeddings` is
  consumed; there are no orphan ops referencing fast weights under a wrong name).

## Section 4 — Fix routing (build-side, escalated)

The fast transformer is 4 standard transformer layers — fully traceable (unlike
Kokoro's `pack_padded_sequence` LSTM, which structurally cannot trace and is
therefore legitimately hand-rolled). So the correct fix is **build-side**:
re-trace the `model` component so the dual-AR generation forward (slow hidden →
fast transformer → 10 codebook logits) is captured into the graph, exposing the
fast path as graph outputs the runtime can drive. Tracked as
`P-BUILD-OPENAUDIO-DUALAR-TRACE`.

A runtime hand-rolled fast-AR in `core/flow/dual_ar.py` is **rejected as the
default**: it would re-implement compute that is already vendored, and would only
be justified if the trace structurally could not capture the fast path (it can).
The architectural build-vs-runtime fork was escalated to Hocine, who confirmed
the build-side route.

Follow-up: `P-AUDIO-OPENAUDIO-CARRIER-TONE` (root cause recorded here);
`P-BUILD-OPENAUDIO-DUALAR-TRACE` (the build-side fix).

## Section 5 — R16: the generation+decode path the re-trace must capture

Vendor reference (Fish-Speech `DualARTransformer`, public). The inference
generation path that the build trace must record (it currently records only the
slow training-style backbone forward):

1. **Slow step** — `forward_generate(x)` runs the slow backbone and returns the
   semantic `logits` AND the per-position `hidden_states` (projected to the fast
   dim). The current trace exposes only `logits [1,23,155776]`; it must ALSO
   expose `hidden_states`.
2. **Sample** the semantic token from the slow logits (stop on the end token —
   note the runtime currently never emits EOS, the secondary 23.8 s symptom).
3. **Fast loop** — reset the fast KV cache, then for `codebook_idx` in
   `0..N-1` (N=10, from `codebook_embeddings [40960,1024] = 10×4096`):
   `forward_generate_fast(hidden, input_pos=codebook_idx)` → codebook logits
   `[1,1,4096]`; sample token; `hidden = fast_embeddings(token)` feeds the next
   codebook step. The fast transformer is 4 standard layers (RMSNorm + attention
   + FFN) — fully traceable.
4. **Codes → waveform** — the stacked codes `[N, T]` must be dequantized to the
   `[1,1024,T]` acoustic features the `codec.decoder` consumes.

Trace-gap inventory the build re-trace must close:
- **Gap 1 (confirmed)**: the fast transformer forward (`forward_generate_fast`)
  — zero graph consumers today.
- **Gap 2 (confirmed)**: `forward_generate` must expose `hidden_states` as a
  graph output (only `logits` is exposed today).
- **Gap 3 (CONFIRMED)**: the codec codes→features dequantize is un-traced. The
  codec is a Residual-VQ (`quantizer.quantizers.0-8.codebook.weight`,
  `semantic_quantizer`); vendor `DAC.decode(indices)` dequantizes codes→z
  (RVQ codebook lookup + sum) then runs the decoder. The traced `codec.quantizer`
  is encode-only (`z→codes`), and `codec.decoder` takes the already-dequantized
  `z=[1,1024,T]`. The `codes→z` dequantize step has no traced path → the re-trace
  must capture it.

**Design simplification (no KV-cache tracing needed)**: the slow backbone is
already run in NeuroBrix as a *full re-forward per step* over the trace window
(no KV cache — `dual_ar.py` slides the window and re-runs). The fast transformer
can use the same pattern: its training-forward already processes the
`[hidden, cb0_emb, …, cb(N-2)_emb]` sequence under a causal mask, so the runtime
can re-forward the growing codebook sequence and take the last position per
codebook step — mirroring the slow path, avoiding KV-cache replay. This makes
the re-trace target a plain full forward (slow→hidden+logits, fast→codebook
logits, codec dequantize), not an AR-with-cache capture.

Runtime consequence (after the re-trace): `core/flow/dual_ar.py` drives the
nested AR loop — slow step → fast loop over N codebooks → dequantize → codec
decode — replacing the current "text-embed → codec.decoder" shortcut. This is a
build + runtime co-design, scoped under `P-BUILD-OPENAUDIO-DUALAR-TRACE`.
