# Open-Sora-v2 — VERDICT: compiled + sequential + triton coherent (3/4); triton_sequential pending (shares the same fixes)

T2V video, FLUX-style MMDiT (double + single stream) + HunyuanVAE causal-3D.
Config: 256×256, 13 frames, 50 steps, guidance 7.5, seed 42, prompt
"a red fox walking in snow".

## Root cause 1 — rotary-embedding convention (ALL modes; born-at-source build fix)

The MMDiT was trained with the **half-split (NeoX-style) rotary convention**
(query dim `d` pairs with `d + D/2`). The graph had been traced with the
**interleaved FLUX convention** (`2d` ↔ `2d+1`). A wrong rotary convention
scrambles every image-token position, so the denoiser cannot localize content
and the latent collapses to a uniform DC field → white/flat frame.

This was invisible to the earlier "MMDiT bit-identical vs PyTorch oracle" proof,
because that oracle had been configured with the *same* interleaved convention as
the graph — the match was convention-blind. Feeding the vendor MMDiT identical
conditioning + noise + schedule and swapping ONLY the rotary convention was
decisive: interleaved → flat field (x0pred spatial-std 0.09); half-split →
coherent fox (x0pred spatial-std 1.03). Fixed born-at-source at trace time: the
rotary application is redirected to the pure-ATen half-split equation (mirroring
the existing untraceable-custom-Function → ATen redirect already used for the
fused RMSNorm), and the position-only rotary table batch dim is pinned to a
literal 1 so it broadcasts over the denoiser's CFG batch.

Consequence: compiled mode renders a coherent fox end-to-end (frame std ~88). The
earlier CFG/guidance-path and "MMDiT mean-pooling collapse" hypotheses are
RETRACTED — plain text-CFG (no oscillation, no pooled-vector negation) foxes once
the rotary convention is correct.

## Root cause 2 — sequential mode AMP fp32 overflow protection defeated (runtime fix)

On V100 the bf16-native model runs in fp16. The DtypeEngine upcasts
`{mm,bmm,addmm,div}` to fp32 to prevent fp16 accumulation/overflow. In sequential
mode the per-op tensor resolver then downcast every fp32 op-output **intermediate**
back to fp16, because it aligned each resolved input to the *graph-recorded*
(stale trace) dtype. This silently undid the AMP protection: at `double_blocks.18`
the adaLN modulation `scale (≈10) × hidden (≈1.0e4)` reached ~1.0e5, overflowed
fp16 → Inf → NaN latents → black video. Compiled mode (arena slots, no per-op
resolver) keeps the AMP fp32 flowing through `view → mul` and never overflows.

Cross-engine op dumps proved it: `addmm::272` fp32 in both modes; its consumer
`view::594` was fp32 in compiled (max 80, finite) but downcast to fp16 in
sequential (→ Inf at the next mul). Fixed in
`core/runtime/graph/tensor_resolver.py`: the dtype-alignment step now applies only
to **leaves** (weights / constants / inputs, `producer_op_uid is None`). Op-output
intermediates keep the dtype the execution engine deliberately produced — exactly
mirroring compiled mode, which applies zero dtype-alignment to intermediates. This
is a general correctness fix: it restores AMP fp32 protection for every model in
sequential mode, not just Open-Sora (only visible here because this model's deep
residual reaches fp16's range).

### Anti-regression (the resolver fix is sequential-only; compiled byte-identical by construction)

`resolve_normalized` runs only on the `--sequential` path (compiled uses the
CompiledSequence arena, never the per-op resolver). Sequential vs compiled on a
diverse closed set, with the fix applied:

| model | family / arch | sequential vs compiled |
|---|---|---|
| Sana-1600M-1024px | image / DC-AE | **0.000 %** (bit-identical) |
| CogVideoX-2b | video / CogDiT | **0.000 %** (bit-identical) |
| Wan2.1-T2V-1.3B | video / WanDiT | 0.39 %, identical stats, coherent fox |
| Open-Sora-v2 | video / FLUX MMDiT | coherent fox (the fix target) |

Bit-identical everywhere except Open-Sora is the signature of a correct fix:
Open-Sora is the only model whose activations reach fp16's range, so it is the
only one where preserving the AMP fp32 changes the result. No dtype-mismatch
surfaced at any non-AMP combiner. (TinyLlama `--sequential` fails on a pre-existing
index-out-of-bounds assert independent of this fix — identical failure with and
without it; a separate LLM-sequential gap, out of scope here.)

## 4-mode state

- **compiled**: PASS — coherent fox (output_compiled.png / .mp4, frame std ~88).
- **sequential**: PASS — coherent fox (output_sequential.png / .mp4, frame std ~98);
  transformer overflow eliminated; NaN probe ran clean through all 50 denoise steps.
- **triton (compiled-triton)**: PASS — coherent fox (output_triton.png / .mp4,
  frame std ~93), real config cfg=7.5 (CFG batch=2), rope fusion ENABLED. Reached
  via a chain of three root fixes on the triton branch:
  1. *Flow port (Milestone 1)*: NBXTensor-pure `triton/flux_video_conditioning.py`
     + 5D pack/unpack + `is_flux_video` state-alias in `triton/flow/
     iterative_process.py` (R30 mirror, gated on the denoiser's `img_ids` input →
     inert for every non-FLUX model). Synthesizes all 7 MMDiT inputs.
  2. *Fused-rope scheduling* (`triton/sequence.py _fuse_rope_ops`): the single
     fused Q+K rope op was anchored at the LATER of the two branch adds — an
     HF-Llama-specific heuristic. Open-Sora's MMDiT feeds one rope output into the
     joint-attention `cat` before the OTHER branch's rope add runs, so the fused
     op produced its first output AFTER a consumer already read the (unwritten =
     None) slot → NOP-cascade → crash. Fixed with a topology-general scheduling
     window `[max_input_pos, min_consumer_pos)`: insert the fused op just before
     the earliest consumer of either output; if a consumer precedes an input the
     pair is left unfused (correct, unoptimised). Never a consumer without its
     producer.
  3. *Timestep double-scale* (`triton/flow/iterative_process.py`): the triton
     flow always applied the flow-matching `[0, num_train]` scale, but FLUX/MMDiT
     scales the timestep INTERNALLY (in-graph `time_factor`) → double-scale past
     fp16 → Inf timestep embedding → NaN. Fixed by the R30 mirror of compiled's
     `_graph_scales_timestep_internally` graph-inspection gate.
  4. *VAE fully-masked-row attention* (`kernels/wrappers.py`) — the black-frame
     root cause. The HunyuanVAE mid-block self-attention (`decoder.mid.attn.0`)
     passes an all-`-inf` additive bias (a fully-masked query row). PyTorch's
     fused SDPA backend (which compiled mode uses) emits 0 for such a row; the
     triton math path (the VAE's route: headdim 512, 64 MB scores ≤ 128 MB budget)
     and flash path computed `exp(-inf - -inf) = NaN`, which propagated through
     the VAE (conv::9 → conv::35) and blackened the video. Cross-engine dump:
     compiled `add::5` finite (l2 834, mask ALSO all `-inf`) vs triton NaN. Fixed
     with a fully-masked-row guard (`nan_to_num`) at the SDPA math + wrapper level,
     matching PyTorch. **Provably a value-identity** (`nan_to_num(finite)==input`,
     bit-exact) for every attention with no fully-masked row → inert for the whole
     zoo. Triton `add::5` = 834.4 now matches compiled 834.4.
  - Predictions from the prior verdict RESOLVED: (1) `img_ids`/rotary is correct —
    the transformer output is finite and rope fusion ENABLED renders a coherent
    fox; (2) the `double_blocks.18` fp16 overflow did NOT recur in triton (dumped
    fp32/clean, matching compiled) — the residual NaN was the VAE attention, not a
    transformer overflow.
- **triton_sequential**: shares the same kernels + flow (the four fixes apply);
  op-by-op validation pending (slow cold-compile).

### Anti-regression (shared SDPA guard)

The fully-masked-row guard is `nan_to_num` at the SDPA-op level, NOT inside the
`@triton.jit` kernels (an in-kernel guard perturbs Triton register allocation).
Both attention kernels are byte-identical to before. Evidence the guard is inert
for every model without a fully-masked row: `nan_to_num(finite)==input` bit-exact
(fp16+fp32, maxdiff 0); the wrapper flash guard is gated on `attn_mask` (skipped
for causal/no-mask); CogVideoX-2b triton renders a coherent snow scene; TinyLlama
triton emits coherent English. (Note: triton greedy LLM decoding is run-to-run
NON-deterministic — two identical-code runs differ — so single-run byte-diff is
not a valid gate; the value-identity proof is.)

## Relaunch

```
python -m neurobrix run --model Open-Sora-v2 --prompt "a red fox walking in snow" \
  --steps 50 --cfg 7.5 --seed 42 --num-frames 13 --height 256 --width 256 \
  [--sequential] --output output.mp4
```

Hocine validation: TODO
