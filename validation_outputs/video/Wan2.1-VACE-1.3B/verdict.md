# Wan2.1-VACE-1.3B — VERDICT: compiled + sequential CLOSED; triton OPEN (root = vace control stream; text-context length REFUTED as cause)

**Date:** 2026-06-28 (supersedes the 2026-06-16 bring-up note and the earlier
2026-06-28 "text-context length" localization, now refuted) · **Family:** video
control · **Arch:** WanVACETransformer3DModel (30 blocks, 15 VACE control layers)
+ UMT5 + AutoencoderKLWan.

## CLOSED
- **compiled, CFG batch=2 (default guidance), f9 480×832**: coherent red fox in
  snow (the Wan batch=1-trace-absorption fix landed; batch symbol s0 runs at 2 ≠
  trace 1). **sequential**: coherent (op-by-op oracle). The control brick (96ch =
  cat[inactive,reactive,mask], scale=ones(15)) is consumed through all 30 VACE
  blocks; rotary symbolic-T proven.

## triton — OPEN: deterministic value-degeneracy (washed-out, std≈13)

triton produces a **deterministic** washed-out pale speckle (std ≈13 vs a coherent
frame), NOT a crash and NOT a race:
- **async vs sync 12-step outputs are byte-identical** (both std 13.15, mean 172.8)
  → `CUDA_LAUNCH_BLOCKING` changes nothing → deterministic numerical bug, not a race.
- The earlier `error 700` crash was a **diagnostic-tool artifact**: `NBX_DUMP_TIDS`'
  per-op stats illegal-addressed on an empty VACE control tensor (`new_zeros
  [1,0,1536]`). Fixed: `nbx_tid_stats` 0-numel guard (commit `c9b751b`, R33-pure).

## CORRECTION — the "cross-attention text-context 512 vs 226" root is REFUTED

A prior pass localized the washed-out to the triton text context being 512 (pad)
vs 226 (oracle) and attributed it to diluted cross-attention. **That was wrong.**
The op-diff that produced it was invalid (triton 512 vs oracle 226 → every
downstream op shape-mismatched, so "first divergence = view::10" was an artifact),
and the conclusion was reached by elimination without inspecting the site. Two
decisive checks (on data already captured) refute it:

- **Check A — boundary value-identity.** At the transformer's real text input
  (`view::10`, the first encoder_hidden_states consumer), triton and the oracle
  are **byte-identical to fp16 precision on the real 226 rows** (head10 match,
  l2=15.118 both). Triton's 226→512 is a **pure zero-pad** (the extra rows are
  zeros — l2 preserved across the larger row count). There is NO value corruption
  in the text representation. (The "85 vs 41" text_encoder-output "value bug" I
  had recorded was misaligned-`op_uid` noise: UMT5 lowers to 2240 ops in triton
  vs 4816 in the sequential oracle, so `text_encoder::46` is not the same logical
  op in each engine. Cross-engine op-diff is only valid at contract-pinned
  component boundaries, not at internal op indices.)

- **Check B — config parity + T2V control.** VACE and T2V have **identical** text
  config (`zero_pad_embeddings: True`, `max_sequence_length=512`). Measured: BOTH
  **compiled** T2V and VACE run the text context at **226** (coherent); BOTH
  **triton** T2V and VACE pad to **512** (T2V triton `view::2 = [1024,4096]` =
  2×512). **T2V triton at 512 is coherent (closed 4/4).** Therefore 512-zero-pad
  is correct, designed Wan behavior (the DiT cross-attends to the full
  max_sequence_length UNMASKED — the trailing-zero count is part of the trained
  conditioning; see `text_encoder_handler.finalize_embeddings`), and it is NOT
  the washed-out cause.

- **Step-1 drift corroboration.** triton vs oracle noise_pred at step 1 (same
  seed, same input latent) differ only **1.2% in global l2** (820.3 vs 810.3) —
  subtle, not the catastrophic divergence a wrong text length would cause. The
  washed-out builds over the 12 steps / through the VAE.

**Benign R30 note (separate from the washed-out):** triton pads the text context
to `max_sequence_length`=512 (via `zero_pad_embeddings`), compiled stays at the
tokenized 226 (it reads `max_sequence_length` from `extracted_values[tokenizer]`,
a different source). Both are coherent (T2V proves both lengths work), so this is
a benign mode asymmetry, not the bug. Worth reconciling for strict R30 hygiene but
it does not gate VACE.

## CURRENT ROOT — the VACE control stream (the directive's target)

The conditioning bricks are R30-mirrored (`triton/vace_control_conditioning.py`
≡ `core/runtime/resolution/vace_control_conditioning.py`, same semantics — one
minor diff: triton normalizes the latent in fp32 via `.float()`). The degeneracy
is therefore either (a) the vae_encoder encode-of-zeros control INPUT, or (b) the
transformer's vace-block INJECTION (the 15 vace layers add `scale[i] · hint[i]`
to the main hidden). Differential in flight (`NBX_VACE_SCALE_ZERO=1`, commit
`284fca1`): injection-off → pure-T2V. **coherent at scale=0 ⇒ bug is in the
vace-block hints/injection; still washed-out ⇒ bug is in the main path's response
to the vace inputs (e.g. patch-embed of control).** `NBX_DIAG_VACE=1` captures
the control-input stats to compare triton vs compiled.

## REMAINING (resume here)
1. Read the `NBX_VACE_SCALE_ZERO=1` triton render (R29 visual): coherent ⇒ localize
   to the vace-block injection ops (op-diff the vace-block kernels triton vs
   oracle); washed-out ⇒ localize to the control patch-embed / main-path response.
2. Compare `NBX_DIAG_VACE=1` control-input stats (triton vs compiled) — rules in/out
   the vae_encoder encode-of-zeros as the corruption source.
3. Fix at root, re-render triton f9 CFG batch=2, drift-gate vs the compiled oracle,
   then triton_sequential mirror.

NOTE: triton VACE wall-clock is heavy (~18 min/render at f9 12-step; a full per-op
dump run is ~6.5 h). Iterate with 1-step drift + the scale-zero differential, not
full dumps.

## Reproduce
```bash
# triton (washed-out):
python3 -m neurobrix run --model Wan2.1-VACE-1.3B-diffusers --triton \
  --prompt "a red fox walking in a snowy forest" --cfg 5.0 \
  --height 480 --width 832 --num-frames 9 --steps 12 --seed 42 --output vace_triton.mp4
# control-stream differential (injection off → pure-T2V):
NBX_VACE_SCALE_ZERO=1 NBX_DIAG_VACE=1 python3 -m neurobrix run --model Wan2.1-VACE-1.3B-diffusers \
  --triton --prompt "a red fox walking in a snowy forest" --cfg 5.0 \
  --height 480 --width 832 --num-frames 9 --steps 12 --seed 42 --output vace_scalezero.mp4
```

## Hocine validation: compiled/sequential = TODO (coherent frames); triton = N/A (OPEN)
