# Wan2.1-VACE-1.3B — VERDICT: compiled + sequential CLOSED; triton OPEN (washed-out). REFUTED so far: text-context-512, vace injection, flash attention. Localized to a 34% self-attn output divergence from identical inputs (math==flash) — compiled-hot-loop vs kernel/baseline split in flight.

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

## DIFFERENTIAL CHAIN — injection + flash-attention also REFUTED

Two more renders ruled out the obvious control-stream suspects:
- **`NBX_VACE_SCALE_ZERO=1` (injection off → pure-T2V): STILL washed-out**
  (frame std 11.2, same gray mesh-speckle). So the bug is NOT the 15-layer
  vace injection — per the differential's own logic, it is the **main path's
  response to the vace inputs**, not the hints. The control INPUT is well-formed
  (`NBX_DIAG_VACE`: vae_latent norm mean=-0.279 std=1.019, control [1,96,3,60,104]).
- **`NBX_FORCE_MATH_ATTENTION=1` (flash off → deterministic math attn): STILL
  washed-out** (std 8.8). So the flash kernel is NOT the cause.

## ELEMENT-WISE LOCALIZATION (metric correction)

The earlier "drift" numbers were **norm-ratio** (`|‖t‖−‖o‖|/‖o‖`), which measures
energy not divergence and badly understated the error. Recomputed **element-wise**
(`‖t−o‖/‖o‖`, cosine) on the head10 pairs, block-0 self-attention, triton vs the
PyTorch oracle:
- q/k/v projections, RMS-norm(q,k), full RoPE chain, RoPE'd q/k (the actual SDPA
  inputs): **edrift ≈ 0.000, cos 1.000 — bit-identical.**
- **self-attention output (`permute::3`): edrift 0.339 (34%), cos 0.943** — wrong
  from identical inputs.
- final noise_pred: edrift 0.125 (12.5%), cos 0.993 (NOT the 1.2% norm-ratio).

**Under math attention, `permute::3` is STILL 0.339 off the oracle AND identical
to flash (math-vs-flash edrift 0.000).** So both triton attention paths agree with
each other but diverge 34% from the PyTorch oracle, from bit-identical q/k/v, with
the graph SDPA carrying **no mask, is_causal=False, default scale**. This points to
a systematic triton-vs-PyTorch attention difference (layout/scale/accumulation),
NOT the kernel choice.

## OPEN QUESTION (resume here) — is the 34% the bug, or a triton-vs-PyTorch baseline?

T2V uses the same triton attention and is coherent, so the 34% may be a tolerated
baseline and the washed-out a separate cause. Two tests decide it:
1. **triton vs triton_sequential** (same kernels, same graph, same 512 pad, aligned
   op_uids) — IN FLIGHT. triton_seq **coherent** ⇒ bug is the **compiled hot-loop**
   (slot reuse / kill_slots / fusion / a vace-block output slot aliasing a main-path
   slot — fits scale-zero staying broken). triton_seq **washed-out** ⇒ kernel/graph.
2. **T2V triton self-attn baseline**: if T2V's self-attn output is also ~34% off its
   own oracle (and T2V is coherent), 34% is baseline ⇒ look past attention.

NOTE: triton VACE wall-clock ~18 min/render (12-step f9); full per-op dump ~6.5 h.
Use 1-step **op_uid-filtered** dumps (`NBX_DUMP_TIDS_FILTER` matches op_uid
substrings — seconds) + element-wise drift, not full dumps.

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
