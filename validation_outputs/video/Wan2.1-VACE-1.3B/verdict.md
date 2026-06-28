# Wan2.1-VACE-1.3B — VERDICT: compiled + sequential CLOSED; triton OPEN (root localized: cross-attention text-context length)

**Date:** 2026-06-28 (supersedes the 2026-06-16 bring-up note) · **Family:** video control · **Arch:** WanVACETransformer3DModel (30 blocks, 15 VACE control layers) + UMT5 + AutoencoderKLWan.

## CLOSED
- **compiled, CFG batch=2 (default guidance), f9 480×832**: coherent red fox in
  snow (the Wan batch=1-trace-absorption fix landed; batch symbol s0 runs at 2 ≠
  trace 1). **sequential**: coherent (op-by-op oracle). The control brick (96ch =
  cat[inactive,reactive,mask], scale=ones(15)) is consumed through all 30 VACE
  blocks; rotary symbolic-T proven.

## triton — OPEN, root LOCALIZED to the cross-attention text context

triton produces a **deterministic value-degeneracy** (washed-out pale speckle, std
≈13 vs a coherent frame), NOT a crash and NOT a race:
- **async vs sync 12-step outputs are byte-identical** (both std 13.15, mean 172.8)
  → `CUDA_LAUNCH_BLOCKING` changes nothing → deterministic numerical bug, not a race.
- The earlier `error 700` crash was a **diagnostic-tool artifact**: `NBX_DUMP_TIDS`'
  per-op stats illegal-addressed on an empty VACE control tensor (`new_zeros
  [1,0,1536]`). Fixed: `nbx_tid_stats` 0-numel guard (commit `c9b751b`, R33-pure).
  That unblocked the cross-engine op-diff.

**Op-diff (triton vs sequential oracle, transformer, component-aware):** the first
divergence is `aten.view::10` — the cross-attention **text context seq is 512 in
triton vs 226 in the oracle** (`[2×226,4096]` vs `[2×512,4096]`), cascading through
the caption-projection MLP + all 30 blocks' cross-attention (~318 ops each side
carry 226/512). The UMT5 text_encoder **outputs [1,226,4096] in BOTH** modes, so the
226→512 happens in the triton **encoder_hidden_states preparation** (the CFG engine
asserts pos==neg seq and did not raise, so triton carries 512 on both branches).
Net effect: the cross-attention attends over the full padded 512 (≈286 extra
positions) instead of the 226 meaningful ones → **diluted text conditioning →
washed-out output**. (The text_encoder output l2 also differs, 41 oracle vs 85
triton, consistent with the padding tail not matching/being zeroed.)

**Trace shape note (key):** `transformer.encoder_hidden_states` traces at **[1,
226, 4096]** (text_encoder input_ids trace = [1,226]) — so 226 IS the trace value
and the oracle correctly uses it. `max_sequence_length=512` is only a config
constant. Triton forces **512**, i.e. it sizes the text context at
`max_sequence_length` rather than the trace/runtime length. This is why T2V triton
is coherent (its validation text length matched) but VACE (226 ≠ 512) is not.

## REMAINING (resume here)
1. **Nail the 226→512 site** in the triton text path. The text_encoder outputs 226
   in both modes (no 512 anywhere in the triton text_encoder dump), so the 512 is
   introduced at the **transformer's encoder_hidden_states binding** in triton (the
   InputResolver/arena sizing it to `max_sequence_length`=512 instead of the actual
   last_hidden_state length 226), NOT in the tokenizer or text encoder. The oracle
   resolves the encoder seq to the runtime 226; triton must do the same (R30), or
   mask the pad positions in cross-attention. (CFG asserts pos==neg seq and did not
   raise, so triton carries 512 on both branches.)
2. Fix at root, re-render triton f9 CFG batch=2, drift-gate vs the compiled oracle.
3. triton_sequential mirror.

NOTE: triton VACE wall-clock is heavy (~18 min/render at f9 12-step; a full per-op
dump run is ~6.5 h — the "throughput-blocked" note). Iterate with the op-diff at
1 step (deterministic; the degeneracy is per-forward, present at 1 step), not full
renders.

## Reproduce
```bash
# triton (degenerate, washed-out):
python3 -m neurobrix run --model Wan2.1-VACE-1.3B-diffusers --triton \
  --prompt "a red fox walking in a snowy forest" --cfg 5.0 \
  --height 480 --width 832 --num-frames 9 --steps 12 --seed 42 --output vace_triton.mp4
# op-diff localization (1 step, both engines, then opdiff by (component,op_uid)):
NBX_DUMP_TIDS=seq.jsonl  python3 -m neurobrix run ... --sequential --steps 1 ...
NBX_DUMP_TIDS=tri.jsonl  python3 -m neurobrix run ... --triton     --steps 1 ...
```

## Hocine validation: compiled/sequential = TODO (coherent frames); triton = N/A (OPEN)
