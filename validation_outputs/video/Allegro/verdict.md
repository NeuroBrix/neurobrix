# Allegro (T2V) — VERDICT: OPEN (odd-H scanline root-caused to the NATIVE-frame-count regime; closure entangled with DETTE D2)

**Date:** 2026-06-28 · **Family:** video T2V · **Arch:** AllegroTransformer3DModel
(32 blocks, full-3D RoPE) + AutoencoderKLAllegro (tiling-only decode) + T5 text encoder.
Native config: 720×1280, 88 frames (latent T=22 at temporal_compression=4).

This verdict records an honest OPEN with a **substantial, decisive diagnosis** that
**overturns the prior working hypothesis** for the "odd-H scanline" (#30).

## The prior hypothesis is DISPROVEN

The chantier was framed as "the even trace value (H_tok=46) did not exercise the odd
native H (H_tok=45) → a frozen symbolic dim → symbolic fix for all parity." This is
**false for the current .nbx**:

- **Op attributes carry zero frozen even-H literals.** A full scan of every
  view/reshape/expand/slice/split size attribute in the transformer graph finds
  **no** 46/81/92/162 — every spatial reshape is symbolic.
- **Runtime shapes resolve correctly at odd-H.** A native-resolution sequential run
  (NBX_DUMP_TIDS) resolves the patchified token sequence to **3600 = 45×80** (T=1)
  and **14400 = 4×3600** (T=4), with **2×** for the CFG batch — **zero** occurrences
  of the trace products (3726 = 46×81, etc.).
- **The temporal RoPE tables resize symbolically** (`aten.cos/sin` `[1,16]→[4,16]`
  as T goes 1→4); all 384 RoPE position embeddings resize `3600→14400`. So the unit
  trace dim T=1 is **not** absorbed — the symbolic tracker handles T fully.
- **RoPE interpolation_scale is config-fixed** (h=2.0, w=2.0, t=2.2), not
  trace-derived → not a value freeze.
- **The transformer is loaded with the correct architecture** (`norm_type =
  ada_norm_single` from config, `AdaLayerNormSingle` path).

Conclusion: there is **no symbolic-shape-freeze** at odd-H. The native trace already
symbolizes H, W and T correctly.

## What the scanline actually is: a frame-count-dependent NATIVE-regime artifact

- At a reduced **13-frame** config (latent T=4), the transformer **output is
  isotropic** (built-in `NBX_HASYM_THW` row/col ratio ≈ **0.98–1.01** at the final
  ops) — i.e. the scanline does **not** reproduce. The only large hasym values are
  the RoPE position tables themselves (inherently anisotropic by construction; they
  do **not** propagate to the output).
- A 13-frame / 20-step native-resolution render is **degenerate** (saturated
  blue/orange blocks) — **out of the model's distribution**. The diffusers vendor
  pipeline at 13 frames is likewise degenerate (uniform gray). Per the R29
  vendor-reproduce rule and the existing project note, **Allegro is native-config
  locked** (the vendor degenerates at non-native configs too).
- The canonical crisp horizontal scanline (`scanline_native_f0.png`) appears only at
  the **native 720×1280 × 88-frame** regime. The remaining bug is therefore a
  **frame-count-dependent numerical artifact in the native regime**, not a shape bug.

## Why this is entangled with DETTE D2 (and not closable at 13f-class)

- Allegro is out-of-distribution at 13f, so the family's **"13f-class single-GPU"
  closure config does not apply** — a coherent Allegro frame requires the native
  88-frame regime.
- The Allegro VAE decode at native resolution already consumes **~28 GB at latent
  T=4**; at the native latent T=22 it **exceeds a single 32 GB GPU** (OOM) — this is
  exactly **DETTE D2** (5D-VAE long-clip / native-resolution single-GPU).
- Both the scanline op-localization (needs the native regime to reproduce) and the
  coherent-frame verdict (needs native 88f) sit in the **D2-deferred regime**.

## Remaining (resume when D2 capacity makes the native regime tractable)

1. **Localize the scanline op** with the native-88f `NBX_HASYM_THW` probe (find the
   first transformer op whose row/col ratio departs from ~1.0 and grows
   monotonically per block — the documented "0.99→2.02 per-layer compounding"). This
   needs the native regime to reproduce (it is isotropic at 13f).
2. **Coherent-frame + cross-engine drift gate** at native 88f — requires the D2
   capability (multi-GPU placement and/or 5D-VAE temporal tiling).
3. A valid vendor reference at native 88f (fp32 VAE; the diffusers VAE decode is
   tiling-only) for the value op-bisect.

## Reproduce

```bash
# Canonical scanline (native regime — heavy, ~native VAE > 1 GPU at full length):
python3 -m neurobrix run --model Allegro --compiled --mode t2v \
  --prompt "a red fox walking in a snowy forest" \
  --height 720 --width 1280 --num-frames 88 --steps 20 --cfg 7.5 --seed 42 \
  --output allegro_native.mp4

# Shape-correctness proof at odd-H (cheap, sequential dump):
NBX_DUMP_TIDS=dump.jsonl python3 -m neurobrix run --model Allegro --sequential \
  --mode t2v --prompt "..." --height 720 --width 1280 --num-frames 13 --steps 1 \
  --seed 42 --output /tmp/probe.mp4   # token seq resolves to 14400=4x45x80, no 3726
```

## Hocine validation: N/A (OPEN — no coherent frame; native regime is D2-deferred)

Artifacts: `scanline_native_f0.png` (canonical native scanline),
`nbx_13f_outofdistribution_f0.png` (13f out-of-distribution degeneracy).
