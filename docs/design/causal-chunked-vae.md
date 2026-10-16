# Causal VAEs run in the vendor's temporal chunks — design (2026-10-16, no code before review)

## Why

Wan2.2-I2V-A14B renders a wrong video (the open defect of the stable list). Root cause, measured 2026-10-16
(campaign `stable/wan22_anchor/agent/`): Prism cuts the **VAE encoder into spatial tiles** (component tiling, 352 px
tiles, 48 px overlap) whenever the whole encode does not fit the card — on a 16 GB card at 17 frames, on a 32 GB card
at 81. The Wan encoder's mid-block attention spans the whole frame, so a tiled encode is a different function: the I2V
condition latent is 5.4e-3 off the vendor's float32 encode (2.8 % on a seam column) against 1.1e-6 untiled, and the
error, amplified by guidance 5.0 and 40 steps, becomes the wrong video. The same plans tile the VAE **decoder**
spatially too. diffusers says it of its own tiled path: "different from non-tiled encoding because each tile uses a
different encoder" (`autoencoder_kl.py:259`); Wan encodes untiled by default (`autoencoder_kl_wan.py:1066`).

The engine's untiled encode is exact (1.1e-6), so the trace is right. What is wrong is the only way the plan has to fit
a big clip: space tiles. A proposal that forbids space tiles for a spatially coupled encoder
(`prism-no-spatial-tile-of-a-coupled-encoder` f4794dbd4) is correct but not landable: Wan at 81 frames would need the
whole-clip encode (~72 GB) and falls to cpu_streaming; Allegro-TI2V would be refused (its vendor VAE has no untiled
encode at all, `autoencoder_kl_allegro.py:804`).

## What the vendor does (diffusers 0.37, `autoencoder_kl_wan.py`)

- `_encode` (:1126): the first frame alone, then chunks of 4 frames, each `encoder(chunk, feat_cache, feat_idx)`;
  outputs concatenated on time; `quant_conv` once.
- `_decode` (:1180): one latent frame at a time, `decoder(x_i, feat_cache, feat_idx, first_chunk=i==0)`.
- `feat_cache`: one entry per causal conv (`WanCausalConv3d`, :131), the last `CACHE_T = 2` frames of that conv's
  input (:31, :349-351), the first chunk's special cases ("Rep" in the temporal resample, :271-276).

So the vendor's memory is one chunk's activations plus the caches; and the chunked result IS the reference (our
untiled single pass equals it to 1.1e-6, which is why the trace is right).

## Proposal

1. **Forge traces the chunk, not the clip.** For a component whose vendor runs it in causal chunks (declared in the
   registry with its source: the chunk schedule — first chunk, chunk length — and the cache depth `CACHE_T`), the
   tracer captures one chunk's forward with every causal conv's cache as an explicit graph input AND output (a cache in,
   the updated cache out), and the first-chunk variant when the vendor's first chunk differs. Every dimension symbolic
   (the chunk's spatial extent included). No per-model code: the cache is a tensor the graph reads and writes.
2. **The runtime loops the schedule.** One brick, both engines (R30): it walks the declared chunk schedule over the
   time axis, feeds each chunk and the carried caches, concatenates the outputs — the vendor's loop, with no vendor
   import (R34). The caches are state: the state engine declares them (`kind: recurrent`, design §4.1).
3. **Prism prices one chunk** plus the caches instead of the clip, so a Wan encode/decode fits a 16 GB card untiled.
   Spatial tiling stays possible only where the vendor's own reference tiles (step 4).
4. **A vendor whose reference is tiled is declared, not inferred.** Allegro's VAE encodes in tiles in its vendor
   pipeline: the registry states its tile geometry with its source, and Prism tiles exactly so. Prism declines to
   space-tile any other component whose graph couples height × width (the proposal branch's detector, kept).

## Proofs

- Wan2.1/2.2 VAE encode and decode: chunked = the vendor's float32 to fp-noise at 17 and 81 frames, on a 16 GB card.
- Wan2.2-I2V-A14B at its judging request (40 steps, 81 frames): the 17-frame step diff against the float32 truth at the
  resident plan's precision on every plan; the 81-frame video judged from outside against the vendor.
- Allegro-TI2V: unchanged outputs (its declared tiles are the vendor's).
- Gate.

## Questions

- The registry key names for the chunk schedule and the tiled-reference geometry (Forge side).
- Whether other causal video VAEs in the catalogue (HunyuanVideo, Mochi, LTX, CogVideoX) take the same mechanism — to
  be read from their vendor code before the Forge rule is written, so the rule is one, not Wan's.
