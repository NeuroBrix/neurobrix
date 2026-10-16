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
- Read 2026-10-16 in the vendor code (diffusers 0.37, `models/autoencoders/`), so the rule is one, not Wan's:
  `autoencoder_kl_wan.py`, `autoencoder_kl_cogvideox.py` and `autoencoder_kl_mochi.py` run temporal chunks carrying a
  conv cache (`feat_cache` / `conv_cache`: 78, 92 and 70 sites) — the same mechanism, three catalogue families;
  `autoencoder_kl_hunyuan_video.py` and `autoencoder_kl_ltx.py` batch frames without a cache (framewise
  encoding/decoding: to be read for exactness before they take it); `autoencoder_kl_allegro.py` has neither, and its
  encode is tiled by construction. Every one defaults to untiled (`self.use_tiling = False`).

## Derived from the graph (decision of 2026-10-16 18:40)

The decision validates the mechanism with one condition: the temporal cut and the causal convolutions' caches are
DERIVED from the graph by Forge, never a schedule written per model; the registry carries only the vendor source. So
step 1 of the proposal changes: Forge does not trace a chunk under a declared schedule, it reads the traced clip graph.
Read 2026-10-17 in the four committed VAE graphs (`components/vae*/graph.json`, the counts below are from the graphs):

| container | causal convs (front-only temporal pad feeding a conv) | norm over time? | vendor reference |
|---|---|---|---|
| Wan2.2-I2V-A14B encoder | 22 of 32: `pad [1,1,1,1,2,0]` -> conv | no: RMS norm per position; mid-block attention per frame (`b*t` folded, `autoencoder_kl_wan.py:403`) | chunks with `feat_cache`, whole pass equal to 1.1e-6 |
| Wan2.2-I2V-A14B decoder | 32 of 39: `pad [1,1,1,1,2,0]`; the temporal resample convs `pad [0,0,0,0,2,0]` | no | same |
| mochi-1-preview decoder | 38 of 39: `replication_pad3d [1,1,1,1,2,0]` -> conv | no: group norm on `[B*T, C, H, W]` (`MochiChunkedGroupNorm3D`, `autoencoder_kl_mochi.py:35-63`) | whole clip by default (`:761`), framewise optional (`:891`) |
| CogVideoX-2b decoder | 38 convs read `cat(slice(x, t, 0, 1), x)` on time then a spatial pad (first frame replicated) | YES: `native_group_norm` on `[B, C, T, H, W]` (37) | chunks of 2 latent / 8 sample frames with `conv_cache` (`autoencoder_kl_cogvideox.py:1072-1073, 1131-1146, 1185-1198`) |
| Allegro decoder | none (no temporal front pad) | YES: `native_group_norm` on `[B, C, T, H, W]` (94) | spatial-temporal tiles (`autoencoder_kl_allegro.py:786-796`) |

What this gives, as one rule over the graph:

1. **A causal conv is read, not declared**: a convolution whose input passes through a pad (or a cat on the conv's
   time axis) that adds k-1 frames at the FRONT of time and none at the back. Its cache is the last k-1 frames of the
   pad's input; at the first chunk, the pad as traced. Forge rewrites each such site into `cat(cache_in, x)` with the
   cache as a graph input and `slice(x, time, -(k-1))` as a graph output — symbolic, no new op, no model name.
2. **The cut is exact only where no op is global over time** between the component's input and output — the
   reachability analysis of decision (1), on the time axis this time. Wan and mochi pass: their norms and attention
   fold time into the batch. CogVideoX and Allegro do not: a group norm reduces over time, so a temporal cut is a
   different function... EXCEPT that their vendor's own reference cuts (CogVideoX's 2-frame batches; Allegro's tiles).
   That is the single declared exception: the registry states the vendor's chunk length (or tile geometry) WITH its
   source, and the engine then runs the vendor's function, not an approximation of it.
3. **The chunk granule is read from the graph's temporal strides**: the product of the time strides of the convs and
   the time scale of the resamples between input and output, with the causal first frame (the `1 + 4k` frame counts of
   Wan and CogVideoX) the graph's own `(t - 1) // s + 1` arithmetic.
4. Prism prices one chunk plus the caches; it walks chunk lengths down from the whole clip, every one exact by 2.

Consequence for CogVideoX: our untiled whole-clip decode normalises over all frames. It is bit-exact against the
vendor's own single-chunk path (`num_latent_frames_batch_size = num_frames`), and 8.2e-2 max / 2.7e-3 mean off the
vendor's DEFAULT 2-frame batches (Forge's record, `forge/tracer/video_vae_patches.py:156-173`). Decision 21:30 names the
vendor's temporal batches as CogVideoX's declared reference, so the declared chunking reproduces the vendor's default
pipeline; the proof (chunked = vendor float32 at its default, to dtype noise, both branches) covers it.
