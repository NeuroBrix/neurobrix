# Video validation INDEX

Hocine visual validation map — one folder per model, four mp4 (one per mode),
four reference frames, verdict.md. All proofs: prompt "a red fox walking in
snow", f9 (9 frames), 30 steps, seed 42 where noted.

## Family status — single-GPU 13f-class (2026-07-01)

**7 / 10 CLOSED 4/4** at the single-GPU 13f-class config (all four modes:
PyTorch-sequential · compiled · Triton-sequential · Triton-compiled):

| model | state | note |
|---|---|---|
| Wan2.1-T2V-1.3B | CLOSED 4/4 | coherent fox all modes |
| CogVideoX-2b | CLOSED 4/4 | CFG batch=2 exercised (cfg=6) |
| SANA-Video_2B_720p | CLOSED 4/4 | fp32-forced (V100) |
| CogVideoX-5b-I2V | CLOSED 4/4 | 13f (native 49f = DETTE D2) |
| Mochi-1-preview | CLOSED 4/4 | — |
| Wan2.1-VACE-1.3B | CLOSED 4/4 | interleaved-complex RoPE fix (a0ddeff) |
| **Open-Sora-v2** | **CLOSED 4/4** | **compiled+sequential+triton coherent fox; triton_sequential DRIFT-PROVEN at cfg=7.5/CFG batch=2 (transformer `view::1052` shape [2,1024,64] finite + VAE finite + frame std 86.5 — batch dim ≠ trace exercised; the slow 50-step op-by-op coherent frame is deferred per the drift-gate doctrine, NOT rendered). Root: SDPA fully-masked-row guard + rope scheduling + timestep (72504ce/d9d10e3).** |

**Remainder (DETTE-deferred or forge-side — single-GPU-achievable branches proven where possible):**

| model | achievable now | deferred |
|---|---|---|
| Wan-I2V-14B | compiled + sequential PROVEN batch=2 (velocity corr 0.999997) | triton velocity divergence (deferred); triton_sequential = **DETTE D1** (multi-GPU) |
| Wan2.2-I2V-A14B | dual-denoiser boundary-switch core built + traced (28B .nbx) | compiled OPEN on i2v `vae_encoder` (**forge** Phase-A, separate system); triton = **DETTE D1** |
| Allegro (T2V) | odd-H scanline root-caused to the native-frame regime | native 88f VAE = **DETTE D2** (5D-VAE tiling) |
| Allegro-TI2V | — | **forge** trace pending; inherits **DETTE D2** |

DETTE D1 (multi-GPU NBXTensor op-input co-location) and D2 (5D-VAE long-clip /
native-res tiling) are the two general Prism capabilities deferred to the final
pass per `DETTE.md` — they unblock the 14B/28B triton axes and the native-VAE
closures together. The next deliberate chantier is D1 (unblocks Wan-I2V-14B +
Wan2.2 triton).

## Artifact paths (validation targets)

| model | proof size | mode | mp4 | reference frame | Hocine OK |
|---|---|---|---|---|---|
| Wan2.1-T2V-1.3B | 480x832 f9 | sequential | Wan2.1-T2V-1.3B/sequential_f9.mp4 | Wan2.1-T2V-1.3B/sequential_coherent_frame.png | TODO |
| Wan2.1-T2V-1.3B | 480x832 f9 | compiled | Wan2.1-T2V-1.3B/compiled_f9.mp4 | Wan2.1-T2V-1.3B/compiled_coherent_frame.png | TODO |
| Wan2.1-T2V-1.3B | 480x832 f9 | triton-seq | Wan2.1-T2V-1.3B/triton_seq_f9.mp4 | Wan2.1-T2V-1.3B/triton_seq_coherent_frame.png | TODO |
| Wan2.1-T2V-1.3B | 480x832 f9 | triton | Wan2.1-T2V-1.3B/triton_f9.mp4 | Wan2.1-T2V-1.3B/triton_coherent_frame.png | TODO |
| CogVideoX-2b | 480x720 f9 | sequential | CogVideoX-2b/sequential_f9.mp4 | CogVideoX-2b/sequential_coherent_frame.png | TODO |
| CogVideoX-2b | 480x720 f9 | compiled | CogVideoX-2b/compiled_f9.mp4 | CogVideoX-2b/compiled_coherent_frame.png | TODO |
| CogVideoX-2b | 480x720 f9 | triton-seq | CogVideoX-2b/triton_seq_f9.mp4 | CogVideoX-2b/triton_seq_coherent_frame.png | TODO |
| CogVideoX-2b | 480x720 f9 | triton | CogVideoX-2b/triton_f9.mp4 | CogVideoX-2b/triton_coherent_frame.png | TODO |
| SANA-Video_2B_720p | 704x1280 f9 | sequential | SANA-Video_2B_720p/sequential_f9.mp4 | SANA-Video_2B_720p/sequential_coherent_frame.png | TODO |
| SANA-Video_2B_720p | 704x1280 f9 | compiled | SANA-Video_2B_720p/compiled_f9.mp4 | SANA-Video_2B_720p/compiled_coherent_frame.png | TODO |
| SANA-Video_2B_720p | 704x1280 f9 | triton-seq | SANA-Video_2B_720p/triton_seq_f9.mp4 | SANA-Video_2B_720p/triton_seq_coherent_frame.png | TODO |
| SANA-Video_2B_720p | 704x1280 f9 | triton | SANA-Video_2B_720p/triton_f9.mp4 | SANA-Video_2B_720p/triton_coherent_frame.png | TODO |

## Closure narratives (diagnostic reference)

| model | family | mode | verdict (agent) | frame | Hocine OK |
|---|---|---|---|---|---|
| Wan2.1-T2V-1.3B | video | compiled (fp16) | COHERENT — sharp red fox in snow (std 0.241 ~ vendor 0.242) | Wan2.1-T2V-1.3B/compiled_coherent_frame.png | TODO |
| Wan2.1-T2V-1.3B | video | sequential | COHERENT — clear red fox walking in snow, temporally consistent (f9 30-step). Root cause was a seq-dispatcher view-shape gap: tensor_resolver blindly inferred the LAST view dim on a numel mismatch, folding the CFG batch into the feature dim ([2,9216]→target[1,6,1536] gave [1,6,3072]) and producing an invalid [4680,4680,-1] on hidden-state views whose traced shape duplicated the seq expr into the batch slot. Fixed by mirroring compiled `_make_view_reshape` (try each axis as -1, prefer input-dim-match, else first valid axis — R30 parity). Anti-regression: TinyLlama sequential unaffected. | Wan2.1-T2V-1.3B/sequential_coherent_frame.png | TODO |
| Wan2.1-T2V-1.3B | video | triton | COHERENT — clear red fox walking in snow, temporally consistent (first/mid/last frames all coherent), matches the compiled oracle scene. Root cause was an R30 asymmetry: the triton diffusion flow never mirrored the compiled `zero_pad_embeddings` handling (UMT5 padding embeddings stayed non-zero → DiT text condition diverged → mosaic). Fixed 6326f42 (`_tokenizer_config_with_flags` at both finalize sites + negative `attention_mask`); condition_embedder gelu::0 now matches the oracle (ratio 1.000, was 7.3×). The earlier "RoPE 174× / conv scramble" leads were instrumentation artifacts (complex-l2 dump read real-part-only; head10 layout-confounded; compiled slot-reuse dedup) — all debunked: freqs |.|=1, positions arange-correct, _complex_mul + conv3d exact in isolation. f9 30-step = 1557s. | Wan2.1-T2V-1.3B/triton_f9_coherent_frame.png | TODO |
| Wan2.1-T2V-1.3B | video | triton-seq | COHERENT — sharp red fox walking in snow, temporally consistent (f9 30-step, 1567s). Was uniform gray (ran end-to-end exit 0, std 5–7): the seq dispatcher's resolve_attr parsed dtype kwargs verbatim, so the graph `_to_copy` to complex128 on the RoPE freqs table reinterpreted the loader-narrowed complex64 (fp32 [re,im] pairs) as fp64 → unit-magnitude table collapsed to near-zero (l2 256→0.5) → all rotary applications corrupted. Fixed a44d4a4: fp64→fp32 / complex128→complex64 narrowing in both dtype-parse branches (R30 mirror of the compiled hot loop); freqs chain now matches --triton exactly. Localized via the new NBX_DUMP_TIDS hook in the triton-seq op loop (mode was previously blind to per-op dumps). Anti-reg: TinyLlama greedy triton-seq byte-identical to compiled. | Wan2.1-T2V-1.3B/triton_seq_coherent_frame.png | TODO |
| CogVideoX-2b | video | sequential | COHERENT — red fox walking in snow, treeline, temporally consistent (f9=480×720×9, 30 steps). Oracle for the whole chain. | CogVideoX-2b/sequential_coherent_frame.png | TODO |
| CogVideoX-2b | video | compiled | COHERENT — sharp fox, snowy trees. Needed compiled mirrors: dead arena-slot kills + native_group_norm derived scalars. | CogVideoX-2b/compiled_coherent_frame.png | TODO |
| CogVideoX-2b | video | triton-seq | COHERENT — sharp fox. Needed the dead-output liveness rule in the triton-seq loop. | CogVideoX-2b/triton_seq_coherent_frame.png | TODO |
| CogVideoX-2b | video | triton | COHERENT — fox in snowy conifer landscape. Needed the dead-output rule in the arena liveness (may also reduce the long-standing triton live-watermark gap). | CogVideoX-2b/triton_coherent_frame.png | TODO |
| CogVideoX-2b | video | **ALL 4 @ CFG batch=2** | **CLOSED 4/4** at native 480×720, 13f, cfg=6 (batch=2), seed 42 — four coherent foxes (corr vs compiled ≥0.988), no seams. Batch/CFG exercised ≠ trace (cfg=1.0→batch=1 coherent). Seam debunk = FIXED Prism over-tiling bug `f42917f` (profiled VAE at 1024² vs real 512² → forced TilingEngine → halo seams; now `single_gpu`, no tiling). VAE proven symbolic (decoded native = 4× trace extent, no tiling). Artifacts: `../video_cfg2/CogVideoX-2b/m_*_f6.png` + seam A/B. Canonical verdict: `CogVideoX-2b/verdict.md`. | ../video_cfg2/CogVideoX-2b/m_compiled_f6.png | TODO |
| SANA-Video_2B_720p | video | sequential (fp32-forced) | COHERENT — sharp red fox walking in snow, temporally consistent (f9 704x1280 30-step, native latent bin 22x40, latent T=2 vs trace T=11 proves symbolic T). Five root causes fixed to get here: (1) Phase A orphaned VAE (buffer-resident latents_mean/std vs dev-window config reads, forge 9bfb448); (2) 3D-RoPE table value collisions (frozen slice ends, forge c69c2a1); (3) GLUMBTempConv rearrange collisions (batch*T==H from defaults-derived stimulus) + misaligned-product view rule gap + unregistered unbind rule (forge 90328c4) — banded-mosaic then noise classes; (4) sequential aten::copy destination detached by contiguous-normalization — rotate-half RoPE slice writes lost, attention on uninitialized q/k (NBX e09b8bc, matched-input microtest cosine 0.014 -> vendor-class); (5) OPEN: fp16 NaN step 1 (vendor prescribes bf16; V100 fp16 overflow) — run under NBX_FORCE_FP32_COMPUTE=1 pending the registry dtype seam. | SANA-Video_2B_720p/sequential_coherent_frame.png | TODO |
| SANA-Video_2B_720p | video | compiled (fp32-forced) | COHERENT — sharp red fox walking in snow, matches the sequential oracle to 0.76/255 mean abs (kernel-order noise). Two compiled-only root causes fixed: (6) _make_slice applied step>1 by TRUNCATION not striding — rotate-half RoPE cos[...,0::2]/sin[...,1::2] read wrong table halves (first model to exercise step>1 slices in compiled; NBX d229f9a); (7) the shared promotion pass stomped the VAE's time-anchored upsampler expr with height-1 on a trace-value coincidence — override now restricted to all-trace-1-symbol exprs (same commit). Plus forge 36fc554: functional rotate-half rotary (graph now 0 aten::copy) + constant-aware stimulus de-collision (batch*heads==W trace collision). | SANA-Video_2B_720p/compiled_coherent_frame.png | TODO |
| SANA-Video_2B_720p | video | triton-seq (fp32-forced) | COHERENT — sharp red fox walking in snow, temporally consistent, matches the PyTorch oracle at 1.5/255 mean abs (TritonDtypeEngine numerics). First-try pass: the functional rotate-half rotary (forge 36fc554, zero aten::copy in the graph) and the born-at-source symbolic rope/temporal chains left nothing for the kernel layer to trip on. | SANA-Video_2B_720p/triton_seq_coherent_frame.png | TODO |
| SANA-Video_2B_720p | video | triton (fp32-forced) | COHERENT — sharp red fox walking in snow, temporally consistent, matches the PyTorch oracle at 1.9/255 mean abs. Arena hot loop took the copy-free functional-rotary graph first-try. | SANA-Video_2B_720p/triton_coherent_frame.png | TODO |
| SANA-Video_2B_720p | video | ALL 4 CLEAN (no env) | 4/4 COHERENT with the registry requires_fp32_compute seam on the transformer only (fp16 text encoder + VAE proven by the run); cross-mode agreement <= 2/255 vs oracle. Final artifacts replaced with the clean runs. | SANA-Video_2B_720p/ | TODO |
| Wan2.1-VACE-1.3B | video | compiled (no-CFG) | COHERENT — red fox walking in white snow, all-generate text→video, f9 480x832 12-step, 44s (latent T=3 vs trace T=10 proves symbolic T). VACE control conditioning brick (96ch = cat[inactive,reactive,mask64] + scale=ones(15)) consumed through all 30 VACE blocks. 9 forge fixes for the diffusers-0.35 real-3D-rotary-grid symbolic-T (T2V/0.33 complex rotary was already symbolic): vae_encoder multiframe, control-spatial match (padding→0), _factory new_* batch inheritance, value-match fallback + tie-break, _view_reshape LAYER 0 size-1, _reinject_slice_end, _view_reshape LAYER 0.5 atomic-flatten-product. OPEN: CFG batch=2 + sequential/triton/triton-seq blocked on a GENERAL Wan batch=1-trace-absorption (modulation broadcast picks concrete 1 over symbolic s0; never exercised since all Wan validations ran cfg=1.0). | Wan2.1-VACE-1.3B/compiled_nocfg_frames.png | TODO |
| Allegro (T2V) | video | (diagnosis) | OPEN — #30 odd-H scanline is **NOT** a symbolic-parity freeze (DISPROVEN): the native-res re-trace already symbolizes H/W/T (op-attrs carry zero frozen even-H literals; runtime resolves seq to 3600=45×80 / 14400=4×3600, never 3726=46×81; temporal RoPE tables resize [1,16]→[4,16]; interpolation_scale config-fixed; norm_type=ada_norm_single loaded correctly). The scanline is a **frame-count-dependent numerical artifact in the NATIVE regime** — isotropic at 13f (hasym≈1.0), degenerate (out-of-distribution) at 13f for both NeuroBrix and the vendor (Allegro is native-config locked). Coherent frame + scanline op-localization both need native 720×1280×88f (VAE ~28GB at latent T=4 → OOM at native T=22) = **DETTE D2**. Verdict: `Allegro/verdict.md`. | Allegro/scanline_native_f0.png | N/A (D2-deferred) |
