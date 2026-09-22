# Apple 0.5.5 — delivery ledger

**Method (doctrine, 2026-09-21, the owner's decision — supersedes the
download-and-run loop this ledger began under):** three stages, three
columns, never one.

- **CENSUS** — the keys the catalogue demands for this profile, kernel x
  BUCKETED-shape x dtype, enumerated by passing the containers' GRAPHS (never
  weights, never model runs) through the engine's dispatch in shadow mode
  (`NBX_CENSUS=1`), from the SHARED CACHE (`Super-NeuroBrix-Cache`, canonical),
  never the hub. The census tool is on main (`tools/certified_census.py`);
  each model's keys are tagged with its graph_sha. NOTE: the launcher key is
  still EXACT on main — the bucketed key is on the Dell's branch; request-
  dependent keys are held until it lands.
- **CERTIFIED** — those keys swept and oracle-proven on synthetic tensors
  under the stability witness (which already refuses a perturbed sweep, so
  the VM may run).
- **VERIFIED** — a claimed model is checked NOT BROKEN: one judged run per
  mode at zero miss. Zero miss at verification is also the census's
  completeness proof. A model that cannot fit is refused BY ARITHMETIC,
  without download.

Machine: M4 Pro, 24 GB unified, 4 GB floor with the Parallels VM running,
22 GiB disk trigger, pin `triton 3.8.0+git4a15f415 mps`, directory
1051 keys. Instruments: LLM literal text; STT transcript; TTS ASR
readback; image external degeneracy judge; video judged by eye.

## Verification table (modes C = compiled, T = triton, S = triton-sequential)

| container | size | verdict | figures |
|---|---|---|---|
| llm/granite-3.1-1b-a400m-instruct | 2.7 GB | **DELIVERED** 2026-09-21 | C/T/S all rc=0, "The capital of France is **Paris.**" (LLM judge, exact), 0 autotune misses in all three modes — its shapes were already covered by the 974-key directory; MoE kernels run fixed configs by design. First mixture-of-experts delivery on Apple; the standing MoE refusal lifted through the driver-selection door on this run's judgment, not on a probe. |
| tts/Kokoro-82M | 366 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, ASR reads the sentence exactly in each mode. |
| tts/chatterbox | 2.1 GB | **DELIVERED** 2026-09-20 | C/T/S rc=0, ASR exact; byte-reproducible under NBX_FORCE_RAND_SEED. |
| llm/TinyLlama-1.1B-Chat-v1.0 | 2.1 GB | **DELIVERED** 2026-09-20 | C/T/S rc=0, "Paris." exact, text byte-identical across C and T. |
| stt/whisper-large-v3-turbo | 1.6 GB local | **DELIVERED** 2026-09-20 | C/T/S rc=0, JFK transcript exact. NOTE: the hub's stt/ directory for it is EMPTY — the local copy is the only reachable one; flagged, not evicted. |
| upscaler/swin2SR-classical-sr-x2-64 | 58 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged non-degenerate (std ~104). |
| upscaler/swin2SR-classical-sr-x4-64 | 58 MB | **DELIVERED** 2026-09-20 | same battery, same judge. |
| upscaler/swin2SR-realworld-sr-x4-64-bsrgan-psnr | 58 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| upscaler/real-esrgan-x2 | 67 MB | **VERIFIED (R29, viewed)** 2026-09-21 | Retraced shared-cache container; @448 → 896², rc=0, **0 misses**, std 103.9. **Looked at (R29):** coherent apple faithful to the input — sharp stem, preserved yellow crown, bright specular, fine lenticel speckle, base spot; no grid, no seam. x8@1024 (tiled) owed in a memory window. |
| upscaler/real-esrgan-x4 | 67 MB | **VERIFIED (R29, viewed)** 2026-09-21 | Retraced shared-cache container; @448 → 1792², rc=0, **0 misses**, std 104.5. **Looked at (R29):** coherent, faithful — same features, crisp speckle; no grid, no seam. x8@1024 (tiled) owed in a memory window. |
| upscaler/real-esrgan-x8 | 67 MB | **VERIFIED (R29, viewed)** 2026-09-21 | Retraced shared-cache container; @448 → 3584², rc=0, **0 misses**, std 104.1. **Looked at (R29):** looked at whole at full 3584² — coherent apple, forked textured stem, yellow crown, smooth specular, the GAN's own lenticel-speckle texture, rounded rim shading, base spot; NO grid, NO tile seam (untiled at 448), NO banding. x8@1024 (tiled) owed in a memory window. |
| upscaler/swinir-classical-x2 | 99 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| upscaler/swinir-classical-x4 | 92 MB | **DELIVERED** 2026-09-20 | C/T/S rc=0, judged. |
| stt/parakeet-tdt-1.1b | 4.2 GB | **DELIVERED** 2026-09-21 | C/T/S all rc=0 (7 s / 22 s / 15 s), JFK transcript exact in each mode (STT judge), 0 misses both triton modes. Extract from hub 350 s (~12 MB/s sequential). Local copy deleted after judgment. |
| tts/openaudio-s1-mini | 4.1 GB | **DELIVERED** 2026-09-21 | C/T/S all rc=0 (12 s / 367 s / 426 s), ASR reads the sentence exactly in each mode. CORRECTION: the harness's miss counter used the wrong pattern — the delivery runs demanded 41 keys (T 15 + S 26); all certified under the witness 2026-09-21. Local copy deleted after judgment. |
| audio_llm/canary-qwen-2.5b | 4.9 GB | **REFUSED — engine defect (Prism)** 2026-09-21 | Both modes tried rc=1 in ~2 s: `ZERO FALLBACK: No allocation for component 'perception_encoder'` — Prism plans no allocation for the perception encoder; mode-independent (planning-time). Re-verified on today's build; first seen 2026-09-19. Prism is the rack side's domain; the model fits this machine by arithmetic (4.9 GB bf16 vs 24 GB) and runs the day the plan allocates its encoder. Local copy deleted. |
| tts/VibeVoice-1.5B | 5.4 GB | **PARTIAL — 2 of 3 modes** 2026-09-21 | C rc=0 11 s and T rc=0 688 s, ASR exact both. CORRECTION: T demanded 38 keys (the miss counter's pattern was wrong); all certified under the witness 2026-09-21. S refuses in 4 s by DESIGN: `ZERO FALLBACK: decode branches need the KV cache path (no KV wrapper on this session)` — sequential mode builds no KV wrapper (deliberate O(n) fallback) and the next_token_diffusion flow needs decode branches. Not scaled down silently: the three-mode criterion is the owner's; this line says exactly which mode is short and why. CANDIDATE WALL CLASS: sequential × branch-decode flows — a second instance stops the loop. Local copy deleted. |
| audio_llm/Voxtral-Mini-3B-2507 | 9.4 GB | **PARTIAL — compiled only** 2026-09-21 | C rc=0 27 s, JFK transcript exact — after fixing a real defect this cycle exposed (audio_llm prefix/suffix embeds looked up on the table's device never joined the context device; "Passed CPU tensor to MPS op"). But the machine holds it at the edge: C's lowest was 4360 MB of the 4096 floor at a 14.5 GB ambient, an earlier attempt floor-stopped from a 13.0 GB ambient, and BOTH triton arms floor-stop at 4086/4084 MB — ten megabytes under the floor, 0 misses, after 118 s / 403 s of real work. The figure that refuses: the triton arms need ~10 MB more than the floor leaves at the ambient band's top (10.2-15.2 GB measured with the VM running). Refused for T/S under standing conditions rather than retried onto a lucky ambient — and **CONDITIONAL on the rack side's TritonSequence retention fix**: the triton arms run through TritonSequence.run(), the object measured retaining ~410 MB live + parking ~332 MB per invocation, and they lose by ten megabytes. A re-run of both arms is OWED when that fix lands — the opposite of a lucky-ambient retry: a re-run owed to a named change. Sits beside the 1024 real-esrgan artefact and the eleventh rung key on the same condition. Local copy deleted. |
| image/Sana_1600M_4Kpx_BF16 | 13.0 GB | **UNVERIFIED — both causes now owned** 2026-09-21 | C rc=1: zero3's torch path is CUDA-hardwired (`torch.cuda.set_device`, streams, events — zero3.py:376 ff) and the plan selects it on mps — engine defect, rack side's domain (the triton branch of zero3 already uses the device abstraction). T rc=42 twice: floor stops at 3912 MB (9.6 GB ambient) and 4083 MB (11.8 GB ambient) — thirteen megabytes under the floor through TritonSequence.run(), **CONDITIONAL on the retention fix** (second instance of the Voxtral class; the loop stopped on this per the two-instances rule). S untried — same object. RE-READ UNDER DOCTRINE: neither cause is arithmetic — the zero3 selection defect is FIXED (red→green, pushed) and the triton stops are the Apple-side retention object (mine); verification re-runs when either the machine resets or the retention moves. Its cycle also exposed and fixed a real loader defect: the parallel extractor shared one ZipFile across 8 threads and raced (CRC-32 failures on the same member twice, single-reader clean; per-thread handles fix it — extract passed first try after). 8 keys demanded by the T attempts. Local copy deleted. |
| upscaler/hat-s-x4 | 51 MB | **REFUSED** (standing) | Blocked by the Prism estimate; stays refused rather than closed by invented demand. |
| upscaler/hat-l-x4 | 182 MB | **REFUSED** (standing) | Floor stop in both triton modes at 1567 MB available with 0 misses after its 8 shapes were certified — the machine genuinely cannot hold it. |

| video/CogVideoX-2b | 13.5 GB | **UNVERIFIED** | Extracted 2026-09-21; the compiled arm was stopped by a fallen ambient (10.4 GB, swap-saturated after 20 h of the old loop), which under doctrine is NOT a refusal — no arithmetic refuses it (largest resident component ~9 GB vs 24 GB unified). Verification owed after the machine resets. Local copy retained pending that. |

## Arithmetic re-reads of the standing refusals (doctrine, 2026-09-21)

**Superseded framing (owner, 2026-09-21):** "refusal STANDS" for a MEMORY reason
is no longer doctrine — NeuroBrix never refuses for lack of memory (see the fit
arithmetic below). hat-l-x4 / hat-s-x4 (floor stop / Prism estimate) become
SCHEDULED under tiling + weight streaming, not refused. canary-qwen and
Voxtral remain as written: canary is an ENGINE DEFECT (no allocation for a
component), not a memory refusal; Voxtral T/S is a CONDITIONAL on a retention
object, not memory. Re-verify all four under the adaptive-memory cascade.

- **hat-l-x4** — refusal STANDS, and it was never ambient: floor stop at
  1567 MB available with 0 misses after its 8 shapes were certified, in both
  triton modes, on the settled machine. The figure is the model's own
  working set against 24 GB unified minus the floor.
- **hat-s-x4** — refusal STANDS (Prism estimate blocks it); not ambient.
- **canary-qwen-2.5b** — refusal STANDS as an engine defect (Prism plans no
  allocation for `perception_encoder`); fits by arithmetic and runs the day
  the plan allocates it.
- **Voxtral-Mini-3B T/S** — CONDITIONAL, not arithmetic: the compiled arm
  fits and verified; the triton arms lose to the Apple-side retention
  object by ten megabytes. Re-verify when the retention moves.
- **real-esrgan-x8 @1024** — CONDITIONAL on the same object.

## RESOLVED: the upscaler "regression" was a STALE LOCAL CONTAINER (2026-09-21)

Not the engine, not the merge. real-esrgan-x2's graph is frozen at the trace
size (pixel-unshuffle H/2 frozen at 32 → the batch symbol inflates to 49 →
`final_as_array` keeps tile 0 → 128px). My engine diagnosis of the defect was
exact — but it is the defect the RETRACE fixed. The Dell retraced the
real-esrgan family on 2026-09-20 (x2 graph `74a2d7ea`, view::0 now
`floordiv(s1, 2)` — SYMBOLIC), and that container lives on the shared cache
(`Super-NeuroBrix-Cache` = 10.0.0.20:/nvme/neurobrix_cache). My local copy was
`626f2e07` — the OLD frozen graph, identical to the hub object because the store
stopped accepting writes, so the hub is behind every retrace.

PROVEN: refreshed x2 from the shared cache (74a2d7ea) → `neurobrix run
real-esrgan-x2 @448 --triton` → **896×896, rc=0**. The engine was never at
fault. ACTION: all verification copies come from the shared cache, never the
hub; the hub is stale for every retraced container. The real-esrgan family is
unblocked; swinir/swin2SR/hat were already symbolic (census harvests their
keys). No Dell datum needed — the correction was the owner's, checked.


## R29 verification of the fitting upscaler family — VIEWED 2026-09-21

All 8 fitting upscalers run from their RETRACED shared-cache containers, @448, rc=0,
**0 misses**, and each artefact was **looked at whole at full resolution** (R29 — std
alone only proves not-blank). Every one is a coherent red apple faithful to the input
(sharp forked stem, preserved yellow crown, smooth specular, natural lenticel-speckle
texture, rounded 3D shading, base dark spot), on a clean white ground, with **no grid,
no tile seam, no banding, no degeneracy**. None tiles at 448 (they fit untiled) so no
tile boundary is present — the across-a-seam look is owed at x8@1024.

| model | out | what I saw |
|---|---|---|
| real-esrgan-x2 | 896² | faithful apple, fine speckle |
| real-esrgan-x4 | 1792² | faithful, crisp |
| real-esrgan-x8 | 3584² | coherent at full res, GAN lenticel texture, sharp stem |
| swinir-classical-x2 | 896² | clean, faithful |
| swinir-classical-x4 | 1792² | clean, crisp white-pink speckle, radial crown streaks |
| swin2SR-classical-sr-x2-64 | 896² | faithful, soft speckle |
| swin2SR-classical-sr-x4-64 | 1792² | faithful, crisp |
| swin2SR-realworld-...-bsrgan-psnr | 1792² | coherent, deeper red, natural speckle |

## Catalogue fit arithmetic (VM off, 24 GB unified, ~22 GB usable) — 2026-09-21
Measured from each container's weight footprint on the hub. No downloads.

**DOCTRINE CORRECTION (owner, 2026-09-21):** NeuroBrix never refuses a model for
lack of memory — a Mac with 4 GB free must run any model, slowly if it must,
never refuse and never crash. So there is NO "refused by arithmetic" class. On a
unified device the Prism cascade must end in a strategy that streams weights
block by block from storage, so resident memory is ONE block rather than the
whole component. The 11 models below are therefore **SCHEDULED**, not refused:
they run by weight block-streaming, throughput bounded by the storage read
(local NVMe / shared cache, not the 7–9 MB/s hub link). Implementing that
cascade tail on unified is the adaptive-memory workstream (this branch; the
`return None` guard in `b23105fe` is the placeholder it replaces).

**Scheduled — weight block-streaming on unified (11, weights exceed 22 GB usable
so they cannot be resident; resident cost = one streamed block):**
- Allegro — 23.6 GB weights
- Flex.1-alpha — 24.5 GB weights
- Wan2.1-T2V-1.3B-Diffusers — 27.0 GB weights
- DeepSeek-Coder-V2-Lite-Instruct — 30.7 GB weights
- mochi-1-preview — 38.2 GB weights
- Open-Sora-v2 — 42.5 GB weights
- Qwen3-Coder-30B-A3B-Instruct — 57.1 GB weights
- Qwen3-30B-A3B-Thinking-2507 — 57.1 GB weights
- Qwen3-VL-30B-A3B-Thinking — 57.9 GB weights
- Wan2.1-I2V-14B-480P-Diffusers — 84.4 GB weights
- Wan2.2-I2V-A14B-Diffusers — 118.1 GB weights

**Batched for one VM-off verification session (12, 12–22 GB — fit only with the Parallels VM off; memory matters at verification, not certification):**
- Sana_1600M_4Kpx_BF16 — 12.1 GB
- CogVideoX-2b — 13.2 GB
- Janus-Pro-7B — 13.8 GB
- orpheus-3b-0.1-ft — 14.1 GB
- orpheus-3b-0.1-ft-snac — 14.2 GB
- granite-speech-3.3-8b — 16.1 GB
- Qwen3-Coder-30B-A3B-Instruct-int4g128-ffnonly — 17.2 GB
- Wan2.1-VACE-1.3B-diffusers — 18.3 GB
- GLM-4.1V-9B-Thinking — 19.2 GB
- PixArt-Sigma-XL-2-1024-MS — 20.3 GB
- PixArt-XL-2-1024-MS — 20.4 GB
- CogVideoX-5b-I2V — 21.5 GB

**Fit with the VM running (18 ≤12 GB weights):** the delivered set plus the frozen upscalers (retrace-queued) and the small TTS/LLM/STT — census + certify + verify these first.

## Census and certification state

**First-stage catalogue census — DONE 2026-09-21, from the CANONICAL shared cache**
(`Super-NeuroBrix-Cache`, not the stale hub), 30 fitting models, `apple_m4_pro`:

- **11,079 keys harvested** (kernel x EXACT-shape x dtype -- the key is exact today; the
  bucketing key change lives on the Dell's origin/bucketed-autotune-keys, NOT main), each
  model tagged with
  its `graph_sha` so a retrace invalidates exactly its own keys. Directory before:
  154 served → **10,925 to certify**.
- **13 OK** (harvest keys): the WHOLE upscaler family (real-esrgan x2/x4/x8, swin2SR
  x2/x4/realworld, swinir x2/x4, hat-l/s — all symbolic on the shared cache),
  TinyLlama (2560), parakeet (16), orpheus-snac (8400).
- **8 RETRACE** (frozen on the shared cache, handed to the Dell's Forge with graph_sha —
  `apple_retrace_queue.md`): GLM-4.1V, Janus-Pro-7B, Sana-4Kpx, VibeVoice, Wan-VACE,
  canary-qwen, chatterbox, granite-speech.
- **9 FAILED** (census request gaps / partial): CogVideoX×2 and PixArt×2 (need a
  resolution in the request), Kokoro/Voxtral/openaudio/orpheus-ft (one mode succeeded,
  one failed — partial), Qwen3-int4 (quantized path). To fix in the request map, not
  keys to sweep.

**CogVideoX/PixArt "FAILED" root-caused — a chain of engine defects, not request gaps
(2026-09-21):** the shadow was diverging from the real plan and dying, not missing a
request field. Each fix let the shadow reach further and harvest more keys.
- **Fix 1 — Prism host-offload on unified (`b23105fe`, red→green):** Strategy 4 placed
  the text_encoder on `cpu` to "save memory" on a unified device, where host==device so
  it frees nothing, and the Metal path then had nothing to trace. Root cause: asymmetric
  accounting — GPU checked against live-free (~10.7 GB, VM up), CPU against the profile's
  24 GB, on ONE physical pool. Same family as zero3-on-unified (`cafaf799`). Guarded:
  on unified, Strategy 4 returns None → refuse-by-arithmetic. Keys: CogVideoX 0→8,
  PixArt 0→5.
- **Fix 2 — NBX dtype read by name (`ac10eddf`, red→green):** Triton input synthesis
  read `str(val.nbx_dtype).split(".")[-1]`; NBXDtype is an IntEnum so this yielded the
  VALUE ("0"), and `np.dtype("0")` raised `data type '' not understood`. Read
  `.name` instead. Latent on ANY triton run that synthesizes an input. Keys:
  PixArt 5→16, CogVideoX 8→10.
- **WALL (open, doctrine call needed) — the census NaN gate blocks coverage:** the run
  now reaches the diffusion main loop and the always-on `_gate_loop_state_finite`
  aborts after step 1 (state NaN/Inf on SYNTHETIC inputs, no real weights). Per-step
  kernels repeat, so step 0 harvested the transformer's keys, but the abort skips the
  post-loop VAE decode → its keys are MISSED (a real census miss). The gate is right for
  a real run and wrong for a census shadow (synthetic garbage is expected). A real run
  can't confirm whether the NaN is synthetic-only: it dies EARLIER, at weight staging —
  a missing `fp32→bf16` conversion in `_load_to_pinned_cpu` (shape 4096×10240). Options
  for the owner: (a) skip/soften the NaN gate under `NBX_CENSUS` (census-mode-specific,
  like `metal_device.runtime()`), so the shadow harvests VAE-decode keys; or (b) census
  the VAE decode as a standalone component. NOT changed unilaterally — it is a
  safety-critical gate.

**Second stage, CORRECTED (2026-09-21):** the keys are EXACT, not bucketed. TinyLlama's
2,560 and orpheus's 8,400 are per-decode-step exact prompt/cache lengths -- the explosion
the buckets collapse (~6 for a short request). Certifying them now unserves them the day
the bucketed key lands on main. So:
- CERTIFIED NOW (request-INDEPENDENT, stable): 31 new keys (+41 served = 72) for the
  upscalers' and hat's convolutions/attention, deriving from the memory ladder and tile
  lattice the bucketing does not touch (conv2d 45, addmm 13, baddbmm 11, matmul 3).
- HELD (request-DEPENDENT, 10,976): every prefill/decode/audio-length family. Re-census
  under the bucketed form and certify ONCE when bucketed-autotune-keys reaches main.
- Request gaps (video/PixArt resolutions, partial audio) fixed in the request map
  meanwhile -- needed under either key form.

- Census basis today: the runtime replay cache (the census tool is being
  built once on main, hardware profile as input; the Apple inputs it needs
  are handed in `census_requirements_apple.md`). Replay-derived and
  delivery-log-derived censuses certified so far: 79 delivery-demand keys +
  2 drift-retries + the partial 430-key cold-demand pass = directory at
  1051 witnessed keys, all stamped `triton 3.8.0+git4a15f415 mps`,
  18 GB memory class.
- Certification under the doctrine runs on synthetic tensors with the VM
  up — the witness refuses perturbed sweeps (measured twice today: 21.2%
  and 16.0% drift refusals, both passed alone).
- Verification at zero miss doubles as census completeness; the merged
  lattice's new families (560-extent at 1024 px; 4 keys still owed) are the
  first case the graph census must catch that the replay census missed.

## Anomalies

- Nine hub entries are empty or unreadable directories (no model.nbx, du=0):
  image/Sana_1600M_1024px_MultiLing(+_diffusers), llm/deepseek-moe-16b-chat,
  multimodal/Ming-Lite-Omni-1.5, multimodal/MiniCPM-o-4_5,
  multimodal/Qwen3-Omni-30B-A3B-Instruct, stt/whisper-large-v3-turbo,
  video/Allegro-TI2V, video/SANA-Video_2B_720p_diffusers. A container that
  is not there can be neither run nor refused by arithmetic; these are
  listed as absent, not skipped.

## Link measurement

Hub-mount transfer measured 2026-09-21: 2663 MB in 340 s = **7 MB/s**
(owner's own figure 8.9 MB/s; no cable yet). Re-planned against 7-9 MB/s.
