# NeuroBrix — Benchmarks

*Every figure on this page is read from a dated result file in this
repository. Nothing is estimated, extrapolated, or rounded in our favour.
Where we are slower, the number is here.*

**Hardware for every measurement: 4 × NVIDIA Tesla V100 (2016-class silicon,
sm_70), SM clocks locked at 1290 MHz.** No A100, no H100, no consumer card. A
newer GPU changes every absolute number on this page and we have not had one.

---

## 1. Capacity — the axis that orders everything else

**45 models. Nine families. One engine, one container format, one command.**

| runtime | of the 45, how many start | what it cannot start |
|---|---:|---|
| **NeuroBrix** | **45** | — |
| ollama | 2–3 | every image, video, upscaler, TTS, STT, audio-LLM, VLM and omni model — no serving for those modalities at all |
| vLLM | **0 on this hardware** | vLLM ≥ 0.7 dropped V100 wheels; 0.7.3 from source refuses Qwen3-MoE on sm_70. No video, image, TTS or upscaler pipelines in any case |
| diffusers | 14 | every LLM, STT, TTS, audio-LLM and VLM; and Open-Sora-v2, for which no pipeline exists |
| faster-whisper | 2 | everything that is not Whisper |
| NeMo | 1 | everything that is not Parakeet |
| vendor `transformers` | ~20 | needs a bespoke environment per model class, and offers no unified serving |

**The claim is the conjunction, not any single row.** The competitor column is
a *union of six tools*, each covering its slice, none crossing families. To
match the left column you install six stacks, maintain six environments, and
still cannot serve them behind one interface.

Distribution state, verified on three surfaces (API, CLI, model page) on
**2026-09-04**: **45 of 45 public, zero deprecated.** The catalogue stood at
37/45 on 1 September, with eight models carrying a badge while their published
defect was fixed; each returned with its fix validated rather than by removing
the badge.

**Portability, measured 2026-09-04:** 867,042 operators across 182 graphs and
56 containers were audited for operators bound to a single vendor's library.
**Zero remain.** Every container is a neutral ATen identity.

---

## 2. Speed, by category

Each cell: our number, the legitimate competitor for that category, the date,
and the competitor's version or configuration. Medians. Interleaved arms.

### Text and code — *slots 2026-08-25 / 08-26*

| row | NeuroBrix | competitor | gap |
|---|---:|---:|---|
| Qwen3-Coder-30B int4, short | 39.8 tok/s | ollama q4_K_M **67.1** | ×1.69 |
| Qwen3-Coder-30B int4, 8.3k ctx | 26.6 tok/s | ollama **62.3** | ×2.34 |
| TinyLlama-1.1B fp16 | 105.6 tok/s | ollama **233.6**, vLLM 205.2 | ×2.2 |

*n = 5, locked 1290 MHz, interleaved.* **We are behind on text decode and say
so.** The number that matters is the movement: this same row was **×84** behind
on 2026-08-05 (0.09 tok/s against 7.56). It is ×1.69 today.

### Image diffusion — *slot 2026-09-05 13:09 (quiet host)*

| row | NeuroBrix (compiled) | vendor `diffusers` | gap |
|---|---:|---:|---|
| PixArt-Sigma 1024, 20 steps | **8.59 s** | 8.67 s | ×0.99 |
| PixArt-XL 1024, 20 steps | **8.31 s** | 8.34 s | ×1.00 |
| Sana-1600M 1024, 20 steps | 9.25 s | **4.17 s** | ×2.2 |

Seconds per image, median of five, warm engine on both arms, same seed, same
GPU, arms run back to back, both arms writing their PNG inside the measured
request (NeuroBrix at PNG level 1, diffusers at PIL's default 6 — both lossless);
every arm sha-identical across its five requests. The medians cross by 1 % and
0.4 %, inside the spread of either arm: parity with a thin edge, not a wide lead.

**Correction of the 08-30 image rows.** The 08-30 NeuroBrix numbers (11.44 s,
13.46 s) were produced with a step cache enabled on our arm and not on the
vendor's; that cache skips 12 of 20 steps and fails our own quality gate
(PSNR 9.8 dB against the cache-off render), so those rows are withdrawn. The
like-for-like 08-30 baseline was 31.35 s (PixArt-Sigma) and 30.25 s (PixArt-XL):
the engine ran the whole transformer in fp32 on a GPU whose fp32 rate is one
eighth of its tensor-core rate. Between 09-04 and 09-05 the compiled engine
gained a per-backbone precision contract (the vendor's fp16 with the vendor's
fp32 islands, recognised from the graph), and lost three copies per attention,
two full-size mask copies per cross-attention and a host stall at every
attention. The renders of every step are byte-identical between the CLI and
the served path; against our own fp32 render the timestep embedding is exact
to 2e-4 where the earlier tier had drifted by 0.27.

**Sana** stays fp32 on our side: its fp16 path is measured at 5.78 s but drifts
21 dB from our fp32 render (the vendor's own fp16 drifts 27 dB) and is refused
under our quality bound for now. **Flex.1-alpha** measures 36.9 s against the
vendor's 99.9 s (the 24 GB fp16 model is resident on one 32 GB card in
NeuroBrix, offloaded to the host in diffusers) but both arms render a blob at
the row's pinned recipe, so that row is not published. Method and every gate:
`validation_outputs/image_fp16_2026_09_04/LEVER.md`.

Our `--triton` engine is far slower here (65–78 s) for a **structural** reason
we can name: on sm_70, Triton does not lower fp16 matmul to the tensor-core
instruction, and the measured ceiling is ~12 % of cuBLAS. That is a property of
this GPU generation, not of the model.

### Upscalers — *slot 2026-08-29 23:21–23:34, six rows, three arms*

NeuroBrix compiled runs the Swin family between **parity and +21 %**:

| row | vendor | NeuroBrix compiled | |
|---|---:|---:|---|
| swinir-x4 | 2.016 s (±0.013) | 2.031 s (±0.006) | **1.007×** — parity |
| swinir-x2 | 1.984 s (±0.008) | **1.638 s** (±0.002) | **0.83×** — we are faster |
| hat-l, realworld | — | — | +10 % |
| swin2sr-x4 | — | — | +21 % |

**The family is not uniformly at parity** — two rows are at or better than the
vendor, three are 10–21 % behind. This is the only category where any row of
ours is faster than a specialised competitor.

### Speech-to-text — *slot 2026-09-05 13:20–13:55 (quiet host)*

Two definitions, both stated, because the two tools do not measure the same
thing. *Warm request*: the engine is loaded and has served once; one request
is timed end to end, including the transcript's return. *Cold execute*: a fresh
process, the transcription phase timed on its own — first-request costs
(kernel loading, buffers, plan compilation) inside, model loading outside;
this is the definition the earlier campaigns published.

| whisper-large-v3-turbo, jfk 11 s | NeuroBrix (compiled) | faster-whisper 1.2.1 fp16 | gap |
|---|---:|---:|---|
| warm request, seconds per clip | **0.27–0.30 s** (RTFx 36–40) | 0.44–0.80 s per call after load (RTFx 14–25; 0.48 s on 08-30) | we lead |
| cold execute, seconds per clip | 2.7 s (RTFx 4.0) | 0.59 s (RTFx 18.8) | ×4.6 behind |

Transcript shas are identical between the two tools on both test clips. The
cold gap is the first request's cost — lazy kernel loading, pinned buffers,
plan compilation — not transcription: the same engine's warm request is
9× faster than its own cold execute. What changed on 09-05: the compiled
engine runs the encoder and decoder at the vendor's half precision (warm
0.72 → 0.47 s), and the decoder keeps its keys and values across tokens in
both engines instead of recomputing the whole transcript at every token
(byte-identical transcripts; the `--triton` engine's 600 s transcription
drops from 1073 s to 45 s). Method and every gate:
`validation_outputs/audio_lot_2026_09_05/RESEARCH.md`.

Parakeet-TDT and the other speech rows keep their 08-30 numbers (RTFx
2.5–3.0 compiled against NeMo's 12–15); they have not been touched yet.

### Text-to-speech — *slot 2026-08-30 10:57–12:50, five rows*

| row | vendor | NeuroBrix compiled | NeuroBrix triton |
|---|---:|---:|---:|
| Kokoro-82M | **1.16 s** | 6.25 s | 11.08 s |
| VibeVoice-1.5B | — | 27.8 s | 78.5 s |
| OpenAudio-S1-mini | — | 36.8 s | 141.1 s |

Every retained audio file was transcribed by an **independent** STT engine and
had to contain the expected phrase. That probe caught two real defects a
stopwatch cannot hear, both since fixed.

### Audio-language models — *slot 2026-08-30 10:00–10:53*

RTFx 0.11–0.86, **NeuroBrix only**: no V100-capable competitor serves these
three as audio-LLMs. Cross-engine outputs are **byte-identical** per cell.

### Video — *re-run 2026-09-01 16:48–21:35, n = 3, dispersion ≤ 6 %*

**9 of 11 hub video models render on at least one NeuroBrix engine.** Each
vendor arm was given its own recipe (tiling, offload, dtype):

| row | vendor `diffusers` | NeuroBrix compiled | NeuroBrix triton |
|---|---:|---:|---:|
| CogVideoX-2b t2v | **8.9 s** (±1.1 %) | 45.4 (±0.7 %) | 231.2 (±0.1 %) |
| CogVideoX-5b i2v | **47.6 s** (±1.5 %) offloaded | 120.2 (±0.6 %) | 502.3 (±0.2 %) |
| Mochi t2v | **72.8 s** (±2.4 %) offloaded | 124.8 (±5.9 %) | 566.4 (±2.4 %) |
| Open-Sora-v2 t2v | *no pipeline exists* | **150.6** (±1.0 %) | — |
| SANA-Video t2v | *vendor version cascade — cannot run this config* | — | **39.5 s** (±2.3 %) |
| Wan-14B i2v | **179.4 s** (±0.9 %) sequential-offload | — | 557.5 (±0.5 %) **sharded across two GPUs** |

**On every row where the vendor runs, the vendor is faster** — by 2.5× to 5×.
Two standouts are ours: **SANA-Video renders in 39.5 s on a configuration the
vendor stack cannot run at all**, and Wan-14B runs sharded across two cards.
Two vendor non-starts are sourced, not assumed.

*An earlier revision of this table claimed our CogVideoX-5b matched the vendor,
on a figure that matched no recorded measurement. It was withdrawn by audit. The
recorded cell — the vendor 2.5× faster — is the one above.*

### Vision-language, multimodal, omni — *slot 2026-08-31 03:20–05:56*

VQA seconds per request (n = 5): MiniCPM-o-4.5 41.4 compiled / 85.6 triton;
GLM-4.1V-9B 54.6 / 130.0; Qwen3-VL-30B-A3B 501 / 1136 — all accuracy-gated.
Text-to-image (n = 3): **Janus-Pro-7B 205 compiled / 107 triton — the first row
where our Triton engine beats our PyTorch engine outright.**

No competitor arm exists for these graphs on this hardware; the non-starts are
sourced.

---

## 3. Limits, stated plainly

- **Text decode is ×1.69–2.34 behind ollama.** Closed from ×84; not closed.
- **Speech-to-text: ahead on a warm request, ×5 behind on a cold execute.**
  faster-whisper's 0.44–0.80 s per call after load against our 0.27–0.30 s warm;
  its 0.59 s against our 2.7 s when a fresh process pays kernel loading,
  buffers and plan compilation inside the timed phase. The cold cost is the
  open item, and it is a one-time cost per process, not transcription.
- **Image diffusion is at parity with diffusers on PixArt** (medians 1 % and
  0.4 % ahead, inside the spread) and **×2.2 behind on Sana**, whose fp16 path we
  refuse under our quality bound for now.
- **The `--triton` engine is much slower than `compiled` on image and
  upscaler work** — a structural sm_70 limit, measured at ~12 % of cuBLAS.
- **Time-to-first-token is ×3.9–4.1 behind ollama** at long contexts on the
  Triton engine. It is the widest open front and its chantier is open.
- **Every number here is 2016-era silicon.** We have no modern GPU and publish
  no number from one.

### Two things we measured and did *not* ship

**Making several cards work on one computation — measured, and rejected on this
hardware.** Coordinating four cards would cost about **half the time of every
word generated** (a cross-card exchange costs 124 µs; 96 are needed per word,
against a 25.1 ms word). The cost sits on the CPU issuing the work, not on the
cables between cards, so a faster GPU does not remove it. We then tested the one
remedy we had identified, and it took the cost from 47 % to 36.5 % — not enough.
The reduction code is written and kept; the feature is not shipped, and the
condition under which it would reopen is recorded.

**A record-and-replay path across several cards — built, and deliberately
locked.** The two capabilities it needs exist. It stays disabled because we have
not been able to *prove* that replaying across cards produces bit-identical
results, and the model that would prove it does not exist on this rig. A lock
that guards against a silently wrong answer is worth more than a capability
opened early.

---

## 4. Method

- **Clocks locked** at 1290 MHz for the whole campaign, not documented
  after the fact. The watchdog refuses a campaign that cannot hold the clock.
- **Interleaved arms** — ours, theirs, ours, theirs — so drift cannot favour
  one side.
- **Repetitions** stated per campaign (n = 5 for the text and VQA rows, n = 3
  for video and text-to-image), with dispersion recorded (≤ 6 % on video).
- **Outputs compared byte for byte** across our two engines wherever both run,
  and content-probed by an independent model for audio.
- **Competitors get their own recipes.** An earlier revision of our video table
  reported that eight vendor arms "cannot start". That was our harness
  withholding their per-model configurations. It was retracted and re-run.
- **Retractions are published.** A claim that cog5b "matched" a vendor arm
  rested on a figure matching no recorded measurement; it was withdrawn by
  audit and the true cell — the vendor 2.5× faster on that row — stands here.

Every cell traces to a dated annex under `benchmarks/results/`, and the
protocol harness (`benchmarks/harness/bench_row.py`) is in this repository.

*Compiled 2026-09-04 from the dated result files. Where a figure has moved
since its campaign, the campaign date is the one shown.*
