# The catalogue, one line per model — 2026-09-12

**47 entries on the registry.** Every cell is read from an artefact on this
machine, and every cell says how it was obtained. A cell that says *not
measured* is not an omission: a blank and a zero read the same, and only one
of them is honest.

The *where a defect would be invisible* column is the census taken **2026-09-16 17:42 UTC**.

The run column is the catalogue pass of **2026-09-11** at engine `4c119b5`
unless a later line overrides it, in which case the override names the
artefact that proves it. The pass's own record is never edited — it stays
what it was on the day it ran.

As the pass left it: **37 met**, **9 failed**, **1 not runnable**. 6 rows carry a later line, and every one of the nine failures was a VIDEO model.

| model | family | GB | on this rack | swept | screened | certified cost | certified for this card's memory | Apple M4 Pro | where a defect would be invisible | debts named | line |
|---|---|---:|---|---:|---:|---|---|---|---|---|---|
| `ibm-granite/Granite-Speech-3.3-8B` | audio_llm | 16.1 | met in 468 s | 117 | 0 | 1353 s, 20.37x, 264/264 keys, base 70 s, bytes same; 67 certified choice(s) contradicted by the runtime sweep (139 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 117/117 · 32 GB 117/117 · proven under triton 3.6.0 | not measured here | none found at the input | `D-AUDIO-LLM-GRANITE-HOST-PLACEMENT` | measured |
| `mistralai/Voxtral-Mini-3B` | audio_llm | 8.7 | met in 121 s | 28 | 0 | 1172 s, 31.23x, 254/254 keys, base 39 s, bytes same; 46 certified choice(s) contradicted by the runtime sweep (186 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 28/28 · 32 GB 28/28 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `nvidia/Canary-Qwen-2.5B` | audio_llm | 4.8 | met in 118 s | 29 | 0 | 1212 s, 54.55x, 251/251 keys, base 23 s, bytes same; 42 certified choice(s) contradicted by the runtime sweep (185 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 29/29 · 32 GB 29/29 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `NVlabs/Sana-1600M-4Kpx-BF16` | image | 12.1 | met in 460 s | 2 | 0 | 2082 s, 5.52x, 59/59 keys, base 460 s, bytes same; 2 certified choice(s) contradicted by the runtime sweep (22 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 2/2 · 32 GB 2/2 · proven under triton 3.6.0 | not measured here | vae `height`@128 (weight-extent); vae `width`@128 (weight-extent) | `D-PRISM-SANA4K-COMPILED-16GB` | measured |
| `NVlabs/Sana-1600M-MultiLing` | image | 12.1 | met in 90 s | 8 | 0 | 325 s, 6.95x, 58/58 keys, base 55 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (26 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 8/8 · 32 GB 8/8 · proven under triton 3.6.0 | not measured here | transformer `height`@32 (weight-extent); transformer `width`@32 (weight-extent) (+2) | none named | measured |
| `PixArt/PixArt-Sigma-XL-1024` | image | 20.3 | met in 154 s | 8 | 0 | 2471 s, 20.90x, 36/36 keys, base 124 s, bytes same | 16 GB 8/8 · 32 GB 8/8 · proven under triton 3.6.0 | not measured here | vae `height`@128 (weight-extent) → bound — spatial differential 2026-09-13 on PixArt-Sigma-XL-2-1024-MS, whose vae graph.json is byte-identical to this container's; vae `width`@128 (weight-extent) → bound — same run, same identical graph | none named | measured |
| `PixArt/PixArt-XL-1024` | image | 20.4 | met in 128 s | 4 | 0 | 2476 s, 21.26x, 36/36 keys, base 122 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (16 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 4/4 · 32 GB 4/4 · proven under triton 3.6.0 | not measured here | transformer `seq_len`@120 (weight-extent); transformer `seq_len`@120 (weight-extent) (+2) | none named | measured |
| `ostris/Flex.1-alpha` | image | 24.5 | met in 317 s | 15 | 0 | 2501 s, 9.24x, 42/42 keys, base 303 s, bytes same | 16 GB 15/15 · 32 GB 15/15 · proven under triton 3.6.0 | not measured here | text_encoder `seq_len`@77 (weight-extent) → bound — differential 2026-09-12 (77 vs 71: 0 dims moved); transformer `seq_len`@4096 (weight-extent) → UNADJUDICATED — the differential's override has no seam on a transformer traced inside a pipeline: both arms recorded trace_value=4096 again on 2026-09-13 16:05 (instrument defect named in the census; 'fixed by the model' on 2026-09-12 read the same silence as a fact) (+2) | none named | measured |
| `Qwen/Qwen3-30B-A3B-Thinking` | llm | 57.1 | met in 154 s | 0 | 0 | 56 s, 0.51x, 7/8 keys, bytes passed | 0 shapes met | not measured here | none found at the input | none named | measured |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | llm | 57.1 | met in 173 s | 1 | 0 | 54 s, 0.47x, 7/8 keys, bytes passed | 16 GB 1/1 · 32 GB 1/1 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-int4g128-ffnonly` | llm | 17.2 | met in 116 s | 0 | 0 | 46 s, 1.43x, 8/8 keys, base 107 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (4 near-ties within the timer's noise) — a finding, keys in the campaign record | 0 shapes met | not measured here | none found at the input | none named | measured |
| `TinyLlama/TinyLlama-1.1B-Chat` | llm | 2.1 | met in 25 s | 3 | 0 | 33 s, 4.18x, 6/6 keys, base 10 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (3 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 3/3 · 32 GB 3/3 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `deepseek-ai/DeepSeek-MoE-16B-Chat` | llm | 30.6 | met in 39 s | 0 | 0 | 46 s, 1.3x, 8/9 keys, bytes passed | 0 shapes met | not measured here | none found at the input | `D-DSMOE-XENGINE-SHA`, `D-TRACE-DEEPSEEK-MOE-ILLEGAL-ACCESS` | measured |
| `deepseek-ai/deepseek-coder-v2-lite-instruct` | llm | 30.7 | met in 140 s | 5 | 0 | 1337 s, 14.38x, 131/137 keys, bytes passed | 16 GB 5/5 · 32 GB 5/5 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `deepseek-ai/janus-pro-7b` | multimodal | 13.8 | met in 100 s | 3 | 0 | 193 s, 3.31x, 25/25 keys, base 84 s, bytes same | 16 GB 3/3 · 32 GB 3/3 · proven under triton 3.6.0 | not measured here | gen_embed `seq_len`@1 (arithmetic) | none named | measured |
| `inclusionai/ming-lite-omni-1.5` | multimodal | 53.0 | met in 175 s | 8 | 0 | 1072 s, 9.82x, 195/203 keys, bytes passed | 16 GB 8/8 · 32 GB 8/8 · proven under triton 3.6.0 | not measured here | image_vae `seq_len`@3 (weight-extent) | `D-DECLARED-MOE-AS-EXECUTED-VIEW` | measured |
| `openbmb/minicpm-o-4_5` | multimodal | 19.7 | met in 301 s | 81 | 0 | 1624 s, 26.98x, 286/287 keys, base 63 s, bytes same; 13 certified choice(s) contradicted by the runtime sweep (224 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 81/81 · 32 GB 81/81 · proven under triton 3.6.0 | not measured here | flow_dit `seq_len`@3 (weight-extent) | none named | measured |
| `qwen/qwen3-omni-30b-a3b-instruct` | multimodal | 65.9 | met in 257 s | 36 | 0 | 1196 s, 12.23x, 183/219 keys, bytes passed | 16 GB 36/36 · 32 GB 36/36 · proven under triton 3.6.0 | not measured here | none found at the input | `D-DEEPSTACK-ZERO-EXTENT`, `D-DECLARED-MOE-AS-EXECUTED-VIEW` | measured |
| `qwen/qwen3-vl-30b-a3b-thinking` | multimodal | 57.9 | met in 365 s | 36 | 0 | 2924 s, 13.87x, 515/585 keys, bytes passed | 16 GB 36/36 · 32 GB 36/36 · proven under triton 3.6.0 | not measured here | none found at the input | `D-DEEPSTACK-ZERO-EXTENT`, `D-QWEN3VL-MOE-RUNS-EVERY-EXPERT-ON-EVERY-TOKEN`, `D-DECLARED-MOE-AS-EXECUTED-VIEW` | measured |
| `nvidia/Parakeet-TDT-1.1B` | stt | 4.1 | met in 32 s | 4 | 0 | 78 s, 8.04x, 17/17 keys, base 11 s, bytes same | 16 GB 4/4 · 32 GB 4/4 · proven under triton 3.6.0 | not measured here | joint `seq_len`@1024 (weight-extent) | none named | measured |
| `openai/Whisper-Large-V2` | stt | 5.8 | met in 23 s | 0 | 0 | 35 s, 3.23x, 10/10 keys, base 16 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (7 near-ties within the timer's noise) — a finding, keys in the campaign record | 0 shapes met | not measured here | none found at the input | none named | **proven** — [transcript.txt](nbx/campaigns/2026_09_16_vitrine/whisper/transcript.txt), judged by the text known in advance (jfk_11s.expected.txt) and faster-whisper 1.2.1, a third-party ASR: word error rate 0.0 against the expected text (22 words, 0 edits); the third-party ASR returns the identical sentence ([verdict](nbx/campaigns/2026_09_16_vitrine/whisper/VERDICT.md)) |
| `openai/Whisper-V3-Turbo` | stt | 1.5 | met in 22 s | 5 | 0 | 35 s, 7.62x, 10/10 keys, base 5 s, bytes same | 16 GB 5/5 · 32 GB 5/5 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `canopylabs/Orpheus-3B` | tts | 14.2 | not runnable — catalogue decision | n/m | n/m | not measured | n/m | not measured here | none found at the input | `D-ORPHEUS-FT-VENDOR-CODEC`, `D-ORPHEUS-SEED-NOT-PINNED` | not measured |
| `fishaudio/OpenAudio-S1-Mini` | tts | 4.0 | met in 1045 s | 238 | 0 | paired cell arm A rc=0, arm B rc=-9 — no cost | 16 GB 238/238 · 32 GB 238/238 · proven under triton 3.6.0 | not measured here | codec.decoder `seq_len`@1 (arithmetic) | none named | measured |
| `hexgrad/Kokoro-82M` | tts | 0.4 | met in 55 s | 6 | 0 | 303 s, 50.72x, 54/54 keys, base 6 s, bytes same; 3 certified choice(s) contradicted by the runtime sweep (7 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 6/6 · 32 GB 6/6 · proven under triton 3.6.0 | not measured here | none found at the input | `D-CPU-COMPLEX-HALF-EXP`, `D-KOKORO-DECODER-PINNED-HOST-READ` | measured |
| `microsoft/VibeVoice-1.5B` | tts | 5.1 | met in 154 s | 15 | 0 | 245 s, 8.28x, 38/38 keys, base 34 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (21 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 15/15 · 32 GB 15/15 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `resemble-ai/Chatterbox` | tts | 2.1 | met in 73 s | 0 | 0 | paired cell arm A rc=0, arm B rc=-9 — no cost | 0 shapes met | not measured here | none found at the input | none named | measured |
| `JingyunLiang/SwinIR-Classical-x2` | upscaler | 0.1 | met in 7 s | 0 | 0 | 37 s, 8.86x, 11/11 keys, base 5 s, bytes same | 0 shapes met | not measured here | none found at the input | none named | measured |
| `JingyunLiang/SwinIR-Classical-x4` | upscaler | 0.1 | met in 9 s | 0 | 0 | 51 s, 11.80x, 12/12 keys, base 5 s, bytes same | 0 shapes met | not measured here | none found at the input | none named | measured |
| `XPixelGroup/HAT-L-x4` | upscaler | 0.2 | met in 16 s | 0 | 0 | 87 s, 7.23x, 18/18 keys, base 14 s, bytes same; 3 certified choice(s) contradicted by the runtime sweep (7 near-ties within the timer's noise) — a finding, keys in the campaign record | 0 shapes met | not measured here | none found at the input | none named | measured |
| `XPixelGroup/HAT-S-x4` | upscaler | 0.1 | met in 10 s | 0 | 0 | 83 s, 14.14x, 18/18 keys, base 6 s, bytes same; 1 certified choice(s) contradicted by the runtime sweep (7 near-ties within the timer's noise) — a finding, keys in the campaign record | 0 shapes met | not measured here | none found at the input | none named | measured |
| `caidas/Swin2SR-Classical-x2` | upscaler | 0.1 | met in 8 s | 0 | 0 | 222 s, 42.82x, 16/16 keys, base 5 s, bytes same | 0 shapes met | not measured here | none found at the input | none named | measured |
| `caidas/Swin2SR-Classical-x4` | upscaler | 0.1 | met in 44 s | 4 | 0 | 438 s, 79.44x, 17/17 keys, base 6 s, bytes same | 16 GB 4/4 · 32 GB 4/4 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `caidas/Swin2SR-RealWorld-x4` | upscaler | 0.1 | met in 8 s | 0 | 0 | 492 s, 80.69x, 17/17 keys, base 6 s, bytes same | 0 shapes met | not measured here | none found at the input | none named | measured |
| `xinntao/Real-ESRGAN-x4` | upscaler | 0.1 | met in 6 s | 0 | 0 | 455 s, 135.49x, 10/10 keys, base 3 s, bytes same | 0 shapes met | not measured here | none found at the input | none named | **proven** — [apple_x4.png](nbx/campaigns/2026_09_16_vitrine/real_esrgan/apple_x4.png), judged by looked at, beside the degeneracy facts and the correlation with the input's bicubic upscale: the input's scene at 1792x1792, sharp: correlation 0.998, std 104.65, 199 358 distinct colours — not white, not flat ([verdict](nbx/campaigns/2026_09_16_vitrine/real_esrgan/VERDICT.md)) |
| `Efficient-Large-Model/SANA-Video-2B-720p` | video | 17.1 | **met (catalogue pass) — paired cell CUT 2026-09-13 11:27, no certified cost** | 25 | 0 | not measured | 16 GB 25/25 · 32 GB 25/25 · proven under triton 3.6.0 | not measured here | transformer `time`@3 (weight-extent) | none named | measured |
| `THUDM/CogVideoX-2b` | video | 13.2 | met in 614 s | 5 | 0 | 403 s, 0.73x, 26/33 keys, bytes **ran, FAILED** | 16 GB 5/5 · 32 GB 5/5 · proven under triton 3.6.0 | not measured here | none found at the input | none named | measured |
| `THUDM/CogVideoX-5b-I2V` | video | 21.6 | **RUNS — corrected at the source, published, installed; run to completion (MEASURED: no artefact judged outside the engine, R29 hardened 2026-09-16)** | 0† | 0† | paired cell PERTURBED (the cell was MEASURED clean at 12:39 on card 3 (48/50 keys, base 103.22 s, sweep 226.99 s, 3.20x, bytes identical, 3 certified choices contradicted, 18 near-ties — quoted from the run log, the record itself was overwritten); the tail's duplicate run at 13:28 wrote into the same directory, found the first run's B_replay_r* caches and REPLAYED instead of sweeping (arm B swept 0), so the record now holds a cell that measured nothing) — no cost | 0 shapes met | not measured here | none found at the input | none named | measured |
| `Wan-AI/Wan2.1-I2V-14B-480P` | video | 84.4 | **INFERRED same debt as Wan2.1-VACE** | 3† | 0† | not measured | 16 GB 3/3 · 32 GB 3/3 · proven under triton 3.6.0 | not measured here | none found at the input | `D-TEMPORAL-UNROLL (inferred)` | inferred |
| `Wan-AI/Wan2.1-T2V-1.3B` | video | 27.0 | FAILED rc=-9 — killed at 2700 s | 11† | 0† | 26 s, 0.01x, 6/36 keys, bytes **not run** | 16 GB 11/11 · 32 GB 11/11 · proven under triton 3.6.0 | not measured here | none found at the input | `D-WAN-T2V-OOM-AT-5D-PAD`, `D-WAN-T2V-VAE-ACTIVATION-12GB` | measured |
| `Wan-AI/Wan2.1-VACE-1.3B` | video | 18.2 | **NAMED DEBT — not corrected, and no stimulus corrects it** | 0† | 0† | 127 s, 4.32x, 34/37 keys, bytes **not run** | 0 shapes met | not measured here | none found at the input | `D-TEMPORAL-UNROLL`, `D-WAN-VACE-BROADCAST-AT-DIV`, `D-NEGATIVE-ALLOCATION-SIZE-WAN-VACE`, `D-WAN-VACE-FRAME-TOKENS-FROZEN` | measured |
| `Wan-AI/Wan2.2-I2V-A14B` | video | 118.1 | **RUNS — compiled ran to completion at the default guidance (MEASURED, R29 hardened 2026-09-16); triton renders at cfg 1.0, does not fit one 32 GB card at batched CFG (Prism finding) — and a second line** | 0† | 0† | not measured | 0 shapes met | not measured here | none found at the input | `D-TEMPORAL-UNROLL`, `D-PRISM-WAN22-TRITON-ONE-CARD` | measured (topology) / inferred (unroll) |
| `genmo/Mochi-1-preview` | video | 38.2 | **RENDERS on triton at 9 frames (1 step, 118 s) — the CUDA 700 was three int32 index wraps past 2^31 elements, a defect of EVERY kernel for ANY model whose tensor exceeds two billion elements (the next family to meet it will not be called Mochi), fixed for the class at the kernels; at its default 84 frames the VAE decoder OOMs where Prism planned 3.2 GB (estimator debt)** | 0† | 0† | not measured | 0 shapes met | not measured here | transformer `seq_len`@256 (weight-extent); transformer `seq_len`@256 (weight-extent) | `D-MOCHI-CUDA-700-AT-MM` | measured |
| `hpcai-tech/Open-Sora-v2` | video | 42.5 | **RE-TRACED, REBUILT, PUBLISHED, INSTALLED, ran to completion (triton, 9 frames, 2026-09-13 15:33) — MEASURED: no artefact judged outside the engine (R29 hardened 2026-09-16)** | 0† | 0† | not measured | 0 shapes met | not measured here | text_encoder_2 `seq_len`@77 (weight-extent) | none named | measured |
| `rhymes-ai/Allegro` | video | 23.6 | FAILED rc=-9 — killed at 2701 s | 12† | 0† | not measured | 16 GB 12/12 · 32 GB 12/12 · proven under triton 3.6.0 | not measured here | none found at the input | `D-ALLEGRO-TRITON-31H-PER-ARM`, `D-VIDEO-CAMPAIGN-STIMULUS` | measured |
| `rhymes-ai/Allegro-TI2V` | video | 24.3 | **RUNS — repaired and delivered** | 27† | 0† | not measured | 16 GB 27/27 · 32 GB 27/27 · proven under triton 3.6.0 | not measured here | none found at the input | `D2 (88 frames: declared limit until cuDNN >= 9.3)`, `D-ALLEGRO-TI2V-FRAME-TOKENS-FROZEN` | measured |
| `zai-org/GLM-4.1V-9B-Thinking` | vlm | 19.2 | met in 913 s | 259 | 0 | 2367 s, 21.14x, 523/523 keys, base 117 s, bytes same; 56 certified choice(s) contradicted by the runtime sweep (423 near-ties within the timer's noise) — a finding, keys in the campaign record | 16 GB 259/259 · 32 GB 259/259 · proven under triton 3.6.0 | not measured here | none found at the input | `D-PRISM-2x16-PIPELINE-OVERFILL` | measured |

## The lines that carry a later verdict

### `Allegro-TI2V` — RUNS — repaired and delivered

Two defects, both fixed at the source: the output size was never read from the container when the backbone's latent is flattened or the flow is named something else, and the conditioning image did not set the resolution. Bounded above: renders to 80 frames, fails at its own declared 88 asking 25.27 GiB in one allocation — decided 2026-09-13: PyTorch's native 3-D conv fallback buffer, taken because cuDNN 9.1 refuses a large non-batch-splittable convolution (needs >= 9.3); a workspace cap and the V8 flag change nothing. 88 is a declared limit on this stack until cuDNN >= 9.3 or a per-tile conv bound lands (DETTE D2).

*Evidence:* validation_outputs/allegro_image_sets_resolution_20260912/out.mp4 (8 frames at 448x448, rc=0, inter-frame diff 23.3); hub replaced 15:13:48; docs/reference/catalogue-repairs.md entry 1  ·  *line:* measured

### `CogVideoX-5b-I2V` — RUNS — corrected at the source, published, installed; run to completion (MEASURED: no artefact judged outside the engine, R29 hardened 2026-09-16)

Its causal temporal pad recorded 2187*s - 2184 against a truth of s + 2, exact at the traced s=1. Profiled at 49 frames the peak was 210.26 GB against 0.09 GB at the trace, a factor of 2237 which is the compound's own coefficient. Re-traced, the temporal axis carries no symbol at all (an I2V encoder conditions on one image) and the same request costs 0.02 GB. 389 -> 265 ops.

*Evidence:* hub record THUDM/CogVideoX-5b-I2V fileSize 23126413914, updatedAt 2026-09-12T22:09:39Z (replace through the internal entry point, 2498 s); installed manifest 22:10:26 UTC; the installed vae_encoder/graph.json holds 265 ops with symbols batch/height/width and NO temporal symbol; regression gate passed component by component (vae_encoder 0.82 -> 0.80 GB, every other component 1.000x); proof by run 22:45 UTC: 9 frames at 448x448, range 9-253, inter-frame diff 3.34 (validation_outputs/proof_by_run_CogVideoX-5b-I2V_20260912_2242/VERDICT.json)  ·  *line:* measured

### `Open-Sora-v2` — RE-TRACED, REBUILT, PUBLISHED, INSTALLED, ran to completion (triton, 9 frames, 2026-09-13 15:33) — MEASURED: no artefact judged outside the engine (R29 hardened 2026-09-16)

The snapshot's arrangement (model_index.json, component dirs, safetensors T5 shards beside the vendor's .bin) is rebuilt from declarations and documented beside the weights. The June container's 11 017-op VAE was an unrolled trace; the new one is flat in T.

*Evidence:* re-trace on the fifth attempt of 2026-09-13 (12:50, unpinned): transformer 5089 ops, vae 237 (June: 11 017 — an unrolled trace; the new graph is FLAT in T, 237 ops at T=9 and T=25, measured on card 0), text encoders 1594/490; rebuild 682 s on the export; regression gate 1.000x on three components, vae 0.959x opened with --allow-shrink on a measurement (248 tensors identical, the graph shrank); upload 13:08:54 -> rc=0 after 1922 s, hub updatedAt 13:40:55Z; install 13:40:56 -> rc=0 after 123 s, 53 files, 42.47 GB; proof by run 15:33: 9 frames at 112x176, range 0-202, inter-frame difference 18.6, rc=0 after 648 s, unpinned; docs/reference/catalogue-repairs.md entry 4  ·  *line:* measured

### `SANA-Video_2B_720p_diffusers` — met (catalogue pass) — paired cell CUT 2026-09-13 11:27, no certified cost

A sweep of video conv keys at 720p costs minutes a key; the 90-minute run timeout that fits every other family cuts this one (and chatterbox's and openaudio's 674/693-key sweeps). Re-measuring needs a per-cell timeout sized by keys — a decision, not tonight's.

*Evidence:* night bench card 3: arm A rc=0; arm B (sweeping, 25 video conv keys at 720p) killed at the campaign's 5400 s run timeout on repetition 0 and cut by hand at 53 min into repetition 1 — 19 keys swept in 45 min, a sweep this cell cannot finish under that clock; the cell was stopped so the night's queue (proofs, Open-Sora, budget gate) could take the rig  ·  *line:* measured

### `Wan2.1-I2V-14B-480P` — INFERRED same debt as Wan2.1-VACE

No local snapshot, so no second trace point and no fitted slope. The claim rests on six module groups agreeing exactly on two numbers each, while the groups ABOVE the loop differ — different VAE sizes carrying the same loop. One snapshot and two 15-second traces convert it.

*Evidence:* its cached encoder carries the measured anchor's chunk-loop groups identically: {2:1, 3:9, 6:12, 8:22, 12:1, 21:22}  ·  *line:* inferred

### `Wan2.1-VACE-1.3B` — NAMED DEBT — not corrected, and no stimulus corrects it

Its VAE encoder unrolls its temporal chunk loop: 517 ops per chunk, measured at two stimuli. The blind count held at 135 through T=17, 25, 33 and 41 while the door's candidate walked 25 -> 33 -> 41. The graph is not symbolic in time however many symbols its table declares.

*Evidence:* validation_outputs/wan_class_e_20260912/VERDICT.md; docs/reference/temporal-unroll-census.md  ·  *line:* measured

### `Wan2.2-I2V-A14B` — RUNS — compiled ran to completion at the default guidance (MEASURED, R29 hardened 2026-09-16); triton renders at cfg 1.0, does not fit one 32 GB card at batched CFG (Prism finding) — and a second line

TWO lines, not one. The rebuild resolves its output size. Its VAE ENCODER stays unrolled over the temporal axis, which no rebuild changes — DETTE D-TEMPORAL-UNROLL. Delivering the first without saying the second would be delivering a fix for the error we found and hiding the one underneath.

*Evidence:* rebuild 22:11-22:22 (676 s, 118.07 GB); regression gate 1.000x on all five components; upload through the internal entry point 22:22:46 -> rc=0 after 5144 s (126.77 GB, ~24.6 MB/s mean, zero SlowDownWrite), hub updatedAt 23:48:30Z; install 23:48:30 -> rc=0 after 312 s, five components in the cache; the proof needs the whole rig and runs after proof 2026-09-13: compiled 9 frames 448x448, diff 3.91, PASSED; triton at cfg 1.0 renders (diff 1.01, byte-identical on both engines); at default CFG triton reaches 31 327 MB on the one 32 GB card Prism chose and OOMs at the first attention (1.77 GB scores), Prism refusing component_placement and weight_sharding on a 96 GB rig — DETTE D-PRISM-WAN22-TRITON-ONE-CARD; VAE encoder UNROLLED, MEASURED 2026-09-13 04:44 on card 0: 1237 ops at T=9, 3305 at T=25 -> 517 ops per chunk (second line)  ·  *line:* measured (topology) / inferred (unroll)

### `mochi-1-preview` — RENDERS on triton at 9 frames (1 step, 118 s) — the CUDA 700 was three int32 index wraps past 2^31 elements, a defect of EVERY kernel for ANY model whose tensor exceeds two billion elements (the next family to meet it will not be called Mochi), fixed for the class at the kernels; at its default 84 frames the VAE decoder OOMs where Prism planned 3.2 GB (estimator debt)

The 9-frame run does not cross 2^31 elements itself (114 480 x 2048 rows); the wraps are proven by the three boundary tests and by the 84-frame op-by-op run that now passes mm::1, add::14 and group_norm::26. The catalogue request (84 frames) waits on Prism's estimate carrying the runtime frame count.

*Evidence:* two compute-sanitizer runs (7 200 s 09-13, 18 000 s 09-14) measured nothing; --triton-sequential + CUDA_LAUNCH_BLOCKING=1 named aten.mm::1 of the VAE in 19 min (M=1 068 480 x N=2048: 2.19e9 output elements, stride_cm * offs_cm wrapped in int32 — triton-lang/triton#832); then aten.add::14, then aten.native_group_norm::26, the same wrap in the flat and tile forms — every GEMM offset, every flat offset and every program id now 64-bit (90fefd4, 7ee3d7c, aa60c5c; register 58; beyond-2^31 tests red then green; four models byte-identical, timings within 3 %). With the wraps gone the op-by-op run reaches the decoder and OOMs at aten.silu::26: 8.75 GB asked, 25.97 GB live, 5.6 GB free on a 32 GB card, where --explain-plan says vae activations 3 209 MB, tiling none planned (D-PRISM-MOCHI-VAE-ACTIVATION-UNDERESTIMATED). Bounded proof by run 2026-09-14 09:11: --triton --steps 1 --num-frames 9, rc=0 in 118 s, 7 decoded frames at 480x848, range 0-154, inter-frame difference 1.3-2.6, a warm field with a red centre (one step), nbx/campaigns/2026_09_12_night_catalogue/mochi_proof_9frames_aa60c5c/  ·  *line:* measured

## How to read the columns

**line** — the last column is the verdict on the line itself, and since
2026-09-16 it distinguishes two things that were being written as one.
*proven* means an ARTEFACT of a real request — not a trace stimulus — was
judged by an instrument OUTSIDE this engine, and the cell links the file and
the written verdict: a transcription against a text known in advance and a
third-party ASR, a synthesised voice read back by that ASR, code that was
executed, an image or a sequence of frames looked at. *measured* is
everything else that rests on an artefact of this machine: a wall clock, a
shape count, a PSNR — and every AGREEMENT between two arms of this engine.
Bytes identical between the Triton and the PyTorch path say the two paths
agree; a graph broken upstream breaks both the same way and the matrix still
reads *identical*. Lines that said "proven by run" before that date and rest
on a run or an agreement were demoted here to *measured* and keep their
evidence; each returns to *proven* when its artefact is produced and judged
(`nbx/campaigns/2026_09_16_vitrine`).

**debts named** — the entries of `DETTE.md` that hold this line (the A rows of
`docs/reference/debts-triage.md`), so a reader of the line sees what it waits on
without opening the debt file. A line that runs and measures may still name one:
a debt that bounds it (frames, a card class) rather than blocks it.

**certified for this card's memory** — of the shapes this model's catalogue
run met (its own log, directory off), how many the directory certifies for a
16 GB card and how many for a 32 GB card, read on the day this document was
rendered. Since 2026-09-13 an entry serves only the memory class it was
proven on (register 56): a shape proven on a 16 GB card sweeps at runtime on a
32 GB card until it is certified there, and a shape proven on the rig with the
card unknown serves no card until re-proven. The two numbers are what a
request on each SKU of this rack is served without a sweep — not what the
directory holds. *Proven under* names the code generator (the Triton version)
each served proof was made with, so a rank reads with its date: a setting stays
correct under any generator — the fp64 oracle proved the SOURCE, not the
compiler — and what a Triton upgrade may age is its rank as the fastest, by a few
percent. A re-proof under a new generator is an optimisation pass on this rack,
incremental, checkpointed, invisible to a request; the old proofs serve meanwhile
(owner, 2026-09-16).

**Apple M4 Pro** — every cell says *not measured here*, and that is the
whole truth of this rack: it has no Apple device, and a Mac's shape keys are
not this rack's (the dtype policy differs), so nothing is inferred from the
Volta census either. What the trunk carries for Apple since the Mac branch
merged is the certified directory the Mac itself wrote, read from the live tree:
`apple_m4_pro`: **137 shapes** (addmm_kernel.fp32 33, baddbmm_kernel.fp32 81, matmul_kernel.fp16 7, matmul_kernel.fp32 16), proofs 2026-09-11..2026-09-12, platform `macOS-26.6.2-arm64-arm-64bit`. A model's Apple line is measured on the machine
that carries the card, by its own matrix runner (`tools/apple_matrix*.py`), and
lands here as a row when it does.

**swept** — shape keys this model had to sweep AT RUNTIME because the
certified directory did not hold them. On a row that MET, `0` is the
per-model measure of certified coverage: it was served entirely from the
directory. `n/m` is a model that was never run.

**† marks a row whose run did not complete**, and it changes what the two
columns mean there. They count what the run REACHED, and a run that died
in five seconds reached nothing — so a `0†` is not coverage, it is the
shape of the failure. Reading it as coverage would credit the directory
for work no one asked it to do.

**screened** — candidate configurations the correctness screen excluded
before timing. Zero across the whole catalogue, on 998 keys.

**certified cost** — from the paired certified-directory campaigns: the
hand-kept table of 2026-09-11, then every campaign listed in `CAMPAIGNS`
read straight from its cells through the table's own arithmetic. A night
cell also carries its base time (arm A median) and its byte gate — `same`,
`N dB`, `DIFFER`, `nondet both` or `did not run`; a cell whose arm failed
or whose lever did not move says so and carries no cost. The 2026-09-12
night ran one model per card with the other cards busy: its numbers are
comparable among themselves, not with a cell that had the rig alone.
The rest of the paragraph describes the 2026-09-11 table, which
covers eleven cells and not the catalogue. The ratio is what runtime
sweeping costs relative to a served run, on this rack, at the shapes these
requests meet. It is not a throughput figure and it says nothing about
other hardware.

**What a sweep costs per shape, measured tonight (sweep cost / keys swept,
per family, from the cells above with both arms at rc=0 and not perturbed):**

| family | cells | s per shape (min – max) | keys per cell (min – max) |
|---|---:|---|---|
| audio_llm | 3 | 5 – 5 | 251 – 264 |
| image | 6 | 6 – 69 | 36 – 59 |
| llm | 2 | 6 – 6 | 6 – 8 |
| multimodal | 2 | 6 – 8 | 25 – 287 |
| stt | 3 | 3 – 5 | 10 – 17 |
| tts | 2 | 6 – 6 | 38 – 54 |
| upscaler | 8 | 3 – 45 | 10 – 18 |
| vlm | 1 | 5 – 5 | 523 – 523 |

The 2026-09-11 table quoted 3–12 s a shape on GEMM-class keys. Tonight the
spread runs from 3 s a shape (`swinir-classical-x2`) to
69 s (`PixArt-XL-1024`, conv2d shapes at 448² screened
against the fp64 oracle) — so a 122 s run of the latter pays
2476 s of sweep. The per-shape cost is a property of the kernel
class and the shape, not a constant; the law (shapes met per second of
served run) holds with that coefficient per cell, not a single one.

**where a defect would be invisible** — axes traced at a value where two
distinct rules give the same number, so the trace-point check cannot tell
them apart. This is NOT a defect list. An axis here needs its rule asserted
structurally or a re-trace outside the collision; a test at the flagged
value is green for the reason that blinds it.

## Verdicts that cut across the lines

* **The strategy change moves no byte** (budget-unified gate, 2026-09-13 15:35-16:03, after-arm rebuilt on the trunk at run time): five pinned pairs whose Prism strategy changes between the arms — PixArt-XL-1024, PixArt-XL-2-1024-MS, PixArt-Sigma-XL-1024, PixArt-Sigma-XL-2-1024-MS on a 16 GB card, Flex.1-alpha on a 32 GB card — rendered byte-identical images on both arms, three cold repetitions each, triton. The byte matrix over the whole catalogue ran 2026-09-14 09:11-15:12 (register 52's re-arm): 30 cells identical, 0 adjudicated differences, 2 unadjudicated on the two models whose own nondeterminism is on record — orpheus-3b-0.1-ft (its sampler draws off the executor's RNG stream, D-ORPHEUS-SEED-NOT-PINNED) and CogVideoX-2b (differs run to run on both engines with no RNG op in its graph; cause not yet named, D-COGVIDEOX-2B-NONDETERMINISTIC-PER-RUN) — and 16 cells unmeasurable on this pair because its before tree (5ca23b1) cannot load today's containers. **Re-posed on 2026-09-16** (before 46479ae, the loader fix; after the merged trunk 291c3b8; two 16 GB cards, triton, cold pairs): GLM-4.1V-9B-Thinking, Janus-Pro-7B and MiniCPM-o-4_5 IDENTICAL (833eb058f3ae, 9a30d277cf21, 35d48303f4e7; each faster on the trunk, 143→127 s, 93→87 s, 59→49 s). Still unmeasurable, said by name: Qwen3-VL-30B-A3B-Thinking and Qwen3-Omni-30B-A3B-Instruct (both arms refuse at the deepstack zero-length bind until their retrace), Ming-Lite-Omni-1.5 triton (the before arm refuses at a freed arena — the defect the trunk fixed today, so the pair has no before). Determinism per mode is a public claim: the two nondeterministic models are its named exceptions until their causes are. `nbx/campaigns/prepared/budget_unified_gate_20260913_1535/RUN.md`, `nbx/campaigns/2026_09_16_converge/matrix_repose/`.
* **The engine suite on the trunk** (`pytest tests/unit tests/regression`, 2026-09-13 13:28-15:20, 1 h 52): 2100 passed, 21 failed. Ten of the 21 were one defect in the triton weight loader's consumed-weight filter (register 50, fixed the same afternoon, measured by run on three cells), one a GPU-less host planned on a GPU (register 51, fixed), eight out-of-memory against a foreign process on the cards, two Qwen3-Omni triton cells to re-read after the fix. The 21 are re-run from a worktree frozen at `8a92312` on a quiet rig; the verdict line is written here when it exists, not before. `nbx/logs/full_suite_night.log`, `nbx/logs/suite_rerun_21.log`.

## What this document does not say

* **Whether a model is CORRECT.** The run column says it produced output
  without failing, on one request, at one moment. Numerical agreement
  against a vendor pipeline is a different instrument and covers five of
  nine families.
* **What every model costs with the certified directory.** 39 of 47
  rows carry a measured or stated cost; the other 8 say *not
  measured* and that is the whole point of the cell.
* **Whether the flagged axes are wrong.** They are places a defect could
  not be seen. Converting one into a verdict costs a second trace at a
  value outside the collision, and the instrument that does it refuses when
  the stimulus does not actually move — a tree compared with itself agrees
  with itself.
* **Anything about hardware other than this rack**: four V100s, two of 16 GB
  and two of 32, at 1290/877 MHz.

