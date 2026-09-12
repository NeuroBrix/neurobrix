# The catalogue, one line per model — 2026-09-12

**47 entries on the registry.** Every cell is read from an artefact on this
machine, and every cell says how it was obtained. A cell that says *not
measured* is not an omission: a blank and a zero read the same, and only one
of them is honest.

The *where a defect would be invisible* column is the census taken **2026-09-12 23:11 UTC** (rendered from its cache: the export was carrying one stream, and the rule is one at a time).

The run column is the catalogue pass of **2026-09-11** at engine `4c119b5`
unless a later line overrides it, in which case the override names the
artefact that proves it. The pass's own record is never edited — it stays
what it was on the day it ran.

As the pass left it: **37 met**, **9 failed**, **1 not runnable**. 6 rows carry a later line, and every one of the nine failures was a VIDEO model.

| model | family | GB | on this rack | swept | screened | certified cost | where a defect would be invisible | line |
|---|---|---:|---|---:|---:|---|---|---|
| `ibm-granite/Granite-Speech-3.3-8B` | audio_llm | 16.1 | met in 468 s | 117 | 0 | not measured | none found at the input | measured |
| `mistralai/Voxtral-Mini-3B` | audio_llm | 8.7 | met in 121 s | 28 | 0 | not measured | none found at the input | measured |
| `nvidia/Canary-Qwen-2.5B` | audio_llm | 4.8 | met in 118 s | 29 | 0 | not measured | none found at the input | measured |
| `NVlabs/Sana-1600M-4Kpx-BF16` | image | 12.1 | met in 460 s | 2 | 0 | not measured | vae `height`@128 (weight-extent); vae `width`@128 (weight-extent) | measured |
| `NVlabs/Sana-1600M-MultiLing` | image | 12.1 | met in 90 s | 8 | 0 | not measured | transformer `height`@32 (weight-extent); transformer `width`@32 (weight-extent) (+2) | measured |
| `PixArt/PixArt-Sigma-XL-1024` | image | 20.3 | met in 154 s | 8 | 0 | not measured | vae `height`@128 (weight-extent); vae `width`@128 (weight-extent) | measured |
| `PixArt/PixArt-XL-1024` | image | 20.4 | met in 128 s | 4 | 0 | not measured | transformer `seq_len`@120 (weight-extent); transformer `seq_len`@120 (weight-extent) (+2) | measured |
| `ostris/Flex.1-alpha` | image | 24.5 | met in 317 s | 15 | 0 | not measured | text_encoder `seq_len`@77 (weight-extent); transformer `seq_len`@4096 (weight-extent) (+2) | measured |
| `Qwen/Qwen3-30B-A3B-Thinking` | llm | 57.1 | met in 154 s | 0 | 0 | 56 s, 0.51x, 7/8 keys, bytes passed | none found at the input | measured |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | llm | 57.1 | met in 173 s | 1 | 0 | 54 s, 0.47x, 7/8 keys, bytes passed | none found at the input | measured |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-int4g128-ffnonly` | llm | 17.2 | met in 116 s | 0 | 0 | not measured | none found at the input | measured |
| `TinyLlama/TinyLlama-1.1B-Chat` | llm | 2.1 | met in 25 s | 3 | 0 | not measured | none found at the input | measured |
| `deepseek-ai/DeepSeek-MoE-16B-Chat` | llm | 30.6 | met in 39 s | 0 | 0 | 46 s, 1.3x, 8/9 keys, bytes passed | none found at the input | measured |
| `deepseek-ai/deepseek-coder-v2-lite-instruct` | llm | 30.7 | met in 140 s | 5 | 0 | 1337 s, 14.38x, 131/137 keys, bytes passed | none found at the input | measured |
| `deepseek-ai/janus-pro-7b` | multimodal | 13.8 | met in 100 s | 3 | 0 | not measured | gen_embed `seq_len`@1 (arithmetic) | measured |
| `inclusionai/ming-lite-omni-1.5` | multimodal | 53.0 | met in 175 s | 8 | 0 | 1072 s, 9.82x, 195/203 keys, bytes passed | image_vae `seq_len`@3 (weight-extent) | measured |
| `openbmb/minicpm-o-4_5` | multimodal | 19.7 | met in 301 s | 81 | 0 | not measured | flow_dit `seq_len`@3 (weight-extent) | measured |
| `qwen/qwen3-omni-30b-a3b-instruct` | multimodal | 65.9 | met in 257 s | 36 | 0 | 1196 s, 12.23x, 183/219 keys, bytes passed | talker.code_predictor.model.codec_embedding `seq_len`@1 (arithmetic) | measured |
| `qwen/qwen3-vl-30b-a3b-thinking` | multimodal | 57.9 | met in 365 s | 36 | 0 | 2924 s, 13.87x, 515/585 keys, bytes passed | none found at the input | measured |
| `nvidia/Parakeet-TDT-1.1B` | stt | 4.1 | met in 32 s | 4 | 0 | not measured | joint `seq_len`@1024 (weight-extent) | measured |
| `openai/Whisper-Large-V2` | stt | 5.8 | met in 23 s | 0 | 0 | not measured | none found at the input | measured |
| `openai/Whisper-V3-Turbo` | stt | 1.5 | met in 22 s | 5 | 0 | not measured | none found at the input | measured |
| `canopylabs/Orpheus-3B` | tts | 14.2 | not runnable — catalogue decision | n/m | n/m | not measured | none found at the input | not measured |
| `fishaudio/OpenAudio-S1-Mini` | tts | 4.0 | met in 1045 s | 238 | 0 | not measured | codec.decoder `seq_len`@1 (arithmetic) | measured |
| `hexgrad/Kokoro-82M` | tts | 0.4 | met in 55 s | 6 | 0 | not measured | none found at the input | measured |
| `microsoft/VibeVoice-1.5B` | tts | 5.1 | met in 154 s | 15 | 0 | not measured | none found at the input | measured |
| `resemble-ai/Chatterbox` | tts | 2.1 | met in 73 s | 0 | 0 | not measured | none found at the input | measured |
| `JingyunLiang/SwinIR-Classical-x2` | upscaler | 0.1 | met in 7 s | 0 | 0 | not measured | none found at the input | measured |
| `JingyunLiang/SwinIR-Classical-x4` | upscaler | 0.1 | met in 9 s | 0 | 0 | not measured | none found at the input | measured |
| `XPixelGroup/HAT-L-x4` | upscaler | 0.2 | met in 16 s | 0 | 0 | not measured | none found at the input | measured |
| `XPixelGroup/HAT-S-x4` | upscaler | 0.1 | met in 10 s | 0 | 0 | not measured | none found at the input | measured |
| `caidas/Swin2SR-Classical-x2` | upscaler | 0.1 | met in 8 s | 0 | 0 | not measured | none found at the input | measured |
| `caidas/Swin2SR-Classical-x4` | upscaler | 0.1 | met in 44 s | 4 | 0 | not measured | none found at the input | measured |
| `caidas/Swin2SR-RealWorld-x4` | upscaler | 0.1 | met in 8 s | 0 | 0 | not measured | none found at the input | measured |
| `xinntao/Real-ESRGAN-x4` | upscaler | 0.1 | met in 6 s | 0 | 0 | not measured | none found at the input | measured |
| `Efficient-Large-Model/SANA-Video-2B-720p` | video | 17.1 | met in 675 s | 25 | 0 | not measured | transformer `time`@3 (weight-extent) | measured |
| `THUDM/CogVideoX-2b` | video | 13.2 | met in 614 s | 5 | 0 | 403 s, 0.73x, 26/33 keys, bytes **ran, FAILED** | none found at the input | measured |
| `THUDM/CogVideoX-5b-I2V` | video | 21.6 | **RUNS — corrected at the source, published, installed, PROVEN by run** | 0† | 0† | not measured | none found at the input | measured |
| `Wan-AI/Wan2.1-I2V-14B-480P` | video | 84.4 | **INFERRED same debt as Wan2.1-VACE** | 3† | 0† | not measured | none found at the input | inferred |
| `Wan-AI/Wan2.1-T2V-1.3B` | video | 27.0 | FAILED rc=-9 — killed at 2700 s | 11† | 0† | 26 s, 0.01x, 6/36 keys, bytes **not run** | none found at the input | measured |
| `Wan-AI/Wan2.1-VACE-1.3B` | video | 18.2 | **NAMED DEBT — not corrected, and no stimulus corrects it** | 0† | 0† | 127 s, 4.32x, 34/37 keys, bytes **not run** | none found at the input | measured |
| `Wan-AI/Wan2.2-I2V-A14B` | video | 118.1 | **REBUILT (118.07 GB, gate 1.000x), upload in flight — and a second line** | 0† | 0† | not measured | none found at the input | measured (topology) / inferred (unroll) |
| `genmo/Mochi-1-preview` | video | 38.2 | FAILED rc=-9 — killed at 2700 s | 0† | 0† | not measured | transformer `seq_len`@256 (weight-extent); transformer `seq_len`@256 (weight-extent) | measured |
| `hpcai-tech/Open-Sora-v2` | video | 42.5 | **DIAGNOSED — rebuild refused at entry, re-trace queued first** | 0† | 0† | not measured | text_encoder_2 `seq_len`@77 (weight-extent) | measured |
| `rhymes-ai/Allegro` | video | 23.6 | FAILED rc=-9 — killed at 2701 s | 12† | 0† | not measured | none found at the input | measured |
| `rhymes-ai/Allegro-TI2V` | video | 24.3 | **RUNS — repaired and delivered** | 27† | 0† | not measured | none found at the input | measured |
| `zai-org/GLM-4.1V-9B-Thinking` | vlm | 19.2 | met in 913 s | 259 | 0 | not measured | none found at the input | measured |

## The lines that carry a later verdict

### `Allegro-TI2V` — RUNS — repaired and delivered

Two defects, both fixed at the source: the output size was never read from the container when the backbone's latent is flattened or the flow is named something else, and the conditioning image did not set the resolution. Bounded above: renders to 80 frames, fails at its own declared 88 asking 25.27 GiB in one allocation (DETTE D2).

*Evidence:* validation_outputs/allegro_image_sets_resolution_20260912/out.mp4 (8 frames at 448x448, rc=0, inter-frame diff 23.3); hub replaced 15:13:48; docs/reference/catalogue-repairs.md entry 1  ·  *line:* measured

### `CogVideoX-5b-I2V` — RUNS — corrected at the source, published, installed, PROVEN by run

Its causal temporal pad recorded 2187*s - 2184 against a truth of s + 2, exact at the traced s=1. Profiled at 49 frames the peak was 210.26 GB against 0.09 GB at the trace, a factor of 2237 which is the compound's own coefficient. Re-traced, the temporal axis carries no symbol at all (an I2V encoder conditions on one image) and the same request costs 0.02 GB. 389 -> 265 ops.

*Evidence:* hub record THUDM/CogVideoX-5b-I2V fileSize 23126413914, updatedAt 2026-09-12T22:09:39Z (replace through the internal entry point, 2498 s); installed manifest 22:10:26 UTC; the installed vae_encoder/graph.json holds 265 ops with symbols batch/height/width and NO temporal symbol; regression gate passed component by component (vae_encoder 0.82 -> 0.80 GB, every other component 1.000x); proof by run 22:45 UTC: 9 frames at 448x448, range 9-253, inter-frame diff 3.34 (validation_outputs/proof_by_run_CogVideoX-5b-I2V_20260912_2242/VERDICT.json)  ·  *line:* measured

### `Open-Sora-v2` — DIAGNOSED — rebuild refused at entry, re-trace queued first

The runtime repair is not enough for this one: the container predates the builder that writes component shapes, so the output size cannot be read from it whatever the runtime does. Snapshot re-downloaded 2026-09-12 16:38 (64.43 GB, the build door's predicate satisfied); the rebuild is queued behind the CogVideoX upload, staged on the root filesystem rather than the export.

*Evidence:* the shipped topology.json carries shapes=NONE for transformer and vae while .cache/graphs holds them (vae: z [1,16,9,14,22]); container dated 2026-06-30; the 22:10 rebuild was refused in 5 s: "component 'scheduler' has no cached graph.json -- topology/graph-cache desync" (its trace cache is also from 2026-06-30); snapshot present (64.43 GB, build door satisfied)  ·  *line:* measured

### `Wan2.1-I2V-14B-480P` — INFERRED same debt as Wan2.1-VACE

No local snapshot, so no second trace point and no fitted slope. The claim rests on six module groups agreeing exactly on two numbers each, while the groups ABOVE the loop differ — different VAE sizes carrying the same loop. One snapshot and two 15-second traces convert it.

*Evidence:* its cached encoder carries the measured anchor's chunk-loop groups identically: {2:1, 3:9, 6:12, 8:22, 12:1, 21:22}  ·  *line:* inferred

### `Wan2.1-VACE-1.3B` — NAMED DEBT — not corrected, and no stimulus corrects it

Its VAE encoder unrolls its temporal chunk loop: 517 ops per chunk, measured at two stimuli. The blind count held at 135 through T=17, 25, 33 and 41 while the door's candidate walked 25 -> 33 -> 41. The graph is not symbolic in time however many symbols its table declares.

*Evidence:* validation_outputs/wan_class_e_20260912/VERDICT.md; docs/reference/temporal-unroll-census.md  ·  *line:* measured

### `Wan2.2-I2V-A14B` — REBUILT (118.07 GB, gate 1.000x), upload in flight — and a second line

TWO lines, not one. The rebuild resolves its output size. Its VAE ENCODER stays unrolled over the temporal axis, which no rebuild changes — DETTE D-TEMPORAL-UNROLL. Delivering the first without saying the second would be delivering a fix for the error we found and hiding the one underneath.

*Evidence:* rebuild 22:11-22:22 (676 s, 118.07 GB, only writer on the pool); regression gate 1.000x on all five components; upload through the internal entry point from 22:22:46 at 37 MB/s, zero SlowDownWrite; cached encoder carries the unroll signature (inferred line)  ·  *line:* measured (topology) / inferred (unroll)

## How to read the columns

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

**where a defect would be invisible** — axes traced at a value where two
distinct rules give the same number, so the trace-point check cannot tell
them apart. This is NOT a defect list. An axis here needs its rule asserted
structurally or a re-trace outside the collision; a test at the flagged
value is green for the reason that blinds it.

## What this document does not say

* **Whether a model is CORRECT.** The run column says it produced output
  without failing, on one request, at one moment. Numerical agreement
  against a vendor pipeline is a different instrument and covers five of
  nine families.
* **What most models cost with the certified directory.** Eleven cells were
  measured; the other thirty-six say *not measured* and that is the whole
  point of the cell.
* **Whether the flagged axes are wrong.** They are places a defect could
  not be seen. Converting one into a verdict costs a second trace at a
  value outside the collision, and the instrument that does it refuses when
  the stimulus does not actually move — a tree compared with itself agrees
  with itself.
* **Anything about hardware other than this rack**: four V100s, two of 16 GB
  and two of 32, at 1290/877 MHz.

