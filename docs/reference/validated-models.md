# Validated models — judged from outside the engine (R29)

One line per model of the NVIDIA (Volta) gate list; one verdict per mode and memory class, written only after a
person or an outside instrument looked at the artefact the cell produced: images read, texts read, audio transcribed
by an independent speech-to-text model and its level measured, videos as five-frame contact sheets. A verdict is
`works` (the artefact is a correct answer to the request), `broken` (it is not, whatever the exit code said),
`rc N` (the cell did not finish) or `UNJUDGED` (a row that landed after the last pass). The layer in parentheses
names the gate whose row was judged: the mq25 candidate tree, the mq25 frozen tree, or the mq24 gate carried forward.
A pass is never pronounced from metrics or exit codes alone; the exit-code gate had called every `broken` cell below
green. The per-cell records live beside each gate's rows (`judgments.jsonl`, written by
`tools/regression_matrix.py judge`); this table is generated from them.

Last pass: 2026-10-12 (Dell, four Tesla V100: two 16 GB, two 32 GB).

| model | 16g native | 16g triton | 32g native | 32g triton |
|---|---|---|---|---|
| Allegro | works (mq24) | works (mq24) | works (cand) | works (cand) |
| Allegro-TI2V | works (mq24) | rc -15 (mq24) | works (mq24) | rc -15 (mq24) |
| CogVideoX-2b | broken (cand) | broken (cand) | broken (cand) | broken (cand) |
| CogVideoX-5b-I2V | broken (cand) | UNJUDGED (cand) | broken (cand) | broken (cand) |
| DeepSeek-Coder-V2-Lite-Instruct | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Flex.1-alpha | UNJUDGED (cand) | works (mq25) | broken (cand) | works (cand) |
| GLM-4.1V-9B-Thinking | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Janus-Pro-7B | works (mq25) | works (mq25) | works (mq25) | works (mq25) |
| Kokoro-82M | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Ming-Lite-Omni-1.5 | works (mq25) | works (mq25) | works (mq25) | works (mq25) |
| MiniCPM-o-4_5 | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Open-Sora-v2 | works (cand) | works (cand) | works (mq24) | works (mq24) |
| PixArt-Sigma-XL-2-1024-MS | works (cand) | works (cand) | works (cand) | works (cand) |
| PixArt-XL-2-1024-MS | works (cand) | works (cand) | works (cand) | works (cand) |
| Qwen3-30B-A3B-Thinking-2507 | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Qwen3-Coder-30B-A3B-Instruct | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Qwen3-Omni-30B-A3B-Instruct | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Qwen3-VL-30B-A3B-Thinking | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Real-ESRGAN-RealESRGAN_x2 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| Real-ESRGAN-RealESRGAN_x4plus | works (mq24) | works (mq24) | works (cand) | works (cand) |
| Real-ESRGAN-RealESRGAN_x8 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| SANA-Video_2B_720p | works (cand) | works (cand) | works (cand) | works (cand) |
| Sana_1600M_1024px_MultiLing | works (cand) | works (cand) | works (cand) | works (cand) |
| Sana_1600M_4Kpx_BF16 | works (cand) | works (cand) | works (cand) | works (cand) |
| SwinIR-swinir_x2 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| SwinIR-swinir_x4 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| TinyLlama-1.1B-Chat-v1.0 | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| VibeVoice-1.5B | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Voxtral-Mini-3B-2507 | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| Wan2.1-I2V-14B-480P | broken (cand) | works (mq25) | broken (cand) | works (cand) |
| Wan2.1-T2V-1.3B | broken (cand) | broken (cand) | broken (cand) | broken (cand) |
| Wan2.1-VACE-1.3B | UNJUDGED (cand) | UNJUDGED (cand) | broken (cand) | broken (cand) |
| Wan2.2-I2V-A14B | broken (mq25) | broken (mq25) | broken (mq25) | broken (mq25) |
| canary-qwen-2.5b | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| chatterbox | works (mq25) | works (mq25) | works (mq25) | works (mq25) |
| deepseek-moe-16b-chat | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| granite-3.1-1b-a400m-instruct | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| granite-speech-3.3-8b | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| mochi-1-preview | works (mq25) | rc 1 (mq25) | works (cand) | broken (cand) |
| openaudio-s1-mini | works (mq25) | works (mq25) | works (mq25) | works (mq25) |
| orpheus-3b-0.1-ft | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| parakeet-tdt-1.1b | works (mq25) | works (mq25) | works (mq25) | works (mq25) |
| swin2SR-classical-sr-x2-64 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| swin2SR-classical-sr-x4-64 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| swin2SR-realworld-sr-x4-64-bsrgan-psnr | works (mq24) | works (mq24) | works (cand) | works (cand) |
| upscale_models-hat_l_x4 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| upscale_models-hat_s_x4 | works (mq24) | works (mq24) | works (cand) | works (cand) |
| whisper-large | works (mq24) | works (mq24) | works (mq24) | works (mq24) |
| whisper-large-v3-turbo | works (mq24) | works (mq24) | works (mq24) | works (mq24) |

models all-works: 40; with a broken/unjudged/red cell: 9; total 49

## What the broken cells showed

* **Native only** (the Triton cell of the same request is correct): Flex.1-alpha renders red translucent petals
  instead of the prompt's apple; Wan2.1-I2V-14B-480P dissolves the input apple into a soft blob.
* **Triton only** (native correct): mochi-1-preview renders magenta and green blobs with horizontal glitch bars.
* **Both modes, both classes** (the cause is upstream of the kernels: the traced graph or the request): CogVideoX-2b
  gives a flat orange field and Wan2.1-T2V-1.3B a red cloud at the gate's 4-step request; Wan2.1-VACE-1.3B smears the
  input into vertical colour bands with block glitches; Wan2.2-I2V-A14B turns the input apple into an inverted,
  washed silhouette; CogVideoX-5b-I2V washes it into a glowing blob (known since its retrace).
* Allegro-TI2V triton is the carried red of the previous gate (killed at its time cap), replaying on the candidate.

Next for each: the vendor's own pipeline run at the same request tells a too-small request from a defect of the
traced graph; a native-vs-Triton op-level diff names the kernel for the single-mode ones.
