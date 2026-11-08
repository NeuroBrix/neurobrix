# NeuroBrix — Benchmarks

*Every number on this page is a cell of one dated register, written by the benchmark itself; nothing is estimated,
extrapolated or rounded in our favour. Where the Triton engine is slower than the PyTorch engine, the number is here.*

**Register:** `benchmarks/results/showcase_2026-11-08_v0.6.2.tsv` — measured 2026-11-07/08, NeuroBrix **0.6.2** (commit `99d17b0`), 18 models x 2 engines x 2 cards, medians of 3 runs.

**What "run" means here.** Every time compared on this page is the *run* time: the end-to-end time of the command
minus the engine's own weight-load time — start-up, planning, generation and writing the output, the part the two
engines execute differently. Loading reads the same weight files for both engines (each through its own loader); its
time is set mostly by the storage the files come from, so it is given apart, in the tables and in the register.

## Method

- **One code, two engines.** Every model runs from the same container with the same request on both NeuroBrix engines: the
  **PyTorch engine** (`compiled`: ATen, cuDNN, cuBLAS) and the **Triton engine** (NeuroBrix's own kernels, no PyTorch at
  run time). The ratio below is PyTorch time / Triton time: above 1, the Triton engine is faster.
- **Through the entry point**, exactly as a user runs it: `python -m neurobrix run --model <model> <request>`; the PyTorch
  engine is the default when PyTorch is installed, `--triton` selects the Triton engine. Every command is listed below.
- **Certified kernels only**: every Triton kernel setting is read from the certified directory shipped in the package,
  none is tuned during a measured run.
- **One run on the whole machine at a time**: one card used, the three others idle, nothing else running on the host.
  A first pass ran the four cards at once (each run alone on its card); the shared host (CPU, memory, PCIe, storage)
  spread repeated runs by up to 2x, most of all for the 30B models whose weights stream from host memory, so it was
  replaced by this one. The spread column gives the fastest and slowest of the three runs of each cell.
- **3 runs per cell, the median kept.** The three outputs of every cell are byte-identical (deterministic), and identical
  to the outputs judged by eye (text read, image and video looked at, audio transcribed) at the 0.6.0 release gate.
- **Short requests** — the release gate's: 32 new tokens for language models, 8 steps at the sizes given for images, 2-4
  steps for video. They measure the engine at the floor of a request; a longer request moves every number.
- **Measured**: *run* = the time after the weights are loaded (the command's end-to-end time minus the engine's own load
  time: start-up, plan, generation, writing the output) — the engines differ here; *load* = the engine's load time, set
  by the storage; *first token* = time to the first generated token (language models); *tok/s* = decode rate;
  *peak* = the card's memory peak.

## Hardware

- 4 x NVIDIA Tesla V100-SXM2 (2017, sm_70): two 32 GB, two 16 GB; application clocks locked at 1290 MHz (SM) / 877 MHz
  (memory); driver 535.
- Host: 2 x Intel Xeon Gold 6230 (80 threads). Models are read from an NVMe cache over NFS/RDMA.
- A 30B mixture-of-experts model does not fit whole in a V100: its weights are held in host memory and streamed to the
  card as each layer runs (the plan NeuroBrix's planner chooses). Those rows measure that path, not a whole-card run.

## The result in one line

Of the 36 model-and-card pairs, the Triton engine is faster in **30** and slower in **6**.

## Text and code (LLM)

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Instruct | 19.3 | 5.3 | **x3.64** | 19.1–20.6 / 5.3–5.4 | 56.5 / 76.7 | 7.42 / 2.53 | 2.76 / 7.82 | 29,092 / 31,314 |
| Qwen3-30B-A3B-Thinking-2507 | 51.8 | 23.8 | **x2.17** | 51.3–53.1 / 23.6–31.3 | 136.2 / 236.8 | 15.90 / 8.93 | 1.14 / 2.28 | 27,870 / 29,597 |
| Qwen3-Coder-30B-A3B-Instruct | 59.7 | 21.8 | **x2.75** | 54.2–68.4 / 20.8–23.4 | 137.2 / 235.6 | 16.59 / 8.17 | 1.16 / 2.70 | 27,870 / 29,727 |
| TinyLlama-1.1B-Chat-v1.0 | 5.0 | 2.3 | **x2.23** | 5.0–5.4 / 2.3–2.6 | 6.2 / 4.9 | 0.52 / 0.93 | 11.60 / 18.72 | 2,232 / 2,352 |
| deepseek-moe-16b-chat | 23.8 | 9.4 | **x2.54** | 21.1–26.4 / 9.2–9.6 | 43.2 / 29.8 | 7.73 / 1.36 | 2.51 / 3.87 | 28,452 / 31,580 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Instruct | 24.9 | 20.2 | **x1.24** | 24.1–26.0 / 19.1–20.3 | 60.0 / 100.8 | 7.28 / 6.03 | 1.89 / 2.01 | 14,052 / 13,394 |
| Qwen3-30B-A3B-Thinking-2507 | 65.1 | 49.8 | **x1.31** | 63.2–68.1 / 34.4–59.1 | 135.6 / 238.5 | 17.63 / 9.75 | 0.88 / 1.44 | 12,830 / 13,213 |
| Qwen3-Coder-30B-A3B-Instruct | 83.8 | 29.0 | **x2.89** | 80.6–124.0 / 28.7–29.2 | 133.6 / 235.9 | 17.15 / 8.47 | 0.94 / 1.68 | 12,830 / 13,407 |
| TinyLlama-1.1B-Chat-v1.0 | 5.2 | 2.5 | **x2.08** | 5.0–5.4 / 2.4–2.5 | 5.6 / 4.8 | 0.63 / 0.88 | 11.23 / 17.93 | 2,232 / 2,320 |
| deepseek-moe-16b-chat | 27.6 | 20.4 | **x1.36** | 27.3–33.3 / 20.2–20.9 | 43.3 / 40.1 | 7.35 / 6.10 | 1.71 / 1.97 | 13,432 / 13,604 |

## Vision-language

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| GLM-4.1V-9B-Thinking | 9.8 | 7.6 | **x1.29** | 9.8–12.1 / 7.5–8.7 | 24.3 / 15.7 | 4.50 / 3.75 | 4.01 / 5.08 | 17,990 / 19,354 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| GLM-4.1V-9B-Thinking | 58.2 | 50.2 | **x1.16** | 57.8–58.4 / 49.5–50.3 | 22.0 / 22.5 | 8.21 / 7.73 | 0.59 / 0.67 | 10,610 / 14,604 |

## Multimodal (image in, text out; text to image)

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| Janus-Pro-7B | 46.0 | 31.3 | **x1.47** | 44.5–51.7 / 30.5–32.4 | 15.5 / 11.2 | 2.79 / 1.20 | 13.70 / 19.17 | 14,342 / 14,217 |
| Qwen3-Omni-30B-A3B-Instruct | 58.1 | 42.0 | **x1.38** | 56.0–69.6 / 39.5–42.5 | 111.7 / 233.4 | 25.46 / 20.11 | 0.70 / 1.07 | 27,236 / 29,676 |
| Qwen3-VL-30B-A3B-Thinking | 289.1 | 265.8 | **x1.09** | 283.9–292.6 / 262.6–267.9 | 71.3 / 66.1 | 19.55 / 18.12 | 0.12 / 0.12 | 23,360 / 26,703 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| Janus-Pro-7B | 156.7 | 29.9 | **x5.24** | 155.7–158.9 / 29.8–30.3 | 14.6 / 11.2 | 5.64 / 1.08 | 3.81 / 20.12 | 13,162 / 14,185 |
| Qwen3-Omni-30B-A3B-Instruct | 58.7 | 43.6 | **x1.35** | 58.0–60.0 / 43.1–47.1 | 109.7 / 229.2 | 25.85 / 20.61 | 0.67 / 0.95 | 12,196 / 13,292 |
| Qwen3-VL-30B-A3B-Thinking | 381.2 | 369.2 | **x1.03** | 376.0–386.6 / 367.7–370.8 | 70.2 / 72.3 | 18.67 / 18.72 | 0.09 / 0.09 | 9,094 / 10,045 |

## Speech to text

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| whisper-large-v3-turbo | 3.6 | 1.9 | **x1.85** | 3.5–4.2 / 1.8–2.2 | 4.3 / 4.1 | 1,604 / 2,637 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| whisper-large-v3-turbo | 3.6 | 1.8 | **x2.02** | 3.6–4.0 / 1.7–1.8 | 4.5 / 4.1 | 1,604 / 2,637 |

## Audio LLM

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| Voxtral-Mini-3B-2507 | 13.3 | 4.1 | **x3.23** | 13.2–15.6 / 4.1–4.5 | 14.8 / 10.6 | 4.91 / 1.80 | 2.29 / 8.64 | 11,046 / 9,270 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | first token PyTorch / Triton (s) | tok/s PyTorch / Triton | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|---|---|
| Voxtral-Mini-3B-2507 | 13.1 | 4.0 | **x3.32** | 12.7–13.2 / 3.7–4.1 | 14.3 / 10.1 | 5.20 / 1.85 | 2.33 / 8.65 | 11,046 / 9,238 |

## Text to speech

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| Kokoro-82M | 5.2 | 7.1 | x0.73 | 5.0–6.3 / 6.5–10.4 | 4.0 / 4.4 | 154 / 1,206 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| Kokoro-82M | 5.0 | 7.2 | x0.70 | 5.0–5.1 / 6.8–7.3 | 3.9 / 4.6 | 154 / 1,174 |

## Image generation

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| Flex.1-alpha | 9.7 | 11.4 | x0.85 | 9.6–9.7 / 11.3–11.6 | 23.8 / 17.7 | 15,750 / 16,936 |
| Sana_1600M_1024px_MultiLing | 11.2 | 8.7 | **x1.30** | 11.1–11.7 / 8.6–8.7 | 15.0 / 11.6 | 5,246 / 6,022 |
| Sana_1600M_4Kpx_BF16 | 60.2 | 67.7 | x0.89 | 60.0–60.4 / 67.6–68.2 | 18.0 / 13.3 | 30,284 / 26,997 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| Flex.1-alpha | 18.4 | 24.4 | x0.76 | 18.2–19.3 / 24.4–24.5 | 24.5 / 25.4 | 13,070 / 12,982 |
| Sana_1600M_1024px_MultiLing | 11.3 | 8.6 | **x1.31** | 11.0–11.4 / 8.6–9.0 | 15.7 / 11.5 | 5,246 / 5,958 |
| Sana_1600M_4Kpx_BF16 | 59.3 | 65.2 | x0.91 | 58.8–59.5 / 64.7–65.8 | 18.8 / 16.7 | 15,206 / 13,016 |

## Video generation

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| CogVideoX-5b-I2V | 310.8 | 266.5 | **x1.17** | 310.7–311.3 / 266.2–266.7 | 28.6 / 22.3 | 15,452 / 18,041 |
| Wan2.1-T2V-1.3B | 16.1 | 15.8 | **x1.02** | 16.1–16.2 / 15.6–16.0 | 19.4 / 21.1 | 11,224 / 11,929 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| CogVideoX-5b-I2V | 310.8 | 267.9 | **x1.16** | 310.5–311.3 / 267.2–268.0 | 28.7 / 28.0 | 15,472 / 14,745 |
| Wan2.1-T2V-1.3B | 17.3 | 15.9 | **x1.08** | 16.8–17.5 / 15.8–16.0 | 19.0 / 21.3 | 11,224 / 11,897 |

## Upscaling

**V100 32 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| Real-ESRGAN-RealESRGAN_x4plus | 3.7 | 2.9 | **x1.28** | 3.5–3.9 / 2.8–2.9 | 2.6 / 3.8 | 2,312 / 2,466 |

**V100 16 GB**

| model | PyTorch run (s) | Triton run (s) | ratio | spread of 3 PyTorch / Triton (s) | load PyTorch / Triton (s) | peak PyTorch / Triton (MB) |
|---|---|---|---|---|---|---|
| Real-ESRGAN-RealESRGAN_x4plus | 3.7 | 2.9 | **x1.27** | 3.6–3.8 / 2.9–3.0 | 2.4 / 3.8 | 2,312 / 2,466 |

## Where the Triton engine wins, and where it loses

Faster (ratio > 1): Janus-Pro-7B 16 GB x5.24, DeepSeek-Coder-V2-Lite-Instruct 32 GB x3.64, Voxtral-Mini-3B-2507 16 GB x3.32, Voxtral-Mini-3B-2507 32 GB x3.23, Qwen3-Coder-30B-A3B-Instruct 16 GB x2.89, Qwen3-Coder-30B-A3B-Instruct 32 GB x2.75, deepseek-moe-16b-chat 32 GB x2.54, TinyLlama-1.1B-Chat-v1.0 32 GB x2.23, Qwen3-30B-A3B-Thinking-2507 32 GB x2.17, TinyLlama-1.1B-Chat-v1.0 16 GB x2.08, whisper-large-v3-turbo 16 GB x2.02, whisper-large-v3-turbo 32 GB x1.85, Janus-Pro-7B 32 GB x1.47, Qwen3-Omni-30B-A3B-Instruct 32 GB x1.38, deepseek-moe-16b-chat 16 GB x1.36, Qwen3-Omni-30B-A3B-Instruct 16 GB x1.35, Sana_1600M_1024px_MultiLing 16 GB x1.31, Qwen3-30B-A3B-Thinking-2507 16 GB x1.31, Sana_1600M_1024px_MultiLing 32 GB x1.30, GLM-4.1V-9B-Thinking 32 GB x1.29, Real-ESRGAN-RealESRGAN_x4plus 32 GB x1.28, Real-ESRGAN-RealESRGAN_x4plus 16 GB x1.27, DeepSeek-Coder-V2-Lite-Instruct 16 GB x1.24, CogVideoX-5b-I2V 32 GB x1.17, CogVideoX-5b-I2V 16 GB x1.16, GLM-4.1V-9B-Thinking 16 GB x1.16, Qwen3-VL-30B-A3B-Thinking 32 GB x1.09, Wan2.1-T2V-1.3B 16 GB x1.08, Qwen3-VL-30B-A3B-Thinking 16 GB x1.03, Wan2.1-T2V-1.3B 32 GB x1.02.

Slower (ratio <= 1): Kokoro-82M 16 GB x0.70, Kokoro-82M 32 GB x0.73, Flex.1-alpha 16 GB x0.76, Flex.1-alpha 32 GB x0.85, Sana_1600M_4Kpx_BF16 32 GB x0.89, Sana_1600M_4Kpx_BF16 16 GB x0.91.

Loading, given apart: the Triton engine loads slower than the PyTorch engine (by more than 1.25x) on DeepSeek-Coder-V2-Lite-Instruct 32 GB (76.7 s against 56.5 s), Qwen3-30B-A3B-Thinking-2507 32 GB (236.8 s against 136.2 s), Qwen3-Coder-30B-A3B-Instruct 32 GB (235.6 s against 137.2 s), Qwen3-Omni-30B-A3B-Instruct 32 GB (233.4 s against 111.7 s), Real-ESRGAN-RealESRGAN_x4plus 32 GB (3.8 s against 2.6 s), DeepSeek-Coder-V2-Lite-Instruct 16 GB (100.8 s against 60.0 s), Qwen3-30B-A3B-Thinking-2507 16 GB (238.5 s against 135.6 s), Qwen3-Coder-30B-A3B-Instruct 16 GB (235.9 s against 133.6 s), Qwen3-Omni-30B-A3B-Instruct 16 GB (229.2 s against 109.7 s), Real-ESRGAN-RealESRGAN_x4plus 16 GB (3.8 s against 2.4 s).

## The commands

Each cell ran exactly this command (the PyTorch engine; the Triton cell adds `--triton`), from the repository root;
`--hardware` named the card's profile, the one NeuroBrix generates at a first run, and `--certified-only` refuses any
kernel setting the certified directory does not serve (no tuning during a measured run):

```
python -m neurobrix run --model CogVideoX-5b-I2V --prompt 'a red apple rolling slowly across a wooden table' --seed 42 --input-image benchmarks/assets/apple_448.png --steps 4 --height 480 --width 720 --hardware <this card's profile> --certified-only --output <out>/output.mp4
python -m neurobrix run --model DeepSeek-Coder-V2-Lite-Instruct --prompt 'Explain in three sentences why the sky is blue and the sunset red.' --max-tokens 32 --temperature 0.0 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Flex.1-alpha --prompt 'a red apple on a wooden table' --seed 42 --steps 8 --height 384 --width 512 --hardware <this card's profile> --certified-only --output <out>/output.png
python -m neurobrix run --model GLM-4.1V-9B-Thinking --input-image benchmarks/assets/apple_448.png --prompt 'Describe this image in one sentence.' --max-tokens 32 --temperature 0.0 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Janus-Pro-7B --prompt 'a red apple on a wooden table' --seed 42 --max-tokens 32 --temperature 0.0 --mode image --hardware <this card's profile> --certified-only --output <out>/output.png
python -m neurobrix run --model Kokoro-82M --prompt 'The quick brown fox jumps over the lazy dog.' --seed 42 --speaker af_alloy --hardware <this card's profile> --certified-only --output <out>/output.wav
python -m neurobrix run --model Qwen3-30B-A3B-Thinking-2507 --prompt 'Explain in three sentences why the sky is blue and the sunset red.' --max-tokens 32 --temperature 0.0 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Qwen3-Coder-30B-A3B-Instruct --prompt 'Explain in three sentences why the sky is blue and the sunset red.' --max-tokens 32 --temperature 0.0 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Qwen3-Omni-30B-A3B-Instruct --prompt 'a red apple on a wooden table' --seed 42 --max-tokens 32 --temperature 0.0 --mode text --input-image benchmarks/assets/apple_448.png --prompt 'Describe this image in one sentence.' --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Qwen3-VL-30B-A3B-Thinking --prompt 'a red apple on a wooden table' --seed 42 --max-tokens 32 --temperature 0.0 --mode text --input-image benchmarks/assets/apple_448.png --prompt 'Describe this image in one sentence.' --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Real-ESRGAN-RealESRGAN_x4plus --input-image benchmarks/assets/apple_448.png --hardware <this card's profile> --certified-only --output <out>/output.png
python -m neurobrix run --model Sana_1600M_1024px_MultiLing --prompt 'a red apple on a wooden table' --seed 42 --steps 8 --height 320 --width 512 --hardware <this card's profile> --certified-only --output <out>/output.png
python -m neurobrix run --model Sana_1600M_4Kpx_BF16 --prompt 'a red apple on a wooden table' --seed 42 --steps 8 --height 1408 --width 2048 --hardware <this card's profile> --certified-only --output <out>/output.png
python -m neurobrix run --model TinyLlama-1.1B-Chat-v1.0 --prompt 'Explain in three sentences why the sky is blue and the sunset red.' --max-tokens 32 --temperature 0.0 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Voxtral-Mini-3B-2507 --audio benchmarks/assets/jfk_11s.wav --prompt 'Transcribe this audio.' --max-tokens 32 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model Wan2.1-T2V-1.3B --prompt 'a red apple rolling slowly across a wooden table' --seed 42 --steps 2 --height 480 --width 832 --num-frames 9 --hardware <this card's profile> --certified-only --output <out>/output.mp4
python -m neurobrix run --model deepseek-moe-16b-chat --prompt 'Explain in three sentences why the sky is blue and the sunset red.' --max-tokens 32 --temperature 0.0 --hardware <this card's profile> --certified-only --output <out>/output.txt
python -m neurobrix run --model whisper-large-v3-turbo --audio benchmarks/assets/jfk_11s.wav --hardware <this card's profile> --certified-only --output <out>/output.txt
```

The raw register (every run's three values, output hashes and command) is `benchmarks/results/showcase_2026-11-08_v0.6.2.tsv`; the chart
`docs/assets/benchmark_0.6.2.png` is drawn from it alone.
