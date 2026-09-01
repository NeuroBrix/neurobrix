> **HISTORICAL STARTING POINT — pre-mandate numbers, kept verbatim as the zero-point of the optimization curve. The current one-pager is `../SUMMARY_2026_08_30.md` (capacity first, trajectory section names this file as its starting column).**

# Benchmark matrix — 2026_08_05_ref

## image_diffusion_sana1024 — config: pinned

| column | result |
|---|---|
| diffusers | 4.08s · 0.20s/step · 20.0G |
| neurobrix_pytorch | 9.33s · 0.47s/step · 15.2G |
| neurobrix_triton | 47.77s · 2.39s/step · 12.4G |

## llm_heavy_qwen3_coder — config: pinned

| column | result |
|---|---|
| neurobrix_pytorch | 0.09 tok/s · TTFT 29.93s · 7.5G |
| ollama | 7.56 tok/s · TTFT 0.40s · 31.6G |

## stt_whisper_turbo — config: pinned

| column | result |
|---|---|
| neurobrix_pytorch | 0.73s · RTF 0.07 · 2.4G |
| neurobrix_triton | 3.80s · RTF 0.35 · 3.5G |
| vendor_transformers | 0.27s · RTF 0.02 · 2.3G |

## video_wan13b_t2v — config: pinned

| column | result |
|---|---|
| diffusers | 18.22s · 1.82s/step · 27.4G |
| neurobrix_pytorch | 133.84s · 13.38s/step · 11.6G |
| neurobrix_triton | error — `RuntimeError('Daemon error: Failed at aten.convolution::0 (aten::convolution): Pointer argument (at 0) cannot be accesse` |

