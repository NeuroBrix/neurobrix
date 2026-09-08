# Calibration rows B — 2026-09-05

The per-cell JSONs beside this file are the measurement; this index is generated from them
(`row` × `column`, the status, the median wall per request, the cold start, the peak VRAM and
how many distinct output hashes the repetitions produced). The run logs of these cells are not
tracked — no log is tracked anywhere under `benchmarks/results/`.

| row | ? | diffusers | neurobrix_pytorch | ollama | vendor_transformers | vllm |
|---|---|---|---|---|---|---|
| env_manifest.json | None, n=0, wall Nones | — | — | — | — | — |
| image_diffusion_flex1 | — | — | ok, n=5, wall 35.87s, cold 4.0s, 29875 MiB, 1 distinct output(s) | — | — | — |
| image_diffusion_pixart_xl | — | — | ok, n=5, wall 7.72s, cold 4.0s, 14819 MiB, 1 distinct output(s) | — | — | — |
| llm_dense_tinyllama | — | — | ok, n=5, wall 4.3s, cold 6.0s, 2743 MiB | — | — | — |
| omni_ming_t2i | — | — | ok, n=5, wall 28.71s, cold 31.0s, 12999 MiB, 1 distinct output(s) | — | dnr, n=0, wall Nones | — |
| omni_minicpmo_voice | — | — | ok, n=5, wall 2.36s, cold 9.0s, 21477 MiB, 5 distinct output(s) | — | — | — |
| omni_qwen3omni | — | — | ok, n=5, wall 337.95s, cold 16.0s, 10479 MiB, 2 distinct output(s) | — | dnr, n=0, wall Nones | — |
| stt_whisper_turbo | — | — | ok, n=5, wall 0.27s, cold 3.1s, 2311 MiB | — | — | — |
| upscale_hat_l_x4 | — | — | ok, n=5, wall 6.61s, cold 4.0s, 10891 MiB, 1 distinct output(s) | — | — | — |
| upscale_realesrgan_x4 | — | — | ok, n=5, wall 0.93s, cold 3.1s, 2775 MiB, 1 distinct output(s) | — | — | — |
| upscale_swin2sr_realworld_x4 | — | — | ok, n=5, wall 1.5s, cold 3.0s, 3735 MiB, 1 distinct output(s) | — | — | — |
| upscale_swin2sr_x4 | — | — | ok, n=5, wall 1.56s, cold 3.0s, 3373 MiB, 1 distinct output(s) | — | — | — |
| upscale_swinir_x2 | — | — | ok, n=5, wall 1.52s, cold 3.0s, 2461 MiB, 1 distinct output(s) | — | — | — |
| upscale_swinir_x4 | — | — | ok, n=5, wall 1.92s, cold 3.0s, 3731 MiB, 1 distinct output(s) | — | — | — |
| video_allegro_t2v | — | — | ok, n=5, wall 22.32s, cold 5.0s, 13441 MiB, 1 distinct output(s) | — | — | — |
| video_allegro_ti2v | — | dnr, n=0, wall Nones | ok, n=5, wall 56.65s, cold 6.0s, 9953 MiB, 1 distinct output(s) | — | — | — |
| video_cog2b_t2v | — | — | ok, n=5, wall 21.54s, cold 6.3s, 10019 MiB, 1 distinct output(s) | — | — | — |
| video_cog5b_i2v | — | — | ok, n=5, wall 111.8s, cold 9.0s, 13025 MiB, 1 distinct output(s) | — | — | — |
| video_mochi_t2v | — | — | ok, n=5, wall 114.46s, cold 6.0s, 32443 MiB, 1 distinct output(s) | — | — | — |
| video_opensora_t2v | — | dnr, n=0, wall Nones | ok, n=5, wall 139.62s, cold 7.0s, 27095 MiB, 1 distinct output(s) | — | — | — |
| video_sana_video | — | dnr, n=0, wall Nones | ok, n=5, wall 13.26s, cold 5.0s, 8759 MiB, 1 distinct output(s) | — | — | — |
| video_wan13b_t2v | — | — | ok, n=5, wall 72.06s, cold 5.0s, 11773 MiB, 1 distinct output(s) | — | — | — |
| video_wan14b_i2v | — | — | error, n=0, wall Nones, 12321 MiB | — | — | — |
| video_wan22_a14b_i2v | — | — | error, n=0, wall Nones, 11943 MiB | — | — | — |
| video_wan_vace | — | — | error, n=0, wall Nones, 11757 MiB | — | — | — |
| vlm_glm41v | — | — | ok, n=5, wall 6.66s, cold 5.0s, 20197 MiB | dnr, n=0, wall Nones | — | dnr, n=0, wall Nones |
| vlm_qwen3vl | — | — | ok, n=5, wall 437.69s, cold 6.0s, 7019 MiB | — | — | dnr, n=0, wall Nones |
