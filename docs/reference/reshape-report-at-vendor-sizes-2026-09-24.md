# Reshape report-only pass at vendor-sourced sizes (engine `c02751c8`)

Hocine's rule 1: every shape proof at two sizes other than the trace size, one far from it. Sizes per container are the vendor's, with sources in `vendor-sizes-2026-09-24.md`; `largeDefault` / `frames_largeDefault` mark the vendor's default where it documents nothing larger. Each size ran at 6 rungs x 2 modes in census shadow (no weights read).

**Coverage:** 164 of 708 shadow runs completed (23 %). A 0 below is a 0 over the completed runs of that size only. Refused runs are uncovered and named at the end.

101949 invention records, 44 on a negative-extent input (counted apart), 0 unreadable lines.

| container | size | inventions | distinct (in, target) | ratios | negative-extent input | done | refused | failed |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Allegro | small368x640 | 1536 | 1 | 0.388 | 0 | 6 | 6 | 0 |
| Allegro | largeDefault720x1280 | 1536 | 1 | 1.5183 | 0 | 6 | 6 | 0 |
| Allegro | frames_small40 | 1536 | 1 | 0.6901 | 0 | 6 | 6 | 0 |
| Allegro | frames_largeDefault88 | 1536 | 1 | 1.5183 | 0 | 6 | 6 | 0 |
| Allegro-TI2V | small368x640 | 0 (0 over 0 done) | 0 |  | 12 | 0 | 6 | 6 |
| Allegro-TI2V | largeDefault720x1280 | 0 (0 over 0 done) | 0 |  | 8 | 0 | 8 | 4 |
| Allegro-TI2V | frames_small40 | 0 (0 over 0 done) | 0 |  | 12 | 0 | 6 | 6 |
| Allegro-TI2V | frames_largeDefault88 | 0 (0 over 0 done) | 0 |  | 12 | 0 | 6 | 6 |
| CogVideoX-2b | small320x480 | 0 | 0 |  | 0 | 6 | 6 | 0 |
| CogVideoX-2b | largeDefault480x720 | 0 | 0 |  | 0 | 6 | 6 | 0 |
| CogVideoX-2b | frames_small25 | 0 | 0 |  | 0 | 8 | 4 | 0 |
| CogVideoX-2b | frames_largeDefault49 | 0 | 0 |  | 0 | 6 | 6 | 0 |
| CogVideoX-5b-I2V | largeDefault480x720 | 72 | 3 | 0.5 | 0 | 6 | 6 | 0 |
| CogVideoX-5b-I2V | frames_small25 | 96 | 3 | 0.5 | 0 | 8 | 4 | 0 |
| CogVideoX-5b-I2V | frames_largeDefault49 | 72 | 3 | 0.5 | 0 | 6 | 6 | 0 |
| Flex.1-alpha | small512x512 | 10 | 1 | 4.0 | 0 | 0 | 2 | 10 |
| Flex.1-alpha | largeDefault1024x1024 | 11 | 6 | 0.1094, 0.1111 | 0 | 0 | 2 | 10 |
| Open-Sora-v2 | small256x256 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Open-Sora-v2 | large576x1024 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Open-Sora-v2 | frames_small17 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 4 | 8 |
| Open-Sora-v2 | frames_large125 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| PixArt-Sigma-XL-1024 | small512x512 | 0 | 0 |  | 0 | 6 | 0 | 6 |
| PixArt-Sigma-XL-1024 | large512x2048 | 0 | 0 |  | 0 | 6 | 0 | 6 |
| PixArt-Sigma-XL-2-1024-MS | small512x512 | 0 | 0 |  | 0 | 8 | 2 | 2 |
| PixArt-Sigma-XL-2-1024-MS | large512x2048 | 0 | 0 |  | 0 | 8 | 2 | 2 |
| PixArt-XL-1024 | small512x512 | 0 | 0 |  | 0 | 6 | 0 | 6 |
| PixArt-XL-1024 | large512x2048 | 0 | 0 |  | 0 | 6 | 0 | 6 |
| PixArt-XL-2-1024-MS | small512x512 | 0 | 0 |  | 0 | 8 | 2 | 2 |
| PixArt-XL-2-1024-MS | large512x2048 | 0 | 0 |  | 0 | 8 | 2 | 2 |
| SANA-Video_2B_720p_diffusers | small480x832 | 0 | 0 |  | 0 | 2 | 8 | 2 |
| SANA-Video_2B_720p_diffusers | large672x1344 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 10 | 2 |
| SANA-Video_2B_720p_diffusers | frames_small41 | 0 | 0 |  | 0 | 2 | 8 | 2 |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 10 | 2 |
| Sana-1600M-MultiLing | small512x512 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 0 | 12 |
| Sana-1600M-MultiLing | large512x2048 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 0 | 12 |
| Sana_1600M_1024px_MultiLing | small512x512 | 33852 | 9 | 0.5, 1.9412, 2.0 | 0 | 0 | 0 | 12 |
| Sana_1600M_1024px_MultiLing | large512x2048 | 33852 | 9 | 0.5, 1.9412, 2.0 | 0 | 0 | 0 | 12 |
| Sana_1600M_4Kpx_BF16 | small1024x1024 | 27840 | 1 | 0.0625 | 0 | 12 | 0 | 0 |
| Sana_1600M_4Kpx_BF16 | large2048x8192 | 0 | 0 |  | 0 | 6 | 0 | 6 |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | 0 | 0 |  | 0 | 4 | 8 | 0 |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | 0 | 0 |  | 0 | 4 | 8 | 0 |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | 0 | 0 |  | 0 | 4 | 8 | 0 |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | 0 | 0 |  | 0 | 4 | 8 | 0 |
| Wan2.1-VACE-1.3B-diffusers | small512x512 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 0 | 12 |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 10 | 2 |
| Wan2.1-VACE-1.3B-diffusers | frames_small33 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 0 | 12 |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 10 | 2 |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 8 | 4 |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| mochi-1-preview | small320x576 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| mochi-1-preview | largeDefault480x848 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| mochi-1-preview | frames_small31 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |
| mochi-1-preview | frames_large163 | 0 (0 over 0 done) | 0 |  | 0 | 0 | 12 | 0 |

## Failures by (container, size) and class

| container | size | runs | class |
|---|---|---:|---|
| Allegro-TI2V | frames_largeDefault88 | 6 | zero or negative extent downstream |
| Allegro-TI2V | frames_small40 | 6 | zero or negative extent downstream |
| Allegro-TI2V | largeDefault720x1280 | 4 | zero or negative extent downstream |
| Allegro-TI2V | small368x640 | 6 | zero or negative extent downstream |
| Flex.1-alpha | largeDefault1024x1024 | 6 | a streamed piece cannot bind a symbol |
| Flex.1-alpha | largeDefault1024x1024 | 2 | broadcast / matmul shape mismatch |
| Flex.1-alpha | largeDefault1024x1024 | 1 | other: [ERROR] Pipeline failed: Failed at aten.cat::31 (aten::cat): 'str' object has no attribute 'ndim' | None args at positio |
| Flex.1-alpha | largeDefault1024x1024 | 1 | other: [ERROR] Pipeline failed: Failed at aten.cat::7 (aten::cat): 'str' object has no attribute 'ndim' | None args at position |
| Flex.1-alpha | small512x512 | 5 | other: [ERROR] Pipeline failed: Failed at aten.addmm::7 (aten::addmm): addmm shape mismatch: (2048, 1024) @ (4096, 3072) — the  |
| Flex.1-alpha | small512x512 | 5 | other: [ERROR] Pipeline failed: [triton-sequential] Failed at aten.addmm::7 (aten::addmm): AssertionError: addmm shape mismatch |
| Open-Sora-v2 | frames_small17 | 5 | a streamed piece cannot bind a symbol |
| Open-Sora-v2 | frames_small17 | 3 | broadcast / matmul shape mismatch |
| PixArt-Sigma-XL-1024 | large512x2048 | 6 | a streamed piece cannot bind a symbol |
| PixArt-Sigma-XL-1024 | small512x512 | 6 | a streamed piece cannot bind a symbol |
| PixArt-Sigma-XL-2-1024-MS | large512x2048 | 2 | broadcast / matmul shape mismatch |
| PixArt-Sigma-XL-2-1024-MS | small512x512 | 2 | broadcast / matmul shape mismatch |
| PixArt-XL-1024 | large512x2048 | 6 | a streamed piece cannot bind a symbol |
| PixArt-XL-1024 | small512x512 | 6 | a streamed piece cannot bind a symbol |
| PixArt-XL-2-1024-MS | large512x2048 | 2 | broadcast / matmul shape mismatch |
| PixArt-XL-2-1024-MS | small512x512 | 2 | broadcast / matmul shape mismatch |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | 2 | a streamed piece cannot bind a symbol |
| SANA-Video_2B_720p_diffusers | frames_small41 | 2 | a streamed piece cannot bind a symbol |
| SANA-Video_2B_720p_diffusers | large672x1344 | 2 | a streamed piece cannot bind a symbol |
| SANA-Video_2B_720p_diffusers | small480x832 | 2 | a streamed piece cannot bind a symbol |
| Sana-1600M-MultiLing | large512x2048 | 12 | broadcast / matmul shape mismatch |
| Sana-1600M-MultiLing | small512x512 | 12 | broadcast / matmul shape mismatch |
| Sana_1600M_1024px_MultiLing | large512x2048 | 12 | broadcast / matmul shape mismatch |
| Sana_1600M_1024px_MultiLing | small512x512 | 12 | broadcast / matmul shape mismatch |
| Sana_1600M_4Kpx_BF16 | large2048x8192 | 6 | broadcast / matmul shape mismatch |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | 2 | broadcast / matmul shape mismatch |
| Wan2.1-VACE-1.3B-diffusers | frames_small33 | 12 | broadcast / matmul shape mismatch |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | 2 | broadcast / matmul shape mismatch |
| Wan2.1-VACE-1.3B-diffusers | small512x512 | 12 | broadcast / matmul shape mismatch |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | 4 | broadcast / matmul shape mismatch |

## Every refused run, with its component and rung

Classes: 254 x activations alone exceed the rung (tiling); 114 x weights over the rung, still refused

| container | size | mode | rung | component | MB | activations | class |
|---|---|---|---:|---|---:|---:|---|
| Allegro | frames_largeDefault88 | triton | 4096 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | frames_largeDefault88 | triton | 6144 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | frames_largeDefault88 | triton | 8192 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | frames_largeDefault88 | triton-sequential | 4096 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | frames_largeDefault88 | triton-sequential | 6144 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | frames_largeDefault88 | triton-sequential | 8192 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | frames_small40 | triton | 4096 | `vae` | 113622 | 108000 | activations alone exceed the rung (tiling) |
| Allegro | frames_small40 | triton | 6144 | `vae` | 113622 | 108000 | activations alone exceed the rung (tiling) |
| Allegro | frames_small40 | triton | 8192 | `vae` | 113622 | 108000 | activations alone exceed the rung (tiling) |
| Allegro | frames_small40 | triton-sequential | 4096 | `vae` | 113622 | 108000 | activations alone exceed the rung (tiling) |
| Allegro | frames_small40 | triton-sequential | 6144 | `vae` | 113622 | 108000 | activations alone exceed the rung (tiling) |
| Allegro | frames_small40 | triton-sequential | 8192 | `vae` | 113622 | 108000 | activations alone exceed the rung (tiling) |
| Allegro | largeDefault720x1280 | triton | 4096 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | largeDefault720x1280 | triton | 6144 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | largeDefault720x1280 | triton | 8192 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | largeDefault720x1280 | triton-sequential | 4096 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | largeDefault720x1280 | triton-sequential | 6144 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | largeDefault720x1280 | triton-sequential | 8192 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro | small368x640 | triton | 4096 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro | small368x640 | triton | 6144 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro | small368x640 | triton | 8192 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro | small368x640 | triton-sequential | 4096 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro | small368x640 | triton-sequential | 6144 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro | small368x640 | triton-sequential | 8192 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_largeDefault88 | triton | 4096 | `vae` | 54553 | 51744 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_largeDefault88 | triton | 6144 | `vae` | 54553 | 51744 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_largeDefault88 | triton | 8192 | `vae` | 54553 | 51744 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_largeDefault88 | triton-sequential | 4096 | `vae` | 54553 | 51744 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_largeDefault88 | triton-sequential | 6144 | `vae` | 54553 | 51744 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_largeDefault88 | triton-sequential | 8192 | `vae` | 54553 | 51744 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_small40 | triton | 4096 | `vae` | 24918 | 23520 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_small40 | triton | 6144 | `vae` | 24918 | 23520 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_small40 | triton | 8192 | `vae` | 24918 | 23520 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_small40 | triton-sequential | 4096 | `vae` | 24918 | 23520 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_small40 | triton-sequential | 6144 | `vae` | 24918 | 23520 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | frames_small40 | triton-sequential | 8192 | `vae` | 24918 | 23520 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton | 4096 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton | 6144 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton | 8192 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton | 11264 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton-sequential | 4096 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton-sequential | 6144 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton-sequential | 8192 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | largeDefault720x1280 | triton-sequential | 11264 | `vae` | 249702 | 237600 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | small368x640 | triton | 4096 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | small368x640 | triton | 6144 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | small368x640 | triton | 8192 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | small368x640 | triton-sequential | 4096 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | small368x640 | triton-sequential | 6144 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| Allegro-TI2V | small368x640 | triton-sequential | 8192 | `vae` | 63978 | 60720 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_largeDefault49 | triton | 4096 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_largeDefault49 | triton | 6144 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_largeDefault49 | triton | 8192 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_largeDefault49 | triton-sequential | 4096 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_largeDefault49 | triton-sequential | 6144 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_largeDefault49 | triton-sequential | 8192 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_small25 | triton | 4096 | `vae` | 58719 | 55688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_small25 | triton | 6144 | `vae` | 58719 | 55688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_small25 | triton-sequential | 4096 | `vae` | 58719 | 55688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | frames_small25 | triton-sequential | 6144 | `vae` | 58719 | 55688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | largeDefault480x720 | triton | 4096 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | largeDefault480x720 | triton | 6144 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | largeDefault480x720 | triton | 8192 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | largeDefault480x720 | triton-sequential | 4096 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | largeDefault480x720 | triton-sequential | 6144 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | largeDefault480x720 | triton-sequential | 8192 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | small320x480 | triton | 4096 | `vae` | 38835 | 36750 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | small320x480 | triton | 6144 | `vae` | 38835 | 36750 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | small320x480 | triton | 8192 | `vae` | 38835 | 36750 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | small320x480 | triton-sequential | 4096 | `vae` | 38835 | 36750 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | small320x480 | triton-sequential | 6144 | `vae` | 38835 | 36750 | activations alone exceed the rung (tiling) |
| CogVideoX-2b | small320x480 | triton-sequential | 8192 | `vae` | 38835 | 36750 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_largeDefault49 | triton | 4096 | `vae` | 50668 | 48020 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_largeDefault49 | triton | 6144 | `vae` | 50668 | 48020 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_largeDefault49 | triton | 8192 | `vae` | 50668 | 48020 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_largeDefault49 | triton-sequential | 4096 | `vae` | 50668 | 48020 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_largeDefault49 | triton-sequential | 6144 | `vae` | 50668 | 48020 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_largeDefault49 | triton-sequential | 8192 | `vae` | 50668 | 48020 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_small25 | triton | 4096 | `vae` | 34204 | 32340 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_small25 | triton | 6144 | `vae` | 34204 | 32340 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_small25 | triton-sequential | 4096 | `vae` | 34204 | 32340 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | frames_small25 | triton-sequential | 6144 | `vae` | 34204 | 32340 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | largeDefault480x720 | triton | 4096 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | largeDefault480x720 | triton | 6144 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | largeDefault480x720 | triton | 8192 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | largeDefault480x720 | triton-sequential | 4096 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | largeDefault480x720 | triton-sequential | 6144 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| CogVideoX-5b-I2V | largeDefault480x720 | triton-sequential | 8192 | `vae` | 87069 | 82688 | activations alone exceed the rung (tiling) |
| Flex.1-alpha | largeDefault1024x1024 | triton | 12288 | `transformer` | 16977 | 599 | weights over the rung, still refused |
| Flex.1-alpha | largeDefault1024x1024 | triton-sequential | 12288 | `transformer` | 16977 | 599 | weights over the rung, still refused |
| Flex.1-alpha | small512x512 | triton | 12288 | `transformer` | 16977 | 599 | weights over the rung, still refused |
| Flex.1-alpha | small512x512 | triton-sequential | 12288 | `transformer` | 16977 | 599 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton | 4096 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton | 6144 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton | 8192 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton | 11264 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton | 12288 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton | 16384 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton-sequential | 4096 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton-sequential | 6144 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton-sequential | 8192 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton-sequential | 11264 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton-sequential | 12288 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_large125 | triton-sequential | 16384 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_small17 | triton | 11264 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_small17 | triton | 12288 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_small17 | triton-sequential | 11264 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | frames_small17 | triton-sequential | 12288 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | large576x1024 | triton | 4096 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton | 6144 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton | 8192 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton | 11264 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton | 12288 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton | 16384 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton-sequential | 4096 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton-sequential | 6144 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton-sequential | 8192 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton-sequential | 11264 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton-sequential | 12288 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | large576x1024 | triton-sequential | 16384 | `vae` | 116265 | 110450 | activations alone exceed the rung (tiling) |
| Open-Sora-v2 | small256x256 | triton | 4096 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton | 6144 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton | 8192 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton | 11264 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton | 12288 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton | 16384 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton-sequential | 4096 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton-sequential | 6144 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton-sequential | 8192 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton-sequential | 11264 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton-sequential | 12288 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| Open-Sora-v2 | small256x256 | triton-sequential | 16384 | `transformer` | 24043 | 217 | weights over the rung, still refused |
| PixArt-Sigma-XL-2-1024-MS | large512x2048 | triton | 4096 | `text_encoder` | 9763 | 215 | weights over the rung, still refused |
| PixArt-Sigma-XL-2-1024-MS | large512x2048 | triton-sequential | 4096 | `text_encoder` | 9763 | 215 | weights over the rung, still refused |
| PixArt-Sigma-XL-2-1024-MS | small512x512 | triton | 4096 | `text_encoder` | 9763 | 215 | weights over the rung, still refused |
| PixArt-Sigma-XL-2-1024-MS | small512x512 | triton-sequential | 4096 | `text_encoder` | 9763 | 215 | weights over the rung, still refused |
| PixArt-XL-2-1024-MS | large512x2048 | triton | 4096 | `text_encoder` | 9723 | 177 | weights over the rung, still refused |
| PixArt-XL-2-1024-MS | large512x2048 | triton-sequential | 4096 | `text_encoder` | 9723 | 177 | weights over the rung, still refused |
| PixArt-XL-2-1024-MS | small512x512 | triton | 4096 | `text_encoder` | 9723 | 177 | weights over the rung, still refused |
| PixArt-XL-2-1024-MS | small512x512 | triton-sequential | 4096 | `text_encoder` | 9723 | 177 | weights over the rung, still refused |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton | 4096 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton | 6144 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton | 11264 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton | 12288 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton | 16384 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton-sequential | 4096 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton-sequential | 6144 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton-sequential | 11264 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton-sequential | 12288 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_largeDefault81 | triton-sequential | 16384 | `vae` | 151208 | 143000 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton | 4096 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton | 6144 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton | 11264 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton | 12288 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton-sequential | 4096 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton-sequential | 6144 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton-sequential | 11264 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | frames_small41 | triton-sequential | 12288 | `vae` | 121640 | 114840 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton | 4096 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton | 6144 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton | 11264 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton | 12288 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton | 16384 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton-sequential | 4096 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton-sequential | 6144 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton-sequential | 11264 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton-sequential | 12288 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | large672x1344 | triton-sequential | 16384 | `vae` | 151549 | 143325 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton | 4096 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton | 6144 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton | 11264 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton | 12288 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton-sequential | 4096 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton-sequential | 6144 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton-sequential | 11264 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| SANA-Video_2B_720p_diffusers | small480x832 | triton-sequential | 12288 | `vae` | 67602 | 63375 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton | 4096 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton | 6144 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton | 8192 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton | 11264 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton | 12288 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton | 16384 | `transformer` | 47405 | 13876 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton-sequential | 4096 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton-sequential | 6144 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton-sequential | 8192 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton-sequential | 11264 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton-sequential | 12288 | `transformer` | 47405 | 13876 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | frames_large161 | triton-sequential | 16384 | `transformer` | 47405 | 13876 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton | 4096 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton | 6144 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton | 8192 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton | 11264 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton | 12288 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton | 16384 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton-sequential | 4096 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton-sequential | 6144 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton-sequential | 8192 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton-sequential | 11264 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton-sequential | 12288 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | frames_small33 | triton-sequential | 16384 | `transformer` | 37106 | 4068 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton | 4096 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton | 6144 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton | 8192 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton | 11264 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton | 12288 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton | 16384 | `transformer` | 45771 | 12320 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton-sequential | 4096 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton-sequential | 6144 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton-sequential | 8192 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton-sequential | 11264 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton-sequential | 12288 | `transformer` | 45771 | 12320 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | large832x480 | triton-sequential | 16384 | `transformer` | 45771 | 12320 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton | 4096 | `transformer` | 42539 | 9243 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton | 6144 | `transformer` | 42539 | 9243 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton | 8192 | `transformer` | 42539 | 9243 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton | 11264 | `transformer` | 42539 | 9243 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton | 12288 | `transformer` | 42539 | 9243 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton | 16384 | `transformer` | 42539 | 9243 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton-sequential | 4096 | `transformer` | 42539 | 9243 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton-sequential | 6144 | `transformer` | 42539 | 9243 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton-sequential | 8192 | `transformer` | 42539 | 9243 | activations alone exceed the rung (tiling) |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton-sequential | 11264 | `transformer` | 42539 | 9243 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton-sequential | 12288 | `transformer` | 42539 | 9243 | weights over the rung, still refused |
| Wan2.1-I2V-14B-480P-Diffusers | small624x624 | triton-sequential | 16384 | `transformer` | 42539 | 9243 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton | 4096 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton | 6144 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton | 8192 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton | 11264 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton-sequential | 4096 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton-sequential | 6144 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton-sequential | 8192 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_large161 | triton-sequential | 11264 | `vae` | 74789 | 71088 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton | 4096 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton | 6144 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton | 8192 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton | 11264 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton-sequential | 4096 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton-sequential | 6144 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton-sequential | 8192 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | frames_small33 | triton-sequential | 11264 | `vae` | 15692 | 14805 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton | 4096 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton | 6144 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton | 8192 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton | 11264 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton-sequential | 4096 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton-sequential | 6144 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton-sequential | 8192 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | large720x1280 | triton-sequential | 11264 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton | 4096 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton | 6144 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton | 8192 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton | 11264 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton-sequential | 4096 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton-sequential | 6144 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton-sequential | 8192 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-T2V-1.3B-Diffusers | small240x416 | triton-sequential | 11264 | `text_encoder` | 11571 | 185 | weights over the rung, still refused |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton | 4096 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton | 6144 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton | 8192 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton | 11264 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton | 12288 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton-sequential | 4096 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton-sequential | 6144 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton-sequential | 8192 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton-sequential | 11264 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | frames_large161 | triton-sequential | 12288 | `vae` | 37689 | 35755 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton | 4096 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton | 6144 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton | 8192 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton | 11264 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton | 12288 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton-sequential | 4096 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton-sequential | 6144 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton-sequential | 8192 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton-sequential | 11264 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.1-VACE-1.3B-diffusers | large832x480 | triton-sequential | 12288 | `vae` | 37853 | 35911 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton | 4096 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton | 6144 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton | 8192 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton | 11264 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton | 12288 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton | 16384 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton-sequential | 4096 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton-sequential | 6144 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton-sequential | 8192 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton-sequential | 11264 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton-sequential | 12288 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_largeDefault81 | triton-sequential | 16384 | `transformer` | 30838 | 2115 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton | 4096 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton | 6144 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton | 8192 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton | 16384 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton-sequential | 4096 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton-sequential | 6144 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton-sequential | 8192 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | frames_small33 | triton-sequential | 16384 | `transformer` | 29588 | 925 | weights over the rung, still refused |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton | 4096 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton | 6144 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton | 8192 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton | 11264 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton | 12288 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton | 16384 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton-sequential | 4096 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton-sequential | 6144 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton-sequential | 8192 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton-sequential | 11264 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton-sequential | 12288 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | large720x1280 | triton-sequential | 16384 | `vae` | 87096 | 82809 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton | 4096 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton | 6144 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton | 8192 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton | 11264 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton | 12288 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton | 16384 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton-sequential | 4096 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton-sequential | 6144 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton-sequential | 8192 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton-sequential | 11264 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton-sequential | 12288 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| Wan2.2-I2V-A14B-Diffusers | small624x624 | triton-sequential | 16384 | `vae` | 36909 | 35011 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton | 4096 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton | 6144 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton | 8192 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton | 11264 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton | 12288 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton | 16384 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton-sequential | 4096 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton-sequential | 6144 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton-sequential | 8192 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton-sequential | 11264 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton-sequential | 12288 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_large163 | triton-sequential | 16384 | `vae` | 106191 | 100788 | activations alone exceed the rung (tiling) |
| mochi-1-preview | frames_small31 | triton | 4096 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton | 6144 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton | 8192 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton | 11264 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton | 12288 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton | 16384 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton-sequential | 4096 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton-sequential | 6144 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton-sequential | 8192 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton-sequential | 11264 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton-sequential | 12288 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | frames_small31 | triton-sequential | 16384 | `transformer` | 42496 | 2220 | weights over the rung, still refused |
| mochi-1-preview | largeDefault480x848 | triton | 4096 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton | 6144 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton | 8192 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton | 11264 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton | 12288 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton | 16384 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton-sequential | 4096 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton-sequential | 6144 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton-sequential | 8192 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton-sequential | 11264 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton-sequential | 12288 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | largeDefault480x848 | triton-sequential | 16384 | `vae` | 53487 | 50594 | activations alone exceed the rung (tiling) |
| mochi-1-preview | small320x576 | triton | 4096 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton | 6144 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton | 8192 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton | 11264 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton | 12288 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton | 16384 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton-sequential | 4096 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton-sequential | 6144 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton-sequential | 8192 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton-sequential | 11264 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton-sequential | 12288 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
| mochi-1-preview | small320x576 | triton-sequential | 16384 | `transformer` | 41649 | 1413 | weights over the rung, still refused |
