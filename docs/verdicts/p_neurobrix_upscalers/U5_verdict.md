# P-NEUROBRIX-UPSCALERS — U5 verdict (Real-ESRGAN)

## Outcome: 12/12 real-esrgan cells ✓ — ZERO NeuroBrix code change

Real-ESRGAN x2 / x4 / x8 (RRDBNet pure-CNN, family upscaler_cnn)
operational across all 4 execution modes. **No NeuroBrix runtime
code change was required** — the generic `nbx upscale` CLI (U3)
+ roll/reflection_pad2d R33-pure kernels (U4) + data-driven
forward_pass handling consumed the new containers transparently.
This is the R34 model-agnostic design paying off: a brand-new
architecture family integrated with registry + vendor changes
only (container side), zero runtime branching.

## Matrix (input 64×64, 32g)

| model | compiled | sequential | triton | triton-seq |
|---|---|---|---|---|
| real-esrgan-x2 | 128² 0.85s | 128² 0.68s | 128² 21s† | 128² 1.8s |
| real-esrgan-x4 | 256² 0.92s | 256² 0.81s | 256² 17s | 256² 2.1s |
| real-esrgan-x8 | 512² 1.16s | 512² 0.85s | 512² 29s | 512² 2.0s |

† first-run Triton JIT compile; scale ratios all exact
(x2→128, x4→256, x8→512).

## R32 cross-mode

cosine vs compiled = **1.00000** for all 9 non-compiled cells
(sequential / triton / triton-seq × x2/x4/x8). R30 dualité
confirmed.

## Anti-régression matrice v2 (PRESERVED)

- TinyLlama compiled 32g : 4.21 s, coherent haiku (= 4.24 s ref)
- Sana 4Kpx 32g compiled : 23.43 s (= ~23 s baseline)

## Visual (R29)

real-esrgan-x4 triton: coherent red-apple upscale, sharper than
swin2sr (RRDBNet GAN detail synthesis). Sample committed:
`real_esrgan_x4_triton_sample.png`.

## Hub

Future publication target: neurobrix.es (proprietary). No
huggingface.co publication.
