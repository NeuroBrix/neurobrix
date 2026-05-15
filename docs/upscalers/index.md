# NeuroBrix Upscalers

Pure image super-resolution models, runnable with the
`neurobrix upscale` subcommand. Every model upscales by a fixed
intrinsic factor and produces numerically equivalent output
across the supported execution modes.

```bash
neurobrix upscale \
    --model <name> \
    --input low_res.png \
    --output upscaled.png \
    --mode compiled        # compiled (default) / sequential / triton / triton-sequential
```

Output is a PNG at `input × scale`.

## Catalogue

| Model | Scale | Class | Best for | Modes |
|-------|-------|-------|----------|-------|
| `swin2SR-classical-sr-x2-64` | 2× | Swin2SR transformer | clean images | all 4 |
| `swin2SR-classical-sr-x4-64` | 4× | Swin2SR transformer | clean images | all 4 |
| `swin2SR-realworld-sr-x4-64-bsrgan-psnr` | 4× | Swin2SR transformer | degraded / real-world photos | all 4 |
| `real-esrgan-x2` | 2× | RRDBNet CNN (GAN) | photos, detail synthesis | all 4 |
| `real-esrgan-x4` | 4× | RRDBNet CNN (GAN) | photos, detail synthesis | all 4 |
| `real-esrgan-x8` | 8× | RRDBNet CNN (GAN) | large upscales | all 4 |
| `swinir-classical-x2` | 2× | SwinIR transformer | high-fidelity clean images | all 4 |
| `swinir-classical-x4` | 4× | SwinIR transformer | high-fidelity clean images | all 4 |
| `hat-s-x4` | 4× | HAT (hybrid attention) | sharp detail, compact | compiled / sequential |
| `hat-l-x4` | 4× | HAT (hybrid attention) | maximum quality | compiled / sequential |

## Choosing a model

- **Fastest, robust**: Real-ESRGAN — pure-CNN, no fp16
  constraints, every mode.
- **Best fidelity on clean sources**: SwinIR classical.
- **Highest quality (heavier)**: HAT-L.
- **Real-world / degraded photos**: Swin2SR realworld or
  Real-ESRGAN.

## Notes

- Transformer SR models (SwinIR, HAT) run their compute in
  float32 on fp16-preferred hardware — their activation range
  exceeds fp16; this is handled automatically.
- HAT currently runs in `compiled` and `sequential` modes; its
  `triton` / `triton-sequential` support is in progress.
- Input is processed at each container's native tile size; the
  output is the input upscaled by the model's intrinsic factor.

## Distribution

These models are published on the NeuroBrix hub
(neurobrix.es). They are not distributed via third-party model
hosts.
