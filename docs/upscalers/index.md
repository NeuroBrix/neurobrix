# NeuroBrix Upscalers

Pure image super-resolution models, runnable with the
`neurobrix upscale` subcommand. Every model upscales by a fixed
intrinsic factor. Each one has been run, and its output judged,
on both engines (PyTorch and Triton), on NVIDIA and on Apple GPUs.

```bash
neurobrix upscale \
    --model <name> \
    --input low_res.png \
    --output upscaled.png
```

`--input-image` is accepted as an alias of `--input`. The engine is
chosen as `neurobrix run` chooses it: the PyTorch engine where PyTorch
is installed, the Triton engine where it is not. To choose it yourself,
pass one of `--compiled`, `--sequential`, `--triton`,
`--triton-sequential`, or the same choice as
`--mode compiled|sequential|triton|triton-sequential`.

Output is a PNG at `input × scale`.

## Catalogue

| Model | Scale | Class | Best for |
|-------|-------|-------|----------|
| `caidas/swin2SR-classical-sr-x2-64` | 2× | Swin2SR transformer | clean images |
| `caidas/swin2SR-classical-sr-x4-64` | 4× | Swin2SR transformer | clean images |
| `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | 4× | Swin2SR transformer | degraded / real-world photos |
| `ai-forever/Real-ESRGAN-RealESRGAN_x2` | 2× | RRDBNet CNN (GAN) | photos, detail synthesis |
| `xinntao/Real-ESRGAN-RealESRGAN_x4plus` | 4× | RRDBNet CNN (GAN) | photos, detail synthesis |
| `ai-forever/Real-ESRGAN-RealESRGAN_x8` | 8× | RRDBNet CNN (GAN) | large upscales |
| `JingyunLiang/SwinIR-001_classicalSR_DF2K_s64w8_SwinIR-M_x2` | 2× | SwinIR transformer | high-fidelity clean images |
| `JingyunLiang/SwinIR-001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` | 4× | SwinIR transformer | high-fidelity clean images |
| `XPixelGroup/HAT-HAT-S_SRx4` | 4× | HAT (hybrid attention) | sharp detail, compact |
| `XPixelGroup/HAT-HAT-L_SRx4_ImageNet-pretrain` | 4× | HAT (hybrid attention) | maximum quality |

## Choosing a model

- **Robust, every engine**: Real-ESRGAN — a pure CNN.
- **Best fidelity on clean sources**: SwinIR classical.
- **Highest quality (heavier)**: HAT-L.
- **Real-world / degraded photos**: Swin2SR realworld or
  Real-ESRGAN.

## Notes

- Values beyond fp16's range are computed in float32 where the
  hardware needs it; there is nothing to set.
- The input image is padded to the alignment the container
  declares, and the output is cropped to exactly the input
  upscaled by the model's intrinsic factor.

## Distribution

All ten models above are in the NeuroBrix catalogue and are
downloaded from the NeuroBrix hub (neurobrix.org):
`neurobrix hub --category UPSCALER` lists them, and
`neurobrix import <name>` installs one.
