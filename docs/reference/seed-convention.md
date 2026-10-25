# The seed convention — one seed, one image, on every hardware

Decided 2026-10-24 (owner's supervisor, 23:44), after the Sana_1600M_1024px_MultiLing comparison against the vendor.

**A NeuroBrix seed draws the stream of PyTorch's CPU generator** (`torch.Generator("cpu").manual_seed(seed)`:
mt19937 + Box-Muller), bit for bit, on every platform and in both engines: the compiled engine threads one such
generator through the run; the Triton engine's house random kernels (`kernels/rng_stream`) reproduce the same stream
without torch (R33). The same seed therefore renders the same image on NVIDIA, on Metal and on a CPU — the only
convention that can: a device generator (CUDA Philox, MPS) gives a different stream per platform and per vendor
version.

A vendor example that draws its noise on the device (`torch.Generator("cuda")`, the usual model-card snippet) renders
another sample for the same integer seed **by construction, not by defect**. Comparing the engine with a vendor
pipeline therefore uses either `torch.Generator("cpu").manual_seed(seed)` on the vendor's side, or the engine's dumped
initial noise handed to the vendor (`NBX_DUMP_INIT_LATENT`, `tools/vendor_image_repro.py --latents`).

## The measurement that decided it

Sana_1600M_1024px_MultiLing, "a red apple on a wooden table", 1024×1024, 20 steps, guidance 4.5, seed 42; NeuroBrix
main 7bfe9077 on a V100-32GB (certified-only), diffusers 0.37 in float32 on the same card:

| vendor render | PSNR against NeuroBrix native | by eye |
|---|---|---|
| `torch.Generator("cuda")`, seed 42 | 11.2 dB | another composition (framing, light, background) |
| the engine's own initial noise handed to the vendor | 44.4 dB | the same image |
| `torch.Generator("cpu")`, seed 42, nothing pinned | 44.5 dB | the same image |
| NeuroBrix triton, seed 42 | 43.8 dB | the same image |

The engine's initial latent correlates 1.0000 with `torch.randn` under the CPU generator for seed 42 (0.001 under the
CUDA generator). Everything after the noise draw — prompt pre-processing (the complex human instruction), the Gemma
text encoder, tokenisation and mask, the flow-match scheduler, the transformer, the DC-AE decoder — equals the vendor's.
Sheet (vendor cuda | vendor cpu | native | triton) on the model's hub page and in the Dell's register
(`registres/verdicts/sana_multiling_r29_2026_10_24*`).
