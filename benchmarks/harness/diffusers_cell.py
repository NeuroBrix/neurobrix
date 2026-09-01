"""HF diffusers competitor cell (image + video rows).

Runs inside the pinned ~/bench_venvs/diffusers venv (backends.yml:
diffusers==0.35.2, torch 2.5.1+cu121 — era pins, V100 dtype doctrine).
Prints ONE JSON object on the last stdout line:
  {"cold_start_s": float, "requests": [...], "pins": {...}}

Fairness contract (backends.yml `diffusers.volta_notes`):
- fp16 transformer; Sana TE+VAE fp32, Wan VAE fp32 (never bf16 on
  sm_70 — CPU-emulation trap, sourced).
- Stock attention = torch SDPA (AttnProcessor2_0, mem-efficient on
  sm_70). No slicing/tiling/offload/compile unless listed in `pins`.
- Checkpoint-shipped scheduler; steps/resolution/seed pinned by the
  row on BOTH columns; per-request seeded CUDA Generator.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def load_pipeline(row: dict):
    import torch
    from diffusers import DiffusionPipeline

    ckpt = row.get("diffusers_checkpoint") or row["checkpoint"]
    snap = Path.home() / "hf_snapshots" / ckpt.split("/")[-1]
    src = str(snap if snap.exists() else ckpt)

    t0 = time.perf_counter()
    # Base dtype is fp16 (sm_70 doctrine) unless the row's recipe pins
    # another (SANA-Video: upstream mandates bf16 transformer — fp16
    # tests are skipped for numerical instability; bf16 runs
    # unaccelerated on sm_70 and the annex row says so).
    _dtype_name = (row.get("diffusers_recipe") or {}).get(
        "torch_dtype", "float16")
    pipe = DiffusionPipeline.from_pretrained(
        src, torch_dtype=getattr(torch, _dtype_name))
    # V100 dtype doctrine (sourced, backends.yml): TE/VAE precision.
    fixes = []
    offloaded = False
    if row["metric_class"] == "image" and hasattr(pipe, "text_encoder"):
        # Sana class: TE (Gemma2) + DC-AE VAE must not run fp16.
        if "sana" in type(pipe).__name__.lower():
            pipe.text_encoder = pipe.text_encoder.to(torch.float32)
            pipe.vae = pipe.vae.to(torch.float32)
            fixes.append("text_encoder=fp32, vae=fp32 (Sana doctrine)")
        # FLUX class (Flex.1): HF Flux docs — fp16 text encoders change
        # outputs on Turing/Volta, fp32 TEs remove the difference; and
        # 26.3 GB fp16 weights do not sit resident with activations on
        # one 32 GB card, so the vendor cpu-offload path applies (S3
        # R16 synthesis, sourced).
        if "flux" in type(pipe).__name__.lower():
            pipe.text_encoder = pipe.text_encoder.to(torch.float32)
            if getattr(pipe, "text_encoder_2", None) is not None:
                pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.float32)
            pipe.enable_model_cpu_offload()
            offloaded = True
            fixes.append("text encoders=fp32 + enable_model_cpu_offload "
                         "(Flux/V100 doctrine)")
    if row["metric_class"] == "video":
        # Per-row vendor recipe (rows.yml `diffusers_recipe`), replacing
        # the 2026-08-30 generic "vae=fp32 on every video pipeline"
        # which was itself the campaign's harness hole: it broke
        # CogVideoX (fp16 latents met a float VAE bias in F.conv3d)
        # while Allegro/Mochi/Wan-14B lacked the tiling/offload their
        # own model cards mandate at 32 GB. Each flag is the vendor's
        # documented recipe, recorded in `pins`; absence = stock.
        recipe = row.get("diffusers_recipe") or {}
        if recipe.get("vae_fp32") and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to(torch.float32)
            fixes.append("vae=fp32 (vendor card / sm_70 doctrine)")
        if recipe.get("vae_tiling") and hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()
            fixes.append("vae.enable_tiling() (vendor card)")
        if (recipe.get("image_encoder_fp32")
                and getattr(pipe, "image_encoder", None) is not None):
            # Wan-I2V vendor recipe: the CLIP image encoder runs fp32
            # (all-fp16 CLIP emits a float pooled_output into a half
            # visual_projection — 2026-09-01 cell x2); the pipeline
            # casts image_embeds AND condition latents to the
            # transformer dtype downstream, so fp32 stays contained.
            pipe.image_encoder = pipe.image_encoder.to(torch.float32)
            fixes.append("image_encoder=fp32 (Wan-I2V vendor recipe)")
        if recipe.get("cpu_offload"):
            pipe.enable_model_cpu_offload()
            offloaded = True
            fixes.append("enable_model_cpu_offload() (vendor card — "
                         "weights exceed one-card residency)")
    if not offloaded:
        pipe = pipe.to("cuda")

    # Fairness-arm cache weapon (drift-discipline clause 6): when the
    # campaign sets BENCH_DIFFUSERS_FBC=<threshold>, enable diffusers'
    # own FirstBlockCache on the denoiser (CacheMixin at the 0.35.2
    # pin — Wan qualifies, Sana does not). Absent = stock pipeline,
    # byte-unchanged. The pins block records the weapon.
    cache_note = None
    fbc_thr = os.environ.get("BENCH_DIFFUSERS_FBC")
    if fbc_thr:
        from diffusers.hooks import FirstBlockCacheConfig
        denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet")
        denoiser.enable_cache(FirstBlockCacheConfig(threshold=float(fbc_thr)))
        cache_note = f"FirstBlockCache(threshold={float(fbc_thr)})"
    cold = time.perf_counter() - t0
    return pipe, cold, fixes, src, cache_note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--media-dir", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    row = json.loads(args.row)
    media_dir = Path(args.media_dir)

    import torch
    import diffusers

    pipe, cold, fixes, src, cache_note = load_pipeline(row)
    mclass = row["metric_class"]
    steps = row["steps"]
    _dtype_name = (row.get("diffusers_recipe") or {}).get(
        "torch_dtype", "float16")

    call_kwargs: dict = {
        "prompt": row["prompt"],
        "num_inference_steps": steps,
    }
    if row.get("height"):
        call_kwargs["height"] = row["height"]
    if row.get("width"):
        call_kwargs["width"] = row["width"]
    if mclass == "video" and row.get("num_frames"):
        call_kwargs["num_frames"] = row["num_frames"]
    if row.get("input_image"):
        # I2V rows carry a versioned conditioning image; the 08-30
        # campaign's cells never consumed it (cog5b failed on the
        # missing required `image` argument — harness hole, not DNR).
        # The kwarg name is per-pipeline (WanVACE conditions through
        # `reference_images=[img]`, not `image=`) — mapped by the row's
        # recipe so both columns run the same task.
        from PIL import Image
        _img = Image.open(
            Path(args.repo) / row["input_image"]).convert("RGB")
        _recipe = row.get("diffusers_recipe") or {}
        _kwarg = _recipe.get("image_kwarg", "image")
        call_kwargs[_kwarg] = [_img] if _recipe.get("image_as_list") else _img

    # Flux/V100 doctrine, second half: with the text encoders held fp32,
    # FluxPipeline builds its latents at prompt_embeds.dtype, so fp32
    # embeds meet the fp16 transformer in x_embedder ("mat1 Float,
    # mat2 Half" — Flex.1 arm, 2026-08-30). The sourced recipe encodes
    # at fp32 and hands the transformer fp16 embeds. Encoding stays
    # INSIDE the measured request — the pipe(prompt=...) path also
    # encodes per request, and so does the NeuroBrix column.
    flux_fp16_embeds = "flux" in type(pipe).__name__.lower()

    def one(idx: int) -> dict:
        gen = torch.Generator(device="cuda").manual_seed(row["seed"])
        t0 = time.perf_counter()
        with torch.inference_mode():
            if flux_fp16_embeds:
                pe, ppe, _ = pipe.encode_prompt(
                    prompt=call_kwargs["prompt"], prompt_2=None,
                    device=pipe._execution_device)
                kw = {k: v for k, v in call_kwargs.items()
                      if k != "prompt"}
                out = pipe(prompt_embeds=pe.to(torch.float16),
                           pooled_prompt_embeds=ppe.to(torch.float16),
                           generator=gen, **kw)
            else:
                out = pipe(generator=gen, **call_kwargs)
        # Materialize + save (the NeuroBrix column's request also pays
        # its file write — same measurement boundary).
        if mclass == "image":
            path = media_dir / f"diffusers_r{idx}.png"
            out.images[0].save(str(path))
        else:
            from diffusers.utils import export_to_video
            path = media_dir / f"diffusers_r{idx}.mp4"
            export_to_video(out.frames[0], str(path), fps=8)
        wall = time.perf_counter() - t0
        rec = {"wall_s": wall, "s_per_step": wall / steps,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if mclass == "video" and row.get("num_frames"):
            rec["s_per_frame"] = wall / row["num_frames"]
        return rec

    one(-1)  # warmup (cudnn autotune, first-call graph inits)
    requests = [one(i) for i in range(args.n)]

    print(json.dumps({
        "cold_start_s": cold,
        "requests": requests,
        "pins": {
            "diffusers": diffusers.__version__,
            "torch": torch.__version__,
            "dtype": (_dtype_name + " (+ " + "; ".join(fixes) + ")"
                      if fixes else _dtype_name),
            "attention": "torch SDPA / AttnProcessor2_0 "
                         "(mem-efficient backend on sm_70)",
            "enabled_optims": cache_note or
                              "none (stock pipeline; slicing/tiling/"
                              "offload/compile all off)",
            "scheduler": type(pipe.scheduler).__name__
                         + " (checkpoint-shipped config)",
            "source": src,
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
