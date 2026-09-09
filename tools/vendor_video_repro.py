#!/usr/bin/env python3
"""Vendor reproduction of one video request (diffusers, the model's own snapshot).

The video half of the vendor-correctness cell. Same contract as
`vendor_image_repro.py`: the vendor renders the request our engine rendered, at
the SAME prompt, steps, frame count and resolution, and — with `--latents` — from
OUR initial noise, so the comparison is between two computations and not between
two valid samples.

Pinning matters more here than for a still. An independent sample differs in
every frame, which would mask exactly the defects this cell exists to catch: a
frozen frame count, a conv3d decomposition that drops frames, a temporal axis
that silently collapses. Read from the pipeline (diffusers 0.38.0):
WanPipeline.__call__ accepts `latents`, so the video family uses the PRIMARY
latent-pinned gate and does not fall back to the semantic net.

Runs under the diffusers_video bench venv (torch is allowed under tools/).

  python tools/vendor_video_repro.py --snapshot ~/hf_snapshots/Wan2.1-T2V-1.3B-Diffusers \
      --prompt "a red apple rolling slowly across a wooden table" --steps 4 \
      --latents ours.npy --out vendor_first_frame.png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--latents", default=None,
                    help="path to a .npy initial latent (ours) the vendor must start from")
    ap.add_argument("--out", required=True, help="PNG of the FIRST frame")
    ap.add_argument("--frames-dir", default=None,
                    help="write every frame as a lossless PNG, straight from the "
                         "pipeline's array. This is the arm a decode differential "
                         "compares against: no codec on either side.")
    ap.add_argument("--clip-out", default=None,
                    help="also write the clip as mp4. The cell then extracts frame 0 from "
                         "BOTH arms' mp4 by the same path: our engine writes a clip, so a "
                         "vendor PNG taken straight from the array is compared across a lossy "
                         "boundary only one side crossed (measured: 36.6 dB of the gap).")
    ap.add_argument("--no-offload", action="store_true")
    a = ap.parse_args()

    dtype = getattr(torch, a.dtype)
    # Wan's VAE is trained in fp32 and the model card keeps it there; only the
    # transformer takes the reduced dtype. Read from the vendor, not chosen.
    vae = AutoencoderKLWan.from_pretrained(a.snapshot, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(a.snapshot, vae=vae, torch_dtype=dtype)
    if a.no_offload:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    extra = {}
    if a.latents:
        arr = np.load(a.latents)
        extra["latents"] = torch.from_numpy(np.ascontiguousarray(arr)).to(
            device="cuda", dtype=dtype)
        print(f"[vendor] starting from our latent {tuple(arr.shape)} {arr.dtype}")

    t0 = time.time()
    result = pipe(prompt=a.prompt, num_inference_steps=a.steps,
                  guidance_scale=a.guidance, num_frames=a.frames,
                  height=a.height, width=a.width, generator=gen,
                  output_type="np", **extra)
    frames = result.frames[0]
    first = frames[0]
    if first.dtype != np.uint8:
        first = (np.clip(first, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(first).save(a.out)
    if a.frames_dir:
        import os
        os.makedirs(a.frames_dir, exist_ok=True)
        for i, f in enumerate(frames):
            fr = f if f.dtype == np.uint8 else (np.clip(f, 0.0, 1.0) * 255).round().astype(np.uint8)
            Image.fromarray(fr).save(os.path.join(a.frames_dir, f"frame{i:03d}.png"))
        print(f"[vendor] {len(frames)} lossless frame(s) -> {a.frames_dir}")
    if a.clip_out:
        import imageio.v2 as iio
        w = iio.get_writer(a.clip_out, fps=16)
        for f in frames:
            fr = f if f.dtype == np.uint8 else (np.clip(f, 0.0, 1.0) * 255).round().astype(np.uint8)
            w.append_data(fr)
        w.close()

    meta = {"snapshot": a.snapshot, "prompt": a.prompt, "seed": a.seed,
            "steps": a.steps, "guidance": a.guidance, "frames": a.frames,
            "height": a.height, "width": a.width, "dtype": a.dtype,
            "offload": not a.no_offload, "latents_from": a.latents,
            "gate": "latent_pinned" if a.latents else "independent_sample",
            "rendered_frames": int(len(frames)),
            "seconds": round(time.time() - t0, 1),
            "diffusers": __import__("diffusers").__version__,
            "torch": torch.__version__}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[vendor] {len(frames)} frame(s), first saved to {a.out} "
          f"in {meta['seconds']} s ({meta['diffusers']}, {meta['torch']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
