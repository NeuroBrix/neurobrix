#!/usr/bin/env python3
"""Vendor reproduction of one image request (diffusers, the model's own snapshot), for a retrace
gate whose two ATen arms differ: the arm closer to the vendor's render at the SAME prompt, seed,
steps and guidance is the right one. Runs under the diffusers bench venv (torch is allowed
under tools/); CPU offload so a 16 GB card carries a T5-XXL pipeline.

  python tools/vendor_image_repro.py --snapshot ~/hf_snapshots/PixArt-XL-2-1024-MS \
      --prompt "a red apple on a wooden table" --seed 42 --steps 20 --guidance 4.5 --out vendor.png
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-offload", action="store_true")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    pipe = AutoPipelineForText2Image.from_pretrained(a.snapshot, torch_dtype=dtype)
    if a.no_offload:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    t0 = time.time()
    img = pipe(prompt=a.prompt, num_inference_steps=a.steps, guidance_scale=a.guidance,
               height=a.height, width=a.width, generator=gen).images[0]
    img.save(a.out)
    meta = {"snapshot": a.snapshot, "prompt": a.prompt, "seed": a.seed, "steps": a.steps, "guidance": a.guidance,
            "height": a.height, "width": a.width, "dtype": a.dtype, "offload": not a.no_offload,
            "seconds": round(time.time() - t0, 1), "diffusers": __import__("diffusers").__version__,
            "torch": torch.__version__}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[vendor] saved {a.out} in {meta['seconds']} s ({meta['diffusers']}, {meta['torch']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
