"""Upscaler competitor cell — the model's vendor/reference stack, pinned.

Runs inside the row's pinned venv (`upscaler_venv`, invoked by
run_bench.cell_vendor_upscaler) and prints ONE JSON object on its last
stdout line: {"cold_start_s": float, "requests": [...], "pins": {...}}
— same shape as every other cell. Media artifacts go under the R29
tree; shas inside the JSON.

Stacks implemented:
  realesrgan — realesrgan==0.3.0 RealESRGANer (basicsr 1.4.2 RRDBNet),
               half=True on CUDA (the vendor's CUDA default -> fp16
               class, matches the nbx arms' hardware dtype).
  swin2sr    — transformers Swin2SRForImageSuperResolution, vendor
               default fp32 (precision class LABELED in pins; the
               model is 12M params).
  swinir/hat — NOT implemented yet: refused loudly with the missing
               pieces named (SwinIR repo vendoring; HAT basicsr
               1.3.4.9 venv + Drive-hosted weights). The refusal is a
               capability gate, never a silent skip.

Weights/snapshots come from the NAS per the S3 disk policy
(upscaler_weights row field; HF_HOME exported by the parent).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _record(out_path: Path, wall: float) -> dict:
    rec = {"wall_s": wall}
    if out_path.exists():
        rec["sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return rec


def run_realesrgan(row: dict, n: int, repo: Path, media_dir: Path) -> dict:
    import cv2
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    scale = int(row.get("scale", 4))
    t0 = time.perf_counter()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale, model_path=row["upscaler_weights"], model=model,
        tile=0, tile_pad=10, pre_pad=0, half=True)
    cold = time.perf_counter() - t0

    img = cv2.imread(str(repo / row["input_image"]), cv2.IMREAD_COLOR)

    def one(idx: int) -> dict:
        out_path = media_dir / f"vendor_realesrgan_r{idx}.png"
        t = time.perf_counter()
        out, _ = upsampler.enhance(img, outscale=scale)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        cv2.imwrite(str(out_path), out)
        return _record(out_path, wall)

    one(-1)  # warmup
    return {"cold_start_s": cold,
            "requests": [one(i) for i in range(n)],
            "pins": {"stack": "realesrgan==0.3.0 basicsr==1.4.2",
                     "torch": torch.__version__,
                     "precision_class": "fp16 (vendor CUDA default half=True)"}}


def run_swin2sr(row: dict, n: int, repo: Path, media_dir: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
    import transformers

    ckpt = row["swin2sr_checkpoint"]
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(ckpt)
    model = Swin2SRForImageSuperResolution.from_pretrained(ckpt).cuda().eval()
    cold = time.perf_counter() - t0

    image = Image.open(repo / row["input_image"]).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda()

    def one(idx: int) -> dict:
        out_path = media_dir / f"vendor_swin2sr_r{idx}.png"
        t = time.perf_counter()
        with torch.no_grad():
            output = model(pixel_values=pixel_values)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        arr = output.reconstruction.squeeze(0).clamp(0, 1).cpu().numpy()
        arr = np.moveaxis(arr, 0, -1)
        Image.fromarray((arr * 255.0).round().astype("uint8")).save(out_path)
        return _record(out_path, wall)

    one(-1)  # warmup
    return {"cold_start_s": cold,
            "requests": [one(i) for i in range(n)],
            "pins": {"stack": f"transformers=={transformers.__version__} "
                              "Swin2SRForImageSuperResolution",
                     "torch": torch.__version__,
                     "precision_class": "fp32 (vendor default; class-labeled "
                                        "vs the nbx hardware dtype)"}}


STACKS = {"realesrgan": run_realesrgan, "swin2sr": run_swin2sr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--media-dir", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    row = json.loads(args.row)
    stack = row.get("upscaler_stack", "")
    fn = STACKS.get(stack)
    if fn is None:
        print(f"upscaler_cell: stack '{stack}' not implemented yet — "
              f"swinir needs the official repo vendored into the bench "
              f"assets; hat needs the basicsr==1.3.4.9 venv plus the "
              f"Drive-hosted weights (see the S3 R16 synthesis). The "
              f"row stays unmeasured until its stack lands.",
              file=sys.stderr)
        return 3
    out = fn(row, args.n, Path(args.repo), Path(args.media_dir))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
