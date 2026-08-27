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


def run_swinir(row: dict, n: int, repo: Path, media_dir: Path) -> dict:
    """Official JingyunLiang/SwinIR route: network_swinir.py imported
    from the NAS-cloned repo, classical-SR M-model config (the
    main_test_swinir.py define_model args for task=classical_sr,
    training_patch_size 48, window 8), vendor default fp32 —
    precision class labeled."""
    import sys as _sys

    import numpy as np
    import torch
    from PIL import Image

    repo_dir = Path.home() / "hf_snapshots" / "vendor_repos" / "SwinIR"
    _sys.path.insert(0, str(repo_dir))
    from models.network_swinir import SwinIR

    scale = int(row.get("scale", 4))
    t0 = time.perf_counter()
    model = SwinIR(
        upscale=scale, in_chans=3, img_size=48, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv")
    ckpt = torch.load(row["upscaler_weights"], map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt.get("params", ckpt), strict=True)
    model = model.cuda().eval()
    cold = time.perf_counter() - t0

    img = Image.open(repo / row["input_image"]).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    # Reflect-pad to a multiple of the window (official test-script
    # convention); crop back after SR.
    _, _, h, w = x.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    def one(idx: int) -> dict:
        out_path = media_dir / f"vendor_swinir_r{idx}.png"
        t = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        y = y[..., :h * scale, :w * scale].squeeze(0).clamp(0, 1)
        out = (y.permute(1, 2, 0).cpu().numpy() * 255.0).round()
        Image.fromarray(out.astype("uint8")).save(out_path)
        return _record(out_path, wall)

    one(-1)  # warmup
    return {"cold_start_s": cold,
            "requests": [one(i) for i in range(n)],
            "pins": {"stack": "JingyunLiang/SwinIR official repo + "
                              "timm==0.9.16",
                     "torch": torch.__version__,
                     "precision_class": "fp32 (vendor default; "
                                        "class-labeled vs the nbx "
                                        "hardware dtype)"}}


def run_hat(row: dict, n: int, repo: Path, media_dir: Path) -> dict:
    """Official XPixelGroup/HAT route (hat venv: basicsr 1.3.4.9 +
    repo setup.py develop). HAT-L SRx4 arch args from the repo's
    options/test/HAT-L_SRx4_ImageNet-pretrain.yml. Weights: community
    HF mirror of the Drive-hosted official checkpoint
    (anchuang/HAT-L_SRx4_ImageNet-pretrain) — labeled in pins;
    byte-verification vs the Drive original pending. Vendor default
    fp32, class-labeled."""
    import numpy as np
    import torch
    from PIL import Image
    from hat.archs.hat_arch import HAT

    scale = int(row.get("scale", 4))
    t0 = time.perf_counter()
    model = HAT(
        upscale=scale, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.,
        depths=[6] * 12, embed_dim=180, num_heads=[6] * 12,
        mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
    ckpt = torch.load(row["upscaler_weights"], map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)),
                          strict=True)
    model = model.cuda().eval()
    cold = time.perf_counter() - t0

    img = Image.open(repo / row["input_image"]).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    _, _, h, w = x.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    def one(idx: int) -> dict:
        out_path = media_dir / f"vendor_hat_r{idx}.png"
        t = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        y = y[..., :h * scale, :w * scale].squeeze(0).clamp(0, 1)
        out = (y.permute(1, 2, 0).cpu().numpy() * 255.0).round()
        Image.fromarray(out.astype("uint8")).save(out_path)
        return _record(out_path, wall)

    one(-1)  # warmup
    return {"cold_start_s": cold,
            "requests": [one(i) for i in range(n)],
            "pins": {"stack": "XPixelGroup/HAT official repo, "
                              "basicsr==1.3.4.9",
                     "torch": torch.__version__,
                     "weights": "HF mirror anchuang/"
                                "HAT-L_SRx4_ImageNet-pretrain "
                                "(byte-verify vs Drive pending)",
                     "precision_class": "fp32 (vendor default; "
                                        "class-labeled)"}}


STACKS = {"realesrgan": run_realesrgan, "swin2sr": run_swin2sr,
          "swinir": run_swinir, "hat": run_hat}


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
