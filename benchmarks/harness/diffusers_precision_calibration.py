#!/usr/bin/env python3
"""Vendor precision calibration: the SAME diffusers pipeline rendered at fp32
and at its fp16 recipe, same seed — the PSNR between the two is what the
vendor's own half precision costs on this model. Our fp16 render is judged
against our fp32 render with that number as the yardstick (an engine that
drifts no more than the vendor's own fp16 is at vendor precision).
Runs inside ~/bench_venvs/diffusers. Writes <out_dir>/vendor_{fp32,fp16}.png."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusers_cell import load_pipeline  # noqa: E402


def render(row, dtype_name, out):
    import torch, yaml  # noqa: F401
    r = dict(row); r["diffusers_recipe"] = {**(row.get("diffusers_recipe") or {}), "torch_dtype": dtype_name}
    pipe, cold, fixes, src, note = load_pipeline(r)
    kw = {"prompt": row["prompt"], "num_inference_steps": row["steps"], "height": row.get("height"), "width": row.get("width")}
    kw = {k: v for k, v in kw.items() if v is not None}
    gen = torch.Generator(device="cuda").manual_seed(row["seed"])
    t0 = time.perf_counter()
    with torch.inference_mode():
        img = pipe(generator=gen, **kw).images[0]
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    img.save(out); print(f"[calib] {dtype_name}: {wall:.2f}s fixes={fixes} -> {out}", flush=True)
    del pipe; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--row", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-yml", default=str(Path(__file__).resolve().parents[1] / "config" / "rows.yml"))
    a = ap.parse_args(); import yaml
    row = {r["id"]: r for r in yaml.safe_load(open(a.rows_yml))["rows"]}[a.row]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    render(row, "float32", out / "vendor_fp32.png")
    render(row, "float16", out / "vendor_fp16.png")


if __name__ == "__main__":
    main()
