"""Microtest for `_tiled_conv2d_spatial_nbx` correctness at small/mid scales.

Hypothesis (P-NBX-TILED-CONV2D-SMALL-SCALE): the NBX tiled conv wrapper
produces correct output at 4096^2 but striped/garbage at 1024^2 / 2048^2.

Method: for each (spatial, in_c, out_c, kh) combo and each tile_factor,
run two paths on the SAME random input/weight:
  - ref  = conv2d_wrapper(input, weight, bias)           # single-pass NBX
  - tiled = _tiled_conv2d_spatial_nbx(... tile_factor)   # band-streamed

Compare ref vs tiled element-wise. Both go through the same Triton
kernel; the only difference is whether the conv runs once on the full
H or N times on H/N bands with halo. tile_factor=1 must be bit-equal.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/microtest_tiled_conv2d_small_scale.py
"""
from __future__ import annotations

import os
import sys
import time

# Add src/ to path so we can import neurobrix without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.kernels.wrappers import conv2d_wrapper, set_hardware_profile
from neurobrix.kernels.ops.fused_upsample_conv import (
    _tiled_conv2d_spatial_nbx, _tiled_conv2d_spatial_torch,
)
from neurobrix.core.prism.loader import load_profile


def torch_to_nbx(t: torch.Tensor) -> NBXTensor:
    """Materialise a torch.Tensor as an NBXTensor on the same device."""
    assert t.is_cuda
    t = t.contiguous()
    dtype_map = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }
    nbx = NBXTensor.empty(
        tuple(t.shape), dtype_map[t.dtype], f"cuda:{t.device.index}"
    )
    # Copy raw bytes
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    DeviceAllocator.memcpy(nbx.data_ptr(), t.data_ptr(), nbx._nbytes)
    return nbx


def nbx_to_torch(n: NBXTensor) -> torch.Tensor:
    """Materialise an NBXTensor as a torch.Tensor for comparison."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    n = n.contiguous()
    t = torch.empty(
        n._shape, dtype=dtype_map[n._dtype.name], device=f"cuda:{n._device_idx}"
    )
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    DeviceAllocator.memcpy(t.data_ptr(), n.data_ptr(), n._nbytes)
    return t


def compare(ref: torch.Tensor, cmp: torch.Tensor, label: str) -> dict:
    """Element-wise comparison ref vs cmp. Returns stats + verdict."""
    assert ref.shape == cmp.shape, f"shape mismatch: {ref.shape} vs {cmp.shape}"
    # fp32 promote for stable stats
    r = ref.detach().to(torch.float32).flatten()
    c = cmp.detach().to(torch.float32).flatten()
    diff = (r - c).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rmag = float(r.abs().mean())
    rel = mean_abs / max(rmag, 1e-9)
    # cosine on a head sample (cheaper than full)
    head = min(100000, r.numel())
    rh = r[:head]
    ch = c[:head]
    cos = float(
        torch.dot(rh, ch) / (torch.norm(rh) * torch.norm(ch) + 1e-9)
    )
    # element-divergence count (|diff| > 0.01)
    diverging = int((diff > 0.01).sum())
    pct_diverging = 100.0 * diverging / r.numel()
    verdict = "OK" if (cos >= 0.999 and max_abs < 0.1) else "DIVERGENT"
    return {
        "label": label,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel": rel,
        "cos": cos,
        "pct_diverging": pct_diverging,
        "verdict": verdict,
    }


def run_one(spatial: int, in_c: int, out_c: int, kh: int, tile_factor: int,
            seed: int = 42, with_bias: bool = True) -> dict:
    torch.manual_seed(seed)
    device = "cuda:0"
    dtype = torch.float16

    # Synthetic Sana-VAE-shaped input: N=1, in_c, spatial, spatial
    x = torch.randn((1, in_c, spatial, spatial), dtype=dtype, device=device)
    # kh × kw conv weight (square kernel, common in Sana VAE)
    w = torch.randn((out_c, in_c, kh, kh), dtype=dtype, device=device) * 0.01
    b = torch.randn((out_c,), dtype=dtype, device=device) * 0.01 if with_bias else None

    pad = kh // 2  # padding=same for stride=1, kh=3 → pad=1

    # NBX inputs
    nx = torch_to_nbx(x)
    nw = torch_to_nbx(w)
    nb = torch_to_nbx(b) if b is not None else None

    # Reference path: single-pass conv2d_wrapper
    t0 = time.time()
    ref_nbx = conv2d_wrapper(
        nx, nw, nb,
        stride=(1, 1), padding=(pad, pad),
        dilation=(1, 1), groups=1,
    )
    torch.cuda.synchronize()
    t_ref = time.time() - t0
    ref = nbx_to_torch(ref_nbx)

    # Free reference NBX intermediate before tiled path
    del ref_nbx

    # Tiled path
    t0 = time.time()
    tiled_nbx = _tiled_conv2d_spatial_nbx(
        nx, nw, nb,
        sh_st=1, sw_st=1, pad_h=pad, pad_w=pad,
        dh=1, dw=1, groups=1, tile_factor=tile_factor,
    )
    torch.cuda.synchronize()
    t_tile = time.time() - t0
    tiled = nbx_to_torch(tiled_nbx)

    stats = compare(ref, tiled,
                    f"spatial={spatial} in_c={in_c} out_c={out_c} "
                    f"kh={kh} tile={tile_factor}")
    stats["t_ref_ms"] = t_ref * 1000
    stats["t_tile_ms"] = t_tile * 1000
    return stats


def sanity_check_torch_tiled():
    """Test `_tiled_conv2d_spatial_torch` directly vs F.conv2d. Test at
    multiple (pad_h, kh) combos to identify whether the bug is universal
    or specific to edge-padded convs (off-by-one halo hypothesis).
    """
    import torch.nn.functional as F
    torch.manual_seed(42)
    device = "cuda:0"
    dtype = torch.float16

    x = torch.randn((1, 64, 1024, 1024), dtype=dtype, device=device)

    print("\n[SANITY torch-tiled] _tiled_conv2d_spatial_torch vs F.conv2d "
          "at 1024^2 in=64:")
    # Test combos: (kh, pad_h)
    # pad_h=0: no edge padding, halo math should not double-count
    # pad_h=1, kh=1: pad has no effect on conv (kh=1 doesn't need halo)
    # pad_h=1, kh=3: classical same-padded 3x3 conv (suspected bug)
    combos = [(1, 0), (3, 0), (1, 1), (3, 1), (5, 2)]
    for kh, pad_h in combos:
        w = torch.randn((64, 64, kh, kh), dtype=dtype, device=device) * 0.01
        b = torch.randn((64,), dtype=dtype, device=device) * 0.01
        torch_ref = F.conv2d(x, w, b, stride=1, padding=pad_h)
        print(f"  kh={kh} pad={pad_h}:")
        for tf in [1, 2]:
            out = _tiled_conv2d_spatial_torch(
                x, w, b,
                sh_st=1, sw_st=1, pad_h=pad_h, pad_w=pad_h,
                dh=1, dw=1, groups=1, tile_factor=tf,
            )
            s = compare(torch_ref, out, f"torch_tiled kh={kh} pad={pad_h} tf={tf}")
            print(f"    tf={tf}: cos={s['cos']:.4f}  "
                  f"max_abs={s['max_abs']:.4f}  "
                  f"%div={s['pct_diverging']:.2f}%  verdict={s['verdict']}")


def sanity_check_conv2d_wrapper():
    """Confirm the NBX conv2d_wrapper itself matches torch.F.conv2d. This
    isolates the test-harness layer from any tiled-wrapper bug. If this
    fails, the harness (torch_to_nbx / nbx_to_torch) has a bug, not the
    wrapper under test.
    """
    import torch.nn.functional as F
    torch.manual_seed(42)
    device = "cuda:0"
    dtype = torch.float16

    x = torch.randn((1, 64, 1024, 1024), dtype=dtype, device=device)
    w = torch.randn((64, 64, 3, 3), dtype=dtype, device=device) * 0.01
    b = torch.randn((64,), dtype=dtype, device=device) * 0.01

    torch_ref = F.conv2d(x, w, b, stride=1, padding=1)

    nx, nw, nb = torch_to_nbx(x), torch_to_nbx(w), torch_to_nbx(b)
    nbx_ref = conv2d_wrapper(nx, nw, nb, stride=(1, 1), padding=(1, 1),
                             dilation=(1, 1), groups=1)
    nbx_back = nbx_to_torch(nbx_ref)

    stats = compare(torch_ref, nbx_back, "sanity: conv2d_wrapper vs F.conv2d")
    print(f"\n[SANITY] conv2d_wrapper vs torch.F.conv2d at 1024^2 in=64 k=3:")
    print(f"  cos={stats['cos']:.4f}  max_abs={stats['max_abs']:.4f}  "
          f"%div={stats['pct_diverging']:.2f}%  verdict={stats['verdict']}")
    return stats["verdict"] == "OK"


def main():
    # Hardware profile first (needed for conv2d_wrapper)
    cap = torch.cuda.get_device_capability(0)
    mem_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    if cap == (7, 0):
        hw_id = "v100-32g" if mem_mb >= 24000 else "v100-16g"
    elif cap == (8, 0):
        hw_id = "a100-40g"
    else:
        hw_id = "v100-32g"
    profile = load_profile(hw_id)
    set_hardware_profile(profile)
    print(f"[microtest] HW profile: {hw_id}, cap={cap}, mem={mem_mb} MB")

    # First: does the algorithm (torch path) work?
    sanity_check_torch_tiled()

    if not sanity_check_conv2d_wrapper():
        print("\n[ABORT] Harness sanity check failed — torch_to_nbx / "
              "nbx_to_torch / set_hardware_profile not wired correctly.")
        return 2

    # SMOKE-TEST phase: isolate the bug by varying with_bias.
    # If with_bias=False is OK but True is DIVERGENT → bug in nbx_add of
    # broadcast bias. If both DIVERGENT → bug is in halo/padding/setitem.
    configs = [
        # (spatial, in_c, out_c, kh, with_bias)
        (1024, 64, 64, 3, True),
        (1024, 64, 64, 3, False),   # same config, no bias
        (1024, 128, 128, 3, True),
        (1024, 128, 128, 3, False),
    ]
    tile_factors = [1, 2]

    print(f"\n{'CONFIG':<40} {'TILE':>5} {'COS':>8} {'MAX_ABS':>10} "
          f"{'%DIV':>7} {'VERDICT':<10} {'REF(ms)':>9} {'TILED(ms)':>10}")
    print("=" * 110)

    n_ok = 0
    n_div = 0
    for spatial, in_c, out_c, kh, with_bias in configs:
        for tf in tile_factors:
            try:
                s = run_one(spatial, in_c, out_c, kh, tf, with_bias=with_bias)
                n_ok += (s["verdict"] == "OK")
                n_div += (s["verdict"] == "DIVERGENT")
                bias_str = "+b" if with_bias else "  "
                cfg = f"{spatial}^2 in={in_c} out={out_c} k={kh} {bias_str}"
                print(f"{cfg:<40} {tf:>5} {s['cos']:>8.4f} "
                      f"{s['max_abs']:>10.4f} {s['pct_diverging']:>6.2f}% "
                      f"{s['verdict']:<10} {s['t_ref_ms']:>9.1f} "
                      f"{s['t_tile_ms']:>10.1f}")
            except Exception as e:
                print(f"{spatial}^2 in={in_c} out={out_c} k={kh} "
                      f"tile={tf} bias={with_bias}: "
                      f"EXCEPTION {type(e).__name__}: {e}")
                n_div += 1
            # Free between runs to keep VRAM low
            torch.cuda.empty_cache()

    print("=" * 110)
    print(f"\nSUMMARY: {n_ok} OK / {n_div} DIVERGENT")
    return 0 if n_div == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
