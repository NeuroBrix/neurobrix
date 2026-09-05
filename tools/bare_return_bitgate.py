#!/usr/bin/env python3
"""Bit-identity gate for the bare-return fixes in the kernel library.

`capture --out DIR` runs the three kernel families that carried a bare
`return` (fft_op, fused_moe, grid_sampler) on seeded inputs at the engine's
tiles and stores every output as .npz; `compare --before DIR1 --after DIR2`
asserts byte equality of every array. The BEFORE capture is taken on the
unfixed kernels, the AFTER on the mask-based ones — the proof the 2026-09-05
addendum asks for. Diagnostic, off the execution path.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def _nbx(arr):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _host(t, np_dtype):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = np.empty(t._nbytes, dtype=np.uint8)
    DeviceAllocator.memcpy(buf.ctypes.data, t.contiguous().data_ptr(), t._nbytes, 2)
    return buf.view(np_dtype).reshape(t.shape)


def capture_fft(rng, out):
    """The kernels driven directly (the inverse wrapper carries an unrelated
    latent defect — a torch-style `mul_` on an NBXTensor — fixed separately)."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import _triton_fft_forward, _set_device
    from neurobrix.kernels.ops.fft_op import bit_reverse_kernel, ifft_stage_kernel
    for N in (256, 1024, 4096):
        xr = _nbx((rng.standard_normal((2, N))).astype(np.float32))
        xi = _nbx((rng.standard_normal((2, N))).astype(np.float32))
        r, i = _triton_fft_forward(xr, xi)
        out[f"fft_fwd_N{N}_real"] = _host(r, np.float32)
        out[f"fft_fwd_N{N}_imag"] = _host(i, np.float32)
        yr = _nbx((rng.standard_normal((N,))).astype(np.float32))
        yi = _nbx((rng.standard_normal((N,))).astype(np.float32))
        tr, ti = NBXTensor.empty_like(yr), NBXTensor.empty_like(yi)
        _set_device(yr)
        bit_reverse_kernel[(N,)](yr, yi, tr, ti, N)
        for stage in range(1, N.bit_length()):
            ifft_stage_kernel[(N // 2,)](tr, ti, N, stage)
        out[f"ifft_stages_N{N}_real"] = _host(tr, np.float32)
        out[f"ifft_stages_N{N}_imag"] = _host(ti, np.float32)


def capture_grid_sampler(rng, out):
    from neurobrix.kernels.wrappers import grid_sampler_2d_wrapper
    inp = _nbx((rng.standard_normal((2, 5, 7, 9))).astype(np.float32))      # C=5: a masked channel block
    grid = _nbx((rng.uniform(-1.2, 1.2, size=(2, 11, 13, 2))).astype(np.float32))
    for mode in (0, 2):
        for pad in (0, 1):
            for align in (False, True):
                o = grid_sampler_2d_wrapper(inp, grid, mode, pad, align)
                out[f"grid_sampler_mode{mode}_pad{pad}_align{int(align)}"] = _host(o, np.float32)


def capture_fused_moe(rng, out):
    import triton
    import triton.language as tl
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
    from neurobrix.kernels.ops.fused_moe import (fused_moe_kernel, fused_moe_wna16_kernel,
                                                 fused_moe_fp32b_kernel)
    from neurobrix.kernels.ops.dequant_gemv import dequant_int4_kernel
    from neurobrix.triton.moe import moe_align_block_size
    from tests.unit.kernels.test_fused_moe_wna16_parity import (_quantize, _ptr_table, GROUP, PACK,
                                                                BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS)
    for M, K, N, E, top_k, routed in ((1, 2048, 768, 8, 2, False), (5, 768, 2048, 8, 2, True)):
        a = _nbx((rng.standard_normal((M, K)) * 0.5).astype(np.float16))
        DeviceAllocator.set_device(a._device_idx)
        qws, scs, mns, denses, halves = [], [], [], [], []
        for e in range(E):
            w_t = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
            pk, sc, mn = _quantize(w_t)
            qw_t, sc_t, mn_t = _nbx(pk), _nbx(sc), _nbx(mn)
            qws.append(qw_t); scs.append(sc_t); mns.append(mn_t)
            dense = NBXTensor.empty((K, N), dtype=NBXDtype.float32, device="cuda")
            dequant_int4_kernel[(triton.cdiv(K, 128), triton.cdiv(N, 64))](
                qw_t, sc_t, mn_t, dense, K, N, qw_t.stride(0), qw_t.stride(1), sc_t.stride(0), sc_t.stride(1),
                dense.stride(0), dense.stride(1), BLOCK_K_C=128, BLOCK_N_C=64, GROUP_C=GROUP, PACK_C=PACK,
                num_warps=4, num_stages=2)
            denses.append(dense)
            halves.append(_nbx(w_t.astype(np.float16)))
        topk_ids = rng.integers(0, E, size=(M * top_k,)).astype(np.int64)
        topk_w = _nbx(rng.random(M * top_k).astype(np.float32))
        sorted_ids, expert_ids, num_post = moe_align_block_size(_nbx(topk_ids), BLOCK_M, E, a._device_idx)
        EM = int(_host(num_post, np.int32).reshape(-1)[0]) if num_post.nbx_dtype == NBXDtype.int32 \
            else int(_host(num_post, np.int64).reshape(-1)[0])
        n_valid = M * top_k
        grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        tag = f"M{M}_K{K}_N{N}_routed{int(routed)}"
        common = dict(BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K, GROUP_SIZE_M=GROUP_M,
                      MUL_ROUTED_WEIGHT=routed, top_k=top_k, compute_type=tl.float16, num_warps=WARPS, num_stages=2)
        o = NBXTensor.empty((n_valid, N), dtype=NBXDtype.float16, device="cuda")
        fused_moe_wna16_kernel[grid](a, _ptr_table(qws), _ptr_table(scs), _ptr_table(mns), o, topk_w, sorted_ids,
                                     expert_ids, num_post, N, K, EM, n_valid, a.stride(0), a.stride(1),
                                     qws[0].stride(0), qws[0].stride(1), scs[0].stride(0), scs[0].stride(1),
                                     o.stride(0), o.stride(1), QGROUP=GROUP, QPACK=PACK, **common)
        out[f"moe_wna16_{tag}"] = _host(o, np.float16).view(np.uint16)
        o = NBXTensor.empty((n_valid, N), dtype=NBXDtype.float16, device="cuda")
        fused_moe_fp32b_kernel[grid](a, _ptr_table(denses), o, topk_w, sorted_ids, expert_ids, num_post, N, K, EM,
                                     n_valid, a.stride(0), a.stride(1), denses[0].stride(0), denses[0].stride(1),
                                     o.stride(0), o.stride(1), **common)
        out[f"moe_fp32b_{tag}"] = _host(o, np.float16).view(np.uint16)
        o = NBXTensor.empty((n_valid, N), dtype=NBXDtype.float16, device="cuda")
        fused_moe_kernel[grid](a, _ptr_table(halves), o, topk_w, sorted_ids, expert_ids, num_post, N, K, EM,
                               n_valid, a.stride(0), a.stride(1), halves[0].stride(0), halves[0].stride(1),
                               o.stride(0), o.stride(1), TOPK_DIVIDE=True, **common)
        out[f"moe_fp16_{tag}"] = _host(o, np.float16).view(np.uint16)


def capture(out_dir: Path) -> int:
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    DeviceAllocator.set_device(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fft", capture_fft), ("grid_sampler", capture_grid_sampler), ("fused_moe", capture_fused_moe)):
        arrays = {}
        fn(np.random.default_rng(1234), arrays)
        np.savez(out_dir / f"{name}.npz", **arrays)
        print(f"[bitgate] {name}: {len(arrays)} array(s) → {out_dir / (name + '.npz')}")
    return 0


def compare(before: Path, after: Path) -> int:
    bad = 0
    for f in sorted(before.glob("*.npz")):
        a = np.load(f); b = np.load(after / f.name)
        for k in a.files:
            same = k in b.files and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes()
            if not same:
                bad += 1
                print(f"[bitgate] DIFFERENT {f.name}:{k}")
        print(f"[bitgate] {f.name}: {len(a.files)} array(s), {sum(1 for k in a.files if k in b.files and a[k].tobytes() == b[k].tobytes())} identical")
    print("[bitgate] BIT-IDENTICAL" if not bad else f"[bitgate] {bad} array(s) DIFFER")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--out", required=True)
    d = sub.add_parser("compare"); d.add_argument("--before", required=True); d.add_argument("--after", required=True)
    args = ap.parse_args()
    if args.cmd == "capture":
        return capture(Path(args.out))
    return compare(Path(args.before), Path(args.after))


if __name__ == "__main__":
    sys.exit(main())
