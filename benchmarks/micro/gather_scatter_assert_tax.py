#!/usr/bin/env python3
"""Decode-row tax of the in-kernel index bound checks (D-GATHER-SCATTER-
OOB-SILENT, 2026-09-02): times the three gather/scatter wrappers on
decode-shaped inputs with the shipped kernels (`tl.device_assert`, kept by
`@triton.jit(debug=True)`) against the same kernels rebuilt from their
source with the asserts stripped and debug off. Measured, not asserted
("isolated ranks, never predicts"): the number that matters is the per-
launch delta on the shapes the decode loop issues per token.

  python3 benchmarks/micro/gather_scatter_assert_tax.py --gpu 1
"""
import argparse
import importlib
import importlib.util
import os
import re
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--vocab", type=int, default=151936)
ap.add_argument("--reps", type=int, default=200)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
sys.path.insert(0, "src")

import numpy as np
import triton
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers as W
from neurobrix.kernels.ops import embedding as emb_mod, index_select as isel_mod, index_put_op as iput_mod


def stripped(mod):
    """Rebuild a kernel module from source with the device_asserts removed
    and debug=False (the pre-2026-09-02 binary). Triton's @jit needs a
    real file: the stripped source is written next to this script under
    `_noassert/` (generated, gitignored) and imported from there."""
    src = open(mod.__file__).read()
    src = re.sub(r"^\s*tl\.device_assert\([^\n]*\)\s*$", "", src, flags=re.M)
    src = src.replace("@triton.jit(debug=True)", "@triton.jit")
    assert not re.search(r"^\s*tl\.device_assert\(", src, flags=re.M)
    assert "@triton.jit(debug=True)" not in src
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_noassert")
    os.makedirs(out_dir, exist_ok=True)
    name = mod.__name__.rsplit(".", 1)[-1] + "_noassert"
    path = os.path.join(out_dir, name + ".py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bench(fn):
    ms = triton.testing.do_bench(fn, rep=a.reps, warmup=25)
    return ms * 1e3  # µs


DeviceAllocator.set_device(0)
H, V = a.hidden, a.vocab
weight = NBXTensor.from_numpy(np.random.randn(V, H).astype(np.float16)).to_device(0) if hasattr(NBXTensor, "to_device") else NBXTensor.from_numpy(np.random.randn(V, H).astype(np.float16))
ids = NBXTensor.from_numpy(np.array([[V // 2]], dtype=np.int64))
x = NBXTensor.from_numpy(np.random.randn(4096, H).astype(np.float16))
sel = NBXTensor.from_numpy(np.arange(0, 2048, 2, dtype=np.int64))  # 1024 column ids < H (the kernel gathers along the last dim)
tgt = NBXTensor.from_numpy(np.zeros((4096, H), dtype=np.float16))
rows = NBXTensor.from_numpy(np.array([7], dtype=np.int64))
val = NBXTensor.from_numpy(np.random.randn(1, H).astype(np.float16))

# move to device if from_numpy lands on host
def dev(t):
    return t.cuda(0) if hasattr(t, "cuda") and getattr(t, "_device", None) in (None, "cpu", -1) else t
weight, ids, x, sel, tgt, rows, val = map(dev, (weight, ids, x, sel, tgt, rows, val))

# Time the KERNEL launches on pre-allocated buffers (the wrappers' output
# allocation and host logic are the same on both sides and would dominate
# a ~µs kernel). Same grids as the wrappers. ABAB order against drift.
from neurobrix.kernels.ops.embedding import embedding_kernel as K_emb
from neurobrix.kernels.ops.index_select import index_select_kernel as K_isel
from neurobrix.kernels.ops.index_put_op import index_put_kernel as K_iput
N_emb = stripped(emb_mod).embedding_kernel
N_isel = stripped(isel_mod).index_select_kernel
N_iput = stripped(iput_mod).index_put_kernel

emb_out = NBXTensor.empty((1, H), dtype=weight.nbx_dtype, device="cuda")
BLOCK = triton.next_power_of_2(H)
isel_out = NBXTensor.empty((4096, 1024), dtype=x.nbx_dtype, device="cuda")
# the wrapper's own formula (kernels/wrappers.py index_select_wrapper):
BM = min(64, triton.next_power_of_2(4096))
BN = min(64, triton.next_power_of_2(1024))
grid_isel = (triton.cdiv(4096, BM), triton.cdiv(1024, BN))
T = H
n_put = H
grid_put = (triton.cdiv(n_put, 1024),)

def emb(k):
    return lambda: k[(1,)](emb_out, ids, weight, V, H, BLOCK)
def isel(k):
    return lambda: k[grid_isel](x, isel_out, 4096, H, sel, 1024, BM, BN)
def iput(k):
    return lambda: k[grid_put](tgt, rows, val, T, n_put, 4096, VAL_SCALAR=False, ACCUMULATE=False, BLOCK_SIZE=1024, num_warps=4)

cases = {
    "embedding kernel (1 token, H)": (emb(K_emb), emb(N_emb)),
    "index_select kernel (4096 rows x 1024 cols of H)": (isel(K_isel), isel(N_isel)),
    "index_put kernel (1 row, H)": (iput(K_iput), iput(N_iput)),
}
print(f"{'case':40s} {'shipped µs':>11s} {'no-assert µs':>13s} {'delta µs':>9s} {'delta %':>8s}   (median of 3 ABAB rounds)")
import statistics
for k, (fa, fb) in cases.items():
    sa, sb = [], []
    for _ in range(3):
        sa.append(bench(fa)); sb.append(bench(fb))
    A, B = statistics.median(sa), statistics.median(sb)
    print(f"{k:40s} {A:11.2f} {B:13.2f} {A-B:9.2f} {100*(A-B)/B:8.1f}")
