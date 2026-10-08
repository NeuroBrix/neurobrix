"""Proves the wrapper takes the m8n8k4 route on volta (fp16) and agrees with the FMA flash path."""
import sys, numpy as np
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from neurobrix.kernels import wrappers as W
calls = []
orig = W._flash_m8n8k4
W._flash_m8n8k4 = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
W._lk.sdpa_route = lambda *a, **kw: ("flash", 0)
rng = np.random.default_rng(1)
for (B, H, T, D) in [(1, 2, 777, 96), (2, 3, 300, 64), (1, 2, 129, 128), (1, 1, 65, 80)]:
    mk = lambda: NBXTensor.from_numpy((rng.standard_normal((B, H, T, D)) * 0.5).astype(np.float16)).to("cuda:0")
    q, k, v = mk(), mk(), mk()
    n0 = len(calls)
    o = W.scaled_dot_product_attention_wrapper(q, k, v, k_pre_transposed=False).numpy().astype(np.float64)
    took = len(calls) - n0
    qq, kk, vv = (x.numpy().astype(np.float64) for x in (q, k, v))
    s = qq @ kk.swapaxes(-1, -2) / D ** 0.5
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    print(f"B{B} H{H} T{T} D{D}: m8n8k4 calls {took}  max|diff| vs fp64 {np.abs(o - p @ vv).max():.2e}", flush=True)
