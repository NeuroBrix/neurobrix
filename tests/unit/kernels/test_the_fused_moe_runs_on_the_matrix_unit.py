"""The fused MoE grouped GEMM on the m8n8k4 matrix unit (`fused_moe_m8n8k4_kernel`) is the FMA kernel's contract:
every valid (token, expert) pair's row is its activation row times its expert's weights (times the routed weight on
the down projection), written at the pair; padding and sentinel rows are written nowhere. Judged against an fp64
oracle at the profile's unit tiles, the routing aligned at the tile's rows, both projections' flags.

What this test would do if the code were wrong: a wrong expert pointer, a wrong activation row (the top-k division),
a row written at the block's slot instead of the pair, the routed weight missing or doubled — RED at the parity; a
sentinel row written — RED at the untouched rows (NaN-poisoned). Needs a device whose profile declares the unit."""
from __future__ import annotations

import numpy as np
import pytest


def _card_with_unit():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        if DeviceAllocator.device_count() < 1:
            return False
        from neurobrix.kernels.autotune_certify import _bind_hardware_profile
        _bind_hardware_profile()
        from nbxtensor.ops._configs import matrix_unit
        return bool((matrix_unit() or {}).get("mm"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _card_with_unit(), reason="needs a device whose profile declares the matrix unit")


@pytest.mark.parametrize("T,top_k,E,K,N,mul,divide", [
    (20, 6, 64, 2048, 1408, False, True),      # a gate projection, a short prompt: most experts with 0-3 rows
    (300, 8, 16, 512, 768, False, True),       # many rows per expert
    (300, 8, 16, 768, 512, True, False),       # the down projection: the pair's own row, the routed weight
])
@pytest.mark.parametrize("tile", [0, -1])     # the profile's first and last mm rows
def test_parity_with_the_oracle(T, top_k, E, K, N, mul, divide, tile):
    import triton
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops._configs import matrix_unit
    from nbxtensor.ops.matmul_m8n8k4 import mm_layouts
    from nbxtensor.ops.fused_moe_m8n8k4 import fused_moe_m8n8k4_kernel
    from neurobrix.triton.moe import moe_align_block_size
    from nbxtensor.memory.allocator import DeviceAllocator
    row = matrix_unit()["mm"][tile]
    BM, BN, BK, WM, WN = (int(row[k]) for k in ("block_m", "block_n", "block_k", "warps_m", "warps_n"))
    rng = np.random.default_rng(0)
    nv = T * top_k
    x = (rng.standard_normal((T if divide else nv, K)) * 0.1).astype(np.float16)
    ws = [(rng.standard_normal((N, K)) * 0.05).astype(np.float16) for _ in range(E)]
    route = rng.integers(0, E, size=nv).astype(np.int64)
    score = rng.random(nv).astype(np.float16)
    wt = [NBXTensor.from_numpy(w) for w in ws]
    ptrs = NBXTensor.from_numpy(np.array([t.data_ptr() for t in wt], dtype=np.int64))
    sids, eids, npad = moe_align_block_size(NBXTensor.from_numpy(route), BM, E, DeviceAllocator.get_device())
    EM = int(sids.shape[0])
    out = NBXTensor.from_numpy(np.full((nv + 7, N), np.nan, dtype=np.float16))       # 7 rows no pair may write
    lay = mm_layouts(BM // 16, BN // 16, WM, WN, BK, True)
    grid = (triton.cdiv(EM, BM) * triton.cdiv(N, BN),)
    a = NBXTensor.from_numpy(x)
    fused_moe_m8n8k4_kernel[grid](a, ptrs, out, NBXTensor.from_numpy(score), sids, eids, npad, N, K, EM, nv,
                                  a.stride(0), a.stride(1), 1, K, out.stride(0), out.stride(1),
                                  MUL_ROUTED_WEIGHT=mul, TOP_K=top_k, TOPK_DIVIDE=divide, B_K_CONTIG=True,
                                  MB=BM // 16, NB=BN // 16, BK=BK, GROUP_M=int(row["group_m"]),
                                  num_warps=WM * WN, **lay)
    got = out.numpy().astype(np.float64)
    ref = np.empty((nv, N))
    for p in range(nv):
        r = x[p // top_k if divide else p].astype(np.float64)
        ref[p] = r @ ws[route[p]].astype(np.float64).T * (float(score[p]) if mul else 1.0)
    assert np.isnan(got[nv:]).all(), "a row past the valid pairs was written"
    err = np.abs(got[:nv] - ref).max() / np.abs(ref).max()
    assert err < 2e-3, f"relative error {err:.2e} against float64"


def test_the_layer_takes_the_unit_where_its_certificate_names_it(monkeypatch):
    """`_fused_moe_pass` runs a projection on the unit, the routing aligned again at the tile's rows, when the
    certificate names the unit (`fused_moe_unit_tile`, stubbed here) — the layer's output agrees with the FMA layer's.
    Red if the route ignores the choice (the unit never launched) or launches at the wrong alignment (parity)."""
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops import moe_wrappers as MW
    from nbxtensor.ops._configs import matrix_unit
    from neurobrix.triton.moe import execute_moe_fused
    T, H, F, E, top_k = 40, 256, 128, 8, 2
    rng = np.random.default_rng(3)
    h = (rng.standard_normal((T, H)) * 0.5).astype(np.float16)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    g = [(rng.standard_normal((F, H)) * 0.1).astype(np.float16) for _ in range(E)]
    u = [(rng.standard_normal((F, H)) * 0.1).astype(np.float16) for _ in range(E)]
    d = [(rng.standard_normal((H, F)) * 0.1).astype(np.float16) for _ in range(E)]

    def run(key):
        return execute_moe_fused(NBXTensor.from_numpy(logits), NBXTensor.from_numpy(h),
                                 [NBXTensor.from_numpy(x) for x in g], [NBXTensor.from_numpy(x) for x in u],
                                 [NBXTensor.from_numpy(x) for x in d], top_k=top_k, num_experts=E,
                                 norm_topk_prob=False, cache_key=key).numpy().astype(np.float64)
    monkeypatch.setattr(MW, "fused_moe_unit_tile", lambda *a, **k: None)
    fma = run("fma")
    tile = {f: int(v) for f, v in matrix_unit()["mm"][-1].items()}
    launched = []
    real = MW.invoke_fused_moe_m8n8k4
    monkeypatch.setattr(MW, "fused_moe_unit_tile", lambda *a, **k: tile)
    monkeypatch.setattr(MW, "invoke_fused_moe_m8n8k4", lambda *a, **k: (launched.append(1), real(*a, **k))[1])
    unit = run("unit")
    assert len(launched) == 3, f"the unit ran {len(launched)} of the layer's three projections"
    err = np.abs(unit - fma).max() / np.abs(fma).max()
    assert err < 5e-3, f"the unit layer against the FMA layer: relative {err:.2e}"
