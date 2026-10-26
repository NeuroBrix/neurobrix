"""The small-M GEMM runs its certified configuration (workstream 1): the key the SIMT GEMV (`mv_wrapper`, which
`mm` and `matmul` reach at M <= 4 and on a vector) and `addmv_wrapper` (which `addmm` reaches at M <= 4) form at the
launch (`triton/autotune_cache.key_of`, the launcher's own) IS the key `launch_keys` computes from shapes and dtypes —
the functions the derived census calls. On the device: every route, shape and dtype pair below.

Injection (seen red, then restored green, 2026-10-22): `mm_launches` keyed on the activation where the GEMV reads the
weight as its matrix -> red."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from neurobrix.kernels import launch_keys as LK  # noqa: E402
from neurobrix.kernels import wrappers as W  # noqa: E402
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype  # noqa: E402
from neurobrix.triton.autotune_cache import key_of  # noqa: E402

F32, F16, BF16 = NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16


class _Launched(Exception):
    """The launch reached: its key is read, the sweep never runs (the key is what is judged here)."""


def _capture(monkeypatch, tuner):
    seen = []

    def run(*args, **kwargs):
        seen.append(key_of(tuner, args, kwargs))
        raise _Launched
    monkeypatch.setattr(tuner, "run", run)
    return seen


def _t(shape, dt):
    x = NBXTensor.from_numpy(np.random.default_rng(0).standard_normal(shape).astype(np.float32)).to("cuda")
    return x if dt == F32 else x.to(dt)


def _flags():
    return W._NBX_HAS_NATIVE_BF16, W.NBX_FORCE_FP32_ACCUM


@pytest.mark.parametrize("M,K,N", [(1, 2048, 32000), (3, 4096, 512), (4, 130, 100), (1, 896, 151936)])
@pytest.mark.parametrize("a_dt,b_dt", [(F32, F16), (F16, F16), (F32, F32), (BF16, BF16)])
def test_the_small_m_mm_launch_is_its_key(monkeypatch, M, K, N, a_dt, b_dt):
    from neurobrix.kernels.ops.gemv_vec import gemv_vec_kernel as K_
    seen = _capture(monkeypatch, K_)
    a, b = _t((M, K), a_dt), _t((K, N), b_dt)
    with pytest.raises(_Launched):
        W.mm(a, b)
    nb, fa = _flags()
    want = LK.mm_launches(M, K, N, a_dt, b_dt, nb, fa)
    assert seen == [want[0][1]] and want[0][0] == LK.GEMV_VEC, (seen, want)


@pytest.mark.parametrize("N,K", [(32000, 2048), (100, 130)])
@pytest.mark.parametrize("m_dt,v_dt", [(F16, F32), (F32, F32), (BF16, BF16)])
def test_the_vector_matmul_launch_is_its_key(monkeypatch, N, K, m_dt, v_dt):
    from neurobrix.kernels.ops.gemv_vec import gemv_vec_kernel as K_
    seen = _capture(monkeypatch, K_)
    mat, vec = _t((N, K), m_dt), _t((K,), v_dt)
    with pytest.raises(_Launched):
        W.matmul_wrapper(mat, vec)
    nb, fa = _flags()
    want = LK.matmul_launches((N, K), (K,), m_dt, v_dt, nb, fa)
    assert seen == [want[0][1]], (seen, want)


@pytest.mark.parametrize("M,K,N", [(1, 2048, 2560), (2, 1024, 100)])
@pytest.mark.parametrize("a_dt,b_dt,bias_dt", [(F32, F16, F16), (F16, F16, F16), (F32, F32, F32), (BF16, BF16, BF16)])
def test_the_small_m_addmm_launch_is_its_key(monkeypatch, M, K, N, a_dt, b_dt, bias_dt):
    from neurobrix.kernels.ops.addmv_op import addmv_kernel as K_
    seen = _capture(monkeypatch, K_)
    a, b, bias = _t((M, K), a_dt), _t((K, N), b_dt), _t((N,), bias_dt)
    with pytest.raises(_Launched):
        W.addmm(bias, a, b)
    nb, fa = _flags()
    want = LK.addmm_launches(M, K, N, a_dt, b_dt, bias_dt, nb, fa)
    assert seen == [want[0][1]] and want[0][0] == LK.ADDMV, (seen, want)


@pytest.mark.parametrize("M,K,N", [(3, 4096, 512), (1, 2048, 32000)])
def test_under_the_fp16_contract_the_small_m_launch_and_its_key_store_fp16(monkeypatch, M, K, N):
    """The derived census's key function takes the component's contract (`contract_half`), as the launch does
    through the active flag: under it an fp16 GEMM stores fp16 on hardware without native bf16 (the aligned Triton
    branch, 2026-10-26). Seen red on e5aa3894: the census keyed fp32 stores while the runtime launched fp16 ones —
    17 gate cells refused their keys (KeyNotCertified)."""
    from neurobrix.kernels.ops.gemv_vec import gemv_vec_kernel as K_
    seen = _capture(monkeypatch, K_)
    monkeypatch.setattr(W, "_NBX_ACTIVATIONS_FP16_SAFE", True)
    a, b = _t((M, K), F16), _t((K, N), F16)
    with pytest.raises(_Launched):
        W.mm(a, b)
    nb, fa = _flags()
    want = LK.mm_launches(M, K, N, F16, F16, nb, fa, contract_half=True)
    assert seen == [want[0][1]] and want[0][0] == LK.GEMV_VEC, (seen, want)
    if not nb:
        assert want[0][1][-1] == LK.tag(F16), want                 # the store tag: fp16 under the contract on Volta
        assert LK.mm_launches(M, K, N, F16, F16, nb, fa)[0][1][-1] == LK.tag(F32)   # and fp32 without it
