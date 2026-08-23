"""Strided K/V consumption in the decode attention path — bit-equality
plus activation proof.

Copies-at-source (2026-08-23): the bucketed KV cache hands out
prefix-slice views `buffer[:, :, :padded, :]` — non-contiguous in the
T dim — and the SDPA wrapper's unconditional `.contiguous()` copied the
ENTIRE padded K and V per layer, per decode step (24 MB/token at bucket
256, scaling with context). The fix consumes the views directly on the
math path: `NBXTensor.merge_dims(0, 1)` keeps the (B, H) merge a pure
view (legal because the non-contiguity lives in the T dim), and both
bmms walk B's strides (`allow_strided_b=True`).

Three layers of proof here:
  1. `merge_dims` unit semantics: view when legal (same data_ptr, no
     copy), reshape-fallback when not, values always equal reshape.
  2. BIT-equality: the full SDPA wrapper on bucketed-view K/V vs the
     same values pre-materialised — byte-identical output on the
     canonical row's exact geometry (GQA 32/8 heads, D=128, bucket 256
     inside a larger buffer, additive pad mask).
  3. ACTIVATION proof (feedback_byte_gate_needs_activation_proof): the
     strided path must actually run — no strided-copy of K/V's size may
     occur during the wrapper call, else the equivalence is vacuous
     (the copy would just have moved).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_math_attention_strided_kv.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_math_attention_strided_kv.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np


def _gpu_available() -> bool:
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=count",
                        "--format=csv,noheader"], capture_output=True)
    return r.returncode == 0 and r.stdout.strip() != b""


def _download(t) -> bytes:
    """Device bytes of a CONTIGUOUS NBXTensor."""
    import ctypes
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = (ctypes.c_char * t._nbytes)()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(),
                           t._nbytes, 2)
    return bytes(buf)


def test_merge_dims_semantics() -> None:
    if not _gpu_available():
        pytest.skip("no GPU")
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(7)

    # 1a. Contiguous: merge is a view, values equal reshape.
    a_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float16)
    a = NBXTensor.from_numpy(a_np).to("cuda:0")
    m = a.merge_dims(0, 1)
    assert tuple(m.shape) == (6, 4, 5)
    assert m.data_ptr() == a.data_ptr(), "contiguous merge must be a view"
    assert _download(m.contiguous()) == a_np.reshape(6, 4, 5).tobytes()

    # 1b. Bucketed-cache pattern: non-contiguity in T, (B, H) merge
    #     still a VIEW with the buffer's row pitch.
    buf_np = rng.standard_normal((1, 8, 512, 128)).astype(np.float16)
    buf = NBXTensor.from_numpy(buf_np).to("cuda:0")
    view = buf[:, :, :256, :]
    assert not view.is_contiguous()
    mv = view.merge_dims(0, 1)
    assert tuple(mv.shape) == (8, 256, 128)
    assert mv.data_ptr() == buf.data_ptr(), "bucketed merge must be a view"
    assert mv._strides == (512 * 128, 128, 1), mv._strides
    ref = buf_np[:, :, :256, :].reshape(8, 256, 128)
    assert _download(mv.contiguous()) == ref.tobytes()

    # 1c. Illegal merge (transposed first two dims) falls back to
    #     reshape semantics — values still correct.
    t = a.transpose(0, 1)  # [3, 2, 4, 5], strides swapped
    mt = t.merge_dims(0, 1)
    assert tuple(mt.shape) == (6, 4, 5)
    ref_t = np.ascontiguousarray(a_np.transpose(1, 0, 2, 3)).reshape(6, 4, 5)
    assert _download(mt.contiguous()) == ref_t.tobytes()

    # 1d. Non-adjacent dims refuse loudly.
    try:
        a.merge_dims(0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("merge_dims(0, 2) must raise")


def test_sdpa_bucketed_view_bit_equal_with_activation_proof() -> None:
    if not _gpu_available():
        pytest.skip("no GPU")
    import os
    from neurobrix.kernels import nbx_tensor as NT
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import scaled_dot_product_attention_wrapper

    # Pin the MATH route — the route decode takes on Volta via the
    # vendor-yml scores budget. A bare test process has no hardware
    # profile loaded, so without this the wrapper routes to FLASH,
    # whose entry legitimately materialises K/V — the first run of this
    # test proved exactly that (2 K/V-sized copies, equivalence
    # vacuous). The strided-consumption claim is a MATH-path claim.
    os.environ["NBX_FORCE_MATH_ATTENTION"] = "1"

    rng = np.random.default_rng(1234)
    B, H, H_k, D, PAD, CAP = 1, 32, 8, 128, 256, 512
    q_np = (rng.standard_normal((B, H, 1, D)) * 0.1).astype(np.float16)
    buf_k = (rng.standard_normal((B, H_k, CAP, D)) * 0.1).astype(np.float16)
    buf_v = (rng.standard_normal((B, H_k, CAP, D)) * 0.1).astype(np.float16)
    mask_np = np.zeros((1, 1, 1, PAD), dtype=np.float16)
    mask_np[..., 200:] = -np.inf  # pad region masked, like the bucket mask

    q = NBXTensor.from_numpy(q_np).to("cuda:0")
    kb = NBXTensor.from_numpy(buf_k).to("cuda:0")
    vb = NBXTensor.from_numpy(buf_v).to("cuda:0")
    mask = NBXTensor.from_numpy(mask_np).to("cuda:0")

    k_view = kb[:, :, :PAD, :]
    v_view = vb[:, :, :PAD, :]
    assert not k_view.is_contiguous(), "precondition: the bucketed view is strided"

    # Arm A — strided views straight in, counting K/V-sized strided
    # copies (activation proof: there must be NONE).
    kv_numel = B * H_k * PAD * D
    copies = {"kv_sized": 0}
    orig_sc = NT._strided_copy

    def counting_sc(src, dst):
        if getattr(src, "_numel", 0) == kv_numel:
            copies["kv_sized"] += 1
        return orig_sc(src, dst)

    NT._strided_copy = counting_sc
    try:
        out_strided = scaled_dot_product_attention_wrapper(
            q, k_view, v_view, attn_mask=mask, is_causal=False)
    finally:
        NT._strided_copy = orig_sc

    # Arm B — the pre-fix data flow: K/V materialised first.
    out_dense = scaled_dot_product_attention_wrapper(
        q, k_view.contiguous(), v_view.contiguous(),
        attn_mask=mask, is_causal=False)

    a = _download(out_strided.contiguous())
    b = _download(out_dense.contiguous())
    assert a == b, "strided K/V consumption changed the output bytes"

    assert copies["kv_sized"] == 0, (
        f"{copies['kv_sized']} K/V-sized strided copies ran — the "
        f"'strided' path materialised after all; the equivalence above "
        f"is vacuous")
    os.environ.pop("NBX_FORCE_MATH_ATTENTION", None)


if __name__ == "__main__":
    if not _gpu_available():
        raise SystemExit("no GPU")
    test_merge_dims_semantics()
    print("PASS: merge_dims semantics (view / fallback / refusal)")
    test_sdpa_bucketed_view_bit_equal_with_activation_proof()
    print("PASS: SDPA bucketed-view bit-equal, zero K/V-sized copies")
