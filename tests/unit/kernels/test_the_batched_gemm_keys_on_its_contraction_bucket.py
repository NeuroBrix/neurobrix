"""The batched GEMM's autotune key names the CONTRACTION's bucket, not the contraction: in the
attention's second product (probs @ V) the contraction is the key length, which a decode walks
one value at a time — openaudio's census recorded 2 049 distinct K in one request (2026-09-21),
each an exact key nobody could certify. Measured on both V100 classes (M=1, N=64, B=32, 135
sizes): 0.0 % median loss on the ladder, 10.5 % / 20.0 % at worst in three 16-step buckets.

The key list is read from the kernel object; no card is needed.
"""
from __future__ import annotations


def test_the_batched_gemm_key_names_the_contraction_bucket():
    from neurobrix.kernels.ops.baddbmm_op import baddbmm_kernel
    keys = list(getattr(baddbmm_kernel, "keys", None) or getattr(baddbmm_kernel, "key", []))
    assert "K_BUCKET" in keys and "K" not in keys, keys
    assert keys.index("M_BUCKET") < keys.index("N_BUCKET") < keys.index("K_BUCKET"), keys
