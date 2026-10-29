"""The certifier synthesizes a flash key's K in the standard layout (B, H, tk, hd) and must say so to the attention
wrapper: where tk == hd (64 x 64, 128 x 128) K is square, its layout cannot be read from its shape, and the wrapper
refuses to guess — four Apple flash keys failed that way on main 779b084cf + c3-attention-price (Mac, 2026-10-24 21:28).
What this test would do if the code were wrong: the wrapper called without `k_pre_transposed=False` -> red."""
from neurobrix.kernels import autotune_certify as C
from nbxtensor.ops import wrappers as W
from nbxtensor.ops import attention_wrappers as _nbx_attention_wrappers  # the family the moved names live in (M2 split)

Q = "nbxtensor.ops.flash_attention.flash_attention_certified_kernel"


def test_a_square_flash_key_tells_the_wrapper_k_is_not_transposed(monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "_flash_batch_heads", lambda *a, **k: (1, 1))
    monkeypatch.setattr(_nbx_attention_wrappers, "scaled_dot_product_attention_wrapper", lambda q, k, v, **kw: seen.update(kw))
    key = (64, 64, 64, "fp32", "fp32", "fp32", "fp32", "fp32", "fp32", "fp32")
    call, _oracle, out = C.synthesize(Q, None, key, None, values=False)
    call()
    assert seen.get("k_pre_transposed") is False, seen
