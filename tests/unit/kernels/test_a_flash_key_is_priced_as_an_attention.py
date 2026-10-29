"""The certifier's memory price of a flash key (`price_key`, read before every key on unified memory) is the attention
oracle's own — q's rows, k and v whole and the window's scores in float64, every window's result kept — not the conv
geometry the function fell through to: every Apple flash certification (C3) failed on 'geometry' before its first
launch (Mac, 2026-10-23). What this test would do if the code were wrong: KeyError 'geometry' -> red."""
from neurobrix.kernels import autotune_certify as C

Q = "nbxtensor.ops.flash_attention.flash_attention_certified_kernel"


def test_a_flash_key_is_priced(monkeypatch):
    monkeypatch.setattr(C, "_flash_batch_heads", lambda *a, **k: (1, 4))
    key = (2048, 2048, 96, "bf16", "bf16", "bf16", "bf16", "bf16", "fp32", "fp32")
    p = C.price_key(Q, None, key)
    assert p is not None and p["peak"] >= p["draws"] + p["device"]
    assert p["oracle"] >= 4 * 2048 * 96 * 8            # the windows' fp64 results are kept


def test_a_long_flash_key_fits_the_card_on_windows(monkeypatch):
    monkeypatch.setattr(C, "_flash_batch_heads", lambda *a, **k: (1, 1))
    key = (81920, 81920, 96, "bf16", "bf16", "bf16", "bf16", "bf16", "fp32", "fp32")
    assert C.price_key(Q, None, key)["peak"] < 18 * 2**30
