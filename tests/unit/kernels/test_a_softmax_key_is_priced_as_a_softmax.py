"""The certifier's memory price of a softmax key (`price_key`, read before every key on unified memory) is the softmax
oracle's own — the fp64 arrays it holds per row window, every window's result kept — not the conv geometry the
function fell through to: every Apple softmax certification failed on 'geometry' before its first launch (2026-10-21)."""
from neurobrix.kernels import autotune_certify as C

Q = "neurobrix.kernels.ops.softmax.softmax_forward_kernel"


def test_a_softmax_key_is_priced():
    p = C.price_key(Q, None, (1310720, 64, False, "fp32", "fp32"))
    e = 1310720 * 64
    assert p is not None and p["peak"] >= p["draws"] + p["device"]
    assert p["oracle"] >= 8 * e                     # every window's fp64 result is kept


def test_a_small_softmax_is_priced_whole():
    p = C.price_key(Q, None, (128, 64, False, "fp32", "fp32"))
    e = 128 * 64
    assert p["oracle"] == 8 * e + e * 8 * (1 + C._SOFTMAX_ORACLE_F64_ARRAYS)
