"""`f32_to_bf16_bits` streams: its temporaries stay bounded, whatever the operand's size, and its bits are exactly the
round-to-nearest-even top halves.

Why: on 2026-09-28 01:22 the Mac's certifier reached 14 336 MB and was killed by its guard while converting the
1.27e9-element input of CogVideoX-5b-I2V's last convolution to bf16 bits: the conversion built its rounding bias and
its shifted sum as full-size uint32 temporaries, three to four copies of a 5.1 GB array beside the array itself, while
the value rounding just before it (`_arr`) already streams through a 1 MB chunk. The key's tensors fit the certifier's
own unified-memory bound; its conversion did not.

The memory assertion uses tracemalloc, which numpy's allocator reports to: the peak traced during the call must stay
under the output (2 bytes per element) plus a fixed chunk, not a multiple of the input. Seen red on the full-size
implementation (peak about 3.5x the input) before the streaming one.
"""
import tracemalloc
import numpy as np
from neurobrix.kernels.autotune_certify import f32_to_bf16_bits


def _reference(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bias = 0x7FFF + ((u >> 16) & 1)
    bits = ((u + bias) >> 16).astype(np.uint16)
    bits[np.isnan(a)] = np.uint16(0x7FC0)
    return bits


def test_bits_are_round_to_nearest_even_with_ties_and_nan():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(200_003, dtype=np.float32) * 40).astype(np.float32)
    # exact ties: a bf16 value plus exactly half an ulp, both parities of the kept bit
    u = a.view(np.uint32)
    u[:1000] = (u[:1000] & np.uint32(0xFFFF0000)) | np.uint32(0x8000)
    a[1000:1010] = np.nan
    a[1010:1020] = np.inf
    a[1020:1030] = -np.inf
    got = f32_to_bf16_bits(a.reshape(7, -1) if False else a)
    assert got.dtype == np.uint16 and got.shape == a.shape
    assert np.array_equal(got, _reference(a))


def test_temporaries_stay_bounded_whatever_the_size():
    n = 16 * 2**20                                   # 64 MB of float32 in, 32 MB of bits out
    a = np.random.default_rng(1).standard_normal(n, dtype=np.float32)
    tracemalloc.start()
    tracemalloc.reset_peak()
    bits = f32_to_bf16_bits(a)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out_bytes = bits.nbytes
    budget = out_bytes + 16 * 2**20                  # the output plus a fixed chunk of temporaries
    assert peak <= budget, f"peak {peak/2**20:.0f} MiB during the conversion of a {a.nbytes/2**20:.0f} MiB array, budget {budget/2**20:.0f} MiB: the conversion holds full-size temporaries"
    assert np.array_equal(bits, _reference(a))
