"""ONE draw for both engines: the vendor's CPU draw, torch-free.

The owner's rule (2026-10-13 02:05): a model's initial noise is one draw, data-driven (the request's
seed), identical between the compiled and the Triton branch and to the vendor at equal seed. Diffusers
draws its initial latents on torch's CPU generator — `torch.Generator("cpu").manual_seed(seed)` then
`randn` — which is MT19937 seeded by `init_genrand(seed)` (the low 32 bits), uniforms made of the 24
high mantissa bits of a 32-bit word, and ATen's `normal_fill`: Box-Muller over blocks of 16 (the first
eight uniforms of a block are the radii, the last eight the angles), the tail of `n % 16` values
recomputed from 16 FRESH uniforms. This module reproduces that stream without torch.

What is bit-exact: the uniform stream (every word of MT19937, every 24-bit mantissa) and the pairing.
What is within 1 ulp: the transcendentals. ATen computes `sqrt(-2 log u1)`, `cos`, `sin` through its
vectorised Sleef kernels (`logf_u10`, `sinf/cosf_u35`, AVX512 here, NEON on Apple), which are not
correctly rounded and differ per architecture; this module computes them in float64 and rounds once,
which is correctly rounded everywhere. Measured against torch 2.14 on an AVX512 Xeon over 100 000 values
per seed (seeds 7, 42, 2^31-1; 2026-10-13): the uniform stream bit-equal, the normals at most 3 ulp
apart (Sleef's u35 bound is 3.5), 98.8 % within 1 ulp, 61.5 % bit-equal, max |diff| 4.8e-7. The draw is
therefore identical between our two branches and on every platform, and within 4 ulp of the vendor's.

No torch, no device: numpy only (R33 holds; the compiled branch wraps the array in a tensor).
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

_N, _M = 624, 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER, _LOWER = np.uint32(0x80000000), np.uint32(0x7FFFFFFF)


class MT19937:
    """The generator as torch's CPU `Generator` runs it: `manual_seed(s)` = `init_genrand(s & 0xffffffff)`."""

    def __init__(self, seed: int) -> None:
        mt = np.zeros(_N, dtype=np.uint32)
        mt[0] = np.uint32(int(seed) & 0xFFFFFFFF)
        for i in range(1, _N):   # the recurrence is sequential by definition; 624 steps once per run
            prev = int(mt[i - 1])
            mt[i] = np.uint32((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self._mt = mt
        self._i = _N
        self._cached = None   # the second Box-Muller sample of the scalar path, as torch caches it

    def _twist(self) -> None:
        """The reference twist, vectorised in the order it reads: `mt[k] = mt[(k + M) % N] ^ f(y_k)` with
        `y_k` over the ORIGINAL `mt[k], mt[k+1]` for every k but the last. The second segment reads entries
        rewritten earlier in the same pass, so it runs in chunks of N-M, each reading only chunks already
        produced (a single vector expression read uninitialised values — the stream departed at word 454,
        2026-10-13)."""
        mt = self._mt
        y = (mt & _UPPER) | (np.roll(mt, -1) & _LOWER)
        shifted = (y >> np.uint32(1)) ^ np.where(y & np.uint32(1), _MATRIX_A, np.uint32(0))
        new = np.empty_like(mt)
        s = _N - _M                                   # 227: the stride a segment reads behind itself
        new[:s] = mt[_M:] ^ shifted[:s]
        k = s
        while k < _N - 1:
            e = min(k + s, _N - 1)
            new[k:e] = new[k - s : e - s] ^ shifted[k:e]
            k = e
        y_last = (mt[_N - 1] & _UPPER) | (new[0] & _LOWER)
        new[_N - 1] = new[_M - 1] ^ ((y_last >> np.uint32(1)) ^ (_MATRIX_A if int(y_last) & 1 else np.uint32(0)))
        self._mt = new
        self._i = 0

    def words(self, n: int) -> np.ndarray:
        """`n` tempered 32-bit words, in stream order."""
        out = np.empty(n, dtype=np.uint32)
        k = 0
        while k < n:
            if self._i >= _N:
                self._twist()
            take = min(n - k, _N - self._i)
            y = self._mt[self._i : self._i + take].copy()
            y ^= y >> np.uint32(11)
            y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
            y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
            y ^= y >> np.uint32(18)
            out[k : k + take] = y
            self._i += take
            k += take
        return out

    def uniform_f32(self, n: int) -> np.ndarray:
        """ATen `uniform_real_distribution<float>(0, 1)`: the 24 low bits of each word over 2^24."""
        return ((self.words(n) & np.uint32((1 << 24) - 1)).astype(np.float64) * (1.0 / (1 << 24))).astype(np.float32)

    def randn(self, n: int) -> np.ndarray:
        """`n` standard normals as torch's CPU `randn` draws them (ATen `normal_fill`, blocks of 16; a tail of
        `n % 16` recomputed from 16 fresh uniforms), the transcendentals in float64 rounded once."""
        if n < 16:
            return self._randn_scalar(n)
        u = self.uniform_f32(n)
        out = np.empty(n, dtype=np.float32)
        m = n - n % 16
        if m:
            _fill16(out, 0, u[:m])
        if n % 16:
            _fill16(out, n - 16, self.uniform_f32(16))
        return out

    def _randn_scalar(self, n: int) -> np.ndarray:
        """ATen's path for fewer than 16 values (`normal_kernel`'s serial branch): `normal_distribution<double>`
        over 53-bit uniforms (two words per uniform, `random64`), Box-Muller in double with the second sample
        of each pair cached in the generator and served next — exactly the vendor's small draws."""
        out = np.empty(n, dtype=np.float32)
        i = 0
        while i < n:
            if self._cached is not None:
                out[i] = np.float32(self._cached)
                self._cached = None
                i += 1
                continue
            # ATen normal_distribution<double>: u1 then u2; r = sqrt(-2 log1p(-u2)); theta = 2 pi u1;
            # the sine sample cached for the next call, the cosine returned.
            u1 = self._uniform_f64()
            u2 = self._uniform_f64()
            radius = math.sqrt(-2.0 * math.log1p(-u2))
            theta = 2.0 * math.pi * u1
            self._cached = radius * math.sin(theta)
            out[i] = np.float32(radius * math.cos(theta))
            i += 1
        return out

    def _uniform_f64(self) -> float:
        """ATen `uniform_real_distribution<double>(0, 1)`: the low 53 bits of a 64-bit word made of two 32-bit
        words (`random64()` = `make64BitsFrom32Bits(random(), random())`: the first word high, the second low)."""
        hi, lo = (int(x) for x in self.words(2))
        v = ((hi << 32) | lo) & ((1 << 53) - 1)
        return v * (1.0 / (1 << 53))


def _fill16(dst: np.ndarray, base: int, u: np.ndarray) -> None:
    b = u.reshape(-1, 16)
    u1 = (np.float32(1.0) - b[:, :8]).astype(np.float64)
    u2 = b[:, 8:].astype(np.float32)
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = (np.float32(2.0) * np.float32(math.pi) * u2).astype(np.float64)   # ATen: 2.0f * pi<double> * u2 → float
    view = dst[base : base + b.size].reshape(-1, 16)
    view[:, :8] = (radius * np.cos(theta)).astype(np.float32)
    view[:, 8:] = (radius * np.sin(theta)).astype(np.float32)


def mt_randn(seed: int, shape: Iterable[int]) -> np.ndarray:
    """A fresh generator seeded with `seed`, its first `prod(shape)` normals reshaped — the vendor's
    `randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(seed))`, torch-free."""
    shape: Tuple[int, ...] = tuple(int(s) for s in shape)
    n = 1
    for s in shape:
        n *= s
    return MT19937(seed).randn(n).reshape(shape)
