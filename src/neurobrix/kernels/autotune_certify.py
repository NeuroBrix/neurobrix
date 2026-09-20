"""`neurobrix autotune certify --profile <profile>` — fills the certified autotune directory.

For one vendor profile, on the machine that carries it: every candidate
config of every autotuned kernel is run on every shape the zoo encountered,
its result compared to the fp64 oracle (the reference bank's definition of
the op, computed here in float64 with numpy — no torch under `src/`), the
configs that diverge beyond the profile's tolerance excluded with their
deviation, the rest timed, and the winner written with its proof into
``config/autotune/<vendor>/<profile>/<kernel>.<dtype>.json``.

The shapes come from the machine's replay cache — the accumulation of every
key the zoo's runs resolved on this machine — or from a census file. Nothing
here names a backend or a model.

Made to be run by us on our machines and by a contributor on theirs: the
directory's gate (`autotune_certified.validate`) refuses a file without a
proof or whose proof does not re-read.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import warnings
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from neurobrix.kernels import autotune_certified as C

ORACLE = "fp64: the op in float64 (numpy), the reference bank's definition"
# The whole oracle of a convolution is computed when it costs at most this many multiply-adds;
# above it, the oracle is computed on WINDOWS of the output — exact float64 on every position
# of the windows, the corners and the centre of the plane, on the first, middle and last batch
# element — and the deviation is measured there. A kernel config's arithmetic is the same at
# every output position (one tile shape, one accumulation order), and a tiling or boundary
# fault reaches a corner: the census's video and 4K shapes (49 frames of 1024², 512 channels)
# would otherwise cost days of float64 each on this machine (2026-09-07: three 896² shapes an
# hour). The proof names the windows.
ORACLE_MAX_MACS = 2_000_000_000
BENCH_WARMUP_MS = 10
BENCH_REP_MS = 40
# Every candidate config runs once against the oracle, and that run is timed; only the
# CONTENDERS — the configs within this factor of the fastest run — go to the stopwatch.
# A register-spilling config on a 1024² convolution runs tens of times slower than the
# winners and cannot win: on 2026-09-07 the bench of all 18 took 557 s of a 672 s shape.
BENCH_CONTENDER_FACTOR = 2.0


def _contenders(results, factor=None):
    """(cfg, deviation, run_s) rows whose single run was within `factor` of the fastest;
    never fewer than two when two exist (the proof records a second-best time)."""
    factor = BENCH_CONTENDER_FACTOR if factor is None else factor
    if not results:
        return []
    fastest = min(r[2] for r in results)
    keep = [r for r in results if r[2] <= factor * fastest]
    if len(keep) < 2 and len(results) >= 2:
        keep = sorted(results, key=lambda r: r[2])[:2]
    return keep


# ---------------------------------------------------------------------------
# the census: which shapes
# ---------------------------------------------------------------------------
def census(path: Optional[str] = None) -> Dict[str, List[tuple]]:
    """{kernel qualname: [key, ...]} from the machine's replay cache (default)
    or a census file of the same shape (`{"entries": {"<qual>::<key>": ...}}`)."""
    from neurobrix.triton import autotune_cache as atc
    src = path or atc._artifact_path()
    if not src or not os.path.exists(src):
        return {}
    doc = json.load(open(src))
    entries = doc.get("entries", doc) if isinstance(doc, dict) else {}
    out: Dict[str, List[tuple]] = {}
    for ident in entries:
        if "::" not in ident:
            continue
        qual, ktext = ident.split("::", 1)
        key = C.parse_key(ktext)
        if key is not None:
            out.setdefault(qual, []).append(key)
    return out


# ---------------------------------------------------------------------------
# inputs from a key, and the fp64 oracle
# ---------------------------------------------------------------------------
_NP = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32, "fp64": np.float64}


class _Synth(np.ndarray):
    """An array that remembers the NBX dtype its values belong to.

    numpy has no bfloat16, and that single fact locked the whole Apple
    certification: `_NP` mapped `"bf16"` to `np.float32`, so a census key
    saying bf16 got an fp32 tensor, the wrapper recomputed an fp32 key, and
    the two could not match. Every shape with a bf16 input failed by
    construction — and Apple models are massively bf16.

    The refusal was right and must not be relaxed: an entry written under a
    key the runtime will never recompute is an entry nobody finds. What was
    needed was to be able to recompute the key.

    bf16 is nothing but the top sixteen bits of an fp32, so its values are
    carried EXACTLY in an fp32 array and this tag says what the kernel must
    receive. The oracle then reads the same values and is exact rather than
    merely close: a bf16 value is exactly representable in fp32 and in fp64.
    No dependency is added — measured 2026-09-11, the integer conversion
    below agrees with `torch.bfloat16` on 200 013 values with ZERO
    divergence, exact halves, subnormals, infinities, NaN and the fp32
    maximum included.
    """

    def __new__(cls, arr, nbx_dtype):
        obj = np.asarray(arr).view(cls)
        obj._nbx_dtype = nbx_dtype
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._nbx_dtype = getattr(obj, "_nbx_dtype", None)


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """The top sixteen bits, rounded to nearest with ties to even.

    `+0x7FFF` alone rounds an exact half down; `((u >> 16) & 1)` adds one ulp
    when the kept bit is odd, which is exactly ties-to-even. A NaN whose
    mantissa lives only in the discarded bits would become an infinity, so it
    is forced to a quiet NaN.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    bits = ((u + bias) >> np.uint32(16)).astype(np.uint16)
    bits[np.isnan(a)] = np.uint16(0x7FC0)
    return bits


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """The exact value those bits represent, as fp32."""
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32)
            << np.uint32(16)).view(np.float32)


def _arr(rng, shape, dtype_name, scale=0.1):
    a = (rng.standard_normal(shape) * scale)
    if dtype_name == "bf16":
        # The VALUES are made exactly representable in bf16, so the oracle
        # reading this array reads what the kernel will receive.
        exact = bf16_bits_to_f32(f32_to_bf16_bits(a.astype(np.float32)))
        return _Synth(exact.reshape(np.shape(a)), "bf16")
    return _Synth(a.astype(_NP.get(dtype_name, np.float32)), dtype_name)


def _conv_out_hw(h, wd, kh, kw, stride, padding, dilation):
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    return (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1


def _conv2d_oracle(x, w, stride, padding, dilation, groups, window=None):
    """Direct convolution in float64 (NCHW, OIHW), the reference bank's definition.
    `window` = (n_idx, r0, r1, c0, c1): only the output block [n_idx, :, r0:r1, c0:c1],
    exact on every position of it (its receptive field is what is read)."""
    w = w.astype(np.float64)
    n, c, h, wd = x.shape
    co, ci_g, kh, kw = w.shape
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    oh, ow = _conv_out_hw(h, wd, kh, kw, stride, padding, dilation)
    if window is None:
        n0, n1, r0, r1, c0, c1 = 0, n, 0, oh, 0, ow
    else:
        ni, r0, r1, c0, c1 = window
        n0, n1 = ni, ni + 1
    # Only the window's receptive field is converted and padded: in padded coordinates the
    # rows [r0·sh, (r1−1)·sh + dh·(kh−1)] and the same for columns — never the whole input
    # (a 1024²×256 input is 2 GB of float64 per window, 109 s of an oracle on 2026-09-07).
    R0, R1 = r0 * sh, (r1 - 1) * sh + dh * (kh - 1) + 1
    C0, C1 = c0 * sw, (c1 - 1) * sw + dw * (kw - 1) + 1
    u0, u1 = max(0, R0 - ph), min(h, R1 - ph)                 # unpadded rows the slab needs
    v0, v1 = max(0, C0 - pw), min(wd, C1 - pw)
    slab = x[n0:n1, :, u0:u1, v0:v1].astype(np.float64)
    top, bottom = max(0, ph - R0), max(0, (R1 - ph) - h)     # padding the slab still needs
    left, right = max(0, pw - C0), max(0, (C1 - pw) - wd)
    xp = np.pad(slab, ((0, 0), (0, 0), (top, bottom), (left, right)))
    nb, rh, rw = n1 - n0, r1 - r0, c1 - c0
    out = np.zeros((nb, co, rh, rw), dtype=np.float64)
    co_g = co // groups
    if groups == c == co and ci_g == 1:
        # depthwise: one broadcast product per tap over every channel — the per-group loop
        # below is thousands of tiny products (a 448² depthwise shape: 157 s of float64)
        for i in range(kh):
            for j in range(kw):
                patch = xp[:, :, i * dh:i * dh + rh * sh:sh, j * dw:j * dw + rw * sw:sw]
                out += patch * w[:, 0, i, j][None, :, None, None]
        return out
    for g in range(groups):
        xg = xp[:, g * ci_g:(g + 1) * ci_g]
        wg = w[g * co_g:(g + 1) * co_g]                       # [co_g, ci_g, kh, kw]
        for i in range(kh):
            for j in range(kw):
                patch = xg[:, :, i * dh:i * dh + rh * sh:sh, j * dw:j * dw + rw * sw:sw]   # [nb, ci_g, rh, rw]
                # one BLAS product per tap: (nb·rh·rw, ci_g) @ (ci_g, co_g)
                prod = patch.transpose(0, 2, 3, 1).reshape(-1, ci_g) @ wg[:, :, i, j].T
                out[:, g * co_g:(g + 1) * co_g] += prod.reshape(nb, rh, rw, co_g).transpose(0, 3, 1, 2)
    return out


def _conv_windows(n, oh, ow, ci_g, co, kh, kw, cap=None):
    """None when the whole oracle fits the cap; else the windows of the output the oracle is
    computed on — the top-left corner of the first batch element, the centre of the middle
    one, the bottom-right corner of the last — each of at most cap/3 multiply-adds."""
    cap = ORACLE_MAX_MACS if cap is None else cap
    per_position = ci_g * co * kh * kw
    if n * oh * ow * per_position <= cap:
        return None
    budget = max(1, int(cap / 3 // per_position))
    side = max(1, int(budget ** 0.5))
    rh, rw = min(oh, side), min(ow, side)
    if rh * rw > budget:
        rw = max(1, budget // rh)
    wins = [(0, 0, rh, 0, rw),
            (n // 2, (oh - rh) // 2, (oh - rh) // 2 + rh, (ow - rw) // 2, (ow - rw) // 2 + rw),
            (n - 1, oh - rh, oh, ow - rw, ow)]
    seen, out = set(), []
    for w in wins:
        if w not in seen:
            seen.add(w); out.append(w)
    return out


class WindowedOracle:
    """The float64 oracle on windows of a convolution's output; the deviation of a kernel's
    result is measured on those windows only, and the proof names them."""

    def __init__(self, blocks, oh, ow, n):
        self.blocks = blocks                              # [(window, float64 array [1, co, rh, rw])]
        self.describe = (f"on {len(blocks)} window(s) of the output (" +
                         "; ".join(f"batch {w[0]} rows {w[1]}-{w[2]} cols {w[3]}-{w[4]}" for w, _ in blocks) +
                         f") of {n}x{oh}x{ow}")

    def slices(self, out):
        for (ni, r0, r1, c0, c1), ref in self.blocks:
            yield out[ni:ni + 1, :, r0:r1, c0:c1], ref


def _conv_oracle_fn(x, wt, stride, padding, dilation, groups):
    n, ci, h, wd = x.shape
    co, ci_g, kh, kw = wt.shape
    oh, ow = _conv_out_hw(h, wd, kh, kw, stride, padding, dilation)
    wins = _conv_windows(n, oh, ow, ci_g, co, kh, kw)
    if wins is None:
        return lambda: _conv2d_oracle(x, wt, stride, padding, dilation, groups)
    return lambda: WindowedOracle([(w, _conv2d_oracle(x, wt, stride, padding, dilation, groups, window=w)) for w in wins], oh, ow, n)


def synthesize(qual: str, tuner, key: tuple, rng) -> Optional[Tuple[Callable[[], Any], Callable[[], np.ndarray], str]]:
    """(call, oracle_fn, output_arg_name): a callable that runs the wrapper on
    inputs of this key's shape and dtypes, a callable computing the fp64
    oracle of the same inputs (LAZY: only once the wrapper's key is known to
    be the census's — an oracle for a mismatched key is minutes wasted), and
    the name of the kernel argument that is the output. None when this
    kernel has no synthesizer here."""
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from neurobrix.kernels import wrappers as W
    dts = C.key_dtypes(key)
    short = C.kernel_short(qual)
    def to(a):
        """The NBXTensor the kernel must receive, with the dtype the key names.

        A bf16 array travels in a uint16 container, which `from_numpy`'s
        `dtype` argument exists for — it names the NBX dtype the BITS already
        are. Passing the fp32 carrier instead is what made every bf16 shape
        uncertifiable.
        """
        if getattr(a, "_nbx_dtype", None) == "bf16":
            bits = f32_to_bf16_bits(np.ascontiguousarray(np.asarray(a)))
            return NBXTensor.from_numpy(bits, dtype=NBXDtype.bfloat16)
        return NBXTensor.from_numpy(np.ascontiguousarray(np.asarray(a)))
    if short in ("matmul_kernel", "addmm_kernel"):
        M, N, K = int(key[0]), int(key[1]), int(key[2])
        a = _arr(rng, (M, K), dts[0] if dts else "fp16")
        b = _arr(rng, (K, N), dts[1] if len(dts) > 1 else "fp16")
        if short == "matmul_kernel":
            return (lambda: W.mm(to(a), to(b))), (lambda: a.astype(np.float64) @ b.astype(np.float64)), "c_ptr"
        bias = _arr(rng, (N,), dts[2] if len(dts) > 2 else "fp16")
        return ((lambda: W.addmm(to(bias), to(a), to(b))),
                (lambda: bias.astype(np.float64)[None, :] + a.astype(np.float64) @ b.astype(np.float64)), "c_ptr")
    if short == "baddbmm_kernel":
        M, N, K = int(key[0]), int(key[1]), int(key[2])
        has_bias = bool(key[5]) if len(key) > 5 and isinstance(key[5], bool) else False
        B = 2
        a = _arr(rng, (B, M, K), dts[0] if dts else "fp16")
        b = _arr(rng, (B, K, N), dts[1] if len(dts) > 1 else "fp16")
        if has_bias:
            bias_dt = dts[3] if len(dts) > 3 else (dts[0] if dts else "fp16")
            bias = _arr(rng, (B, M, N), bias_dt)
            return ((lambda: W.baddbmm_wrapper(to(bias), to(a), to(b))),
                    (lambda: a.astype(np.float64) @ b.astype(np.float64) + bias.astype(np.float64)), "out_ptr")
        return (lambda: W.bmm(to(a), to(b))), (lambda: a.astype(np.float64) @ b.astype(np.float64)), "out_ptr"
    if short == "conv2d_forward_kernel":
        (n, ci, h, w, co, _oh, _ow, kh, kw, sh, sw, ph, pw, dh, dw, groups) = [int(v) for v in key[:16]]
        x = _arr(rng, (n, ci, h, w), dts[0] if dts else "fp16")
        wt = _arr(rng, (co, ci // max(groups, 1), kh, kw), dts[1] if len(dts) > 1 else "fp16")
        return ((lambda: W.conv2d_wrapper(to(x), to(wt), None, (sh, sw), (ph, pw), (dh, dw), False, 0, groups)),
                _conv_oracle_fn(x, wt, (sh, sw), (ph, pw), (dh, dw), groups), "output_pointer")
    if short == "depthwise_conv2d_kernel":
        (c, h, w, _oh, _ow, kh, kw, sh, sw, ph, pw) = [int(v) for v in key[:11]]
        x = _arr(rng, (1, c, h, w), dts[0] if dts else "fp16")
        wt = _arr(rng, (c, 1, kh, kw), dts[1] if len(dts) > 1 else "fp16")
        return ((lambda: W.conv2d_wrapper(to(x), to(wt), None, (sh, sw), (ph, pw), (1, 1), False, 0, c)),
                _conv_oracle_fn(x, wt, (sh, sw), (ph, pw), (1, 1), c), "out_ptr")
    return None


def host_values(t) -> np.ndarray:
    """A kernel's output as NUMBERS, whatever dtype the tensor carries.

    `NBXTensor.numpy()` hands bf16 back as a 2-byte VOID array (`|V2`): numpy
    has no bfloat16, so the bits are correct and nothing can read them as
    numbers. `np.asarray(that, dtype=np.float64)` raises "setting an array
    element with a sequence", and that is how EVERY bf16 shape failed
    certification — reported as "no config could run (10 of 10)" on kernels
    that had in fact run correctly and produced the right answer. Measured
    2026-09-17: `mm` on a (22,2048)x(2048,2048) bf16 pair returns a bf16
    tensor of the right shape, and only the READING of it failed.

    The input half of this defect was found and fixed earlier (`_NP` mapped
    bf16 to fp32, so a bf16 census key was handed an fp32 tensor and the
    wrapper recomputed an fp32 key). This is the output half of the same
    thing, and on Apple it is the more expensive half: 58 of the 84 shapes the
    seven proven cells demand are bf16, conv2d alone accounting for 53.

    Decoded through `bf16_bits_to_f32`, which is exact — a bf16 value is
    exactly representable in fp32 — so the comparison against the fp64 oracle
    loses nothing here.
    """
    host = t.numpy()
    dt = getattr(host, "dtype", None)
    if dt is not None and dt.kind == "V" and dt.itemsize == 2:
        return bf16_bits_to_f32(np.ascontiguousarray(host).view(np.uint16))
    return host


def oracle_deviation(out: np.ndarray, oracle) -> float:
    """max |out - oracle| relative to the oracle's own magnitude; inf when one
    side is not finite where the other is. A windowed oracle is measured on
    its windows, the largest deviation among them."""
    if isinstance(oracle, WindowedOracle):
        return max(oracle_deviation(piece, ref) for piece, ref in oracle.slices(np.asarray(out)))
    x = np.asarray(out, dtype=np.float64).reshape(-1)
    y = np.asarray(oracle, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        return float("inf")
    fx, fy = np.isfinite(x), np.isfinite(y)
    if not np.array_equal(fx, fy):
        return float("inf")
    if not fy.any():
        return 0.0
    scale = float(np.abs(y[fy]).max())
    if scale == 0.0:
        return 0.0 if np.array_equal(x, y) else float("inf")
    return float(np.abs(x[fy] - y[fy]).max() / scale)


# ---------------------------------------------------------------------------
# one key: every config against the oracle, the survivors against the clock
# ---------------------------------------------------------------------------
def _machine() -> Dict[str, Any]:
    info: Dict[str, Any] = {"hostname": socket.gethostname(), "platform": platform.platform(),
                            "python": platform.python_version()}
    try:
        from neurobrix.kernels import wrappers as W
        hw = W.get_hardware_profile()
        if hw is not None:
            info["hardware_profile"] = str(getattr(hw, "id", hw))
            info["has_native_bf16"] = bool(W.has_native_bf16())
    except Exception:
        pass
    try:
        from neurobrix.kernels.launcher import target
        t = target()
        info["target"] = f"{t.backend}-{t.arch}"
    except Exception:
        pass
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        for name in ("device_name", "driver_version"):
            fn = getattr(DeviceAllocator, name, None)
            if callable(fn):
                info[name] = fn()
    except Exception:
        pass
    info["device"] = _certifying_device()
    info["clocks_mhz"] = _clocks_mhz()
    return info


_CERTIFYING_DEVICE: list = []          # read once per run, like the clocks


def _certifying_device() -> Optional[Dict[str, Any]]:
    """Read ONCE for the run (a query per key is the anti-pattern `_clocks_mhz` names)."""
    if not _CERTIFYING_DEVICE:
        _CERTIFYING_DEVICE.append(_read_certifying_device())
    return _CERTIFYING_DEVICE[0]


def _read_certifying_device() -> Optional[Dict[str, Any]]:
    """The card the certifying run executes on — the device `from_numpy`
    places the synthesized inputs on (`DeviceAllocator.get_device()`), read in
    the Prism hardware profile in force: index, name, memory. The proof says
    which card's memory it was made on, and an entry is served only to that
    memory class (register 56). None when the card is not in the profile —
    the certifier then REFUSES to write the entry (`file_certification`)."""
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        idx = int(DeviceAllocator.get_device())
    except Exception:
        return None
    try:
        from neurobrix.kernels import wrappers as W
        prof = W.get_hardware_profile()
        devices = getattr(prof, "devices", None) if prof is not None else None
    except Exception:
        devices = None
    dev = next((d for d in (devices or []) if getattr(d, "index", None) == idx), None)
    if dev is None:
        return None
    # `ordinal` is the CUDA ordinal in the visible set (0 under a pin to any physical card),
    # not the physical card; the visible set is recorded beside it so the pair says which.
    return {"ordinal": idx, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": str(getattr(dev, "name", "?")), "memory_mb": int(getattr(dev, "memory_mb", 0) or 0)}


def _clocks_mhz():
    """The application clock of every card, or None when it cannot be read.

    A certification picks a configuration BY TIMING candidates, so the frequency
    the cards ran at is a condition of the result exactly as the platform and the
    hardware profile already recorded here are. Without it a proof cannot say
    what regime produced its `best_ms`, and the question becomes unanswerable the
    moment the machine reboots — which is how the 2026-09-11 entries came to
    carry timings whose clock is unrecoverable.

    It records what it read and never asserts a protocol: whether a reading is
    the right one is the workshop's question (`tools/rig_clock.py`), not the
    engine's. `None` means the reading failed and is written as such, because a
    field quietly absent is indistinguishable from a machine that had no clocks.

    READ ONCE for the whole run, not once per shape. `_machine()` is called for
    every certified key — 7,137 of them in this directory — and a driver query
    per key is the anti-pattern this project has already paid for once. So this
    is the reading at the run's START, and it says nothing about whether the
    clock HELD: holding is the sampler's job (`ClockWatch` in bench_row.py),
    which watches `clocks.sm` throughout and marks an excursion. Entry condition
    here, ongoing condition there; neither substitutes for the other.
    """
    if _clocks_mhz.cached is not _UNREAD:
        return _clocks_mhz.cached
    _clocks_mhz.cached = _read_clocks_mhz()
    return _clocks_mhz.cached


class UnreachableCensusKey(RuntimeError):
    """The census holds a key the engine can no longer ask for.

    NOT a failure of the certification, and the distinction is the whole point of
    the class. The census accumulates across engine versions; when a wrapper
    changes how it computes its autotune key, every entry recorded under the old
    rule becomes unreachable — no run will ever present that key again, so there
    is nothing to certify and refusing is correct.

    It is separated from a real break because an exit code that conflates the two
    stops being read. On 2026-09-12 the certification refused 184 such keys, all
    of them the known debt D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE, and
    exited 1 — the same 1 a genuine break would produce. A status that says
    "something is wrong" on every run of a directory that is in fact healthy will
    be ignored, and the day it tells the truth nobody will listen.
    """


_UNREAD = object()


def _read_clocks_mhz():
    """Both application clocks of every card.

    Both, because `nvidia-smi -ac <mem>,<gfx>` sets both and a protocol names
    both. A reader that returns only the graphics clock lets a rig satisfy this
    door while failing the workshop's (`tools/rig_clock.py`), and two doors that
    disagree about the same protocol are a false green waiting for its occasion.

    Every card, because `nvidia-smi` speaks real indices and ignores
    CUDA_VISIBLE_DEVICES: a card at the wrong frequency beside a measurement is
    a fact about the rig whether or not this job was pinned away from it.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.applications.graphics,"
             "clocks.applications.memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        out = {}
        for line in r.stdout.splitlines():
            if line.strip():
                idx, gfx, mem = [c.strip() for c in line.split(",")]
                out[idx] = {"graphics": int(gfx), "memory": int(mem)}
        return out or None
    except Exception:
        return None


_clocks_mhz.cached = _UNREAD

_PROTOCOL_ENV = "NEUROBRIX_RIG_PROTOCOL"
OFF_PROTOCOL_OPT = "--allow-off-protocol-clock"


def _current_backend() -> Optional[str]:
    """The backend this machine targets (`cuda`, `metal`, `hip`), or None.

    A protocol is scoped to a backend, so selecting the file needs the backend
    name — and it must be the ENGINE's name for the backend, not the Triton
    target's.

    This used to read `launcher.target().backend`. Those two agree on NVIDIA
    (both say `cuda`), which is why the difference stayed invisible, and they
    diverge on Apple: `_detect_gpu_backend()` says `metal` while the Triton
    target says whatever the installed Triton backend calls itself — `metal`
    under the archived fork, `mps` under triton-ext.

    So swapping the Triton backend silently renamed the thing the protocol is
    scoped by, `rig_protocol.metal.json` stopped being found, and certify ran on
    2026-09-17 reporting "this machine declares no measurement protocol for
    backend 'mps' — the regime is RECORDED in every proof but checked against
    nothing". Six shapes were certified that way. **The witness is the whole
    reason Apple certification is trustworthy** (the clocks cannot be locked, so
    stability is proven by re-timing a reference kernel), and it was not being
    checked.

    A measurement regime is a property of the MACHINE AND ITS GPU — "Apple GPU
    clocks are OS-managed" is true whatever Triton calls the target — so it is
    scoped by the engine's own vendor name, which is stable across Triton
    backend swaps. The old proofs agree: they record `backend.name = "metal"`
    and `target = "metal-apple-m4-pro"`.
    """
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        return _detect_gpu_backend()
    except Exception:
        return None


def _protocol_file(backend: Optional[str] = None) -> Optional[Path]:
    """The machine's declared measurement protocol FOR ITS BACKEND, or None.

    A protocol is a property of the MACHINE AND ITS BACKEND, not of the engine:
    it records the regime this rack's numbers are taken under (a clock lock on
    an NVIDIA rack, a witness on Apple Silicon where the clock cannot be
    locked). It is DISCOVERED, never shipped with a built-in value.

    SCOPED BY BACKEND so it cannot LEAK across machines: the file is
    `rig_protocol.<backend>.json`, and a Mac (metal) never reads the Dell's
    `rig_protocol.cuda.json`. The un-suffixed `rig_protocol.json` was a leak by
    construction — the checkout carried one machine's V100 protocol and every
    other machine inherited it (2026-09-16: a Mac refused certification because
    it could not read the NVIDIA clocks the Dell's committed protocol named).

    Order: an explicit env pointer, then the backend file in a source checkout,
    then the machine's dotfile. A legacy un-suffixed file is honoured ONLY if it
    declares this backend, never blindly — the leak does not come back.
    """
    backend = backend or _current_backend()
    env = os.environ.get(_PROTOCOL_ENV)
    if env:
        return Path(env)                      # named, so a missing one refuses below
    names = [f"rig_protocol.{backend}.json"] if backend else []
    for name in names:
        for parent in Path(__file__).resolve().parents:
            cand = parent / "tools" / name
            if cand.is_file():
                return cand
        cand = Path.home() / ".neurobrix" / name
        if cand.is_file():
            return cand
    # A legacy un-suffixed file counts only if it names THIS backend.
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "rig_protocol.json"
        if cand.is_file() and _protocol_backend(cand) == backend:
            return cand
    cand = Path.home() / ".neurobrix" / "rig_protocol.json"
    if cand.is_file() and _protocol_backend(cand) == backend:
        return cand
    return None


def _protocol_backend(path: Path) -> Optional[str]:
    """The backend a protocol file declares (`_backend`), or None if unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("_backend")
    except Exception:
        return None


def rig_protocol_refusal(allow_off_protocol: bool = False, say=print) -> None:
    """Refuse to certify unless EVERY card sits at the machine's protocol clock.

    WHY A CERTIFICATION TAKES A CAMPAIGN'S ENTRY CONDITION
    -----------------------------------------------------
    Certification picks a configuration BY TIMING candidates. That makes it a
    measurement exactly as a benchmark row is one, and its `best_ms` values carry
    a regime whether or not anyone recorded which. On 2026-09-11 this machine lost
    mains twice in nine minutes, came back at 1312/1312/1290/1290 — each SKU at
    its OWN factory default — and a certification ran across all four cards with
    no harness reading the clocks at entry.

    WHY A DOOR AND NOT A CENSUS
    ---------------------------
    Application clocks do not survive a reboot, so the harmful state recurs on a
    schedule nobody controls. A census says the clocks were right this time; a
    refusal at entry says a number produced behind it was produced on protocol,
    which is a claim about every future run.

    WHY IT NAMES EVERY DIVERGING CARD
    ---------------------------------
    The rack is heterogeneous: 16 GB V100s default to 1312 MHz, 32 GB ones to
    1290, and the protocol value is 1290. After a reboot half the rig is already
    at the protocol value by pure manufacturer coincidence. A check that samples
    one card goes green on a rig that is half wrong, and one that samples card 2
    or 3 here goes green ALWAYS. Both SKUs advertise byte-identical supported-clock
    lists, so no capability query reveals the disagreement — only the reading does.

    A machine that declares NO protocol is not refused: there is nothing for it to
    diverge from, and the proof still records the clocks it read. But it is told
    so in the run's own output, because a silence here would be indistinguishable
    from a door that held.
    """
    path = _protocol_file()
    if path is None:
        be = _current_backend()
        say(f"[certify] this machine declares no measurement protocol for backend "
            f"{be!r} (no {_PROTOCOL_ENV}, no tools/rig_protocol.{be}.json, no "
            f"~/.neurobrix/rig_protocol.{be}.json) — the regime is RECORDED in "
            f"every proof but checked against nothing.")
        return

    try:
        proto = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the protocol from {path} ({exc}). The protocol is not "
            f"optional and has no default: restore the file rather than "
            f"certifying without it.")

    # The stability regime: a recorded LOCK (NVIDIA, clocks held to a protocol
    # and read at entry) or a recorded WITNESS (Apple, where the clock cannot be
    # locked, so stability is PROVEN per-sweep by a reference kernel's drift —
    # checked in `certify_key`, not here). The contract stays whole and single:
    # both are proofs, both are recorded, an entry without a recorded regime is
    # not served.
    regime = proto.get("regime", "clock_lock")
    if regime == "witness":
        _witness_entry_refusal(proto, path, say)
        return
    if regime != "clock_lock":
        raise RuntimeError(
            f"protocol {path} declares an unknown regime {regime!r}: the regimes "
            f"are 'clock_lock' (a held clock, read at entry) and 'witness' (a "
            f"reference kernel's drift, proven per sweep).")

    try:
        clock = proto["clock"]
        want_gfx = int(clock["application_graphics_mhz"])
        want_mem = int(clock["application_memory_mhz"])
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the clock protocol from {path} ({exc}). The protocol "
            f"value is not optional and has no default: restore the file rather "
            f"than certifying without it.")

    cards = _clocks_mhz()
    if cards is None:
        raise RuntimeError(
            "cannot read the rig's clocks, and this machine declares a protocol "
            f"({want_gfx}/{want_mem} MHz) to hold them to. A certification whose "
            "conditions cannot be read is not a measurement.")
    if not cards:
        raise RuntimeError(
            "the driver reported ZERO cards. This check examined nothing, and a "
            "check that examined nothing must not be read as one that found "
            "nothing wrong.")

    off = {i: c for i, c in cards.items()
           if c["graphics"] != want_gfx or c["memory"] != want_mem}
    if not off:
        say(f"[certify] {len(cards)} card(s) read, all at the protocol clock "
            f"{want_gfx}/{want_mem} MHz ({path})")
        return

    lines = [f"the rig is NOT at the protocol clock: {len(off)} of {len(cards)} "
             f"card(s) diverge (protocol {want_gfx}/{want_mem} MHz, from {path})"]
    for i in sorted(cards, key=lambda s: int(s)):
        c = cards[i]
        lines.append(f"    card {i}  {c['graphics']}/{c['memory']} MHz"
                     + ("" if i in off else "  (at protocol)"))
    lines.append("")
    lines.append("  Application clocks do not survive a reboot and each SKU returns "
                 "to its OWN factory default, so half a heterogeneous rack can sit "
                 "at the protocol value by coincidence.")
    lines.append("")
    lines.append("  Restore every card, then re-run:")
    lines.append("      for i in " + " ".join(sorted(cards, key=lambda s: int(s)))
                 + f"; do sudo nvidia-smi -i $i -ac {want_mem},{want_gfx}; done")

    if allow_off_protocol:
        for line in ["WARNING, " + lines[0]] + lines[1:]:
            say("[certify] " + line)
        say(f"[certify] proceeding anyway on {OFF_PROTOCOL_OPT}: every timing "
            f"produced by this run was taken off protocol and may not be compared "
            f"with one that was not.")
        return

    raise RuntimeError("\n".join(lines) +
                       f"\n\n  To certify off protocol deliberately, pass "
                       f"{OFF_PROTOCOL_OPT} — the run then says so in its own output.")


_REGIME = _UNREAD


def _regime():
    """(regime_str, protocol_dict) for this machine's backend, read once.

    ('clock_lock', proto) on an NVIDIA rack, ('witness', proto) on Apple,
    (None, None) when the machine declares no protocol for its backend.
    """
    global _REGIME
    if _REGIME is not _UNREAD:
        return _REGIME
    path = _protocol_file()
    if path is None:
        _REGIME = (None, None)
        return _REGIME
    try:
        proto = json.loads(Path(path).read_text(encoding="utf-8"))
        _REGIME = (proto.get("regime", "clock_lock"), proto)
    except Exception:
        _REGIME = (None, None)
    return _REGIME


def _witness_time_ms(proto: Dict[str, Any]) -> float:
    """Time the protocol's witness kernel — a fixed-shape matmul, always the
    same — with the sweep's own timer (`L.do_bench`). The one number whose only
    variable across a sweep's open and close is the GPU regime: same shape, same
    inputs, same timer, so a change in it is a change in the machine."""
    from neurobrix.kernels import launcher as L
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    spec = proto["witness"]
    if spec.get("kernel", "matmul") != "matmul":
        raise RuntimeError(f"witness kernel {spec.get('kernel')!r} not supported "
                           f"(only 'matmul')")
    M, N, K = int(spec["M"]), int(spec["N"]), int(spec["K"])
    dtype = spec.get("dtype", "fp16")
    npd = _NP.get(dtype, np.float16)
    rng = np.random.default_rng(0)                      # fixed inputs, every time
    a = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((M, K)) * 0.1).astype(npd)))
    b = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((K, N)) * 0.1).astype(npd)))
    return float(L.do_bench(lambda: W.mm(a, b), warmup=BENCH_WARMUP_MS, rep=BENCH_REP_MS))


def _witness_entry_refusal(proto: Dict[str, Any], path: Path, say) -> None:
    """Entry condition for the witness regime: the reference kernel must RUN, so
    the per-sweep drift check has a baseline it can take. It does NOT read or
    hold a clock — Apple's clock is OS-managed and unlockable; stability is
    proven per sweep in `certify_key`, not asserted here."""
    if "witness" not in proto:
        raise RuntimeError(
            f"protocol {path} declares regime 'witness' but carries no 'witness' "
            f"spec (shape, dtype, drift_tolerance): a witness regime with no "
            f"witness is a door with no hinge.")
    try:
        t = _witness_time_ms(proto)
    except Exception as exc:
        raise RuntimeError(
            f"the witness kernel from {path} could not be timed ({exc}). A "
            f"certification whose stability cannot be measured is not a "
            f"measurement — the same rule the clock lock enforces, by its own "
            f"means on this backend.")
    spec = proto["witness"]
    say(f"[certify] regime WITNESS ({path}): reference {spec.get('kernel','matmul')} "
        f"{spec['M']}x{spec['N']}x{spec['K']} {spec.get('dtype','fp16')} opens at "
        f"{t:.4f} ms; each sweep is bracketed and refused if it drifts past "
        f"{float(spec['drift_tolerance'])*100:.0f}%.")


def _backend() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import triton
        out["triton"] = triton.__version__
    except Exception:
        pass
    try:
        from neurobrix.kernels.launcher import target
        out["name"] = target().backend
    except Exception:
        out["name"] = "?"
    return out


def certify_key(qual: str, tuner, key: tuple, tolerance: float, rng, bench=None) -> Dict[str, Any]:
    """Run every candidate of `tuner` on this key's inputs, compare each to the
    fp64 oracle, exclude beyond `tolerance`, time the rest; the entry (config,
    proof, excluded) for the directory. Raises when nothing survives."""
    from neurobrix.kernels import launcher as L
    from neurobrix.triton import autotune_cache as atc
    from triton.runtime.autotuner import Autotuner
    made = synthesize(qual, tuner, key, rng)
    if made is None:
        raise RuntimeError(f"no synthesizer for {qual}")
    call, oracle_fn, out_name = made
    oracle_box: Dict[str, Any] = {}
    bench = bench or (lambda fn: L.do_bench(fn, warmup=BENCH_WARMUP_MS, rep=BENCH_REP_MS))
    upstream_prune = getattr(Autotuner.prune_configs, "_nbx_upstream", Autotuner.prune_configs)
    state: Dict[str, Any] = {}
    saved_run = tuner.run

    def certifying_run(*args, **kwargs):
        # The seam the autotuner itself uses: its own key must be this one.
        tuner.nargs = dict(zip(tuner.arg_names, args))
        seen = atc.key_of(tuner, args, kwargs)
        if tuple(seen) != tuple(key):
            raise UnreachableCensusKey(
                f"the wrapper computed key {seen!r} for inputs synthesized from {key!r}: the census "
                f"and the kernel disagree — nothing certified for this key")
        oracle = oracle_box.get("v")
        t_or = time.time()
        if oracle is None:                              # the key matched: now the fp64 oracle is worth computing
            oracle = oracle_box["v"] = oracle_fn()
        state["t_oracle"] = round(time.time() - t_or, 3)
        state["oracle"] = ORACLE + (" " + oracle.describe if isinstance(oracle, WindowedOracle) else "")
        configs = list(upstream_prune(tuner, kwargs))
        names = list(tuner.arg_names)
        out_idx = next((i for i, n in enumerate(names) if n == out_name), None)
        if out_idx is None or out_idx >= len(args):
            raise RuntimeError(f"{qual}: output argument {out_name!r} not among the kernel's arguments")
        out_tensor = args[out_idx]
        buffers = L._writable_buffers(args)
        if buffers is None:
            raise RuntimeError(f"{qual}: a strided view among the arguments — not certifiable")
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        out_addr, out_nbytes = int(out_tensor.data_ptr()), int(out_tensor._nbytes)

        def poison():
            # Every byte of the output is set to 0xFF (NaN for every float
            # dtype) before a candidate runs, on the device: a config that
            # writes only part of its output — the class the screen found on
            # 2026-09-07 — cannot inherit the previous candidate's correct
            # values. The sanctioned kernels read their inputs and write
            # their output, so nothing else needs restoring; the screen's
            # snapshot/restore of every buffer through the host is what made
            # a shape cost ten seconds.
            DeviceAllocator.memset_cuda(out_addr, 0xFF, out_nbytes)

        results: List[Tuple[Any, float, float]] = []
        excluded: List[Dict[str, Any]] = []
        unrun: List[Any] = []
        t_runs = time.time()
        for cfg in configs:
            poison()
            try:
                t_one = time.time()
                tuner.fn.run(*args, **{**kwargs, **cfg.all_kwargs()})
                DeviceAllocator.stream_synchronize(0)
                run_s = time.time() - t_one                # the run every config makes anyway, timed
                dev = oracle_deviation(host_values(out_tensor), oracle)
            except Exception as exc:                     # a config the backend refuses: counted, never trusted
                unrun.append({"config": atc._config_to_dict(cfg), "error": str(exc)[:200]})
                continue
            if not (dev <= tolerance):
                excluded.append({"config": atc._config_to_dict(cfg), "deviation": dev, "tolerance": tolerance})
            else:
                results.append((cfg, dev, run_s))
        if not results:
            if not excluded and unrun:
                raise RuntimeError(f"{qual} at {key!r}: no config could run ({len(unrun)} of {len(configs)}; "
                                   f"first: {unrun[0]['error']})")
            raise RuntimeError(f"{qual} at {key!r}: every config diverges from the fp64 oracle beyond {tolerance:g} "
                               f"({len(excluded)} excluded, {len(unrun)} could not run"
                               + (f"; first error: {unrun[0]['error']}" if unrun else "") + ")")
        state["t_runs"] = round(time.time() - t_runs, 3)
        timed: List[Tuple[Any, float, float]] = []
        t_bench = time.time()
        contenders = _contenders(results)
        # STABILITY REGIME. The candidate timings below decide `best_ms` by
        # comparison, so they are only comparable if the machine held still
        # across them. On an NVIDIA rack that is the clock lock, read at entry.
        # On Apple the clock is OS-managed and unlockable, so stability is
        # PROVEN, not asserted: a fixed reference kernel is timed just before and
        # just after this sweep, and if it drifts past the profile's tolerance
        # the regime moved while we compared and the sweep is refused.
        _regime_kind, _proto = _regime()

        def _witness_ms():
            """Time the witness on the UNPATCHED path.

            `certifying_run` IS the sweep, and it is installed AS `tuner.run`.
            The witness is itself a matmul (`W.mm`), so on a matmul
            certification it re-enters this very function: the nested call
            computes the witness's own key and compares it to the candidate's.

            Measured 2026-09-17, the first run after the protocol became
            findable at all: every fp16 matmul shape came back "UNREACHABLE —
            the wrapper computed key (512, 512, 512)", which is the witness's
            shape and not the census's, and the (512,512,512) shape itself
            FAILED with all ten configs "diverging from the fp64 oracle"
            because the witness's own launches were being taken for candidate
            runs. 0 certified where the unwitnessed run had certified 6.

            Suspending the patch keeps the witness on the real path — it must
            measure the same machine the candidates are measured on — while
            stopping it from being mistaken for one of them.
            """
            tuner.run = saved_run
            try:
                return _witness_time_ms(_proto)
            finally:
                tuner.run = certifying_run

        _w_open = _witness_ms() if _regime_kind == "witness" else None
        for cfg, dev, _run_s in contenders:
            ms = bench(lambda: tuner.fn.run(*args, **{**kwargs, **cfg.all_kwargs()}))
            timed.append((cfg, dev, float(ms)))
        if _regime_kind == "witness":
            _w_close = _witness_ms()
            _tol = float(_proto["witness"]["drift_tolerance"])
            _drift = abs(_w_close - _w_open) / max(_w_open, 1e-9)
            if _drift > _tol:
                raise RuntimeError(
                    f"{qual} at {key!r}: the witness drifted {_drift*100:.1f}% "
                    f"across the sweep ({_w_open:.4f} -> {_w_close:.4f} ms, "
                    f"tolerance {_tol*100:.0f}%): the GPU regime moved while "
                    f"candidates were being compared, so their times are not "
                    f"comparable. Sweep refused — not a measurement.")
            state["stability_witness"] = {"open_ms": round(_w_open, 4),
                                          "close_ms": round(_w_close, 4),
                                          "drift": round(_drift, 4),
                                          "tolerance": _tol,
                                          "kernel": _proto["witness"].get("kernel", "matmul"),
                                          "shape": [int(_proto["witness"]["M"]),
                                                    int(_proto["witness"]["N"]),
                                                    int(_proto["witness"]["K"])]}
        state["t_bench"] = round(time.time() - t_bench, 3)
        timed.sort(key=lambda t: t[2])
        best, dev, ms = timed[0]
        state.update({"config": atc._config_to_dict(best), "deviation": dev, "best_ms": ms,
                      "second_ms": timed[1][2] if len(timed) > 1 else None,
                      "candidates": len(configs), "accepted": len(results), "benched": len(contenders),
                      "excluded": excluded, "unrun": unrun,
                      "timings": [{"config": atc._config_to_dict(c), "deviation": d, "ms": m} for c, d, m in timed]})
        tuner.cache[key] = best
        poison()
        return tuner.fn.run(*args, **{**kwargs, **best.all_kwargs()})

    # The wrappers decide the kernel's dtypes and flags (PROMOTE_B, IEEE_PRECISION,
    # a conv's output dtype) from the hardware profile and the component's compute
    # dtype the engine hands them — reproduce that context, or the wrapper computes
    # another key than the census's (the first dry run: every matmul key mismatched
    # because a bare process reads `has_native_bf16` as True).
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXDtype
    _prev_dt = W.get_compute_dtype()
    _out_dt = C.output_dtype(tuner, key)
    _nbx = {"fp16": NBXDtype.float16, "bf16": NBXDtype.bfloat16, "fp32": NBXDtype.float32}.get(_out_dt)
    tuner.run = certifying_run
    fell_back = []
    try:
        if _nbx is not None:
            W.set_compute_dtype(_nbx)
        # Did the kernel actually BUILD on the device, or did the backend fail
        # to compile it and fall back to the CPU?
        #
        # The screen cannot answer that. A CPU fallback COMPUTES CORRECTLY, so
        # its deviation against the fp64 oracle is excellent — 1e-6 like any
        # sound path — and an entry certified on it would record a
        # configuration chosen for a path that never runs. Measured
        # 2026-09-11: 30 of 30 entries do build, but the proof did not say so,
        # and a reader six months from now could not redo the check.
        #
        # What is measured and not written does not exist.
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")
            call()
            fell_back = [str(w.message).splitlines()[0][:120] for w in _caught
                         if "fall back to CPU" in str(w.message)
                         or "Metal compilation failed" in str(w.message)]
    finally:
        W.set_compute_dtype(_prev_dt)
        tuner.run = saved_run
        tuner.nargs = None
    if "config" not in state:
        raise RuntimeError(f"{qual} at {key!r}: the wrapper never reached the autotuner")
    proof = {"date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
             "engine_version": _engine_version(), "backend": _backend(), "shape": list(key),
             "deviation": state["deviation"], "tolerance": tolerance, "oracle": state.get("oracle", ORACLE), "machine": _machine(),
             "best_ms": state["best_ms"], "second_ms": state["second_ms"], "candidates": state["candidates"],
             "accepted": state["accepted"], "benched": state["benched"], "could_not_run": len(state["unrun"]),
             "seconds": {"oracle": state.get("t_oracle"), "runs": state.get("t_runs"), "bench": state.get("t_bench")},
             # The stability regime this sweep was measured under: the machine's
             # clock is in `machine.clocks_mhz` (a held lock, NVIDIA), and the
             # witness is here (a proven drift, Apple). One of the two is present
             # whenever a protocol was declared; `proof_records_regime` reads
             # either, and an entry with neither is not served.
             "stability_witness": state.get("stability_witness"),
             "built": {"gpu": not fell_back,
                       "how": "no backend compilation fallback was raised during "
                              "the certifying run",
                       "fallback": fell_back or None}}
    return {"config": state["config"], "proof": proof, "excluded": state["excluded"],
            "could_not_run": state["unrun"], "timings": state["timings"]}


def _bind_hardware_profile() -> str:
    """Hand Prism's hardware profile to the wrappers exactly as the CLI does
    before a request (`set_hardware_profile`): the flag `has_native_bf16` and
    the per-device VRAM the wrappers read for their dtype decisions."""
    from neurobrix.core.prism import load_profile
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.kernels import wrappers as W
    hw_id = get_or_create_default_profile()
    W.set_hardware_profile(load_profile(hw_id))
    return str(hw_id)


def _has_native_bf16() -> bool:
    from neurobrix.kernels import wrappers as W
    return bool(W.has_native_bf16())


def _engine_version() -> str:
    try:
        from neurobrix import __version__
        return str(__version__)
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# the run: a profile, its shapes, its files
# ---------------------------------------------------------------------------
def _tolerance(vendor: str, profile: str, dtype: str) -> float:
    tol = C._tolerance_for(vendor, profile, dtype)
    if tol is None:
        raise RuntimeError(f"the profile {vendor}/{profile} declares no `autotune_screen_rtol` for {dtype}: "
                           f"certification will not invent a tolerance")
    return tol


def _write_file(path: Path, vendor: str, profile: str, qual: str, dtype: str, entries: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": C.format_for(entries), "vendor": vendor, "profile": profile, "kernel": qual, "dtype": dtype,
           "entries": dict(sorted(entries.items()))}     # the stamp is what every entry satisfies, never the writer's era
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_file(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(doc.get("entries") or {}) if isinstance(doc, dict) else {}


_STICKY_MARKS = ("error 700", "rc=700", "illegal memory access", "STICKY")


def sticky_cuda_error(exc: BaseException) -> bool:
    """True when a key's failure says the CUDA context itself is dead. After an
    illegal memory access every later launch and malloc in the process fails
    with the same error, so nothing after it is a measurement: on 2026-09-14
    07:38 a certifier met one such key (a 2.19e9-element matmul on a kernel
    that still wrapped) and went on to report 227 further keys FAILED for
    "GPU malloc failed (error 700) for 256 bytes" — 227 shapes lost to a
    fault that happened once."""
    text = str(exc)
    return any(m in text for m in _STICKY_MARKS)


def after_key_failure(exc: BaseException, summary: Dict[str, Any], key_text: str, log) -> bool:
    """Count a key's failure; return True when the run must STOP because the
    context is poisoned (the summary then names the key and the reason)."""
    summary["failed"] += 1
    if sticky_cuda_error(exc):
        summary["aborted"] = {"key": key_text, "reason": str(exc)[:300]}
        log(f"[certify] ABORTED at {key_text}: the CUDA context is poisoned ({str(exc)[:160]}) — "
            f"every launch after this would fail the same way; nothing after it is a measurement. "
            f"Re-run from this key once the cause is fixed.")
        return True
    return False


def certify(profile: str, vendor: Optional[str] = None, census_path: Optional[str] = None,
            out: Optional[str] = None, kernels: Optional[List[str]] = None, limit: Optional[int] = None,
            only_missing: bool = False, seed: int = 20260907, log=None,
            allow_off_protocol: bool = False, reprove_unclocked: bool = False,
            reprove_generator: bool = False) -> Dict[str, Any]:
    """Certify every census shape for `profile` on this machine; write the files."""
    if log is None:
        def log(*a):                      # a run of hours, read while it runs: never buffered
            print(*a, flush=True)
    # The entry condition, before anything is timed. It lives HERE and not in the
    # CLI because this is the narrowest point: the door then holds for every
    # caller, not only for the one that types the documented command.
    rig_protocol_refusal(allow_off_protocol=allow_off_protocol, say=log)
    active = C.active_profile()
    if active is None:
        raise RuntimeError("no vendor profile is in force on this machine (the launcher resolved none)")
    if vendor is None:
        vendor = active[0]
    if (vendor, profile) != active:
        raise RuntimeError(f"this machine carries the profile {active[0]}/{active[1]}, not {vendor}/{profile}: a "
                           f"certification is measured on the profile it names")
    root = Path(out) if out else C.directory()
    hw_id = _bind_hardware_profile()
    log(f"[certify] hardware profile {hw_id}: has_native_bf16={_has_native_bf16()} — the wrappers' dtype policy in force")
    from neurobrix.triton import autotune_cache as atc
    tuners = {qual: t for qual, t in atc._autotuners()}
    # The card this run certifies on, read once (register 56): every proof
    # names it, `--only-missing` asks per its memory class, and a card the
    # profile in force does not describe is refused at entry — an entry made
    # on it could never say its class and would be served to no card.
    certifying_device = _certifying_device()
    certifying_class = C.memory_class_gb((certifying_device or {}).get("memory_mb"))
    if certifying_class is None:
        raise RuntimeError("the certifying card is not described by the hardware profile in force "
                           f"(device {certifying_device}): refused — a proof must say which card's memory it was made on")
    log(f"[certify] certifying on {certifying_device['name']} ordinal {certifying_device['ordinal']} "
        f"(CUDA_VISIBLE_DEVICES={certifying_device['visible_devices']}), "
        f"{certifying_device['memory_mb']} MB = memory class {certifying_class} GB; entries serve that class only")
    shapes = census(census_path)
    if kernels:
        want = set(kernels)
        shapes = {q: ks for q, ks in shapes.items() if q in want or C.kernel_short(q) in want}
    rng = np.random.default_rng(seed)
    summary: Dict[str, Any] = {"vendor": vendor, "profile": profile, "directory": str(root), "kernels": {},
                               "certified": 0, "skipped": 0, "failed": 0, "unreachable": 0,
                               "excluded_configs": 0, "started": time.time()}
    done = 0
    attempts = 0                                  # `limit` bounds the shapes TRIED, failures included
    for qual, keys in shapes.items():
        tuner = tuners.get(qual)
        if tuner is None:
            log(f"[certify] {qual}: not an autotuner in this engine — skipped ({len(keys)} shapes)")
            summary["skipped"] += len(keys)
            continue
        per_dtype: Dict[str, Dict[str, Dict]] = {}
        for key in keys:
            if limit is not None and attempts >= limit:
                break
            dtype = C.output_dtype(tuner, key)
            path = C.file_for(vendor, profile, qual, dtype, root=root)
            entries = per_dtype.setdefault(dtype, _read_file(path))
            ktext = C.key_repr(key)
            if (only_missing or reprove_unclocked or reprove_generator) and C.entry_covers(
                    entries, ktext, certifying_class, need_clock=reprove_unclocked,
                    need_generator=C.proof_backend({"backend": _backend()}) if reprove_generator else None):
                continue                      # certified FOR THIS CARD's memory class already (and, with
                                              # --reprove-unclocked, at a recorded clock; with
                                              # --reprove-generator, under the running code generator)
            attempts += 1
            t0 = time.time()
            try:
                tol = _tolerance(vendor, profile, dtype)
                entry = certify_key(qual, tuner, key, tol, rng)
            except UnreachableCensusKey as exc:
                # Known debt, not a break: no run will ever present this key
                # again. Counted apart so the exit code can still mean something.
                summary["unreachable"] += 1
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: UNREACHABLE — {exc}")
                continue
            except Exception as exc:
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: FAILED — {exc}")
                if after_key_failure(exc, summary, ktext, log):
                    summary["seconds"] = round(time.time() - summary["started"], 1)
                    return summary                    # a poisoned context: stop, say it, exit non-zero
                continue
            try:
                C.file_certification(entries, ktext, entry)   # by the class its proof names; refused without one
            except ValueError as exc:
                summary["failed"] += 1
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: REFUSED — {exc}")
                continue
            _write_file(path, vendor, profile, qual, dtype, entries)
            done += 1
            summary["certified"] += 1
            summary["excluded_configs"] += len(entry["excluded"])
            k = summary["kernels"].setdefault(qual, {"certified": 0, "excluded_configs": 0})
            k["certified"] += 1; k["excluded_configs"] += len(entry["excluded"])
            p = entry["proof"]
            log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: {entry['config']['kwargs']} "
                f"warps={entry['config']['num_warps']} stages={entry['config']['num_stages']} — deviation {p['deviation']:.2e} "
                f"(tol {tol:g}), {p['best_ms']:.4f} ms, {p['accepted']}/{p['candidates']} accepted, "
                f"{len(entry['excluded'])} excluded, {time.time() - t0:.1f} s "
                f"(oracle {p['seconds']['oracle']}, runs {p['seconds']['runs']}, bench {p['seconds']['bench']})")
    summary["seconds"] = round(time.time() - summary["started"], 1)
    return summary
