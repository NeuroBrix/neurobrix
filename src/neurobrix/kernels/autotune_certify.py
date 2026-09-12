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


def _arr(rng, shape, dtype_name, scale=0.1):
    a = (rng.standard_normal(shape) * scale)
    return a.astype(_NP.get(dtype_name, np.float32))


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
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    dts = C.key_dtypes(key)
    short = C.kernel_short(qual)
    to = lambda a: NBXTensor.from_numpy(np.ascontiguousarray(a))   # noqa: E731
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
    info["clocks_mhz"] = _clocks_mhz()
    return info


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


_UNREAD = object()


def _read_clocks_mhz():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.applications.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        out = {}
        for line in r.stdout.splitlines():
            if line.strip():
                idx, mhz = [c.strip() for c in line.split(",")]
                out[idx] = int(mhz)
        return out or None
    except Exception:
        return None


_clocks_mhz.cached = _UNREAD


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
            raise RuntimeError(f"the wrapper computed key {seen!r} for inputs synthesized from {key!r}: the census "
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
                dev = oracle_deviation(out_tensor.numpy(), oracle)
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
        for cfg, dev, _run_s in contenders:
            ms = bench(lambda: tuner.fn.run(*args, **{**kwargs, **cfg.all_kwargs()}))
            timed.append((cfg, dev, float(ms)))
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
    try:
        if _nbx is not None:
            W.set_compute_dtype(_nbx)
        call()
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
             "seconds": {"oracle": state.get("t_oracle"), "runs": state.get("t_runs"), "bench": state.get("t_bench")}}
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
    doc = {"format": C.FORMAT, "vendor": vendor, "profile": profile, "kernel": qual, "dtype": dtype,
           "entries": dict(sorted(entries.items()))}
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


def certify(profile: str, vendor: Optional[str] = None, census_path: Optional[str] = None,
            out: Optional[str] = None, kernels: Optional[List[str]] = None, limit: Optional[int] = None,
            only_missing: bool = False, seed: int = 20260907, log=None) -> Dict[str, Any]:
    """Certify every census shape for `profile` on this machine; write the files."""
    if log is None:
        def log(*a):                      # a run of hours, read while it runs: never buffered
            print(*a, flush=True)
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
    shapes = census(census_path)
    if kernels:
        want = set(kernels)
        shapes = {q: ks for q, ks in shapes.items() if q in want or C.kernel_short(q) in want}
    rng = np.random.default_rng(seed)
    summary: Dict[str, Any] = {"vendor": vendor, "profile": profile, "directory": str(root), "kernels": {},
                               "certified": 0, "skipped": 0, "failed": 0, "excluded_configs": 0, "started": time.time()}
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
            if only_missing and ktext in entries:
                continue
            attempts += 1
            t0 = time.time()
            try:
                tol = _tolerance(vendor, profile, dtype)
                entry = certify_key(qual, tuner, key, tol, rng)
            except Exception as exc:
                summary["failed"] += 1
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: FAILED — {exc}")
                continue
            entries[ktext] = {"config": entry["config"], "proof": entry["proof"], "excluded": entry["excluded"]}
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
