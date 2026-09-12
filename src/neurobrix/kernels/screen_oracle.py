"""The fp64 oracle the autotune screen consults, computed from the LIVE args.

The screen's consensus asks the candidates whether they agree with each other.
That question has one blind spot, and it is not academic: when the defect is in
the BACKEND'S EMISSION rather than in the choice of tile, every candidate is
wrong in the same way, the vote is unanimous, and the screen seats a wrong
configuration in silence.

Measured 2026-09-11 on four `addmm` shapes the catalogue demands: the emitted
MSL declared the float scalars `alpha`/`beta` as `int`, so the bit pattern of
`1.0f` multiplied the accumulator — wrong by a factor of 1065353216. Eight
runs, four shapes, two arms: the bare vote SEATED a configuration every time;
the same screen with this oracle REFUSED every time.

## Where the cost is paid, and why it is affordable

A CERTIFIED entry is served without a sweep, without a screen and without an
oracle — zero cost. Only an UNCERTIFIED key sweeps, and only then is the screen
run and the oracle computed, once for the key and not once per candidate.

**The cost is therefore paid exactly where the risk lives, and never
elsewhere** — and the certified directory gains a second reason to exist that
was not visible before: it does not only remove the sweep, it removes the
oracle's cost with it.

## What this covers, and what it says when it does not

An oracle exists for the kernels whose mathematics can be recomputed in float64
on the host from the live operands. For anything else this returns None, and
the screen falls back to the consensus — LOUDLY, once per process, naming the
key, because a screen running without an oracle is exactly the situation the
`addmm` measurement showed to be dangerous.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

#: Keys already announced as running without an oracle. Once per process: a
#: warning repeated per launch is a warning nobody reads.
_ANNOUNCED: set = set()

_NP = {"fp16": np.float16, "float16": np.float16,
       "fp32": np.float32, "float32": np.float32,
       "fp64": np.float64, "float64": np.float64}


def _to_f64(t) -> Optional[np.ndarray]:
    """A live operand as float64, or None when it cannot be read here."""
    try:
        from neurobrix.kernels.nbx_tensor import NBXDtype
        # ONE path to host memory, for every dtype. The bf16 case used to read
        # the device pointer directly with `ctypes.string_at`, which crosses NO
        # BARRIER -- so a buffer a kernel had just written was read before the
        # write landed. Measured 2026-09-12 on a kernel-written buffer: 4093 of
        # 4096 elements differed from the copy path, and agreed after an
        # explicit synchronisation.
        #
        # That produced the symptom it is easiest to misread: every candidate
        # compared against the same stale bytes, so the fp64 oracle contradicted
        # all ten and the screen refused the whole space. A reference read at
        # the wrong moment contradicts every correct answer exactly as loudly
        # as a wrong one. Eighteen recorded refusals rested on it.
        #
        # The defect had the shape of a PATH, not of a dtype: bf16 was merely
        # the only traffic on it. So the raw path is gone rather than repaired,
        # and a dtype routed here later cannot inherit it. Removing it also
        # removes the second failure mode `string_at` carries -- it hands its
        # size to `PyBytes_FromStringAndSize` as a C int, so any buffer of
        # 2 GiB or more returns "Negative size", confirmed live on this machine.
        host = t.to_cpu() if getattr(t, "_device", "cpu") != "cpu" else t
        if getattr(t, "_dtype", None) == NBXDtype.bfloat16:
            # numpy has no bf16; the bits are the top half of an fp32 and
            # travel in a uint16 container.
            bits = np.asarray(host.numpy()).view(np.uint16).astype(np.uint32)
            return ((bits << np.uint32(16)).view(np.float32)
                    .reshape(tuple(t.shape)).astype(np.float64))
        return np.asarray(host.numpy(), dtype=np.float64)
    except Exception:                                  # noqa: BLE001
        return None


def _mm(named: Dict[str, Any]) -> Optional[np.ndarray]:
    a, b = _to_f64(named.get("a_ptr")), _to_f64(named.get("b_ptr"))
    if a is None or b is None:
        return None
    out = a @ b
    bias = named.get("bias_ptr")
    if bias is not None:
        bv = _to_f64(bias)
        if bv is None:
            return None
        alpha = float(named.get("alpha", 1.0) or 1.0)
        beta = float(named.get("beta", 1.0) or 1.0)
        out = alpha * out + beta * (bv[None, :] if bv.ndim == 1 else bv)
    return out


def _baddbmm(named: Dict[str, Any]) -> Optional[np.ndarray]:
    a, b = _to_f64(named.get("A_ptr")), _to_f64(named.get("B_ptr"))
    if a is None or b is None:
        return None
    out = a @ b
    alpha = float(named.get("alpha", 1.0) or 1.0)
    beta = float(named.get("beta", 1.0) or 1.0)
    out = alpha * out
    if named.get("HAS_BIAS") and named.get("bias_ptr") is not None:
        bv = _to_f64(named["bias_ptr"])
        if bv is None:
            return None
        out = out + beta * bv
    return out


#: kernel short name -> (reference for its OUTPUT from live args, output arg).
#: A kernel absent from here has no oracle and the screen says so out loud.
#:
#: The OUTPUT ARGUMENT NAME is not decoration. The first version of this
#: provider matched the output buffer by BYTE LENGTH, and for a matmul at
#: (19, 2048, 2048) the operand `a` and the output `c` are both 19x2048 fp32 —
#: the same 155648 bytes. The oracle was therefore placed on `a`, an INPUT, so
#: every candidate "disagreed" with it and the screen refused all ten
#: candidates of a shape whose certified deviation is 1.7e-06.
#:
#: An oracle that refuses correct configurations is worse than no oracle: it
#: turns a silent wrong into a loud stop on healthy work, and it would have
#: been believed because a refusal reads as vigilance. Matched by ADDRESS now.
ORACLES = {
    "matmul_kernel": (_mm, "c_ptr"),
    "addmm_kernel": (_mm, "c_ptr"),
    "baddbmm_kernel": (_baddbmm, "out_ptr"),
}


def provider(tuner, key, buffers) -> Optional[List[bytes]]:
    """The correct contents of every screened buffer, or None.

    The screen snapshots EVERY writable buffer, not just the output — four of
    them for `addmm` (`a`, `b`, `bias`, `c`). The correct content of an INPUT
    after the kernel is its content unchanged; only the output has an oracle.
    Returning the oracle for all of them compares an input to an output.
    """

    name = getattr(getattr(tuner, "base_fn", None), "__name__", "") or ""
    entry = ORACLES.get(name)
    if entry is None:
        announce_no_oracle(name or str(tuner), key)
        return None
    fn, out_name = entry
    named = dict(getattr(tuner, "nargs", None) or {})
    if not named:
        return None
    out_tensor = named.get(out_name)
    out_addr = int(out_tensor.data_ptr()) if out_tensor is not None else None
    if out_addr is None:
        announce_no_oracle(name, key,
                           why=f"its output argument {out_name!r} is not among "
                               f"the live arguments")
        return None
    try:
        import time as _time
        _t0 = _time.perf_counter()
        reference = fn(named)
        # The oracle's own CPU wall time is the one number measured for this
        # key BEFORE any candidate runs, so it is the bootstrap base for the
        # first candidate's time budget: a GPU configuration slower than the
        # single-threaded float64 CPU computation of the same mathematics is
        # not a configuration, whatever else it is.
        record_oracle_ms(key, (_time.perf_counter() - _t0) * 1e3)
    except Exception:                                  # noqa: BLE001
        reference = None
    if reference is None:
        announce_no_oracle(name, key, why="its operands could not be read here")
        return None

    # Every live tensor by its address, so an INPUT is read through the copy
    # path like everything else. The line this replaces read the device
    # pointer with `ctypes.string_at` and justified it as "an input,
    # unchanged" -- an assumption this repository's own measurement destroyed
    # the same day: the bf16 operand of eighteen refusals was the BIAS, an
    # input produced by a conversion kernel earlier in the graph. An input can
    # be exactly as stale as an output, and "unchanged" says nothing about
    # WHEN it was written.
    #
    # The size failure mode does not even need that argument: `string_at`
    # hands its length to `PyBytes_FromStringAndSize` as a C int, so any
    # buffer of 2 GiB or more returns "Negative size" whoever wrote it --
    # confirmed live here at 2 GiB + 1 -- and a matmul input passes 2 GiB
    # without difficulty on the shapes that matter.
    by_addr = {}
    for _v in named.values():
        try:
            by_addr[int(_v.data_ptr())] = _v
        except Exception:                              # noqa: BLE001
            continue

    out: List[bytes] = []
    for addr, nbytes, dtype_name in buffers:
        if int(addr) == out_addr:                      # the output, BY ADDRESS
            want = np.ascontiguousarray(
                reference.astype(_NP.get(dtype_name, np.float32)))
            if want.nbytes != int(nbytes):
                announce_no_oracle(
                    name, key,
                    why=f"the reference is {want.nbytes} bytes and the output "
                        f"buffer is {nbytes}")
                return None
            out.append(want.tobytes())
        else:
            # An input, unchanged -- read through the tensor's own copy to the
            # host, which crosses the barrier a raw pointer read does not.
            src = by_addr.get(int(addr))
            if src is None:
                announce_no_oracle(
                    name, key,
                    why=f"an input buffer at {addr:#x} is not among the live "
                        f"arguments, so it can only be read by raw pointer, "
                        f"which crosses no barrier and caps at 2 GiB")
                return None
            host = src.to_cpu() if getattr(src, "_device", "cpu") != "cpu" else src
            raw = np.ascontiguousarray(host.numpy()).tobytes()
            if len(raw) != int(nbytes):
                announce_no_oracle(
                    name, key,
                    why=f"the input at {addr:#x} copies to {len(raw)} bytes "
                        f"and the buffer is {nbytes}")
                return None
            out.append(raw)
    return out


#: Oracle CPU wall time by key text, for the sweep's time budget. In-process,
#: like `_SCREEN_CACHE`: the budget protects a sweep that runs in this very
#: process, so persisting these would only let a stale machine state speak.
_ORACLE_MS: dict = {}


def record_oracle_ms(key, ms: float) -> None:
    _ORACLE_MS[str(key)] = float(ms)


def oracle_ms_of(key):
    """The oracle's measured CPU time for this key, or None if it never ran."""
    return _ORACLE_MS.get(str(key))


def announce_no_oracle(kernel_name: str, key, why: str = "no oracle is registered "
                                                        "for this kernel") -> None:
    """Say, once per process, that a screen ran on a consensus alone.

    A mechanism that is complete and switched off is the most expensive form of
    a vacuous guard: it costs the price of writing it and returns nothing. The
    only thing worse is one that is silent about being off.
    """
    token = (kernel_name, str(key))
    if token in _ANNOUNCED:
        return
    _ANNOUNCED.add(token)
    print(f"[AUTOTUNE_ORACLE] {kernel_name}: screening at key {key} on the "
          f"CONSENSUS ALONE — {why}. A consensus cannot see a defect that is "
          f"in the backend's emission, because every candidate then shares it "
          f"and the vote is unanimous (measured: addmm wrong by 1e9 with a "
          f"unanimous vote, 2026-09-11).", flush=True)


def install() -> None:
    """Make the screen consult this oracle."""
    from neurobrix.kernels import launcher as L
    L.set_screen_oracle(provider)
