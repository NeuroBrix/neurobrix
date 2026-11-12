"""What would have made it fit.

Addition 4 of `docs/reference/adaptive-memory-a-runtime-controller.md`: a refusal
that only says "out of memory" costs the reader the whole diagnosis. The allocator
knows what it asked for and what the driver had; between those two numbers there
is a concrete answer — how many pieces this work would have to be cut into, and
how much larger a card would have taken it whole.

This module says only what the numbers support. It does not guess at a remedy it
cannot compute: with no shape it talks about bands and card size, and it mentions
a spatial extent only when it is given the shape to derive one from.

It lives under `kernels/` because both execution modes raise the same
`DeviceOOMError` and both need the same sentence — and because `kernels/` is
already free of torch, so the triton branch can import it without an R33
question.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

_MB = 1024 * 1024


def _mb(n: int) -> str:
    return f"{n / _MB:,.0f} MB"


def what_would_have_fit(oom, shape: Optional[Sequence] = None) -> str:
    """One sentence naming what would have made the refused allocation fit.

    `oom` is a `DeviceOOMError` carrying `requested` and `driver_free`. Returns
    an empty string when it carries neither — a sentence built on a figure the
    driver never gave would be a guess wearing the clothes of a measurement.

    `shape` is the allocation's shape when the caller knows it. Its leading
    dimension is the one a band split cuts, so it is the only one this reports
    on; nothing here assumes NCHW or any other layout beyond "the first axis is
    divisible", which is what band streaming actually needs.
    """
    requested = getattr(oom, "requested", None)
    free = getattr(oom, "driver_free", None)
    if not requested or free is None:
        return ""

    parts = []
    shortfall = getattr(oom, "shortfall", None)
    if shortfall:
        parts.append(f"short by {_mb(shortfall)}")

    if free > 0:
        bands = math.ceil(requested / free)
        # `free` is what the driver has RIGHT NOW, with this run's own live set
        # already on the card, so a band of `requested / bands` is a size that
        # fits beside what is already there — not a hypothetical on an empty card.
        parts.append(
            f"it would fit in {bands} bands of about {_mb(requested / bands)}")
        if shape:
            lead = shape[0] if len(shape) else None
            if isinstance(lead, int) and lead >= bands > 1:
                parts.append(
                    f"cutting the leading axis of {tuple(shape)} from {lead} to "
                    f"{lead // bands} per band")
    else:
        parts.append("the card reports nothing free: no band split can help here")

    total = getattr(oom, "driver_total", None)
    if total and requested > total:
        parts.append(
            f"and no band split alone reaches it — the request of {_mb(requested)} "
            f"exceeds the card's entire {_mb(total)}")

    return "; ".join(parts)


def annotate(message: str, cause, shape: Optional[Sequence] = None) -> str:
    """Append the advice to an op-failure message when the cause is an OOM.

    Any other cause passes through untouched: this must not turn an unrelated
    failure into one that reads as a memory problem.
    """
    try:
        from nbxtensor.memory.allocator import DeviceOOMError
    except Exception:          # pragma: no cover - import cycle safety
        return message
    if not isinstance(cause, DeviceOOMError):
        return message
    advice = what_would_have_fit(cause, shape)
    return f"{message} | what would have fit: {advice}" if advice else message


def _describe(r) -> str:
    """One referrer, named by what it is: a frame or a function by its code's place, a module by its name, a container
    by its length, anything else by its class."""
    import os
    kind = type(r).__name__
    if kind == "frame":
        return f"frame@{r.f_code.co_name}:{os.path.basename(r.f_code.co_filename)}:{r.f_lineno}"
    if kind == "function":
        return f"fn:{r.__name__}@{os.path.basename(r.__code__.co_filename)}:{r.__code__.co_firstlineno}"
    if kind == "module":
        return f"module:{r.__name__}"
    if isinstance(r, (list, tuple, dict, set, frozenset)):
        return f"{kind}[len={len(r)}]"
    return f"{type(r).__module__}.{kind}"


def _holders(obj, depth: int, ignore: set) -> str:
    """What refers to `obj` (at most five referrers), each with its own referrers `depth - 1` levels up; the lists
    and frames of this walk itself left out."""
    import gc
    refs = gc.get_referrers(obj)
    ignore = ignore | {id(refs)}
    own = (_holders.__code__, held_memory_report.__code__)
    parts = []
    for r in refs:
        if id(r) in ignore or (type(r).__name__ == "frame" and r.f_code in own):
            continue
        s = _describe(r)
        if depth > 1:
            s += f" <- ({_holders(r, depth - 1, ignore | {id(obj)})})"
        parts.append(s)
        if len(parts) == 5:
            break
    return ", ".join(parts)


def _block_lines(bucket: int) -> List[str]:
    """Every device's blocks in use: the total, and the sizes of at least `bucket` bytes by their totals."""
    from collections import Counter
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    out = []
    for d in range(DA.device_count()):
        blocks = DA.blocks_in_use(d)
        if blocks:
            sizes = Counter(n // _MB * _MB for n in blocks.values() if n >= bucket)
            top = sorted(sizes.items(), key=lambda kv: -kv[0] * kv[1])[:8]
            out.append(f"[LIVE_BLOCKS dev={d}] total={sum(blocks.values()) / _MB:.0f}MB big_blocks(>={bucket / _MB:.0f}MB): "
                       + " ".join(f"{c}x{s / _MB:.0f}MB" for s, c in top))
    return out


def _autotuner_lines() -> List[str]:
    """The Triton autotuners still holding tensors among their arguments (kept when a benchmark fails)."""
    import gc
    from nbxtensor.tensor import NBXTensor
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return []
    tuners = [o for o in gc.get_objects() if isinstance(o, Autotuner)]
    leaks = [(getattr(getattr(a, "base_fn", None), "__name__", "?"), len(a.nargs),
              sum(isinstance(v, NBXTensor) for v in a.nargs.values()))
             for a in tuners if isinstance(getattr(a, "nargs", None), dict)]
    leaks = [x for x in leaks if x[2]]
    return [f"[AUTOTUNE_LEAK_SCAN] {len(tuners)} Autotuner instances; {len(leaks)} with nargs holding NBXTensors"
            + "".join(f"; kernel={k} nargs={n} tensors={c}" for k, n, c in leaks)]


def held_memory_report(named: Iterable = (), big: int = 1 << 30, bucket: int = 100 * _MB) -> List[str]:
    """Who holds the device memory, for a forensic dump after a refused allocation, as log lines: every device's
    blocks in use (the sizes of at least `bucket` bytes, the largest totals first); whether a cycle collection frees
    blocks of at least `big` bytes; every tensor alive on such a block, named from `named` ((name, tensor) pairs: the
    application's own tensors) or ORPHAN, with what refers to it (three levels up for an orphan); the Triton
    autotuners still holding tensors among their arguments. Read-only but for one `gc.collect()`."""
    import gc
    import sys
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    from nbxtensor.tensor import NBXTensor
    out = _block_lines(bucket)
    before = sum(1 for n in DA.blocks_in_use().values() if n >= big)
    collected = gc.collect()
    big_blocks = {p for p, n in DA.blocks_in_use().items() if n >= big}
    out.append(f"[GC_COLLECT_TEST] gc.collect() collected {collected} cycle objects, big_blocks(>={big / _MB:.0f}MB) "
               f"before={before} after={len(big_blocks)}")
    names = {t.storage_address(): name for name, t in named if isinstance(t, NBXTensor)}
    tensors = [o for o in gc.get_objects() if isinstance(o, NBXTensor) and o.storage_address() in big_blocks]
    out.append(f"[BIG_TENSORS] {len(tensors)} NBXTensors on big blocks (>={big / _MB:.0f}MB)")
    for i in range(len(tensors)):                       # no (i, t) tuple of its own among the referrers
        t = tensors[i]
        name = names.get(t.storage_address(), "ORPHAN(not in arena)")
        out.append(f"  tensor#{i} tid={name} shape={tuple(t.shape)} nbytes={t.nbytes() / _MB:.0f}MB "
                   f"address={t.storage_address()} refcount={sys.getrefcount(t)} gc_tracked={gc.is_tracked(t)} "
                   f"referrers: {_holders(t, 3 if name.startswith('ORPHAN') else 1, {id(tensors)})}")
    return out + _autotuner_lines()
