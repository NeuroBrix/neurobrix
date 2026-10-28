"""What an engine's runtime grows to on a card during a model's run, outside every allocator — a measured fact per model.

The runtime holds device memory no allocator counts: its context, read when a plan starts, and what it adds while the run
loads the kernels it needs (their modules and the local memory the driver reserves for them). That second part depends
on the kernels the model's graph runs — 67 MB for HAT-L x4, 911 MB for Wan2.2-I2V-A14B on the rack's V100-SXM2
(2026-10-28, NBX_LIVE_WATERMARK_TRACE) — and a plan cut to the card less the context alone runs out of memory by it.

So the engine measures it on every run it can (Triton on an accelerator: driver used less the allocator's own live bytes,
its pool included, at its height over the run — after each sequence's first pass — and at its end, less what other
processes hold, less the same reading before the executor exists; the driver's tool only at the two ends) and records
the largest seen, per engine and architecture, in the engine store beside the calibration records. Prism reads it for the
model it plans (`PrismSolver._runtime_growth`); a model never measured is planned against the vendor file's measured
envelope, and a device with neither keeps the whole-component fraction (`PrismSolver._part_of_rung_mb`).
stdlib + the engine's own allocator: no torch (R33).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def _root() -> Path:
    from neurobrix.core.dtype.calibration import STORE_ROOT
    return Path(STORE_ROOT)


def path(model: str) -> Path:
    return _root() / model / "runtime_growth.json"


def load(model: Optional[str], engine: str, architecture: str) -> Optional[float]:
    """The recorded growth in MB of `model` under `engine` on `architecture`, or None (never measured)."""
    if not model:
        return None
    p = path(model)
    if not p.is_file():
        return None
    value = ((json.loads(p.read_text()) or {}).get(engine) or {}).get(str(architecture))
    return None if value is None else float(value)


def untracked_raw_mb(device: int) -> float:
    """MB of `device` the driver says used (total less free) that the Triton allocator does not hold live, its pool
    included: one driver call (`mem_get_info`), no tool — cheap enough for the run path (`note`)."""
    from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
    free_b, total_b = DA.mem_get_info(int(device))
    return (int(total_b) - int(free_b) - DA._cuda_live_bytes.get(int(device), 0)) / (1024 * 1024)


def _held_by_others_mb(device: int) -> Optional[float]:
    """What OTHER processes hold of `device` (`autodetect.device_sharing`, the driver's tool), or None when unreadable:
    another run on the card is not this runtime's growth. Read at the run's two ends only, never on its path."""
    from neurobrix.core.prism.autodetect import device_sharing
    facts = device_sharing(int(device))
    if not facts or facts.get("held_by_others_mb") is None:
        return None
    return float(facts["held_by_others_mb"])


#: {device index: the most MB this process held outside the allocator during its run, raw}: noted after the FIRST pass
#: of every Triton sequence (`note`, where its kernels load), so a streamed run's growth is read at its height, every
#: piece seen. {device index: what other processes held there when the run began}. Both reset by `begin`.
_HIGH: Dict[int, float] = {}
_OTHERS: Dict[int, float] = {}


def begin(devices) -> Dict[int, float]:
    """The run's starting reading, before the executor exists: {device: raw MB less what other processes hold} for every
    device the tool can read (one it cannot is left out, and nothing is recorded for it). Resets the run's height. Not
    the process's own per-pid figure: the driver reserves for this run memory no pid is charged with (Wan2.2-I2V-A14B,
    2026-10-28: 128 MB per pid, ~1 GB lost to the card, its pieces out of memory by it)."""
    _HIGH.clear()
    _OTHERS.clear()
    out = {}
    for d in devices:
        others = _held_by_others_mb(int(d))
        if others is None:
            continue
        _OTHERS[int(d)] = others
        out[int(d)] = untracked_raw_mb(int(d)) - others
    return out


def note(device: int) -> None:
    """One reading of `device` folded into the run's height (`_HIGH`): a driver call, nothing else."""
    if int(device) not in _OTHERS:
        return
    mb = untracked_raw_mb(int(device))
    if mb > _HIGH.get(int(device), float("-inf")):
        _HIGH[int(device)] = mb


def record(model: Optional[str], engine: str, architecture: str, before: Dict[int, float]) -> Optional[float]:
    """Reads the run's end, then writes max(recorded, the largest growth of any device this run — its height over the
    run, `note`, or the end reading — less what other processes hold there, the larger of the run's two ends) and
    returns the run's growth in MB."""
    if not model or not before:
        return None
    grown = None
    for d, b in before.items():
        others_now = _held_by_others_mb(d)
        others = max(_OTHERS.get(d, 0.0), others_now if others_now is not None else 0.0)
        raw = max(untracked_raw_mb(d), _HIGH.get(d, float("-inf")))
        g = raw - others - b
        grown = g if grown is None else max(grown, g)
    if grown is None:
        return None
    grown = max(0.0, round(grown, 1))
    p = path(model)
    data = json.loads(p.read_text()) if p.is_file() else {}
    per = data.setdefault(engine, {})
    prior = per.get(str(architecture))
    if prior is None or grown > float(prior):
        per[str(architecture)] = grown
        data["source"] = ("the engine's own reading over a run (after each sequence's first pass and at its end): driver "
                          "used less its allocator's live bytes, less what other processes hold, less the same reading "
                          "before the executor (core/runtime/runtime_growth.py)")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(p)
    return grown
