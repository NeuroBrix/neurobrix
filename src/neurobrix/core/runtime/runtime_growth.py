"""What an engine's runtime grows to on a card during a model's run, outside every allocator — a measured fact per model.

The runtime holds device memory no allocator counts: its context, read when a plan starts, and what it adds while the run
loads the kernels it needs (their modules and the local memory the driver reserves for them). That second part depends
on the kernels the model's graph runs — 67 MB for HAT-L x4, 911 MB for Wan2.2-I2V-A14B on the rack's V100-SXM2
(2026-10-28, NBX_LIVE_WATERMARK_TRACE) — and a plan cut to the card less the context alone runs out of memory by it.

So the engine measures it on every run it can (Triton on an accelerator: driver used less the allocator's own live bytes,
its pool included, at the run's end less the same reading before the executor exists) and records the largest seen, per
engine and architecture, in the engine store beside the calibration records. Prism reads it for the model it plans
(`PrismSolver._runtime_growth`); a model never measured is planned against the vendor file's measured envelope.

A record carries the method that measured it (`METHOD`), and only a record of this method is read: a figure written by
another measuring code, or by none, is not this measurement. Measured 2026-10-29: the store held 0.0 MB for 35 of its
48 models — written by the no-card census shadow (the coverage check), whose allocator answers no device — and 202 MB
for Wan2.2-I2V-A14B, written by a branch's different reading; this code measures 872 MB there (the envelope: 911). Its
16 GB plan, cut to 202 MB, ran out of memory at an attention. A census shadow records nothing.
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


#: the measuring method a record states, and the only one `load` reads
METHOD = "triton-driver-used-less-allocator-live, run end less before the executor, on real devices (v2, 2026-10-29)"


def load(model: Optional[str], engine: str, architecture: str) -> Optional[float]:
    """The recorded growth in MB of `model` under `engine` on `architecture`, or None (never measured)."""
    if not model:
        return None
    p = path(model)
    if not p.is_file():
        return None
    value = ((json.loads(p.read_text()) or {}).get(engine) or {}).get(str(architecture))
    if not isinstance(value, dict) or value.get("method") != METHOD:
        return None              # another method's figure, or one that states none: not this measurement
    return float(value["mb"])


def untracked_mb(devices) -> Dict[int, float]:
    """{device index: MB the driver reports used that the Triton allocator does not hold} for `devices`."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    out = {}
    for d in devices:
        free_b, total_b = DA.mem_get_info(d)
        out[int(d)] = (int(total_b) - int(free_b) - DA._cuda_live_bytes.get(int(d), 0)) / (1024 * 1024)
    return out


def record(model: Optional[str], engine: str, architecture: str, before: Dict[int, float],
           after: Dict[int, float]) -> Optional[float]:
    """Writes max(recorded by this method, the largest growth of any device this run) and returns the run's growth in
    MB. Under the census shadow nothing is measured and nothing is written: its devices are no device."""
    if not model or not before or not after:
        return None
    from neurobrix.kernels import census
    if census.active():
        return None
    grown = max(after[d] - before[d] for d in after if d in before)
    grown = max(0.0, round(grown, 1))
    p = path(model)
    data = json.loads(p.read_text()) if p.is_file() else {}
    per = data.setdefault(engine, {})
    prior = per.get(str(architecture))
    prior_mb = (float(prior["mb"]) if isinstance(prior, dict) and prior.get("method") == METHOD else None)
    if prior_mb is None or grown > prior_mb:
        per[str(architecture)] = {"mb": grown, "method": METHOD}
        data["source"] = ("the engine's own reading at the end of a run, driver used less its allocator's live bytes, "
                          "less the same reading before the executor (core/runtime/runtime_growth.py)")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(p)
    return grown
