"""The Triton arena's factor, observed on the user's own runs (the owner via the supervisor, 2026-11-07 17:05, voie B):
the first run of each component a Triton plan runs WHOLE is watched by one allocator window — its hold (the window's
peak minus its start) over the walk Prism priced for it at that request (`ComponentAllocation.priced_walk_bytes`, the
KV cache apart). Nothing is run in addition and no shape is formed that the run does not form, so no kernel is ever
tuned for it (a separate probe on traced extents measured 1.4 to 28 and swept uncertified kernels for 40 minutes).

The observations are kept in the generated profile of the machine (`measured.arena_observations`, the last ones); when
three of them agree within the storage measure's steadiness, their median is written as
`measured.triton_arena_activation_factor_observed`, which Prism reads after the device's own vendor file and before its
family (`core/config/loader.declared_arena_factor`). Until then the family's declared value is used. Only a generated
profile (`default...`) is written; a profile named by --hardware is the user's and is left as it is."""
from __future__ import annotations

from typing import Any, Optional

#: the observations kept per profile (the last ones)
KEEP = 9
#: the agreeing observations a factor is written from
AGREE = 3


def _wanted(walk: Optional[int]) -> bool:
    if not walk or walk <= 0:
        return False
    from neurobrix.core.prism import autodetect as AD
    hw = getattr(AD, "ACTIVE_HARDWARE_ID", None)
    return bool(hw) and str(hw).startswith("default") and observed_factor(hw) is None


def open_first(executor: Any, walk: Optional[int], device_idx: int):
    """An allocator window around this executor's FIRST run, when the walk is priced and the profile has no stable
    observed factor yet; None otherwise (every later run costs one attribute read)."""
    if getattr(executor, "_arena_observed", False):
        return None
    executor._arena_observed = True
    if not _wanted(walk):
        return None
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.device_synchronize(device_idx)
    return DA.open_window(device_idx)


def close_first(executor: Any, token, walk: Optional[int], device_idx: int) -> None:
    if token is None:
        return
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.device_synchronize(device_idx)
    start, peak = DA.window_start(token), DA.close_window(device_idx, token)
    hold = int(peak) - int(start)
    if hold > 0:
        record(getattr(executor, "_component_name", None) or "?", hold, int(walk))


def observed_factor(hardware_id: str) -> Optional[float]:
    """The profile's stable observed factor, or None."""
    data = _profile(hardware_id)
    dev = ((data or {}).get("devices") or [{}])[0]
    v = (dev.get("measured") or {}).get("triton_arena_activation_factor_observed")
    return float(v) if v is not None else None


def agreed(ratios, steady: float) -> Optional[float]:
    """The median of the last `AGREE` ratios when they agree within `steady`, else None."""
    last = sorted(ratios[-AGREE:])
    if len(last) < AGREE or last[-1] > last[0] * (1 + steady):
        return None
    return last[len(last) // 2]


def record(component: str, hold: int, walk: int) -> None:
    """One observation into the generated profile; the factor written when the last `AGREE` agree."""
    import inspect
    from neurobrix.core.prism import autodetect as AD
    from nbxtensor.io import read_rate
    hw = getattr(AD, "ACTIVE_HARDWARE_ID", None)
    path = AD.HARDWARE_DIR / f"{hw}.yml"
    data = _profile(hw)
    if not data or not data.get("devices"):
        return
    m = data["devices"][0].setdefault("measured", {})
    obs = list(m.get("arena_observations") or [])
    obs.append({"component": component, "hold_bytes": hold, "walked_bytes": walk, "ratio": round(hold / walk, 4)})
    m["arena_observations"] = obs[-KEEP:]
    steady = inspect.signature(read_rate.measure).parameters["within"].default
    factor = agreed([o["ratio"] for o in m["arena_observations"]], steady)
    if factor is not None:
        m["triton_arena_activation_factor_observed"] = round(float(factor), 3)
        print(f"   [Auto-detect] triton_arena_activation_factor observed on this machine's own runs: {factor:.3f} "
              f"(the last {AGREE} agree) — written into {path.name}")
    AD._write_profile_atomically(path, data)


def _profile(hardware_id) -> Optional[dict]:
    import yaml
    from neurobrix.core.prism import autodetect as AD
    if not hardware_id:
        return None
    path = AD.HARDWARE_DIR / f"{hardware_id}.yml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or {}
