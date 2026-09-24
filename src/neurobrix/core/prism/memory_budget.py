"""The memory budget — one law for every device kind, every machine.

Hocine's memory doctrine (the owner, 2026-09-21), as it stands for a Mac, this rack, Windows, a
CPU-only host, all of them:

* On a SHARED device — unified memory, a GPU driving a display, or a card part of whose memory
  another process holds — the budget is the FREE reading rounded DOWN onto the commercial
  memory ladder, never a value off the ladder.
* On a DEDICATED compute card that nothing else uses, the card is used whole, less only the
  runtime's own context.
* On a dedicated card partly held by someone else, the remainder is rounded down onto the ladder
  (the second rule again: a card someone else holds is shared).
* The same law governs host RAM, which is always shared with the operating system.

The ladder is DATA (`PRISM_DEFAULTS["memory_ladder_gb"]`, 4 GB to 512 GB), never a literal
here. The dedicated-or-shared decision is read from the DEVICE — its display activity, the
memory other processes hold, whether its memory is unified — never from a brand or a model
name. A reading that cannot be taken is answered as SHARED: rounding the free figure down is
the safe side, and the plan says which side it took.

What this replaces (measured on main before this brick): a literal ladder that stopped at
128 GB; a rung function that passed any reading under 4 GB through untouched; a capacity door
that took the minimum of (capacity − max(3072 MB, 12 %)) and the rung of the free reading, so an
idle V100-16GB reading ~15.7 GB free was budgeted at 12 GB and an idle V100-32GB at 24 GB — a
quarter of each card never used, the margin stacked on a rounding that already was the
headroom; and host RAM budgeted as 0.7 × the installed figure, never through the ladder.

The tile a plan cuts is derived from the RUNG (Hocine's tiling standard): on a dedicated card
the card's own nominal rung, on a shared device the rung of the free reading — so the tile is a
pure function of (component, rung), the input size creates no extra shapes, and a census can
enumerate every rung up to a card's capacity.

Doors: `NBX_PRISM_BUDGET_MB=<mb>` makes every device's budget and tile rung that value (the
census enumerating rungs; a measurement), said in clear.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from neurobrix.core.config.system import PRISM_DEFAULTS


@dataclass(frozen=True)
class DeviceReading:
    """What the machine says about one memory pool, read once, at the plan's entry."""
    kind: str                       # "device" or "host"
    capacity_mb: float              # the pool's whole size (a card's total, a host's installed RAM)
    free_mb: float                  # the free reading at this moment
    unified: bool = False           # the pool is shared with the host (Apple, an APU)
    display_active: bool = False    # the device drives a display
    held_by_others_mb: float = 0.0  # memory other processes hold on this device
    own_context_mb: float = 0.0     # the runtime's own context on the device (a dedicated card's only deduction)
    measured: bool = True           # False when the sharing facts could not be read: treated as shared
    source: str = ""                # where the reading came from, for the plan's own words


def memory_ladder_mb() -> List[int]:
    """The commercial ladder in MB, ascending, from configuration — never a literal here."""
    rungs = PRISM_DEFAULTS.get("memory_ladder_gb")
    if not rungs:
        raise RuntimeError("ZERO HARDCODE: PRISM_DEFAULTS['memory_ladder_gb'] declares no memory ladder")
    # int(g) TRUNCATED every rung to a whole GB, which silently destroyed the spacing the
    # ladder is derived with: rungs 4.0, 4.102, 4.207, 4.314 ... all became 4, and the set
    # collapsed back to the integers 4, 5, 6, 7 — a linear 1 GB ladder whose first step is
    # 25 %, ten times the measured reading noise, precisely in the low range where the rungs
    # decide between streaming and refusing. The ladder is data; rounding it here re-picked it.
    out = sorted({int(round(float(g) * 1024)) for g in rungs})
    if out[0] <= 0:
        raise RuntimeError("the memory ladder's lowest rung must be positive")
    return out


def rung_down_mb(mb: float) -> int:
    """The largest rung ≤ `mb`; 0 when the reading is under the lowest rung (nothing to size
    against there — the caller refuses or streams in its own words). Never a value off the
    ladder: a reading under 4 GB used to pass through untouched, exactly where standard rungs
    matter most."""
    rung = 0
    for r in memory_ladder_mb():
        if r <= mb:
            rung = r
    return rung


def is_shared(r: DeviceReading) -> bool:
    """Shared: a host pool, unified memory, a display, another process's memory, or a device
    whose sharing could not be read."""
    return (r.kind == "host" or r.unified or r.display_active
            or r.held_by_others_mb > 0 or not r.measured)


def _door_mb() -> Optional[int]:
    v = os.environ.get("NBX_PRISM_BUDGET_MB")
    return int(v) if v else None


def budget_mb(r: DeviceReading) -> int:
    """What a plan may be budgeted against on this pool."""
    door = _door_mb()
    if door is not None:
        return door
    if is_shared(r):
        return rung_down_mb(r.free_mb)
    return int(max(0.0, r.capacity_mb - r.own_context_mb))


def tile_rung_mb(r: DeviceReading) -> int:
    """The rung a tiled component sizes its tile from: a dedicated card's own nominal rung
    (the card whole), a shared pool's free rung."""
    door = _door_mb()
    if door is not None:
        return door
    if is_shared(r):
        return rung_down_mb(r.free_mb)
    return rung_down_mb(r.capacity_mb)


def describe(r: DeviceReading) -> str:
    door = _door_mb()
    if door is not None:
        return f"budget {door} MB by the NBX_PRISM_BUDGET_MB door"
    if is_shared(r):
        why = ("host RAM" if r.kind == "host" else "unified memory" if r.unified
               else "a display on it" if r.display_active
               else f"{r.held_by_others_mb:.0f} MB held by other processes" if r.held_by_others_mb > 0
               else "sharing could not be read")
        return (f"shared ({why}): free {r.free_mb:.0f} MB rounds down to the {rung_down_mb(r.free_mb)} MB rung"
                + (f" [{r.source}]" if r.source else ""))
    return (f"dedicated: the card whole, {r.capacity_mb:.0f} MB less its own context {r.own_context_mb:.0f} MB "
            f"= {budget_mb(r)} MB; tile rung {tile_rung_mb(r)} MB" + (f" [{r.source}]" if r.source else ""))


# ----------------------------------------------------------------------------- readings
def read_device_sharing(index: int) -> DeviceReading:
    """The sharing facts of one card, read from the driver's own tool through the vendor's
    seam (`neurobrix.core.prism.autodetect.device_sharing`): display activity, memory other
    processes hold, the runtime's own context. When no seam answers, the card is read as
    shared (`measured=False`) — the plan then rounds the free figure down, the safe side."""
    from neurobrix.core.prism import autodetect
    facts = autodetect.device_sharing(index)
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    free = DeviceAllocator.free_memory_mb(index)
    total = None
    try:
        total = DeviceAllocator.device_total_bytes(index) / (1024 * 1024)
    except Exception:  # noqa: BLE001 — the profile supplies the capacity when the driver does not
        total = None
    if facts is None:
        return DeviceReading(kind="device", capacity_mb=float(total or 0), free_mb=float(free or 0),
                             measured=False, source="no sharing reading for this device")
    return DeviceReading(kind="device", capacity_mb=float(total or facts.get("total_mb") or 0),
                         free_mb=float(free if free is not None else facts.get("free_mb") or 0),
                         display_active=bool(facts.get("display_active")),
                         held_by_others_mb=float(facts.get("held_by_others_mb") or 0.0),
                         own_context_mb=float(facts.get("own_context_mb") or 0.0),
                         measured=True, source=str(facts.get("source") or ""))


def host_reading() -> DeviceReading:
    from neurobrix.core.host_memory import memory_state
    st = memory_state()
    return DeviceReading(kind="host", capacity_mb=float(getattr(st, "total_mb", 0) or 0),
                         free_mb=float(getattr(st, "available_mb", 0) or 0),
                         measured=bool(getattr(st, "measured", False)), source=str(getattr(st, "source", "")))


def host_budget_mb() -> int:
    """Host RAM through the same law: the free reading rounded down onto the ladder."""
    return budget_mb(host_reading())
