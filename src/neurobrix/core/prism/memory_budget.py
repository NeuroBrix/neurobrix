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
    out = sorted({int(g) * 1024 for g in rungs})
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


def sized_by_the_moment(r: DeviceReading) -> bool:
    """The pools whose budget is still the FREE reading rounded onto the ladder: host RAM (no profile states what a
    plan may hold of it) and a device whose facts could not be read at all. Every device pool with a declared capacity
    — dedicated, unified, a display on it, another process on it — is budgeted by that fact (the owner, 2026-10-25
    15:16: a plan must be reproducible from one run to the next; the moment's free reading is the DOOR at execution,
    `execution_door`, never the budget). Until then a unified pool's budget was its free memory rounded down, and
    Janus-Pro-7B on the Mac moved from a resident plan to layer_streaming (six times slower) for 0.3 GB less free
    at a rung edge (2026-10-25 14:56)."""
    return r.kind == "host" or not r.measured


def budget_mb(r: DeviceReading) -> int:
    """What a plan may be budgeted against on this pool: the pool's declared capacity less the runtime's own context
    (a fact of the profile and of this process), or — host RAM and an unreadable device — the free reading rounded
    down onto the ladder."""
    door = _door_mb()
    if door is not None:
        return door
    if sized_by_the_moment(r):
        return rung_down_mb(r.free_mb)
    return int(max(0.0, r.capacity_mb - r.own_context_mb))


def tile_rung_mb(r: DeviceReading) -> int:
    """The rung a tiled component sizes its tile from: a device pool's own nominal rung (its declared capacity),
    a host's or an unreadable device's free rung."""
    door = _door_mb()
    if door is not None:
        return door
    if sized_by_the_moment(r):
        return rung_down_mb(r.free_mb)
    return rung_down_mb(r.capacity_mb)


def execution_door(r: DeviceReading, planned_mb: float) -> Optional[str]:
    """The door the moment's reading keeps: a plan sized by the pool's facts that does not fit what is FREE now is
    refused by name — what holds the memory is said — never traded for a slower plan. None when the reading is no
    live one (a census shadow, an unreadable device) or the plan fits."""
    if _door_mb() is not None or r.kind != "device" or not r.measured or planned_mb <= r.free_mb:
        return None
    holder = (f"{r.held_by_others_mb:.0f} MB held by other processes" if r.held_by_others_mb > 0
              else "a display on it" if r.display_active
              else "the pool is shared with the host" if r.unified
              else "the device's free memory is below its declared capacity")
    return (f"ZERO FALLBACK: the plan places {planned_mb:.0f} MB on this pool and {r.free_mb:.0f} MB are free now "
            f"({holder}); the plan is sized by the pool's declared {r.capacity_mb:.0f} MB and is not traded for a "
            f"slower one — free the memory or plan under a smaller profile" + (f" [{r.source}]" if r.source else ""))


def describe(r: DeviceReading) -> str:
    door = _door_mb()
    if door is not None:
        return f"budget {door} MB by the NBX_PRISM_BUDGET_MB door"
    if sized_by_the_moment(r):
        why = "host RAM" if r.kind == "host" else "sharing could not be read"
        return (f"sized by the moment ({why}): free {r.free_mb:.0f} MB rounds down to the {rung_down_mb(r.free_mb)} MB rung"
                + (f" [{r.source}]" if r.source else ""))
    kind = ("unified pool" if r.unified else "a display on it" if r.display_active
            else f"{r.held_by_others_mb:.0f} MB held by other processes" if r.held_by_others_mb > 0 else "dedicated")
    return (f"{kind}: the pool's declared {r.capacity_mb:.0f} MB less its own context {r.own_context_mb:.0f} MB "
            f"= {budget_mb(r)} MB; tile rung {tile_rung_mb(r)} MB; free now {r.free_mb:.0f} MB (the door)"
            + (f" [{r.source}]" if r.source else ""))


# ----------------------------------------------------------------------------- readings
def read_device_sharing(index: int) -> DeviceReading:
    """The sharing facts of one card, read from the driver's own tool through the vendor's
    seam (`neurobrix.core.prism.autodetect.device_sharing`): display activity, memory other
    processes hold, the runtime's own context. When no seam answers, the card is read as
    shared (`measured=False`) — the plan then rounds the free figure down, the safe side."""
    from neurobrix.core.prism import autodetect
    facts = autodetect.device_sharing(index)
    from nbxtensor.memory.allocator import DeviceAllocator
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
    from nbxtensor.backend.host_memory import memory_state
    st = memory_state()
    return DeviceReading(kind="host", capacity_mb=float(getattr(st, "total_mb", 0) or 0),
                         free_mb=float(getattr(st, "available_mb", 0) or 0),
                         measured=bool(getattr(st, "measured", False)), source=str(getattr(st, "source", "")))


def host_budget_mb() -> int:
    """Host RAM through the same law: the free reading rounded down onto the ladder."""
    return budget_mb(host_reading())


# ----------------------------------------------------------------------------- the deferred drain
#: The env door that pins the drain cliff (`triton/sequence.py::_resolve_drain_policy`), read here
#: by the same name so the plan prices the cliff the run will hold.
DEFERRED_DRAIN_DOOR = "NBX_DEFERRED_DRAIN_BYTES"


def deferred_drain_policy_of(memory: dict) -> tuple:
    """The Triton engine's deferred-free drain policy from ONE vendor profile's `memory:` section,
    as `(cliff_bytes, floor_bytes, pressure_reserve_bytes, freecheck_interval)` — the pure reading
    shared by the runtime (`kernels.wrappers.deferred_drain_policy`, which drains by it) and the
    plan (`PrismSolver._deferred_drain_mb`, which prices it). One function, so the two can never
    read two figures.

    The hot loops retire every dead or overwritten tensor onto a queue and free it only after a
    device sync, when the queue reaches `cliff` bytes (or, with the pressure branch armed and the
    allocator pool off, earlier). Between two drains the queue's bytes are dead to the program and
    alive to the card. A profile that does not state its cliff is refused by name: the figure is
    both a runtime knob and a planned price, and neither may be guessed."""
    if "deferred_drain_cliff_bytes" not in (memory or {}):
        raise KeyError("ZERO FALLBACK: the vendor profile's memory section declares no "
                       "deferred_drain_cliff_bytes — the Triton engine's dead-tensor queue has "
                       "no stated ceiling, so neither the run nor the plan can size it")
    cliff = int(memory["deferred_drain_cliff_bytes"])
    reserve = int(memory.get("deferred_drain_pressure_reserve_bytes", 0))
    if reserve <= 0:
        return (cliff, cliff, 0, 1 << 30)
    floor = min(int(memory["deferred_drain_floor_bytes"]), cliff)
    interval = max(1, int(memory.get("deferred_drain_freecheck_interval", 8)))
    return (cliff, floor, reserve, interval)


def deferred_drain_retained_bytes(memory: dict) -> int:
    """The most dead bytes a Triton hot loop's queue holds at any allocation: its CLIFF.

    Why the cliff bounds it exactly: the loop appends an op's evicted tensors AFTER the op ran and
    tests the cliff right there, so every allocation an op makes sees a queue below the cliff. The
    pressure branch only drains earlier, and under the allocator pool (the default) it is off — so
    the cliff, not the floor, is what a card must hold room for. The door pins it as it pins the
    run's cliff."""
    door = _drain_door_bytes()
    return door if door > 0 else deferred_drain_policy_of(memory)[0]


def _drain_door_bytes() -> int:
    """The door's pinned cliff in bytes, 0 when the door is shut (unset, empty, unparsable, <= 0)."""
    door = os.environ.get(DEFERRED_DRAIN_DOOR)
    if not door:
        return 0
    try:
        return max(0, int(door))
    except ValueError:
        return 0


def deferred_drain_floor_of(memory: dict) -> int:
    """The SMALLEST queue a plan may leave the Triton hot loop: the profile's
    `deferred_drain_floor_bytes`, clamped to the cliff.

    Source: the same key the runtime's pressure branch drains early at (`deferred_drain_policy_of`)
    — the profile's statement of the queue size below which draining costs more than it frees
    (each drain is a device sync, then a free per dead tensor). A plan leaving less room than this
    beside a component held whole would make the hot loop drain after nearly every op; such a
    component is not held whole. A profile that does not state it is refused by name."""
    if "deferred_drain_floor_bytes" not in (memory or {}):
        raise KeyError("ZERO FALLBACK: the vendor profile's memory section declares no "
                       "deferred_drain_floor_bytes — the smallest drain cliff a plan may leave the "
                       "Triton engine's dead-tensor queue is not stated, so no component can be "
                       "judged whole beside it")
    return min(int(memory["deferred_drain_floor_bytes"]), deferred_drain_policy_of(memory)[0])


def planned_drain_cliff_bytes(memory: dict, room_bytes: float) -> int:
    """THE drain cliff a plan gives one device, read by Prism (which plans and prices it) and, through
    the plan, by the Triton hot loop (which drains at it):

        cliff = min(profile cliff, room),   room = what the device's usable part leaves beside the
                                            largest component the plan holds whole on it.

    Pass a room at or above the profile cliff when nothing is held whole. The door
    (`NBX_DEFERRED_DRAIN_BYTES`) pins the result, as it pins the run. Never below 0; a room under
    the floor (`deferred_drain_floor_of`) is a placement the plan refuses, not a cliff it returns."""
    door = _drain_door_bytes()
    if door > 0:
        return door
    return max(0, min(deferred_drain_policy_of(memory)[0], int(room_bytes)))


def planned_drain_floor_bytes(memory: dict) -> int:
    """The room a component held whole must leave the queue: the floor, or the door's pinned cliff."""
    door = _drain_door_bytes()
    return door if door > 0 else deferred_drain_floor_of(memory)
