"""The facts a plan is priced with that a first launch does not owe (the orchestrator's ruling of 22/11/2026 17:43,
lines 4 and 5): the per-piece reload rate of a streamed candidate and the host's GEMM and copy of a host-placed one,
measured when Prism prices a candidate whose time can decide the plan (`measure_owed`), once, and kept in the generated
profile. The profile's file, its directory and the first-launch sizes are `autodetect`'s: read there at each call,
never copied here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from neurobrix.core.prism import launch_lines

_log = logging.getLogger(__name__)


def measure_owed(mode, ranked, container, profile) -> None:
    """The facts a candidate in the running is priced with that this machine has not measured yet, measured NOW,
    once, and kept in the generated profile — never at every first launch (the orchestrator's ruling of 22/11/2026
    17:43): the per-piece reload rate when a STREAMED candidate (reloaded at every use from the container, not
    host-held) is ranked (`measure_reload_rates`, line 4); the host's GEMM and copy when a candidate
    places a component on the host, the branch of the estimate that reads them (`measure_host_facts`,
    line 5). Each is said in one line where it is measured; what
    was written is read back into `profile`, under the keys the estimate reads. A fact a plan needs is never a
    default: one that cannot be measured here leaves the candidate not estimated, said by name."""
    from neurobrix.core.prism import autodetect
    from neurobrix.core.prism.host_footprint import engine_of
    from neurobrix.core.prism.solver import STRATEGY_RANK
    from neurobrix.core.prism import time_estimate as TE
    if not any(getattr(d, "measured", None) for d in profile.devices):
        return                  # no measured device facts: no candidate is timed, none owes a measure
    hw, engine = autodetect.ACTIVE_HARDWARE_ID, engine_of(mode)
    # only the candidates the estimate can decide between: the best physical rank present (`order_candidates`
    # orders by rank first, the time only inside a rank) — a streamed or host-placed candidate behind a plan of a
    # better rank is never chosen by its time, and owes no measure (every LLM that fits its card is also streamable)
    best = min((STRATEGY_RANK.get(n, 9) for _s, n, _a, _d in ranked), default=None)
    ranked = [c for c in ranked if STRATEGY_RANK.get(c[1], 9) == best]
    measured = False
    if any(n in TE.STREAM_EVERY_USE and n not in TE.HOST_HELD for _s, n, _a, _d in ranked):
        measured |= bool(measure_reload_rates(hw, engine, getattr(container, "cache_path", None)))
    if any(TE.on_host(a) for _s, _n, allocs, _d in ranked for a in (allocs or {}).values()):
        measured |= bool(measure_host_facts(hw))
    if measured:
        from neurobrix.core.prism.loader import load_profile
        profile.host_measured = load_profile(hw).host_measured


def each_rank_priced(ranked, priced, estimates):
    """`ranked` (rank by rank, `order_candidates`) yielded in order, except that a rank the selection reaches after the
    best one is priced before it is ordered: what that rank owes is measured and its candidates timed (`priced(group)`,
    taken into `estimates`). The best rank was measured before the first estimate (`measure_owed` measures the best
    rank present only); when the selection turns every candidate of it down (its host side declined, the arena door, a
    KV cache that does not fit), the streamed or host-placed plan that wins is ordered by its measured time, never by
    the figure taken before its facts existed (qa 1231-2). A rank never reached is never measured."""
    from neurobrix.core.prism.solver import STRATEGY_RANK, order_candidates
    for i, rank in enumerate(dict.fromkeys(STRATEGY_RANK.get(c[1], 9) for c in ranked)):
        group = [c for c in ranked if STRATEGY_RANK.get(c[1], 9) == rank]
        if i:
            estimates.update(priced(group))
            group = order_candidates(group, estimates)
        yield from group


def measure_host_facts(hardware_id: Optional[str]) -> List[str]:
    """The host's GEMM and copy (`host_matmul_gflops`, `host_memory_gbps`), measured when Prism prices a plan that
    places a component on the host (`measure_owed`), never at every first launch — the orchestrator's
    ruling of 22/11/2026 17:43, line 5. The host's block lies over the vendor's key by key (`loader.load_profile`): only
    what neither states is measured, once, and kept in the generated profile. Returns the keys written; [] when none is
    owed, on a committed profile or under a census shadow."""
    from neurobrix.core.prism import autodetect as A
    path = A.generated_profile(hardware_id)
    if path is None:
        return []
    from neurobrix.core.prism.loader import vendor_block
    from nbxtensor.backend import host_measure as H      # the library's measures (qa 1231-4: numpy measures are its own)
    import time as _time
    data = yaml.safe_load(path.read_text()) or {}
    devices = data.get("devices") or []
    stated = {**((vendor_block(devices[0], "host_measured") if devices else None) or {}), **(data.get("host_measured") or {})}
    if "host_matmul_gflops" in stated:
        return []
    launch_lines.say("host")
    q = A.FIRST_LAUNCH
    data.setdefault("host_measured", {}).update({
        "source": f"measured {_time.strftime('%Y-%m-%d %H:%M:%S %Z')} when a host-held plan was first priced on this "
                  f"machine (autodetect, quick)",
        "host_matmul_gflops": round(H.host_matmul_gflops(q["host_gemm_n"])),
        "host_memory_gbps": round(H.host_memory_gbps(q["host_copy_bytes"]))})
    A.write_profile_atomically(path, data)
    return ["host_matmul_gflops", "host_memory_gbps"]


def measure_reload_rates(hardware_id: Optional[str], engine: str, container_dir=None) -> List[str]:
    """The per-piece reload rate a STREAMED candidate is priced with (`time_estimate.reload_bps`), measured when Prism
    prices one (`measure_owed`), never at every first launch — the orchestrator's ruling of
    22/11/2026 17:43, line 4 (a model that fits its card owes none). Per stored dtype of the requested model (of the cache
    without one) that neither the generated profile nor its vendor block states for `engine`: the smallest component
    stored so, at least the loader-rate floor and at most the card, reloaded in its own process (`arena_probe.child`),
    which writes the rate with its spread (`note_reload_rate`): measured once, kept. Returns the stored dtypes probed;
    [] when none is owed, on a committed profile or under a census shadow."""
    from neurobrix.core.prism import autodetect as A
    path = A.generated_profile(hardware_id)
    if path is None:
        return []
    from neurobrix.core.prism.loader import vendor_block
    data = yaml.safe_load(path.read_text()) or {}
    devices = data.get("devices") or []
    if not devices:
        return []
    probes = []                  # the reload probes owed, decided first: what is said is what is done
    from neurobrix.core.paths import cache_dir
    held = (data.get("host_measured"), vendor_block(devices[0], "host_measured"))
    wanted = None if container_dir is None else {
        A.stored_major(container_dir, c.name) for c in (Path(container_dir) / "components").glob("*")}
    bound = int(devices[0].get("memory_mb") or 0) << 20
    for name, st in A.STORED_DTYPES.items():
        if wanted is not None and name not in wanted:
            continue
        if any((((b or {}).get("weight_reload_gbps") or {}).get(engine) or {}).get(st) for b in held):
            continue
        # the asked model's own component first; the rate is the machine's: another's when none is large enough
        for roots in ([Path(container_dir)] if container_dir is not None else [], list(Path(cache_dir()).glob("*"))):
            comps = sorted((sum(f.stat().st_size for f in c.rglob("*.safetensors")), str(c.parent.parent), c.name)
                           for root in roots for c in (root / "components").glob("*")
                           if A.stored_major(c.parent.parent, c.name) == name)
            comps = [x for x in comps if A.LOADER_RATE_MIN_BYTES <= x[0] <= bound]
            if comps:
                probes.append((*comps[0], st))
                break
    if not probes:
        return []
    launch_lines.say("reload", A.work_said(probes, None, 0))
    from neurobrix.core.prism import arena_probe as A
    for _bytes, root, comp, st in probes:
        r = A.child(root, comp, hardware_id, engine)
        _log.debug(f"   [Auto-detect] {engine} {st} reloads of {r['container']}:{r['component']}: "
                   f"{[round(b / s / 1e9, 3) for b, s in r['draws']]} GB/s (written when three agree)")
    return [st for *_x, st in probes]
