"""E2-full — persistable autotune-config artifact (scoping §Phase 4a.4).

Triton's own disk cache (`cache_results=True`) persists autotune
selections keyed by SOURCE HASH — any kernel-file edit invalidates it
and the next run re-benches (the battery kernel-edit false-RED class,
and a run-to-run timing-variance surface). This module captures the
SELECTED configs keyed by OUR fingerprint — (kernel qualname,
autotune-key literal, arch) — and seeds them back into the Autotuner
in-memory caches, immune to source-hash invalidation:

  capture()  after a warmed pass (the replay recording pass), walk the
             five sanctioned Autotuner kernels (mm/bmm/addmm/conv2d/
             depthwise — the Phase 1.5 scope) and merge their selected
             configs into the artifact.
  seed()     before execution, insert stored configs into each
             Autotuner's cache: a hit means run() never benches.

R33: imports triton + stdlib only. The artifact is an optimization,
never a failure source — every I/O error degrades to "no seed".
Storage: ~/.neurobrix/replay_cache/autotune_configs_<arch>.json.
"""

import ast
import json
import os
from typing import Dict, Iterator, Optional, Tuple

# The machine's replay cache (the producer's accumulation across models, and
# the correctness screen's exclusions). NEUROBRIX_REPLAY_CACHE relocates it —
# a gate that needs a truly cold sweep per arm gives every arm its own, else
# sweep mode seeds the autotuners from here and no arm sweeps (2026-09-06:
# a screened arm reported "checked 0 key(s)" for exactly this reason).
_DIR = os.environ.get("NEUROBRIX_REPLAY_CACHE") or os.path.join(os.path.expanduser("~"), ".neurobrix", "replay_cache")

# The sanctioned autotune surface (Phase 1.5 doctrine: mm/bmm/addmm/
# conv2d only) — explicit list, not a gc walk. A new autotuned kernel
# is added here the day its autotune exception is granted. Supervisor
# ruling 2026-08-16: NO autotuner lives outside this regime (the one
# historical candidate outside it, the FlagGems kernels/utils remnant,
# was dead code and was removed 2026-08-17). Since the owner directive of
# 2026-09-06 the regime is the CERTIFIED DIRECTORY (`kernels/
# autotune_certified.py`, `config/autotune/<vendor>/<profile>/`): a shape it
# holds is applied at load; a shape it lacks sweeps at runtime with the
# consensus screen and lands HERE, the machine's local replay cache.
_KERNEL_SITES = (
    ("neurobrix.kernels.ops.matmul", "matmul_kernel"),
    ("neurobrix.kernels.ops.matmul", "addmm_kernel"),
    ("neurobrix.kernels.ops.baddbmm_op", "baddbmm_kernel"),
    ("neurobrix.kernels.ops.conv2d", "conv2d_forward_kernel"),
    ("neurobrix.kernels.ops.depthwise_conv2d", "depthwise_conv2d_kernel"),
)


def _arch_fingerprint() -> Optional[str]:
    """None when the driver query fails — no fingerprint means NO
    artifact (capture/seed no-op), never a shared cross-arch bucket
    (an sm_80 config seeded onto sm_70 fails at compile inside a user
    run)."""
    try:
        from neurobrix.kernels.launcher import target as _nbx_target   # engine data, no driver probe (R33)
        target = _nbx_target()
        return f"{target.backend}-{target.arch}"
    except Exception:
        return None


def _artifact_path() -> Optional[str]:
    arch = _arch_fingerprint()
    if arch is None:
        return None
    return os.path.join(_DIR, f"autotune_configs_{arch}.json")


def _autotuners() -> Iterator[Tuple[str, object]]:
    """Every Autotuner the process has loaded from the kernel library — the
    five sanctioned sites first (imported on demand), then every other
    `neurobrix.kernels.ops.*` module already in sys.modules. Since 2026-09-05
    the engine is the only persistence of a sweep (upstream's on-disk cache
    keyed its files by a torch-importing driver probe and was dropped), so
    every tuner's selections are captured, not only the matmul class's."""
    import importlib
    import sys
    from triton.runtime.autotuner import Autotuner
    seen = set()
    for mod_name, attr in _KERNEL_SITES:
        try:
            obj = getattr(importlib.import_module(mod_name), attr, None)
        except Exception:
            continue
        if isinstance(obj, Autotuner):
            seen.add(id(obj))
            yield f"{mod_name}.{attr}", obj
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("neurobrix.kernels.ops.") or mod is None:
            continue
        for attr, obj in list(vars(mod).items()):
            if isinstance(obj, Autotuner) and id(obj) not in seen:
                seen.add(id(obj))
                yield f"{mod_name}.{attr}", obj


def _config_to_dict(cfg) -> Dict:
    return {"kwargs": dict(cfg.kwargs), "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages, "num_ctas": cfg.num_ctas,
            "maxnreg": cfg.maxnreg}


def _config_from_dict(d):
    import triton
    return triton.Config(dict(d["kwargs"]), num_warps=d["num_warps"],
                         num_stages=d["num_stages"], num_ctas=d["num_ctas"],
                         maxnreg=d.get("maxnreg"))


def _exclusions_path() -> Optional[str]:
    """Where the correctness screen's exclusions are kept, beside the sweep."""
    path = _artifact_path()
    if path is None:
        return None
    return path.replace("autotune_configs_", "autotune_exclusions_")


def record_screen_exclusions(entries) -> int:
    """Persist configs the correctness screen refused, so Forge sees them.

    A config excluded for being WRONG and a config that merely lost on speed
    are indistinguishable in a sweep that records only the winner — and they
    are not the same fact at all. One is a tuning outcome; the other is a
    backend defect with a shape attached.

    Keyed by kernel, key and config so repeated runs merge instead of
    accumulating duplicates. Best-effort: a sweep that cannot be written must
    never fail a launch, and the exclusions are also printed as they happen.
    """
    path = _exclusions_path()
    if path is None or not entries:
        return 0
    try:
        os.makedirs(_DIR, exist_ok=True)
        stored: Dict[str, Dict] = {}
        try:
            with open(path) as f:
                stored = json.load(f)
        except (OSError, ValueError):
            stored = {}
        added = 0
        for entry in entries:
            entry = dict(entry)
            ident = f"{entry.get('kernel')}::{entry.get('key')!r}::{entry.get('config')}"
            if ident not in stored:
                stored[ident] = entry
                added += 1
        if added:
            with open(path, "w") as f:
                json.dump(stored, f, indent=1, default=str)
        return added
    except OSError:
        return 0


def screen_exclusions() -> Dict[str, Dict]:
    """Everything the screen has refused on this machine, for Forge."""
    path = _exclusions_path()
    if path is None:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


#: Keys whose config was chosen WITHOUT measurement -- e.g. the bench was
#: skipped because its arguments alone exceed the machine's available memory,
#: so timing candidates would have measured the swap and not the kernels. A
#: choice made without measurement must never be recorded as if measured:
#: persisted, it would outlive the pressure that forced it and keep deciding
#: on machines and days it knows nothing about.
_UNMEASURED: set = set()


def mark_unmeasured(at, key) -> None:
    _UNMEASURED.add((id(at), tuple(key) if isinstance(key, (list, tuple)) else key))


def is_unmeasured(at, key) -> bool:
    return (id(at), tuple(key) if isinstance(key, (list, tuple)) else key) in _UNMEASURED


def capture() -> int:
    """Merge every selected config into the artifact. Returns the
    number of NEW entries written (0 = artifact already covers this
    process's selections)."""
    path = _artifact_path()
    if path is None:
        return 0
    entries: Dict[str, Dict] = {}
    skipped_unmeasured = 0
    for qual, at in _autotuners():
        for key, cfg in getattr(at, "cache", {}).items():
            if is_unmeasured(at, key):
                skipped_unmeasured += 1
                continue
            rec = _config_to_dict(cfg)
            if (id(at), key) in _TIMINGS:
                rec["timing"] = _TIMINGS[(id(at), key)]
            entries[f"{qual}::{key!r}"] = rec
    if skipped_unmeasured:
        print(f"[AUTOTUNE_CACHE] {skipped_unmeasured} choice(s) made without "
              f"measurement were NOT persisted", flush=True)
    if not entries:
        return 0
    try:
        os.makedirs(_DIR, exist_ok=True)
        stored: Dict[str, Dict] = {}
        try:
            with open(path) as f:
                stored = json.load(f)
        except (OSError, ValueError):
            stored = {}
        new = {k: v for k, v in entries.items() if k not in stored}
        if new:
            stored.update(new)
            with open(path, "w") as f:
                json.dump(stored, f)
        return len(new)
    except OSError:
        return 0


def seed() -> int:
    """Insert stored configs into the Autotuner caches. Returns the
    number of entries seeded.

    MEMBERSHIP GATE (staleness safety): a stored config is seeded ONLY
    if it is a member of the Autotuner's CURRENT declared config space
    (kwargs + num_warps + num_stages equality against at.configs) — a
    member is by definition a config a fresh bench could legally
    select for the current kernel source and current policy. This
    refuses resurrection of configs removed as buggy, refuses configs
    whose constexpr names drifted with a source edit, and honors
    NBX_DISABLE_AUTOTUNE's pinned single-config list (one member).
    Non-members and non-literal keys are skipped: that shape simply
    re-tunes, exactly the pre-E2 behavior."""
    path = _artifact_path()
    if path is None:
        return 0
    try:
        with open(path) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return 0
    return _seed_entries(stored)


def _seed_entries(stored: Dict[str, Dict]) -> int:
    """Insert `stored` configs into the Autotuner caches under the
    membership gate documented on `seed()`. Returns the number seeded."""
    seeded = 0
    for qual, at in _autotuners():
        prefix = f"{qual}::"
        space = {(tuple(sorted(c.kwargs.items())), c.num_warps,
                  c.num_stages) for c in getattr(at, "configs", [])}
        for k, d in stored.items():
            if not k.startswith(prefix):
                continue
            member = (tuple(sorted(dict(d["kwargs"]).items())),
                      d["num_warps"], d["num_stages"])
            if member not in space:
                continue
            try:
                key = ast.literal_eval(k[len(prefix):])
            except (ValueError, SyntaxError):
                continue
            if key not in at.cache:
                try:
                    at.cache[key] = _config_from_dict(d)
                    seeded += 1
                except Exception:
                    continue
    return seeded


# ---------------------------------------------------------------------------
# The Autotuner's key, and the bench margin of a key this process measured
# ---------------------------------------------------------------------------
def key_of(at, args, kwargs) -> tuple:
    """The Autotuner's cache key for a call, as `Autotuner.run` forms it."""
    _args = {**dict(zip(at.arg_names, args)), **kwargs}
    key = [_args[k] for k in at.keys if k in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


def _qual_of(at) -> Optional[str]:
    for qual, obj in _autotuners():
        if obj is at:
            return qual
    return None


_TIMINGS: Dict[Tuple[int, tuple], Dict] = {}     # (id(tuner), key) -> best/second/margin of a bench this process ran


def note_timings(at, key: tuple, timings) -> None:
    """Keep the bench of a key this process just measured: the best time, the
    second-best, and the MARGIN between them. A cache that records only the
    winner cannot say whether the winner was clear or a near-tie the timer
    could flip on the next run — a gate comparing two sweeps needs exactly
    that to tell a change of choice from noise. Written beside the config by
    `capture()`."""
    if not timings:
        return
    try:
        times = sorted(float(t) for t in timings.values() if t is not None and float(t) == float(t))
    except (TypeError, ValueError):
        return
    if not times:
        return
    best = times[0]
    second = times[1] if len(times) > 1 else None
    _TIMINGS[(id(at), key)] = {
        "best_ms": best, "second_ms": second, "candidates": len(times),
        "margin": (second / best - 1.0) if (second is not None and best > 0) else None}
