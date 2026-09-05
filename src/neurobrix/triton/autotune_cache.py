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

_DIR = os.path.join(os.path.expanduser("~"), ".neurobrix", "replay_cache")

# The sanctioned autotune surface (Phase 1.5 doctrine: mm/bmm/addmm/
# conv2d only) — explicit list, not a gc walk. A new autotuned kernel
# is added here the day its autotune exception is granted. Supervisor
# ruling 2026-08-16: NO autotuner lives outside this artifact+gate
# regime (the one historical candidate outside it, the FlagGems
# kernels/utils remnant, was dead code and was removed 2026-08-17).
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
    import importlib
    from triton.runtime.autotuner import Autotuner
    for mod_name, attr in _KERNEL_SITES:
        try:
            obj = getattr(importlib.import_module(mod_name), attr, None)
        except Exception:
            continue
        if isinstance(obj, Autotuner):
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


def capture() -> int:
    """Merge every selected config into the artifact. Returns the
    number of NEW entries written (0 = artifact already covers this
    process's selections)."""
    path = _artifact_path()
    if path is None:
        return 0
    entries: Dict[str, Dict] = {}
    for qual, at in _autotuners():
        for key, cfg in getattr(at, "cache", {}).items():
            entries[f"{qual}::{key!r}"] = _config_to_dict(cfg)
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
