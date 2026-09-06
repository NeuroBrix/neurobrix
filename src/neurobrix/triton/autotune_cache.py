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
    if _ACTIVE is not None and not _ACTIVE["sweep"]:
        return 0          # a model request in default mode: its own artifact is the only source
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
# The per-model sweep artifact — the sweep happens on our side (owner
# directive, 2026-09-06). Its result per kernel, per shape and per hardware
# profile is an artifact delivered with the model (`runtime/autotune/<arch>.json`
# inside the container, embedded by the build from the engine's store) or by the
# hub, and loaded by the engine at each Triton request. A missing artifact for
# this profile is an EXPLICIT refusal, never a silent sweep; the producer runs
# with `--sweep` (NBX_AUTOTUNE=sweep) and its measurements land in the store
# (`~/.neurobrix/autotune/<model>/<arch>.json`) the build embeds.
#
# A request whose shape the artifact never saw does not sweep either: the
# config of the nearest measured shape of the same kernel (same key but the
# leading extent, membership-gated like every seeded config) serves it; a
# kernel with no measured shape at all is the refusal.
# ---------------------------------------------------------------------------
FORMAT = "nbx-autotune-sweep/1"
_STORE = os.path.join(os.path.expanduser("~"), ".neurobrix", "autotune")
_ACTIVE: Optional[Dict] = None     # the model request in force: model, arch, source, entries, sweep


def sweep_mode() -> bool:
    return os.environ.get("NBX_AUTOTUNE", "").strip().lower() == "sweep"


def store_dir() -> str:
    return os.environ.get("NEUROBRIX_AUTOTUNE_STORE") or _STORE


def store_path(model_name: str, arch: Optional[str] = None) -> Optional[str]:
    arch = arch or _arch_fingerprint()
    return None if arch is None else os.path.join(store_dir(), model_name, f"{arch}.json")


def embedded_path(container_path: str, arch: Optional[str] = None) -> Optional[str]:
    arch = arch or _arch_fingerprint()
    return None if arch is None else os.path.join(str(container_path), "runtime", "autotune", f"{arch}.json")


def _read_artifact(path: Optional[str]) -> Optional[Dict[str, Dict]]:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not str(doc.get("format", "")).startswith("nbx-autotune-sweep/"):
        raise RuntimeError(f"{path}: not a sweep artifact (format={doc.get('format') if isinstance(doc, dict) else type(doc).__name__!r})")
    entries = doc.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"{path}: sweep artifact without entries")
    return entries


def load_model_artifact(model_name: str, container_path: Optional[str]) -> Tuple[Optional[Dict[str, Dict]], Optional[str]]:
    """(entries, source path): the container's embedded artifact first, then
    the engine's store; (None, None) when neither exists for this arch."""
    for path in (embedded_path(container_path) if container_path else None, store_path(model_name)):
        entries = _read_artifact(path)
        if entries is not None:
            return entries, path
    return None, None


def refusal(model_name: str, detail: str) -> RuntimeError:
    arch = _arch_fingerprint()
    return RuntimeError(
        f"[autotune] {detail} for hardware profile {arch!r}. The kernel sweep is never run inside a "
        f"request: measure it once on this profile with `neurobrix run --model {model_name} --triton "
        f"--sweep ...` (the result lands in {store_path(model_name) or store_dir()} and the build embeds "
        f"it as runtime/autotune/{arch}.json), or install the model's sweep for this profile from the hub.")


def activate(model_name: str, container_path: Optional[str]) -> int:
    """Arm the policy for one Triton request. Default mode: the model's
    artifact is the only source of kernel configs, and a model without one
    is refused HERE, before any kernel runs. Sweep mode: the artifact and the
    machine cache both seed, everything else is measured and captured by
    `capture_model()`. Returns the number of configs seeded now."""
    global _ACTIVE
    arch = _arch_fingerprint()
    sweep = sweep_mode()
    if _ACTIVE is not None and _ACTIVE["model"] == model_name and _ACTIVE["arch"] == arch and _ACTIVE["sweep"] == sweep:
        return 0
    entries, source = load_model_artifact(model_name, container_path)
    if entries is None and not sweep:
        raise refusal(model_name, "no kernel sweep artifact")
    _ACTIVE = {"model": model_name, "arch": arch, "source": source, "entries": entries or {}, "sweep": sweep,
               "container": str(container_path) if container_path else None, "used": set(), "timings": {}}
    seeded = _seed_entries(entries) if entries else 0
    if sweep:
        seeded += seed()
    if source:
        print(f"[autotune] {model_name}: sweep artifact {source} ({len(entries)} measured shape(s), {seeded} seeded)", flush=True)
    elif sweep:
        print(f"[autotune] {model_name}: SWEEP mode — measuring; the result goes to {store_path(model_name)}", flush=True)
    return seeded


def active() -> Optional[Dict]:
    return _ACTIVE


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


def _shape_distance(a: tuple, b: tuple) -> Optional[int]:
    """L1 distance over the numeric fields of two autotune keys of the same
    kernel; None when a non-numeric field (a dtype name, a flag) differs —
    those select a different kernel variant, never a neighbouring shape."""
    dist = 0
    for x, y in zip(a, b):
        numeric = isinstance(x, int) and not isinstance(x, bool) and isinstance(y, int) and not isinstance(y, bool)
        if numeric:
            dist += abs(x - y)
        elif x != y:
            return None
    return dist


def _nearest(qual: str, at, key: tuple):
    """The config of the nearest measured shape of the same kernel: same
    key length, same non-numeric fields (dtypes, flags), least L1 distance
    over the extents (M for a matmul; batch, rows and columns for a batched
    one); membership-gated. None if none."""
    prefix = f"{qual}::"
    space = {(tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages) for c in getattr(at, "configs", [])}
    best, best_d = None, None
    for k, d in _ACTIVE["entries"].items():
        if not k.startswith(prefix):
            continue
        try:
            stored_key = ast.literal_eval(k[len(prefix):])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(stored_key, tuple) or len(stored_key) != len(key):
            continue
        dist = _shape_distance(stored_key, key)
        if dist is None:
            continue
        member = (tuple(sorted(dict(d["kwargs"]).items())), d["num_warps"], d["num_stages"])
        if member not in space:
            continue
        if best_d is None or dist < best_d:
            best, best_d = _config_from_dict(d), dist
    return best


def note_use(at, key: tuple) -> None:
    """Record that this request resolved `key` on `at` — the sweep artifact
    holds the shapes the MODEL uses, not everything the machine ever
    measured (a producer run seeds the whole machine cache so that known
    shapes are not re-benched; only the used ones are written)."""
    if _ACTIVE is not None and _ACTIVE["sweep"]:
        _ACTIVE["used"].add((id(at), key))


def note_timings(at, key: tuple, timings) -> None:
    """Keep the bench of a key this request just measured: the best time, the
    second-best, and the MARGIN between them. A sweep that records only the
    winner cannot say whether the winner was clear or a near-tie the timer
    could flip on the next run — and a gate comparing two sweeps needs
    exactly that to tell a change of choice from noise."""
    if _ACTIVE is None or not _ACTIVE["sweep"] or not timings:
        return
    try:
        times = sorted(float(t) for t in timings.values() if t is not None and float(t) == float(t))
    except (TypeError, ValueError):
        return
    if not times:
        return
    best = times[0]
    second = times[1] if len(times) > 1 else None
    _ACTIVE["timings"][(id(at), key)] = {
        "best_ms": best, "second_ms": second, "candidates": len(times),
        "margin": (second / best - 1.0) if (second is not None and best > 0) else None}


def resolve_missing(at, key: tuple) -> None:
    """A key the Autotuner holds no config for is about to be benched.
    Outside a model request (tools, tests) or in sweep mode: measure. In a
    request's default mode: the nearest measured shape, else the refusal."""
    if _ACTIVE is None or _ACTIVE["sweep"]:
        return
    qual = _qual_of(at)
    cfg = _nearest(qual, at, key) if qual else None
    if cfg is None:
        raise refusal(_ACTIVE["model"], f"no measured configuration for {qual or getattr(at, 'base_fn', at)} at shape {key!r}")
    at.cache[key] = cfg
    print(f"[autotune] {qual}: shape {key!r} not in the sweep artifact — served by the nearest measured shape "
          f"(no sweep)", flush=True)


def capture_model() -> Optional[str]:
    """Sweep mode, end of a request: write the model's artifact into the
    store — every config the process resolved for the sanctioned kernels,
    merged with what the store already held. Returns the path written."""
    if _ACTIVE is None or not _ACTIVE["sweep"]:
        return None
    path = store_path(_ACTIVE["model"], _ACTIVE["arch"])
    if path is None:
        return None
    entries: Dict[str, Dict] = dict(_ACTIVE["entries"])       # what earlier sweeps of this model measured
    used = _ACTIVE["used"]
    timings = _ACTIVE.get("timings") or {}
    for qual, at in _autotuners():
        for key, cfg in getattr(at, "cache", {}).items():
            if (id(at), key) in used:
                rec = _config_to_dict(cfg)
                if (id(at), key) in timings:
                    rec["timing"] = timings[(id(at), key)]
                entries[f"{qual}::{key!r}"] = rec
    if not used:
        print(f"[autotune] {_ACTIVE['model']}: this request resolved no autotuned kernel — nothing to write", flush=True)
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {"format": FORMAT, "model_name": _ACTIVE["model"], "arch": _ACTIVE["arch"],
           "entries": dict(sorted(entries.items()))}
    try:
        from neurobrix import __version__ as _v
        doc["engine_version"] = _v
    except Exception:
        pass
    import datetime as _dt
    doc["created_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    _ACTIVE["entries"] = entries
    print(f"[autotune] {_ACTIVE['model']}: sweep artifact written {path} ({len(entries)} measured shape(s))", flush=True)
    _announce_screen(_ACTIVE["model"])
    capture()            # the machine cache too — the producer's accumulation across models
    return path


def _announce_screen(model_name: str) -> None:
    """One line per sweep saying what the autotune correctness screen did —
    the activation proof of a gate: a clean screen prints nothing of its own,
    so without this line a sweep with the screen off and a sweep where every
    config passed would read the same."""
    try:
        from neurobrix.kernels import launcher
        cache = getattr(launcher, "_SCREEN_CACHE", None)
        screened = getattr(launcher, "screened_out", None)
    except Exception:
        return
    if cache is None or screened is None:
        return
    keys = sum(len(v) for v in cache.values())
    excluded = screened()
    state = "off" if os.environ.get("NBX_AUTOTUNE_SCREEN", "on").lower() == "off" else "on"
    print(f"[autotune] {model_name}: correctness screen {state}: checked {keys} key(s), "
          f"excluded {len(excluded)} config(s)", flush=True)
