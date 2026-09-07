"""The certified autotune directory — an engine component.

The settings the autotuner would otherwise discover by sweeping at runtime
live WITH the engine, versioned with it, under
``src/neurobrix/config/autotune/<vendor>/<profile>/<kernel>.<dtype>.json``:
one file per kernel and per dtype of the buffer the kernel writes, indexed
inside by the shape key the launcher already computes (``autotune_cache.key_of``
— the Autotuner's own key). Each entry carries the setting retained and its
PROOF — the date, the engine and backend versions, the shape, the deviation
measured against the fp64 oracle, the profile's tolerance, the machine — and
the settings excluded with their deviation.

The directory is filled by ``neurobrix autotune certify --profile <profile>``
(``autotune_certify.py``), never by a request: at load the launcher looks the
certified setting up for the profile Prism detected, the kernel, the dtype and
the shape; if it exists it is applied without a sweep and without the screen;
if it is missing the runtime sweeps with the consensus screen as before, SAYS
SO, and keeps the result in the machine's local replay cache — never in this
directory. A file without a proof, or whose proof does not re-read, is refused
here with its reason and the runtime falls back to the sweep.

Nothing here names a backend or a model: the vendor and profile come from the
vendor profile in force, the kernels from the launcher, the tolerances from the
profile.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

FORMAT = "nbx-autotune-certified/1"
_DTYPES = ("fp16", "bf16", "fp32", "fp64", "int8", "int16", "int32", "int64", "bool")
_PROOF_FIELDS = ("date", "engine_version", "backend", "shape", "deviation", "tolerance", "machine", "oracle")

_LOADED: Dict[Tuple[str, str, str, str], Optional[Dict[str, Dict]]] = {}   # (vendor, profile, kernel, dtype) -> entries
_REFUSED: Dict[str, str] = {}                                                # file -> reason (said once)
_ANNOUNCED: set = set()                                                       # (kernel, key) already said missing
_SERVED: Dict[str, int] = {"certified": 0, "swept": 0, "local": 0}     # local = keys seeded from the machine's replay cache


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------
def directory() -> Path:
    """The engine's directory (NEUROBRIX_AUTOTUNE_CERTIFIED_DIR relocates it — tests, a contributor's draft)."""
    override = os.environ.get("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "config" / "autotune"


def enabled() -> bool:
    """NBX_AUTOTUNE_CERTIFIED=off makes every key sweep at runtime (the proof's "runtime sweep" arm)."""
    return os.environ.get("NBX_AUTOTUNE_CERTIFIED", "on").strip().lower() != "off"


def active_profile() -> Optional[Tuple[str, str]]:
    """(vendor, profile) of the vendor profile in force, from the profile files themselves."""
    try:
        from neurobrix.kernels.ops._configs import active_vendor_profile
        cfg = active_vendor_profile()
    except Exception:
        return None
    vendor, profile = cfg.get("_vendor"), cfg.get("_profile")
    if not vendor or not profile:
        return None
    return str(vendor), str(profile)


def kernel_short(qual: str) -> str:
    return qual.rsplit(".", 1)[-1]


def file_for(vendor: str, profile: str, kernel_qual: str, dtype: str, root: Optional[Path] = None) -> Path:
    return (root or directory()) / vendor / profile / f"{kernel_short(kernel_qual)}.{dtype}.json"


# ---------------------------------------------------------------------------
# the key and its dtype
# ---------------------------------------------------------------------------
def key_dtypes(key: tuple) -> List[str]:
    """The dtype names the launcher appended to the key, in argument order."""
    return [str(k) for k in key if isinstance(k, str) and str(k).lower() in _DTYPES]


def output_dtype(tuner, key: tuple) -> str:
    """The dtype of the buffer the kernel WRITES — the file's dtype and the
    tolerance's. `key_of` appends one dtype per tensor argument in the
    kernel's argument order, so the i-th dtype belongs to the i-th pointer
    argument; the output pointer is the one named for it."""
    dts = key_dtypes(key)
    if not dts:
        return "any"
    names = [n for n in getattr(tuner, "arg_names", []) if n.endswith("_ptr") or n.endswith("_pointer")]
    for i, name in enumerate(names):
        low = name.lower()
        if i < len(dts) and (low.startswith(("out", "c_", "output", "y_")) or low in ("c_ptr", "out_ptr", "output_pointer")):
            return dts[i]
    return dts[-1]


def key_repr(key: tuple) -> str:
    return repr(tuple(key))


def parse_key(text: str) -> Optional[tuple]:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return tuple(value) if isinstance(value, (tuple, list)) else None


def describe_key(tuner, key: tuple) -> str:
    """`M=1500 N=1280 K=1280 IEEE_PRECISION=True … fp32,fp16,fp32` — readable in a log line."""
    names = list(getattr(tuner, "keys", []) or [])
    parts = [f"{n}={v}" for n, v in zip(names, key)]
    rest = [str(v) for v in key[len(names):]]
    return " ".join(parts) + (" " + ",".join(rest) if rest else "")


# ---------------------------------------------------------------------------
# the gate: a file is trusted only if its proof re-reads
# ---------------------------------------------------------------------------
def _tolerance_for(vendor: str, profile: str, dtype: str) -> Optional[float]:
    """The profile's `autotune_screen_rtol[dtype]`, read from the profile file itself."""
    try:
        import yaml
        path = Path(__file__).resolve().parents[1] / "config" / "vendors" / vendor / f"{profile}.yml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    table = cfg.get("autotune_screen_rtol") or {}
    canon = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64"}.get(dtype, dtype)
    val = table.get(canon, table.get(dtype))
    return float(val) if val is not None else None


def validate(doc: Any, path: Path, tuner=None) -> List[str]:
    """Every reason this file cannot be trusted; empty when its proof re-reads.

    Checked: the format, that vendor/profile/kernel/dtype inside match the path,
    that each entry has a config and a complete proof, that the proof's
    deviation does not exceed its tolerance, that the tolerance is the
    profile's for that dtype, that every excluded setting carries a deviation
    ABOVE the tolerance, that the key parses and (with the tuner) has the
    kernel's arity and the file's dtype, that the date parses."""
    problems: List[str] = []
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        return [f"format is not {FORMAT!r}"]
    vendor, profile = path.parent.parent.name, path.parent.name
    stem_kernel, stem_dtype = path.stem.rsplit(".", 1) if "." in path.stem else (path.stem, "")
    if doc.get("vendor") != vendor or doc.get("profile") != profile:
        problems.append(f"vendor/profile inside ({doc.get('vendor')}/{doc.get('profile')}) differ from the path ({vendor}/{profile})")
    if kernel_short(str(doc.get("kernel", ""))) != stem_kernel:
        problems.append(f"kernel inside ({doc.get('kernel')}) differs from the file name ({stem_kernel})")
    if str(doc.get("dtype")) != stem_dtype:
        problems.append(f"dtype inside ({doc.get('dtype')}) differs from the file name ({stem_dtype})")
    entries = doc.get("entries")
    if not isinstance(entries, dict) or not entries:
        problems.append("no entries")
        return problems
    expected_tol = _tolerance_for(vendor, profile, stem_dtype)
    for ktext, entry in entries.items():
        where = f"entry {ktext[:80]}"
        key = parse_key(ktext)
        if key is None:
            problems.append(f"{where}: the key does not parse as a shape key")
            continue
        if tuner is not None:
            arity = len(getattr(tuner, "keys", []) or [])
            if len(key) < arity:
                problems.append(f"{where}: the key has {len(key)} fields, the kernel's key has {arity}")
            if output_dtype(tuner, key) != stem_dtype:
                problems.append(f"{where}: the key's output dtype is {output_dtype(tuner, key)}, the file's is {stem_dtype}")
        if not isinstance(entry, dict):
            problems.append(f"{where}: not an object")
            continue
        cfg = entry.get("config")
        if not isinstance(cfg, dict) or not isinstance(cfg.get("kwargs"), dict) or "num_warps" not in cfg or "num_stages" not in cfg:
            problems.append(f"{where}: no config (kwargs, num_warps, num_stages)")
        proof = entry.get("proof")
        if not isinstance(proof, dict):
            problems.append(f"{where}: no proof")
            continue
        missing = [f for f in _PROOF_FIELDS if f not in proof]
        if missing:
            problems.append(f"{where}: proof without {', '.join(missing)}")
            continue
        try:
            dev, tol = float(proof["deviation"]), float(proof["tolerance"])
        except (TypeError, ValueError):
            problems.append(f"{where}: deviation/tolerance are not numbers")
            continue
        if not (dev == dev) or dev < 0 or dev > tol:
            problems.append(f"{where}: the proof's deviation {dev:.3e} exceeds its tolerance {tol:.1e}")
        if expected_tol is not None and abs(tol - expected_tol) > 1e-15:
            problems.append(f"{where}: tolerance {tol:.1e} is not the profile's {expected_tol:.1e} for {stem_dtype}")
        if list(proof.get("shape") or []) != list(key):
            problems.append(f"{where}: the proof's shape is not the entry's key")
        try:
            _dt.datetime.fromisoformat(str(proof["date"]).replace("Z", "+00:00"))
        except ValueError:
            problems.append(f"{where}: the date does not parse")
        for ex in entry.get("excluded") or []:
            try:
                if float(ex.get("deviation")) <= tol:
                    problems.append(f"{where}: an excluded setting carries a deviation within the tolerance")
            except (TypeError, ValueError, AttributeError):
                problems.append(f"{where}: an excluded setting without a deviation")
    return problems


def files(root: Optional[Path] = None) -> Iterator[Path]:
    base = root or directory()
    if not base.exists():
        return iter(())
    return iter(sorted(base.glob("*/*/*.json")))


# ---------------------------------------------------------------------------
# lookup at load
# ---------------------------------------------------------------------------
def _load(vendor: str, profile: str, kernel_qual: str, dtype: str, tuner=None) -> Optional[Dict[str, Dict]]:
    ident = (vendor, profile, kernel_qual, dtype)
    if ident in _LOADED:
        return _LOADED[ident]
    path = file_for(vendor, profile, kernel_qual, dtype)
    entries: Optional[Dict[str, Dict]] = None
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            doc = None
            problems = [f"unreadable: {exc}"]
        else:
            problems = validate(doc, path, tuner)
        if problems:
            if str(path) not in _REFUSED:
                _REFUSED[str(path)] = problems[0]
                print(f"[autotune] certified file REFUSED {path}: {problems[0]}"
                      + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")
                      + " — its keys sweep at runtime", flush=True)
        elif isinstance(doc, dict):
            entries = dict(doc["entries"])
    _LOADED[ident] = entries
    return entries


def lookup(kernel_qual: str, tuner, key: tuple, ignore_switch: bool = False) -> Optional[Dict[str, Any]]:
    """The certified entry for this profile, kernel, dtype and shape, or None.
    `ignore_switch`: read the directory even when NBX_AUTOTUNE_CERTIFIED=off —
    the switch stops APPLYING a certification, never REPORTING against one."""
    if not enabled() and not ignore_switch:
        return None
    ident = active_profile()
    if ident is None:
        return None
    entries = _load(ident[0], ident[1], kernel_qual, output_dtype(tuner, key), tuner)
    if not entries:
        return None
    return entries.get(key_repr(key))


# A promotion flag that widens a pointer's tile on load: the kernel computes
# that operand in fp32 whatever its memory dtype, so the key that names the
# COMPUTATION carries fp32 for it — the directory's entries were certified
# with the operand widened in memory (the copy the promotion replaced), and
# the same setting applies. PROMOTE_B is NOT here: it was keyed by the memory
# dtype from the first certification (`fp32,fp16,fp32`), and stays so.
_KEYED_AS_COMPUTED = {"PROMOTE_A": "a_ptr", "PROMOTE_BIAS": "bias_ptr"}


def _promotions_by_policy(tuner, tags: List[str]) -> Dict[str, bool]:
    """Which operands the wrappers widen on load for a key, read from the dtype
    tags under the hardware profile's rule — the same rule `mm`/`addmm` apply:
    on a card without native bf16 a narrow activation computes in fp32, and a
    narrow bias too whenever the activation computes in fp32. Used where no
    call is at hand (the local replay cache's keys at load)."""
    names = [n for n in getattr(tuner, "arg_names", []) if n.endswith("_ptr") or n.endswith("_pointer")]
    if "PROMOTE_A" not in getattr(tuner, "arg_names", []):
        return {}
    try:
        from neurobrix.kernels.wrappers import _NBX_HAS_NATIVE_BF16 as native_bf16
    except Exception:
        return {}
    def tag(ptr):
        return str(tags[names.index(ptr)]).lower() if ptr in names and names.index(ptr) < len(tags) else ""
    a_narrow = tag("a_ptr") in ("fp16", "bf16")
    promote_a = bool(a_narrow and not native_bf16)
    a_fp32 = promote_a or tag("a_ptr") == "fp32"
    promote_bias = bool(a_fp32 and tag("bias_ptr") in ("fp16", "bf16"))
    return {"PROMOTE_A": promote_a, "PROMOTE_BIAS": promote_bias}


def computed_key(tuner, key: tuple, kwargs: Optional[Dict[str, Any]] = None) -> Optional[tuple]:
    """The key with every pointer a set promotion flag widens on load tagged
    fp32 — the dtype the kernel computes it in — or None when no tag changes.
    `kwargs` are the call's; None reads the flags from the profile's rule."""
    names = [n for n in getattr(tuner, "arg_names", []) if n.endswith("_ptr") or n.endswith("_pointer")]
    nkeys = len(list(getattr(tuner, "keys", []) or []))
    head, tags = list(key[:nkeys]), list(key[nkeys:])
    if kwargs is None:
        kwargs = _promotions_by_policy(tuner, tags)
    changed = False
    for flag, ptr in _KEYED_AS_COMPUTED.items():
        if kwargs.get(flag) and ptr in names:
            i = names.index(ptr)
            if i < len(tags) and str(tags[i]).lower() in ("fp16", "bf16"):
                tags[i] = "fp32"
                changed = True
    return tuple(head + tags) if changed else None


def apply(kernel_qual: str, tuner, key: tuple, lookup_key: Optional[tuple] = None) -> bool:
    """Put the certified config in the tuner's cache for `key` — no sweep, no
    screen. False when the directory has nothing for it. `lookup_key`: the key
    the directory is read at when it differs from the tuner's (a promoted
    operand keyed by its computed dtype, `computed_key`)."""
    entry = lookup(kernel_qual, tuner, lookup_key or key)
    if entry is None:
        return False
    import triton
    cfg = entry["config"]
    tuner.cache[key] = triton.Config(dict(cfg["kwargs"]), num_warps=int(cfg["num_warps"]),
                                     num_stages=int(cfg["num_stages"]), num_ctas=int(cfg.get("num_ctas") or 1),
                                     maxnreg=cfg.get("maxnreg"))
    _SERVED["certified"] += 1
    return True


def override_seeded(tuners=None) -> int:
    """The directory first, even on a warm machine: every key the local replay
    cache seeded into a tuner that the directory certifies takes the certified
    config instead (no sweep, no screen). Returns how many were overridden —
    counted as certified, no longer as local. Without this the seed ran before
    the lookup and a warm machine never applied a certified setting (Flex.1 on
    2026-09-07: 2,717 entries in the directory, 0 served, 5,547 from the cache)."""
    if not enabled():
        return 0
    if tuners is None:
        from neurobrix.triton.autotune_cache import _autotuners
        tuners = list(_autotuners())
    n = 0
    for qual, tuner in tuners:
        cache = getattr(tuner, "cache", None)
        if not cache:
            continue
        for key in list(cache.keys()):
            if not isinstance(key, tuple):
                continue
            try:
                entry = lookup(qual, tuner, key)
                twin = None
                if entry is None:
                    twin = computed_key(tuner, key)       # a widened-on-load operand, by the profile's rule
                    entry = lookup(qual, tuner, twin) if twin else None
            except Exception:
                entry = None
            if entry is None:
                continue
            cache.pop(key, None)
            if apply(qual, tuner, key, lookup_key=twin):
                n += 1
    _SERVED["local"] = max(0, _SERVED["local"] - n)
    return n


def announce_missing(kernel_qual: str, tuner, key: tuple) -> None:
    """Said once per key, in clear: the runtime is about to sweep."""
    ident = (kernel_qual, key)
    if ident in _ANNOUNCED:
        return
    _ANNOUNCED.add(ident)
    _SERVED["swept"] += 1
    prof = active_profile()
    where = f"{prof[0]}/{prof[1]}" if prof else "the profile in force"
    try:
        from neurobrix.triton.autotune_cache import _DIR as _local
    except Exception:
        _local = "the local replay cache"
    print(f"[autotune] no certified setting for {kernel_short(kernel_qual)} {output_dtype(tuner, key)} "
          f"({describe_key(tuner, key)}) on {where}: sweeping at runtime with the consensus screen; the result is "
          f"kept in {_local}, never in the engine's directory — `neurobrix autotune certify --profile "
          f"{prof[1] if prof else '<profile>'}` certifies it", flush=True)


def served() -> Dict[str, int]:
    """How many keys this process took from the directory, how many it swept, how many came from the local replay cache."""
    return dict(_SERVED)


def note_local(n: int) -> None:
    """Keys the local replay cache seeded into the autotuners (a previous runtime sweep on this machine)."""
    _SERVED["local"] += int(n or 0)


def reset() -> None:
    _LOADED.clear(); _REFUSED.clear(); _ANNOUNCED.clear()
    _SERVED.update({"certified": 0, "swept": 0, "local": 0})


def certified_config_for(kernel_name: str, key_fields: tuple) -> Optional[Dict[str, Any]]:
    """For the screen: the certified entry of a kernel (short name) whose key
    starts with these fields on the profile in force, or None. Used to report
    a runtime exclusion that contradicts a certification."""
    ident = active_profile()
    if ident is None:
        return None
    root = directory() / ident[0] / ident[1]
    if not root.exists():
        return None
    for path in sorted(root.glob(f"{kernel_name}.*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if validate(doc, path):
            continue
        for ktext, entry in (doc.get("entries") or {}).items():
            key = parse_key(ktext)
            if key is not None and tuple(key[:len(key_fields)]) == tuple(key_fields):
                return {"path": str(path), "key": key, **entry}
    return None


# ---------------------------------------------------------------------------
# contradictions: the runtime screen against a certification
# ---------------------------------------------------------------------------
_CONTRADICTIONS: List[Dict[str, Any]] = []


def contradictions() -> List[Dict[str, Any]]:
    """Every runtime exclusion this process saw that contradicts a certified setting."""
    return list(_CONTRADICTIONS)


def _config_matches(cfg: Dict[str, Any], config_text: str) -> bool:
    """`str(triton.Config)` against a stored config dict: every kwarg and num_warps/num_stages present."""
    text = str(config_text)
    for k, v in (cfg.get("kwargs") or {}).items():
        if f"{k}: {v}" not in text:
            return False
    return (f"num_warps: {cfg.get('num_warps')}" in text) and (f"num_stages: {cfg.get('num_stages')}" in text)


def report_contradictions(tuner, dropped) -> List[Dict[str, Any]]:
    """Called by the screen with the configs it just excluded: any of them that
    the directory certifies for this profile, kernel, dtype and shape is
    printed as a CONTRADICTION and recorded beside the screen's exclusions.
    The certified setting stays in force for this run; the finding is the
    directory's to resolve (a re-certification, or a defect in one of the two
    measurements) — silently trusting either would be the actual failure."""
    if not dropped:
        return []
    from neurobrix.triton.autotune_cache import key_of, _qual_of
    qual = _qual_of(tuner)
    if qual is None:
        return []
    try:
        key = key_of(tuner, [], dict(getattr(tuner, "nargs", None) or {}))
    except Exception:
        return []
    entry = lookup(qual, tuner, key, ignore_switch=True)     # the proof's sweep arm is where the two can disagree
    if entry is None:
        return []
    found = []
    for ex in dropped:
        if not _config_matches(entry["config"], getattr(ex, "config", ex)):
            continue
        proof = entry.get("proof") or {}
        rec = {"kernel": qual, "key": key_repr(key), "dtype": output_dtype(tuner, key), "config": entry["config"],
               "certified": {"date": proof.get("date"), "deviation": proof.get("deviation"),
                             "tolerance": proof.get("tolerance"), "machine": proof.get("machine")},
               "runtime": {"deviation": getattr(ex, "deviation", None), "tolerance": getattr(ex, "tolerance", None)},
               "contradiction": True}
        found.append(rec)
        _CONTRADICTIONS.append(rec)
        print(f"[AUTOTUNE_SCREEN] CONTRADICTION — {kernel_short(qual)} {rec['dtype']} ({describe_key(tuner, key)}): the "
              f"setting certified on {proof.get('date')} (deviation {proof.get('deviation')} vs the fp64 oracle, tolerance "
              f"{proof.get('tolerance')}) is excluded by the runtime consensus (deviation {rec['runtime']['deviation']}). "
              f"A finding to report: the certification and this run disagree.", flush=True)
    if found:
        try:
            from neurobrix.triton.autotune_cache import record_screen_exclusions
            record_screen_exclusions([{**r, "config": str(r["config"])} for r in found])
        except Exception:
            pass
    return found
