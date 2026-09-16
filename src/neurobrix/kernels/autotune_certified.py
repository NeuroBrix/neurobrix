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

FORMAT = "nbx-autotune-certified/2"

#: Formats this reader accepts. `/2` added the `built` field, which says the
#: kernel actually COMPILED on the device during its certifying run. A `/1`
#: file pre-dates that field and is READ, not refused: the 5628 shapes the
#: other machine certified before it existed are proven work, and discarding
#: them over a field that did not exist when they were written would be
#: destroying a measurement to tidy a schema. What `/1` cannot say is said
#: instead — `built_unknown_formats` names them to the caller.
_FORMATS = ("nbx-autotune-certified/1", "nbx-autotune-certified/2")

#: Which proof fields each format requires.
_REQUIRED = {
    "nbx-autotune-certified/1": ("date", "engine_version", "backend", "shape",
                                 "deviation", "tolerance", "machine", "oracle"),
    "nbx-autotune-certified/2": ("date", "engine_version", "backend", "shape",
                                 "deviation", "tolerance", "machine", "oracle",
                                 "built"),
}
_DTYPES = ("fp16", "bf16", "fp32", "fp64", "int8", "int16", "int32", "int64", "bool")
#: `built` says the kernel actually COMPILED on the device during the
#: certifying run. It is required because the screen cannot answer it: a CPU
#: fallback computes correctly, so its deviation against the fp64 oracle is
#: excellent — an entry certified on one would record a configuration chosen
#: for a path that never runs, and nothing in the proof would say so.
_PROOF_FIELDS = ("date", "engine_version", "backend", "shape", "deviation",
                 "tolerance", "machine", "oracle", "built")



def format_for(entries: Dict[str, Dict]) -> str:
    """The format a file of these entries may claim: `/2` only when EVERY
    certification in it — primary and variants — carries `built`; `/1`
    otherwise. A writer that stamped `/2` over a `/1` file's entries made the
    gate refuse the whole file (three files, 6 344 entries, 2026-09-13 22:40):
    the stamp is a claim about every entry, and the claim was false."""
    for entry in (entries or {}).values():
        certs = [entry] + list((entry.get("variants") or {}).values()) if isinstance(entry, dict) else [entry]
        for c in certs:
            if not isinstance(c, dict) or "built" not in (c.get("proof") or {}):
                return "nbx-autotune-certified/1"
    return FORMAT


def restamp(path: Path) -> Optional[str]:
    """Repair a file's format claim to what its entries satisfy, entries
    untouched. Returns the new format when it changed, None otherwise."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    want = format_for(doc.get("entries") or {})
    if doc.get("format") == want:
        return None
    doc["format"] = want
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return want


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


# ---------------------------------------------------------------------------
# the memory class a proof covers (register entry 56)
# ---------------------------------------------------------------------------
# One vendor profile can be carried by cards of different memory: this rack's
# `nvidia/volta` is two 16 GB V100s and two 32 GB ones. A proof was made on ONE
# card, and an entry serves only the memory class it covered — a card of
# another class sweeps at runtime, announced. The class is the card's memory
# rounded to GB, the same rounding the auto-profile name uses
# (`core/prism/autodetect.py`: `round(memory_mb / 1024)`), so the class read
# from a proof's device record and the one read from a legacy profile name are
# the same number for the same card.
from neurobrix.core.prism.structure import memory_class_gb, memory_class_from_profile_id  # the ONE rule (no torch there)


def proof_memory_class(proof: Optional[Dict[str, Any]]) -> Optional[int]:
    """The memory class (GB) of the card a proof was made on, or None when
    the proof cannot say. A device record (`machine.device.memory_mb`) says it
    exactly; a legacy profile name says it through the profile grammar's own
    parser (`memory_class_from_profile_id`: a single-card name by its suffix,
    a rig-wide name is unknown — never a reconstruction)."""
    machine = (proof or {}).get("machine") or {}
    device = machine.get("device")
    cls = memory_class_gb(device.get("memory_mb")) if isinstance(device, dict) else None
    if cls is not None:
        return cls
    return memory_class_from_profile_id(machine.get("hardware_profile"))


def _variant_slot(cls: int) -> str:
    return f"{int(cls)}g"


def proof_backend(proof: Optional[Dict[str, Any]]) -> Optional[str]:
    """The code generator a proof was made with, as one label: `triton <version>`
    plus the backend name when it is not cuda (e.g. `triton 3.7.0 metal`). A
    setting stays correct under any generator — the oracle proved the source,
    not the compiler; what a newer generator may age is its rank as the
    fastest, by a few percent — so the directory is re-proven under a new one
    as an optimisation pass and the document reads each rank with its
    generator's date (owner, 2026-09-16)."""
    if not proof:
        return None
    b = proof.get("backend") or {}
    ver = b.get("triton")
    if not ver:
        return None
    name = b.get("name")
    return f"triton {ver}" + (f" {name}" if name and name != "cuda" else "")


def proof_backends(entry: Dict[str, Any]) -> set:
    """Every generator label this entry carries a proof from (primary and variants)."""
    out = set()
    lab = proof_backend(entry.get("proof"))
    if lab:
        out.add(lab)
    for slot, var in (entry.get("variants") or {}).items():
        lab = proof_backend((var or {}).get("proof"))
        if lab:
            out.add(lab)
    return out


def covered_memory_classes(entry: Dict[str, Any]) -> set:
    """Every memory class this entry carries a proof for: its primary proof's
    class and each variant's."""
    out = set()
    cls = proof_memory_class(entry.get("proof"))
    if cls is not None:
        out.add(cls)
    for slot, var in (entry.get("variants") or {}).items():
        vcls = proof_memory_class((var or {}).get("proof"))
        if vcls is not None:
            out.add(vcls)
    return out


def entry_for_memory_class(entry: Optional[Dict[str, Any]], cls: Optional[int]) -> Optional[Dict[str, Any]]:
    """The certification that covers this memory class — the entry itself when
    its primary proof was made at that class, the variant filed under it
    otherwise — or None. None for an unknown class: an engine that cannot say
    which card it is on cannot say it is covered."""
    if entry is None or cls is None:
        return None
    if proof_memory_class(entry.get("proof")) == cls:
        return entry
    var = (entry.get("variants") or {}).get(_variant_slot(cls))
    if var and proof_memory_class(var.get("proof")) == cls:
        return var
    return None


def proof_records_clock(proof: Optional[Dict[str, Any]]) -> bool:
    """True when the proof says the clock every card ran at (`machine.clocks_mhz`).
    5 628 proofs of 2026-09-07 do not: made before the clock door, at a frequency
    nothing recorded — their timings read 1.176× those of a proof at the protocol
    clock (2026-09-14, register 56, adjudicated paragraph)."""
    return bool(((proof or {}).get("machine") or {}).get("clocks_mhz"))


def entry_covers(entries: Dict[str, Dict[str, Any]], ktext: str, cls: Optional[int],
                 need_clock: bool = False, need_generator: Optional[str] = None) -> bool:
    """`--only-missing`'s question, asked per memory class; with `need_clock`
    (`--reprove-unclocked`) a certification whose proof records no clock does
    not count as coverage — it is re-proven at the protocol clock; with
    `need_generator` (`--reprove-generator`, the running code generator's label,
    e.g. `triton 3.8.0`) a certification proven under another generator does
    not count either — a Triton upgrade changes the code it emits, so every
    setting is re-proven under the new one (owner, 2026-09-16)."""
    cert = entry_for_memory_class(entries.get(ktext), cls)
    if cert is None:
        return False
    if need_clock and not proof_records_clock(cert.get("proof")):
        return False
    if need_generator is not None and proof_backend(cert.get("proof")) != need_generator:
        return False
    return True


def file_certification(entries: Dict[str, Dict[str, Any]], ktext: str, cert: Dict[str, Any]) -> None:
    """Place a fresh certification in a file's entries by the class its proof
    names: the primary slot when the key is new or the class is the primary's,
    a variant slot otherwise. A proof that cannot say its card is refused — an
    entry no card is ever served is not worth writing. A legacy primary whose
    class is unknown (a rig-wide proof) is REPLACED by the fresh one: it served
    no card, and its config, deviation and timings leave with it — said here
    because a discarded measurement should be discarded on purpose."""
    cls = proof_memory_class(cert.get("proof"))
    if cls is None:
        raise ValueError("the certification's proof does not say which card's memory it was made on "
                         "(no machine.device.memory_mb): refused — an entry without a memory class is served to no card")
    body = {"config": cert["config"], "proof": cert["proof"], "excluded": cert.get("excluded") or []}
    cur = entries.get(ktext)
    if cur is None or proof_memory_class(cur.get("proof")) in (cls, None):
        variants = (cur or {}).get("variants")
        entries[ktext] = body if not variants else {**body, "variants": variants}
        entries[ktext].get("variants", {}).pop(_variant_slot(cls), None)
        if "variants" in entries[ktext] and not entries[ktext]["variants"]:
            del entries[ktext]["variants"]
        return
    cur.setdefault("variants", {})[_variant_slot(cls)] = body


def executing_memory_class(args=None, device_idx: Optional[int] = None) -> Optional[int]:
    """The memory class of the card a launch executes on: the device of its
    first device-resident tensor argument (or the index given), looked up in
    the Prism hardware profile in force — a static profile read, never a
    driver query in the hot path. Without an index, the class only when every
    device of the profile shares one; on a heterogeneous rig that is None."""
    if device_idx is None and args:
        for a in args:
            idx = getattr(a, "_device_idx", None)
            if idx is not None and getattr(a, "_device", None) != "cpu":
                device_idx = int(idx)
                break
    try:
        from neurobrix.kernels.wrappers import get_hardware_profile
        prof = get_hardware_profile()
    except Exception:
        return None
    devices = getattr(prof, "devices", None) if prof is not None else None
    if not devices:
        return None
    if device_idx is not None:
        dev = next((d for d in devices if getattr(d, "index", None) == int(device_idx)), None)
        return memory_class_gb(getattr(dev, "memory_mb", None)) if dev is not None else None
    classes = {memory_class_gb(getattr(d, "memory_mb", None)) for d in devices}
    return classes.pop() if len(classes) == 1 else None


def _validate_certification(where: str, entry: Any, fmt: str, expected_tol, key, stem_dtype, problems: List[str]) -> None:
    """The per-certification checks, shared by an entry's primary and its variants."""
    if not isinstance(entry, dict):
        problems.append(f"{where}: not an object")
        return
    cfg = entry.get("config")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("kwargs"), dict) or "num_warps" not in cfg or "num_stages" not in cfg:
        problems.append(f"{where}: no config (kwargs, num_warps, num_stages)")
    proof = entry.get("proof")
    if not isinstance(proof, dict):
        problems.append(f"{where}: no proof")
        return
    missing = [f for f in _REQUIRED[fmt] if f not in proof]
    if missing:
        problems.append(f"{where}: proof without {', '.join(missing)}")
        return
    try:
        dev, tol = float(proof["deviation"]), float(proof["tolerance"])
    except (TypeError, ValueError):
        problems.append(f"{where}: deviation/tolerance are not numbers")
        return
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


def validate(doc: Any, path: Path, tuner=None) -> List[str]:
    """Every reason this file cannot be trusted; empty when its proof re-reads.

    Checked: the format, that vendor/profile/kernel/dtype inside match the path,
    that each entry has a config and a complete proof, that the proof's
    deviation does not exceed its tolerance, that the tolerance is the
    profile's for that dtype, that every excluded setting carries a deviation
    ABOVE the tolerance, that the key parses and (with the tuner) has the
    kernel's arity and the file's dtype, that the date parses; and that every
    variant is filed under the memory class its own proof names."""
    problems: List[str] = []
    fmt = doc.get("format") if isinstance(doc, dict) else None
    if fmt not in _FORMATS:
        return [f"format is not one of {list(_FORMATS)!r}"]
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
        _validate_certification(where, entry, fmt, expected_tol, key, stem_dtype, problems)
        if not isinstance(entry, dict):
            continue
        variants = entry.get("variants")
        if variants is None:
            continue
        if not isinstance(variants, dict):
            problems.append(f"{where}: variants is not an object")
            continue
        for slot, var in variants.items():
            vwhere = f"{where} variant {slot}"
            _validate_certification(vwhere, var, fmt, expected_tol, key, stem_dtype, problems)
            vcls = proof_memory_class(var.get("proof")) if isinstance(var, dict) else None
            if vcls is None:
                problems.append(f"{vwhere}: its proof does not say which card's memory it was made on")
            elif _variant_slot(vcls) != str(slot):
                problems.append(f"{vwhere}: filed under {slot}, its proof was made on a {vcls} GB card")
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


def lookup(kernel_qual: str, tuner, key: tuple, ignore_switch: bool = False,
           memory_class: Optional[int] = None, any_class: bool = False) -> Optional[Dict[str, Any]]:
    """The certification for this profile, kernel, dtype, shape AND the memory
    class of the executing card, or None. An entry proven on another class is
    not served (`entry_for_memory_class`); `memory_class` None means the card
    is unknown, and unknown is served nothing. `any_class`: the raw entry
    whatever its class — for a report that must name what exists, never to
    serve. `ignore_switch`: read the directory even when
    NBX_AUTOTUNE_CERTIFIED=off — the switch stops APPLYING a certification,
    never REPORTING against one."""
    if not enabled() and not ignore_switch:
        return None
    ident = active_profile()
    if ident is None:
        return None
    entries = _load(ident[0], ident[1], kernel_qual, output_dtype(tuner, key), tuner)
    if not entries:
        return None
    entry = entries.get(key_repr(key))
    if any_class:
        return entry
    return entry_for_memory_class(entry, memory_class)


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


def apply(kernel_qual: str, tuner, key: tuple, lookup_key: Optional[tuple] = None,
          memory_class: Optional[int] = None) -> bool:
    """Put the certified config in the tuner's cache for `key` — no sweep, no
    screen. False when the directory has nothing for it ON THIS CARD's memory
    class. `lookup_key`: the key the directory is read at when it differs from
    the tuner's (a promoted operand keyed by its computed dtype,
    `computed_key`). `memory_class`: the executing card's, from
    `executing_memory_class(args)` at a launch site; None here means "the
    class every device of the profile in force shares", which is None — served
    nothing — on a heterogeneous rig where the launch site did not say."""
    if memory_class is None:
        memory_class = executing_memory_class()
    entry = lookup(kernel_qual, tuner, lookup_key or key, memory_class=memory_class)
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
    # The card is not known at seed time. When every device of the profile in
    # force shares one memory class (a pinned card, a homogeneous rig) the
    # class is that one and the directory is applied here. On a heterogeneous
    # rig it is None: a seeded key the directory certifies for SOME class is
    # then EVICTED, its local config remembered, so the launch site — which
    # knows the tensor's card — decides: the certified setting for that
    # card's class, or the local config put back (`reseed_evicted`).
    mcls = executing_memory_class()
    n = 0
    for qual, tuner in tuners:
        cache = getattr(tuner, "cache", None)
        if not cache:
            continue
        for key in list(cache.keys()):
            if not isinstance(key, tuple):
                continue
            twin = None
            try:
                entry = lookup(qual, tuner, key, any_class=True)
                if entry is None:
                    twin = computed_key(tuner, key)       # a widened-on-load operand, by the profile's rule
                    entry = lookup(qual, tuner, twin, any_class=True) if twin else None
            except Exception:
                entry = None
            if entry is None:
                continue
            local = cache.pop(key, None)
            if mcls is not None:
                if apply(qual, tuner, key, lookup_key=twin, memory_class=mcls):
                    n += 1
                elif local is not None:
                    cache[key] = local                       # certified elsewhere, not for this card: the local config stays
            elif local is not None:
                _EVICTED[(qual, key)] = (local, twin)
    _SERVED["local"] = max(0, _SERVED["local"] - n)
    return n


#: (kernel, key) -> (local triton.Config, twin key): seeds evicted by
#: `override_seeded` on a rig whose card was unknown at seed time.
_EVICTED: Dict[Tuple[str, tuple], Tuple[Any, Optional[tuple]]] = {}


def served_evicted(kernel_qual: str, key: tuple) -> None:
    """A key evicted at seed time that the launch site served CERTIFIED: it
    is no longer a local key (the seed count was taken at seed time)."""
    if _EVICTED.pop((kernel_qual, key), None) is not None:
        _SERVED["local"] = max(0, _SERVED["local"] - 1)


def reseed_evicted(kernel_qual: str, tuner, key: tuple) -> bool:
    """Put back the local replay config `override_seeded` evicted for this
    key when the directory does not cover the executing card's class — the
    machine's own earlier sweep, not a sweep again. True when one was put back."""
    rec = _EVICTED.pop((kernel_qual, key), None)
    if rec is None:
        return False
    cache = getattr(tuner, "cache", None)
    if cache is None:
        return False
    cache[key] = rec[0]
    return True


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
    # Say WHY when an entry exists and is not served: proven on another card's
    # memory. A silence here would read as "the directory never had it".
    why = ""
    try:
        raw = lookup(kernel_qual, tuner, key, ignore_switch=True, any_class=True)
        if raw is not None:
            covered = sorted(covered_memory_classes(raw))
            here = executing_memory_class(list((getattr(tuner, "nargs", None) or {}).values()) or None)
            why = (f" — certified for {', '.join(f'{c} GB' for c in covered) or 'no known memory class'}, "
                   f"this card is {f'{here} GB' if here is not None else 'of unknown memory'}, not served (register 56)")
    except Exception as exc:                # the reason failed to compute: say that, never an empty reason
        why = f" — (could not read what the directory holds for it: {exc})"
    print(f"[autotune] no certified setting for {kernel_short(kernel_qual)} {output_dtype(tuner, key)} "
          f"({describe_key(tuner, key)}) on {where}{why}: sweeping at runtime with the consensus screen; the result is "
          f"kept in {_local}, never in the engine's directory — `neurobrix autotune certify --profile "
          f"{prof[1] if prof else '<profile>'}` certifies it", flush=True)


def served() -> Dict[str, int]:
    """How many keys this process took from the directory, how many it swept, how many came from the local replay cache."""
    return dict(_SERVED)


def note_local(n: int) -> None:
    """Keys the local replay cache seeded into the autotuners (a previous runtime sweep on this machine)."""
    _SERVED["local"] += int(n or 0)


def reset() -> None:
    _LOADED.clear(); _REFUSED.clear(); _ANNOUNCED.clear(); _EVICTED.clear()
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
    # the proof's sweep arm is where the two can disagree — against the certification
    # made for THIS card's memory class (a contradiction with another class's proof is not one)
    mcls = executing_memory_class(list((getattr(tuner, "nargs", None) or {}).values()) or None)
    entry = lookup(qual, tuner, key, ignore_switch=True, memory_class=mcls)
    if entry is None:
        raw = lookup(qual, tuner, key, ignore_switch=True, any_class=True)
        if raw is not None and mcls is None:
            # a certification exists and this card's class is unknown: neither a
            # contradiction nor a clean pass — said, never dropped
            print(f"[AUTOTUNE_SCREEN] UNADJUDICATED — {kernel_short(qual)} {output_dtype(tuner, key)} "
                  f"({describe_key(tuner, key)}): the directory certifies this shape for "
                  f"{', '.join(f'{c} GB' for c in sorted(covered_memory_classes(raw))) or 'no known memory class'} and the "
                  f"executing card's memory class is unknown, so the runtime exclusion cannot be read against it.", flush=True)
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
