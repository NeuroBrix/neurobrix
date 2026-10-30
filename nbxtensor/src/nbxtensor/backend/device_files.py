"""The library's device files: the kernel-relevant description of each device, one file per device
(`devices/<vendor>/<name>.yml`, beside this module) — API.md section 11.

What a kernel reads of a device lives here: its identification (`compute_capability`, `compute_capability_matches`,
`architecture`), the tiles, the shared memory and registers, the warp size, the matrix unit, the dispatch limits, the
autotune's screen tolerances and buckets. The list is the device-key census (`docs/nbxtensor/device_keys.txt`).

An application may keep its own facts about the same devices in its own files (NeuroBrix: Prism's budgets and measured
rates, in `config/vendors/<vendor>/<name>.yml`) and give the library their directory (`nbxtensor.host`,
`vendors_dir`, optional). The two halves of one device are MERGED, and a key in both is refused by name: a fact is
written once. With no application, the library's half is the whole description.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

DEVICES = Path(__file__).resolve().parent / "devices"

__all__ = ["DEVICES", "names", "read", "merge", "merged", "describe", "param", "in_force", "for_arch", "forget"]


@functools.lru_cache(maxsize=None)
def _parsed(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read(vendor: str, name: str) -> Dict[str, Any]:
    """The library's half of device `vendor/name` (a copy), `{}` when the library describes none."""
    path = DEVICES / vendor / f"{name}.yml"
    return copy.deepcopy(_parsed(str(path))) if path.is_file() else {}


def merge(library: Dict[str, Any], application: Dict[str, Any], where: str = "") -> Dict[str, Any]:
    """The two halves of one device as one description; a key written in both is refused by name."""
    out = copy.deepcopy(library)
    for key, value in application.items():
        if key not in out:
            out[key] = copy.deepcopy(value)
        elif isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge(out[key], value, f"{where}{key}.")
        else:
            raise ValueError(f"device file {where}{key}: written in both the library's device file and the "
                             f"application's — a fact is written once (nbxtensor/backend/devices)")
    return out


def _application_dir() -> Optional[Path]:
    from nbxtensor import host
    if "vendors_dir" not in host.provided():
        return None
    return Path(host.need("vendors_dir")())


def names() -> Iterator[Tuple[str, str, Path]]:
    """(vendor, name, path) of every device described by the library or by the application, once each."""
    seen = {}
    roots = [DEVICES] + ([d] if (d := _application_dir()) is not None else [])
    for root in roots:
        for path in sorted(root.glob("*/*.yml")):
            seen.setdefault((path.parent.name, path.stem), path)
    for (vendor, name), path in sorted(seen.items()):
        yield vendor, name, path


def merged(vendor: str, name: str) -> Dict[str, Any]:
    """Device `vendor/name` whole: the library's half merged with the application's, when it gave its directory."""
    app = _application_dir()
    app_half = {}
    if app is not None and (app / vendor / f"{name}.yml").is_file():
        app_half = copy.deepcopy(_parsed(str(app / vendor / f"{name}.yml")))
    return merge(read(vendor, name), app_half, f"{vendor}/{name}: ")


@functools.lru_cache(maxsize=256)
def _described(vendor: str, name: str, app: Optional[str]) -> Dict[str, Any]:
    app_half = {}
    if app is not None and (Path(app) / vendor / f"{name}.yml").is_file():
        app_half = _parsed(str(Path(app) / vendor / f"{name}.yml"))
    lib_path = DEVICES / vendor / f"{name}.yml"
    if not app_half and not lib_path.is_file():
        known = ", ".join(f"{v}/{n}" for v, n, _ in names()) or "none"
        raise LookupError(f"ZERO FALLBACK: no device file describes {vendor}/{name} (the library's "
                          f"backend/devices/{vendor}/{name}.yml, nor the application's). Known: {known}")
    return merge(_parsed(str(lib_path)) if lib_path.is_file() else {}, app_half, f"{vendor}/{name}: ")


def describe(vendor: str, name: str) -> Dict[str, Any]:
    """Device `vendor/name` whole (both halves), refused by name when neither describes it — the reader a kernel
    asks for its device's facts. Merged once per process; a copy is handed out."""
    app = _application_dir()
    return copy.deepcopy(_described(vendor, name, None if app is None else str(app)))


def param(vendor: str, name: str, section: str, key: str, default):
    """One `<section>.<key>` of device `vendor/name`, `default` when the device states no such key (the device
    itself is refused by name when nothing describes it) — the per-op reader: no copy of the whole description."""
    app = _application_dir()
    value = (_described(vendor, name, None if app is None else str(app)).get(section) or {}).get(key, _ABSENT)
    if value is _ABSENT:
        return default
    return copy.deepcopy(value) if isinstance(value, (dict, list)) else value


_ABSENT = object()


def in_force() -> Dict[str, Any]:
    """The description of the device the launch target names (both halves merged), `{}` when no device file matches
    it — what every kernel setting reads, resolved once per process."""
    from nbxtensor.ops import _configs
    return _configs.active_vendor_profile()


def for_arch(arch: str) -> Dict[str, Any]:
    """The description an arch string resolves to, read WITHOUT a launch target (a caller the target depends on, or
    a derivation that binds no device), and recorded as the one in force."""
    from nbxtensor.ops import _configs
    return _configs.vendor_profile_for_arch(arch)


def forget() -> None:
    """Forget the description in force: the next reader resolves it again (after the launch target is bound)."""
    from nbxtensor.ops import _configs
    _configs._ACTIVE_PROFILE.clear()
