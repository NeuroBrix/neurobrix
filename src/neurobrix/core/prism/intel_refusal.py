"""An Intel GPU refused at RUN time (the owner, 2026-11-10 13:09; the Dell + the Mac 14:3x). The Intel backend is
experimental — not yet validated on hardware by the NeuroBrix lab — so a chip whose Level Zero driver is present is
TRIED, and its first Level Zero error (`nbxtensor.backend.level_zero.IntelDeviceError`: the call, the result by name,
the build log) is handled here, never silently:

1. the refusal is SAID, with its cause, the experimental label and where to report it (`sentence`);
2. the chip is RECORDED as refused under the Level Zero driver and the nbxtensor version it failed with
   (`record`: hardware/intel_runtime_refusals.json), and this environment's generated profiles are removed so the
   machine is detected again: detection then lists the chip as a refused device with that reason (`recorded`), and
   Prism plans on the CPU and says so in the plan (its `refused_devices`, the existing path);
3. the entry point starts the request again ONCE in a fresh process (a backend is process-global): the child carries
   `NBX_INTEL_REFUSED` (the sentence it prints first), and an Intel refusal in a process that carries it is the run's
   failure — by construction, no counter.

The record is not forever: another driver version or another nbxtensor version tries the chip again (`stale` at the
next start removes the record and the profile). No torch, no Level Zero import unless a record or an error exists."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ENV = "NBX_INTEL_REFUSED"
REPORT_URL = "https://github.com/NeuroBrix/neurobrix/issues/new?template=intel_gpu_report.yml"
LABEL = "experimental backend, not yet validated on hardware by the NeuroBrix lab"


def _record_path() -> Path:
    from neurobrix.core.prism.autodetect import HARDWARE_DIR
    return Path(HARDWARE_DIR) / "intel_runtime_refusals.json"


def _versions() -> Dict[str, str]:
    """The Level Zero driver's version and nbxtensor's: what a refusal is recorded under."""
    import nbxtensor
    from nbxtensor.backend import level_zero
    return {"driver": level_zero.runtime().driver_version(), "nbxtensor": str(nbxtensor.__version__)}


def _read() -> Dict[str, Dict[str, str]]:
    path = _record_path()
    return json.loads(path.read_text()) if path.exists() else {}


def intel_error(exc: Optional[BaseException]):
    """The IntelDeviceError `exc` is or was raised from, or None (the module is looked up only when it was loaded: a
    process that never touched Level Zero cannot hold one)."""
    mod = sys.modules.get("nbxtensor.backend.level_zero")
    seen = set()
    while mod is not None and exc is not None and id(exc) not in seen:
        if isinstance(exc, mod.IntelDeviceError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def planned_devices(hardware_id: Optional[str]) -> List[Dict[str, Any]]:
    """The Intel devices the profile `hardware_id` plans (its file's own entries: model, pci_id)."""
    import yaml
    from neurobrix.core.prism.autodetect import HARDWARE_DIR
    path = Path(HARDWARE_DIR) / f"{hardware_id}.yml"
    if not hardware_id or not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return [d for d in data.get("devices") or [] if str(d.get("brand")).lower() == "intel"]


def sentence(devices: List[Dict[str, Any]], err) -> str:
    names = ", ".join(f"{d.get('model') or 'Intel GPU'} ({d.get('pci_id') or 'pci id unknown'})" for d in devices) \
        or "the Intel GPU"
    log = f" [{err.log[:300]}]" if getattr(err, "log", "") else ""
    return (f"Intel GPU {names} refused at run time: {err.call}: {err.result}{log} — {LABEL}; running on the CPU. "
            f"Please report it: {REPORT_URL} with the output of `neurobrix doctor`.")


def record(devices: List[Dict[str, Any]], reason: str) -> None:
    """Record each device (by PCI id) as refused under the versions in force, and remove this environment's generated
    profiles so the next start detects the machine again (their measured facts are measured again: a fact of a
    machine whose GPU is no longer planned)."""
    from neurobrix.core.prism import autodetect
    data, versions = _read(), _versions()
    for d in devices:
        data[str(d.get("pci_id"))] = dict(versions, reason=reason)
    path = _record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
    tag = autodetect._visible_set_tag()
    for p in ([Path(autodetect.HARDWARE_DIR) / f"default-{tag}.yml"] if tag else []) + [Path(autodetect.DEFAULT_PROFILE_PATH)]:
        if p.exists():
            p.unlink()


def recorded(pci_id: str) -> Optional[str]:
    """The reason `pci_id` is refused, when a record holds under the versions in force (detection asks, per chip)."""
    entry = _read().get(str(pci_id))
    if not entry:
        return None
    now = _versions()
    if (entry.get("driver"), entry.get("nbxtensor")) != (now["driver"], now["nbxtensor"]):
        return None
    return (f"{entry['reason']} (recorded under Level Zero driver {entry['driver']}, nbxtensor {entry['nbxtensor']}: "
            f"another version of either tries it again)")


def stale() -> bool:
    """A record made under another driver or nbxtensor version: it is dropped — the caller removes the generated
    profile so the chip is tried again. False, at no cost, when nothing is recorded."""
    data = _read()
    if not data:
        return False
    now = _versions()
    kept = {k: v for k, v in data.items() if (v.get("driver"), v.get("nbxtensor")) == (now["driver"], now["nbxtensor"])}
    if len(kept) == len(data):
        return False
    path = _record_path()
    path.write_text(json.dumps(kept, indent=1, sort_keys=True)) if kept else path.unlink()
    return True


def handle(exc: BaseException, devices: List[Dict[str, Any]], argv: List[str]) -> bool:
    """The entry point's answer to an exception: False when it holds no Intel refusal (the caller raises it as it
    is) or when this process already runs again after one (the run's failure). Otherwise the refusal is said and
    recorded and the request starts again in a fresh process — `os.execve` does not return."""
    err = intel_error(exc)
    if err is None or os.environ.get(ENV):
        return False
    said = sentence(devices, err)
    print(said, file=sys.stderr, flush=True)
    record(devices, f"refused at run time: {err.call}: {err.result}")
    os.execve(sys.executable, [sys.executable] + list(argv), dict(os.environ, **{ENV: said}))
    return True


def announce() -> None:
    """The first line of a process started again after a refusal: why it runs again."""
    said = os.environ.get(ENV)
    if said:
        print(f"Running again on the CPU — {said}", file=sys.stderr, flush=True)


def doctor_lines() -> List[str]:
    """What `neurobrix doctor` says of Intel GPUs: the experimental label when a generated profile of this machine
    plans one, and every chip refused at run time with its record (read from the files: Level Zero is not asked)."""
    from neurobrix.core.prism.autodetect import HARDWARE_DIR
    out = []
    planned = [d for p in sorted(Path(HARDWARE_DIR).glob("default*.yml")) for d in planned_devices(p.stem)]
    if planned:
        names = sorted({f"{d.get('model')} ({d.get('pci_id')})" for d in planned})
        out.append(f"Intel GPU:      {', '.join(names)} — {LABEL}; reports welcome: {REPORT_URL}")
    for pci, entry in sorted(_read().items()):
        out.append(f"Intel GPU:      {pci} {entry['reason']} under Level Zero driver {entry['driver']}, nbxtensor "
                   f"{entry['nbxtensor']} — it runs on the CPU until either version changes; please report it: {REPORT_URL}")
    return out
