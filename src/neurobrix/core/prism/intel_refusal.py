"""An Intel GPU refused at RUN time (the owner, 2026-11-10 13:09; the Dell + the Mac 14:3x, the Dell's review 16:0x).
The Intel backend is experimental — not yet validated on hardware by the NeuroBrix lab — so a chip whose Level Zero
driver is present is TRIED, and a Level Zero error (`nbxtensor.backend.level_zero.IntelDeviceError`: the call, the
result by name, the build log, the device) is handled here, never silently and never said falsely:

1. only a result that says THIS chip under THIS driver cannot do what was asked is a refusal of the chip (`CANNOT`);
   any other result (a device lost, memory, an argument our launcher got wrong) is the run's own failure, named;
2. only the device the error names is refused (`refused_devices`): another Intel GPU of the machine is left alone;
3. the refusal is SAID with its cause, the experimental label and where to report it (`sentence`);
4. the chip is RECORDED as refused under the Level Zero driver and the nbxtensor version it failed with (`record`: one
   file per chip under hardware/intel_runtime_refusals/, written atomically) and the generated profile that planned
   it is removed, so the machine is detected again: detection lists the chip as a refused device with that reason
   (`recorded`), and Prism plans without it and says so in the plan (its `refused_devices`, the existing path);
5. `neurobrix run` starts the request again ONCE in a fresh process (a backend is process-global): the child carries
   `NBX_INTEL_REFUSED` (the sentence it prints first), and a refusal in a process that carries it is the run's failure
   by construction. No restart when the profile was GIVEN (`--hardware`: the user's, it names the chip) or when no
   device can be named: the error stands, said.

The record is not forever: another driver version or another nbxtensor version tries the chip again (`stale`), and so
does a machine whose Level Zero loader is gone (detection then refuses the chip for the missing loader, its own
reason). No torch; Level Zero is not asked unless a record or an error exists."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

ENV = "NBX_INTEL_REFUSED"
REPORT_URL = "https://github.com/NeuroBrix/neurobrix/issues/new?template=intel_gpu_report.yml"
LABEL = "experimental backend, not yet validated on hardware by the NeuroBrix lab"
#: the build log's share of the sentence, in characters (the whole log is in the error and in the daemon's log)
LOG_CHARS = 300
#: ze_result_t values (oneapi-src/level-zero include/ze_api.h, `ze_result_t`, each with its `[Core]` comment) that say
#: the driver cannot do what was asked of this device — the same answer at every run under the same driver and library,
#: which is exactly what a record is keyed by: "generic error code for unsupported versions / features", "error
#: occurred when building / linking module, see build log for details", "native binary is not supported by the
#: device". Every other result is not a verdict on the chip: lost device, memory, uninitialised driver, an argument or
#: a handle our own launcher got wrong (a NeuroBrix defect to report, not a chip to retire).
CANNOT = frozenset({"ZE_RESULT_ERROR_UNSUPPORTED_VERSION", "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE",
                    "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE", "ZE_RESULT_ERROR_MODULE_LINK_FAILURE",
                    "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY"})


def _record_dir() -> Path:
    from neurobrix.core.prism.autodetect import HARDWARE_DIR
    return Path(HARDWARE_DIR) / "intel_runtime_refusals"


def _versions() -> Optional[Dict[str, str]]:
    """The Level Zero driver's version and nbxtensor's — what a refusal is recorded under; None when the machine has
    no Level Zero loader any more (no record can hold there: detection refuses the chip for the missing loader)."""
    import nbxtensor
    from nbxtensor.backend import level_zero
    if level_zero.load_library() is None:
        return None
    return {"driver": level_zero.runtime().driver_version(), "nbxtensor": str(nbxtensor.__version__)}


def _read() -> Dict[str, Dict[str, str]]:
    """{pci id: its record}; a file that is not a record (torn by a crash) is said and removed."""
    out: Dict[str, Dict[str, str]] = {}
    root = _record_dir()
    for path in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            entry = json.loads(path.read_text())
            out[str(entry["pci_id"])] = {k: str(entry[k]) for k in ("driver", "nbxtensor", "reason")}
        except (ValueError, KeyError, TypeError) as e:
            _log.warning(f"{path} is not an Intel refusal record ({e}): removed")
            path.unlink()
    return out


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
    """The Intel devices the GENERATED profile `hardware_id` plans (its file's own entries: index, model, pci_id)."""
    import yaml
    from neurobrix.core.prism.autodetect import HARDWARE_DIR
    path = Path(HARDWARE_DIR) / f"{hardware_id}.yml"
    if not hardware_id or not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return [d for d in data.get("devices") or [] if str(d.get("brand")).lower() == "intel" and d.get("pci_id")]


def refused_devices(devices: List[Dict[str, Any]], err) -> List[Dict[str, Any]]:
    """The device `err` names among `devices` (the profile's index is the runtime's); the only one when the error
    names none; [] when it names none of several, or one the profile does not plan — nothing is refused by guess."""
    if err.device is None:
        return list(devices) if len(devices) == 1 else []
    return [d for d in devices if d.get("index") == err.device]


def sentence(devices: List[Dict[str, Any]], err, then: str) -> str:
    names = ", ".join(f"{d.get('model') or 'Intel GPU'} ({d['pci_id']})" for d in devices) or "the Intel GPU"
    log = f" [{err.log[:LOG_CHARS]}]" if getattr(err, "log", "") else ""
    return (f"Intel GPU {names} refused at run time: {err.call}: {err.result}{log} — {LABEL}; {then} "
            f"Please report it: {REPORT_URL} with the output of `neurobrix doctor`.")


def record(devices: List[Dict[str, Any]], reason: str, hardware_id: Optional[str]) -> None:
    """Record each device as refused under the versions in force (one file per chip, replaced atomically), and remove
    the generated profile that planned it and the machine's shared one, so the next start detects the machine again
    (their measured facts are measured again: facts of a machine whose plan changes)."""
    from neurobrix.core.prism import autodetect
    versions = _versions()
    if versions is None:
        raise RuntimeError("ZERO FALLBACK: an Intel refusal is recorded under its Level Zero driver's version, and the "
                           "Level Zero loader is not installed any more")
    root = _record_dir()
    root.mkdir(parents=True, exist_ok=True)
    for d in devices:
        path = root / f"{str(d['pci_id']).replace(':', '_')}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(dict(versions, pci_id=str(d["pci_id"]), reason=reason), indent=1, sort_keys=True))
        os.replace(tmp, path)
    for p in ([Path(autodetect.HARDWARE_DIR) / f"{hardware_id}.yml"] if str(hardware_id or "").startswith("default")
              else []) + [Path(autodetect.DEFAULT_PROFILE_PATH)]:
        if p.exists():
            p.unlink()


def recorded(pci_id: str) -> Optional[str]:
    """The reason `pci_id` is refused, when a record holds under the versions in force (detection asks, per chip)."""
    entry = _read().get(str(pci_id))
    if not entry:
        return None
    now = _versions()
    if now is None or (entry["driver"], entry["nbxtensor"]) != (now["driver"], now["nbxtensor"]):
        return None
    return (f"{entry['reason']} (recorded under Level Zero driver {entry['driver']}, nbxtensor {entry['nbxtensor']}: "
            f"another version of either tries it again)")


def stale() -> bool:
    """Records made under another driver or nbxtensor version, or on a machine whose loader is gone: dropped — the
    caller removes the generated profile so the chip is tried again. False, at no cost, when nothing is recorded."""
    data = _read()
    if not data:
        return False
    now = _versions()
    old = [k for k, v in data.items() if now is None or (v["driver"], v["nbxtensor"]) != (now["driver"], now["nbxtensor"])]
    for pci in old:
        (_record_dir() / f"{pci.replace(':', '_')}.json").unlink()
    return bool(old)


def handle(exc: BaseException, hardware_id: Optional[str], hardware_given: bool, argv: List[str]) -> None:
    """`neurobrix run`'s answer to an exception. Returns when the exception is not a refusal of an Intel chip this
    module may act on (the caller raises it as it is — after the named sentence, when one is due). Otherwise the
    refusal is said and recorded and the request starts again in a fresh process: this call does not return."""
    err = intel_error(exc)
    if err is None or err.result not in CANNOT or os.environ.get(ENV):
        return
    devices = refused_devices(planned_devices(hardware_id), err)
    if hardware_given:
        print(sentence(devices, err, "the profile given with --hardware plans this chip: run without --hardware (the "
                                     "machine is detected, the chip refused by name) or name a profile without it."),
              file=sys.stderr, flush=True)
        return
    if not devices:               # before a plan, or a device the profile does not hold: nothing is refused by guess
        return
    said = sentence(devices, err, "running again without it.")
    record(devices, f"refused at run time: {err.call}: {err.result}", hardware_id)
    print(said, file=sys.stderr, flush=True)
    sys.stdout.flush()
    cmd, env = [sys.executable] + list(argv), dict(os.environ, **{ENV: said})
    if sys.platform == "win32":   # Windows has no exec that replaces the process: the child's status is this one's
        sys.exit(subprocess.run(cmd, env=env).returncode)
    os.execve(sys.executable, cmd, env)


def announce() -> None:
    """The first line of a process started again after a refusal: why it runs again."""
    said = os.environ.get(ENV)
    if said:
        print(f"Running again — {said}", file=sys.stderr, flush=True)


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
                   f"{entry['nbxtensor']} — planned without it until either version changes; please report it: "
                   f"{REPORT_URL}")
    return out
