"""What a launch keeps of what it measured on this machine, and reuses at the next one (the supervisor, row 167: a
measured fact is kept once measured and never paid twice; a measure that cannot conclude keeps its best figure with its
spread, and is replaced by one that concludes).

Four facts, each measured by a primitive of the library and kept by the engine on the machine's generated profile:
the expandable pool's chunk of each card model (`set_chunk`, `sweep_chunks`), the weight reader's width after a load
(`note_read_width`, `kept_read_width`), and the reload rate of a stored dtype with its spread when its draws did not
agree (`reload_may_be_written`, `put_reload`). The profile's file, its directory and the active hardware id are
`autodetect`'s: read there at each call, never copied here.
"""
from __future__ import annotations

import functools
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def set_chunk(block: dict, measured) -> None:
    """A sweep's answer, (chunk, spread) as `facts_probe.vmm_chunk_bytes` gives it, into a device's measured block: an
    unconcluded sweep's best candidate with its spread beside it (`vmm_chunk_spread`), never a 0 written as if measured
    (row 167); 0 only for a device with no VMM."""
    chunk, spread = measured
    block["vmm_chunk_bytes"] = int(chunk or 0)
    block["vmm_chunk_under"] = card_key()          # what it was measured under: another driver or library measures again
    if spread is not None:
        block["vmm_chunk_spread"] = dict(spread)
    else:
        block.pop("vmm_chunk_spread", None)


def _models_to_sweep(devices, first, measured_of) -> Dict[Any, int]:
    """{card model: the index of the card to sweep it on} for every model of the profile whose block states no chunk:
    card 0 for card 0's model, else the model's first card (by its `index`)."""
    todo: Dict[Any, int] = {}
    for i, d in enumerate(devices):
        own = first if d.get("model") == devices[0].get("model") else measured_of(d)
        if "vmm_chunk_bytes" in own or d.get("model") in todo:
            continue
        if i and "index" not in d:
            raise ValueError(f"profile {d.get('model')!r}: a device of this card model has no 'index' fact, so its "
                             f"expandable-pool chunk cannot be measured on its own card; add 'index' to every device "
                             f"of the profile's devices list")
        todo[d.get("model")] = 0 if i == 0 else int(d["index"])
    return todo


def sweep_chunks(devices, first, measured_of, q, replay) -> Dict[Any, Any]:
    """{card model: its chunk sweep's (chunk, spread)} for every model of the profile whose block states no chunk: each
    swept on ITS OWN first card (every replay child sees that card alone), the models SIDE BY SIDE, a thread each;
    identical cards answer once. The rack's fresh 4-card home ran its 48 replay children one after another: 1024 s of
    its 1236 s to ready (py-spy, 2026-11-19)."""
    from concurrent.futures import ThreadPoolExecutor
    from neurobrix.core.prism import facts_probe as P
    todo = _models_to_sweep(devices, first, measured_of)
    if not todo:
        return {}

    def sweep(item):
        model, index = item
        ahead = replay[0] if replay and model == devices[0].get("model") else None
        return model, P.vmm_chunk_bytes(index, q["vmm_reps"], q["vmm_ks"], off_first=ahead)
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        return dict(pool.map(sweep, todo.items()))


@functools.lru_cache(maxsize=None)
def library_key() -> str:
    """The nbxtensor a fact was measured with: its version, and its commit when it is imported from a source tree (a
    dev install, whose version string does not move between commits: the reader's growth rule changed five times in
    one week under 0.1.11). A source tree whose commit cannot be read keys this process only: nothing kept under it is
    reused by another process."""
    import nbxtensor
    version = str(nbxtensor.__version__)
    here = Path(nbxtensor.__file__).resolve().parent
    for up in (here, *here.parents):
        if (up / ".git").exists():
            r = subprocess.run(["git", "-C", str(up), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
            return f"{version}+{r.stdout.strip()}" if r.returncode == 0 and r.stdout.strip() else \
                f"{version}+unread commit, process {os.getpid()}"
    return version


@functools.lru_cache(maxsize=None)
def driver_key() -> str:
    """The GPU drivers the OS states it runs, read from the OS and never from a device runtime: where the OS lists its
    PCI devices (`/sys/bus/pci`), the kernel module bound to each display-class device (class 0x03) with the version
    that module states; elsewhere the OS's own release and build, which ship or name its drivers."""
    root = Path("/sys/bus/pci/devices")
    found = set()
    if root.is_dir():
        for dev in root.iterdir():
            try:
                if not (dev / "class").read_text().strip().startswith("0x03") or not (dev / "driver").exists():
                    continue
                module = (dev / "driver" / "module").resolve().name
                stated = Path("/sys/module") / module / "version"
                found.add(f"{module} {stated.read_text().strip() if stated.exists() else '(no version stated)'}")
            except OSError:
                continue
    return ", ".join(sorted(found)) if found else f"{platform.system()} {platform.release()} {platform.version()}"


def card_key() -> Dict[str, str]:
    """What a card fact (the expandable pool's chunk) and a reload rate were measured under: the driver and the
    library. A fact kept under another is measured again (`drop_stale`)."""
    return {"driver": driver_key(), "library": library_key()}


def storage_key(source) -> str:
    """The storage a weight file is read from: its device number, the mount the OS states for it (source and type: an
    NFS export names its server) and the block device's model where the OS states one."""
    st = os.stat(source)
    if not hasattr(os, "major"):
        return f"device {st.st_dev}"
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    where = f"device {major}:{minor}"
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                fields = line.split()
                if len(fields) > 2 and fields[2] == f"{major}:{minor}" and "-" in fields:
                    sep = fields.index("-")
                    where += f", {fields[sep + 2]} {fields[sep + 1]}"
                    break
    except OSError:
        pass
    block = Path(f"/sys/dev/block/{major}:{minor}")
    if block.exists():
        for model in (block.resolve() / "device" / "model", block.resolve().parent / "device" / "model"):
            if model.exists():
                where += f", {model.read_text().strip()}"
                break
    return where


def read_key(source) -> str:
    """The key the weight reader's width is kept under: the library that measured it, the storage the weights were read
    from (`source`, a weight file or its directory) and the OS kernel. A width is a fact of the three, not of the GPU
    set: a width kept under another key is never reused, the reader starts at one and keeps the new key."""
    return (f"library {library_key()}; storage {storage_key(source)}; "
            f"kernel {platform.system()} {platform.release()}")


#: the facts `drop_stale` drops with a stale key, per block
_CHUNK_FACTS = ("vmm_chunk_bytes", "vmm_chunk_spread", "vmm_chunk_under")
_RELOAD_FACTS = ("weight_reload_gbps", "weight_reload_spread", "weight_reload_source", "weight_reload_under")


def drop_stale(data: dict) -> list:
    """Drops from a generated profile's content `data` the kept facts measured under another driver or library than
    this process's (`card_key`), so the launch measures them again (qa on 1215, the orchestrator's decision: the chunk
    and the reload rates were kept by card model alone and copied into every sibling profile): each device's chunk
    (`vmm_chunk_bytes` with its spread, keyed by `vmm_chunk_under`), the host's reload rates (keyed by
    `weight_reload_under`), and a reader width kept with no key. A fact with no key is stale. Returns what was dropped,
    worded for the launch's account."""
    key, dropped = card_key(), []
    for i, d in enumerate(data.get("devices") or []):
        m = d.get("measured") or {}
        if "vmm_chunk_bytes" in m and m.get("vmm_chunk_under") != key:
            dropped.append(f"card {d.get('index', i)}: vmm_chunk_bytes measured under {m.get('vmm_chunk_under') or 'no key'}")
            for k in _CHUNK_FACTS:
                m.pop(k, None)
    host = data.get("host_measured") or {}
    if host.get("weight_reload_gbps") and host.get("weight_reload_under") != key:
        dropped.append(f"host: weight_reload_gbps measured under {host.get('weight_reload_under') or 'no key'}")
        for k in _RELOAD_FACTS:
            host.pop(k, None)
    if isinstance(host.get("weight_read"), dict) and "workers" in host["weight_read"]:
        dropped.append("host: weight_read kept with no key")
        host.pop("weight_read")
    return dropped


def _profile_path(hardware_id: Optional[str]) -> Optional[Path]:
    """The generated profile's file of `hardware_id`, or None for a hardware that is not a generated profile or has no
    file: the only profiles the engine keeps measured facts on."""
    from neurobrix.core.prism import autodetect as A
    if not hardware_id or not str(hardware_id).startswith("default"):
        return None
    path = A.HARDWARE_DIR / f"{hardware_id}.yml"
    return path if path.exists() else None


def _kept_widths(data: dict) -> Dict[str, dict]:
    """{key: kept width} of a profile's `host_measured.weight_read`; a width kept with no key (before the key, 1215)
    is none."""
    kept = (data.get("host_measured") or {}).get("weight_read") or {}
    return {} if "workers" in kept else {k: v for k, v in kept.items() if isinstance(v, dict)}


def kept_read_width(hardware_id: Optional[str], source=None) -> Optional[int]:
    """The weight reader's width this machine KEPT (`note_read_width`) under the key of the weights at `source`
    (`read_key`: the library, the storage, the kernel), or None — no source, no width kept, or one kept under another
    key: the reader then starts at one and measures (qa on 1215: a width settled under one library or disk was reused
    under another as if measured today)."""
    path = _profile_path(hardware_id)
    if path is None or source is None:
        return None
    kept = _kept_widths(yaml.safe_load(path.read_text()) or {}).get(read_key(source)) or {}
    return int(kept["workers"]) if kept.get("workers") else None


def note_read_width(hardware_id: Optional[str], measured: Optional[dict], source=None) -> bool:
    """What the weight reader measured on this process's loads (`nbxtensor.io.reader.Reader.measured()`), KEPT on a
    generated profile as `host_measured.weight_read`: the width, the low and high rate of its rounds at that width and
    their count, whether its growth settled, the date. The library measures; the engine keeps and reuses (row 167). A
    growth that did not settle gives its best width so far (`measured["workers"]`, None when no width was measured:
    nothing is kept) with that width's spread. Written when none is kept, or over an unsettled one by a reader that
    searched and settled (the quiet measure replaces the unconcluded); never otherwise: a kept figure is never paid
    twice. Kept under the key of the weights read (`read_key(source)`), beside the widths of other keys; nothing is
    kept without a source. True when written."""
    path = _profile_path(hardware_id)
    if path is None or source is None or not measured or measured.get("workers") is None:
        return False
    from neurobrix.core.prism import autodetect as A
    width = int(measured["workers"])
    at = (measured.get("draws") or {}).get(width) or []
    data = yaml.safe_load(path.read_text()) or {}
    key, widths = read_key(source), _kept_widths(data)
    kept = widths.get(key)
    if not at or (kept and (kept.get("settled") or not measured.get("settled") or measured.get("start") is not None)):
        return False
    widths[key] = {
        "workers": width, "low_gbps": round(min(at) / 1e9, 3), "high_gbps": round(max(at) / 1e9, 3),
        "rounds": len(at), "settled": bool(measured.get("settled")), "measured": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    data.setdefault("host_measured", {})["weight_read"] = widths
    A._write_profile_atomically(path, data)
    return True


def note_weight_load(engine: str, nbx_path, component: str, where: str, landed: int, seconds: float) -> None:
    """What a weight load of a component measured, handed to the active generated profile: the loader's own rate
    (`autodetect.note_loader_rate`, S2: Prism's estimate reads it from the next plan on) and the width the weight reader
    reached (`note_read_width`, row 167: the next launch's reader starts there)."""
    from neurobrix.core.prism import autodetect as A
    from neurobrix.core.workspace import weight_reader_measured
    stored = A.stored_major(nbx_path, component)
    if stored is not None:
        A.note_loader_rate(A.ACTIVE_HARDWARE_ID, engine, stored, where, int(landed), float(seconds),
                           f"{Path(str(nbx_path)).name}:{component}")
    note_read_width(A.ACTIVE_HARDWARE_ID, weight_reader_measured(), nbx_path)


def reload_may_be_written(host, vendor, engine: str, stored: str, spread) -> bool:
    """Whether a reload rate of `engine` and `stored` may be written into the profile's `host_measured` block `host`:
    not over one the vendor's block states, and not over a kept one unless that one is unsteady (kept with its spread)
    and this one is steady (`spread` None): only a steady measure replaces an unsteady one."""
    for block in (host, vendor):
        if (((block or {}).get("weight_reload_gbps") or {}).get(engine) or {}).get(stored) is None:
            continue
        unsteady = (((block or {}).get("weight_reload_spread") or {}).get(engine) or {}).get(stored) is not None
        if block is vendor or spread is not None or not unsteady:
            return False
    return True


def put_reload(host: dict, engine: str, stored: str, gbps: float, spread, stamp: str) -> str:
    """The reload rate `gbps` of `engine` and `stored` into `host` (`host_measured`), with its spread and date in
    `weight_reload_spread` when its draws were unsteady (`spread`: (low, high) GB/s), the spread dropped when they were
    steady. Returns how the rate came, worded for `weight_reload_source`."""
    host.setdefault("weight_reload_gbps", {}).setdefault(engine, {})[stored] = gbps
    host["weight_reload_under"] = card_key()       # every rate kept is of this driver and library (`drop_stale`)
    spreads = host.setdefault("weight_reload_spread", {})
    if spread is None:
        (spreads.get(engine) or {}).pop(stored, None)
        if not spreads.get(engine):
            spreads.pop(engine, None)
        if not spreads:
            host.pop("weight_reload_spread")
        return f"{gbps} GB/s"
    high = round(float(max(spread)), 3)
    spreads.setdefault(engine, {})[stored] = {"low": gbps, "high": high, "measured": stamp}
    return (f"{gbps} GB/s, the LOW draw of an unsteady measure ({gbps}-{high} GB/s: the plan is priced at the "
            f"slowest; a later steady measure replaces it)")
