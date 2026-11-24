"""Adding the Maxwell (5.x), Pascal (6.x) and Blackwell consumer (12.x) classes, and naming cc 8.6 `ampere`, changes
NOTHING for a class that runs today (the project lead, 2026-11-24): for every existing vendor file (nvidia volta, turing,
ampere, ada, hopper, blackwell; every amd, intel and apple file), every shipped hardware profile (the four 8.6 ones,
a10-24g, rtx3060-12g, rtx3080-10g and rtx3090-24g, included: they name ampere, as on main), and every NVIDIA
capability that resolved before, what the engine builds is byte-identical to main's (engine 9787ac25, library 50597afc):
- the merged vendor profile (the engine's half and the library's, `get_vendor_config`), its file by capability
  (`vendor_profile_for_device`) and its Triton arena factor with its source (`declared_arena_factor`);
- the device each twin's detector makes of one nvidia-smi line, and the preferred dtype of the profile it writes;
- the devices of each shipped hardware profile as the engine's loader reads them.
cc 8.6 is the one capability that did not run when auto-detected: main's detector named it `ampere_consumer`, which had
no engine file ("No hardware profile"). Its detected device is fingerprinted with the architecture read as `ampere`
(`AMPERE_86`): on main that is the name substituted, here the detector's own output, so the name is the only change.
The figures below are the SHA-256 of each one's canonical JSON, computed on main by `fingerprints()` itself. What it
catches: any edit that moves a value an existing class reads - red. No card."""
import hashlib
import json
import subprocess

import pytest
import yaml

from neurobrix.core.config.loader import CONFIG_ROOT

VENDORS = ("nvidia", "amd", "intel", "apple")
NEW_CLASSES = {"maxwell", "pascal", "blackwell_consumer"}
# (nvidia-smi name, MiB, capability) of a card of each capability that resolved before, and the RTX 3090's 8.6
NVIDIA_CARDS = [("Tesla V100-SXM2-16GB", 16384, "7.0"), ("Tesla T4", 15360, "7.5"), ("NVIDIA A100-SXM4-80GB", 81920, "8.0"),
                ("NVIDIA GeForce RTX 3090", 24576, "8.6"),
                ("Jetson AGX Orin", 32768, "8.7"), ("NVIDIA L40S", 46068, "8.9"), ("NVIDIA H100 80GB HBM3", 81559, "9.0"),
                ("NVIDIA B200", 183359, "10.0"), ("NVIDIA B300", 270000, "10.3")]
AMPERE_86 = "8.6"


def _sha(obj) -> str:
    """The SHA-256 of an object's canonical JSON (keys sorted, non-JSON values by their str)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _vendor_fingerprint(vendor, arch):
    """The fingerprint of one vendor class: its merged profile, its file by capability and its arena factor."""
    from neurobrix.core.config.loader import declared_arena_factor, get_vendor_config, vendor_profile_for_device
    cfg = get_vendor_config(vendor, arch)
    cc = str(cfg.get("compute_capability", ""))
    return _sha({"config": cfg, "by_capability": vendor_profile_for_device(vendor, arch, cc),
                 "arena": declared_arena_factor(vendor, arch, cc)})


def _detected_fingerprint(twin, name, memory_mb, cc):
    """The fingerprint of the device one twin's detector makes of one nvidia-smi line, and its preferred dtype."""
    import importlib
    detector = importlib.import_module(f"neurobrix.{twin}.prism.autodetect")
    line = f"0, {name}, {memory_mb}, {cc}, 00000000:01:00.0\n"
    real_run, real_pcie = detector.subprocess.run, detector._detect_pcie_version_sysfs
    detector.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=line, stderr="")
    detector._detect_pcie_version_sysfs = lambda bus: "4.0"
    try:
        (device,) = detector._parse_nvidia_smi()
    finally:
        detector.subprocess.run, detector._detect_pcie_version_sysfs = real_run, real_pcie
    if cc == AMPERE_86:
        device = dict(device, architecture="ampere")
    return _sha({"device": device, "preferred": detector._PREFERRED_DTYPE.get(device["architecture"], "float16")})


def _shipped_fingerprint(hardware_id):
    """The fingerprint of a shipped hardware profile's devices as the engine's loader reads them."""
    from neurobrix.pytorch.prism.loader import load_profile
    return _sha([{k: v for k, v in vars(d).items()} for d in load_profile(hardware_id).devices])


def existing_classes():
    """(vendor, arch) of every vendor file of main's classes: today's files less the new classes."""
    return [(v, p.stem) for v in VENDORS for p in sorted((CONFIG_ROOT / "vendors" / v).glob("*.yml"))
            if p.stem not in NEW_CLASSES]


def shipped_profiles():
    """The id of every shipped hardware profile (config/hardware/*.yml)."""
    return [p.stem for p in sorted((CONFIG_ROOT / "hardware").glob("*.yml"))]


def fingerprints() -> dict:
    """Every fingerprint by its name: the vendor classes, the detected cards of both twins, the shipped profiles."""
    out = {f"vendor {v}/{a}": _vendor_fingerprint(v, a) for v, a in existing_classes()}
    out.update({f"detected {t} {cc}": _detected_fingerprint(t, n, m, cc)
                for t in ("pytorch", "triton") for n, m, cc in NVIDIA_CARDS})
    out.update({f"shipped {h}": _shipped_fingerprint(h) for h in shipped_profiles()})
    return out


#: main's fingerprints, written by running `fingerprints()` on main (beside this file)
MAIN_FILE = __import__("pathlib").Path(__file__).with_name("every_class_that_runs_today.main.yml")
MAIN = yaml.safe_load(MAIN_FILE.read_text()) if MAIN_FILE.exists() else {}


@pytest.mark.parametrize("what", sorted(MAIN))
def test_the_class_builds_what_it_built_on_main(what):
    """One class builds the fingerprint main built for it."""
    assert fingerprints_cached().get(what) == MAIN[what], f"{what} differs from main"


def test_every_class_of_main_is_checked():
    """The classes fingerprinted here are exactly main's: none added, none lost."""
    assert MAIN, f"no fingerprint of main: {MAIN_FILE}"
    assert set(fingerprints_cached()) == set(MAIN), sorted(set(fingerprints_cached()) ^ set(MAIN))


def test_an_auto_detected_86_card_now_names_the_file_its_shipped_profiles_name():
    """cc 8.6 auto-detected names `ampere` in both twins, the architecture the shipped 8.6 hardware files carry."""
    for twin in ("pytorch", "triton"):
        detector = __import__(f"neurobrix.{twin}.prism.autodetect", fromlist=["_nvidia_cc_to_arch"])
        assert detector._nvidia_cc_to_arch(AMPERE_86) == "ampere", twin
    for hw in ("a10-24g", "rtx3060-12g", "rtx3080-10g", "rtx3090-24g"):
        devices = yaml.safe_load((CONFIG_ROOT / "hardware" / f"{hw}.yml").read_text())["devices"]
        assert {d["architecture"] for d in devices} == {"ampere"}, hw


_CACHE = {}


def fingerprints_cached() -> dict:
    """`fingerprints()` computed once per process."""
    if not _CACHE:
        _CACHE.update(fingerprints())
    return _CACHE
