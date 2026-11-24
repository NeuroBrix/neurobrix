"""A GTX 10 / P100 (Pascal), Maxwell, RTX 30 / A10 / A40 (Ampere GA10x) or RTX 50 (Blackwell consumer) owner gets a
hardware profile, not "No hardware profile" (2026-11-24).

The detector (pytorch/prism/autodetect.py and its Triton twin) named cc 5.x `maxwell`, 6.x `pascal` and 8.6
`ampere_consumer`, but no engine vendor file described them, and it named cc 12.0 "unknown" (major 12 absent from
`_NVIDIA_ARCH_MAP`): `get_vendor_config` refused all four. 8.6 now names `ampere`, whose library file declares 8.6
through its memory.by_capability "8.6" and which the shipped 8.6 hardware files already name; 5.x, 6.x and 12.x have
their engine vendor file. The Triton branch refuses 5.x and 6.x by name (no code below sm_70) and 12.x as it refuses
ada (no measured dead-tensor queue); 8.6 runs it as every ampere card does. A device is TWO halves, merged by the loader: the library's
(nbxtensor backend/devices/nvidia/<arch>.yml, the documented hardware figures) and the engine's
(config/vendors/nvidia/<arch>.yml, device prefix, allocator keys, precision flags).

Card-less: the detection is patched (one nvidia-smi line per card, as `_parse_nvidia_smi` reads it). What it would do
without the cure: the architecture's engine file is missing (or the name is "unknown"), `get_vendor_config` raises
UnsupportedArchitectureError and no profile resolves - RED."""
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

from neurobrix.core.config.loader import (CONFIG_ROOT, get_device_prefix, get_vendor_config,
                                          vendor_profile_for_device)
from neurobrix.core.dtype.config import device_supports_fp64
from neurobrix.pytorch.prism import autodetect as pytorch_autodetect
from neurobrix.pytorch.prism import loader as profile_loader
from neurobrix.pytorch.prism.memory_budget import deferred_drain_policy_of
from neurobrix.triton.prism import autodetect as triton_autodetect

ENGINE = CONFIG_ROOT / "vendors" / "nvidia"
LIBRARY = Path(importlib.util.find_spec("nbxtensor").origin).parent / "backend" / "devices" / "nvidia"

# (nvidia-smi name, memory MiB, compute capability) -> the architecture the detector names, the dtypes it lists, the
# preferred dtype. Capabilities and names: NVIDIA's lists; the memory is the card's size class.
CARDS = {
    "5.2": (("NVIDIA GeForce GTX 980", 4096), "maxwell", ["float32", "float16"], "float16"),
    "6.1": (("NVIDIA GeForce GTX 1080", 8192), "pascal", ["float32", "float16"], "float16"),
    "8.6": (("NVIDIA GeForce RTX 3090", 24576), "ampere", ["float32", "float16", "bfloat16"], "bfloat16"),
    "12.0": (("NVIDIA GeForce RTX 5090", 32607), "blackwell_consumer",
             ["float32", "float16", "bfloat16", "float8_e4m3fn"], "bfloat16"),
}
TWINS = {"pytorch": pytorch_autodetect, "triton": triton_autodetect}


def _detected(monkeypatch, detector, cc):
    """The device one twin's detector makes of the card's patched nvidia-smi line."""
    (name, memory_mb), *_ = CARDS[cc]
    line = f"0, {name}, {memory_mb}, {cc}, 00000000:01:00.0\n"
    with monkeypatch.context() as m:           # nvidia-smi alone: a process-wide cache must not read the fake later
        m.setattr(detector.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=line, stderr=""))
        m.setattr(detector, "_detect_pcie_version_sysfs", lambda bus: "4.0")
        (device,) = detector._parse_nvidia_smi()
    return device


@pytest.mark.parametrize("twin", sorted(TWINS))
@pytest.mark.parametrize("cc", sorted(CARDS, key=lambda c: tuple(map(int, c.split(".")))))
def test_the_capability_names_an_architecture_both_halves_describe(monkeypatch, twin, cc):
    """The detected capability names an architecture whose engine and library halves both exist and merge."""
    _card, arch, dtypes, _preferred = CARDS[cc]
    device = _detected(monkeypatch, TWINS[twin], cc)
    assert device["architecture"] == arch and device["supports_dtypes"] == dtypes, device
    assert (ENGINE / f"{arch}.yml").is_file(), f"no engine vendor file for {arch} (cc {cc})"
    assert (LIBRARY / f"{arch}.yml").is_file(), f"no library device file for {arch} (cc {cc}): {LIBRARY}"
    cfg = get_vendor_config("nvidia", arch)                      # the two halves merged, a key in both refused
    assert cfg["architecture"] == arch and cfg["memory"]["warp_size"] == 32
    assert cfg["memory"]["max_shared_memory_per_block"] > 0
    # one file by both of the engine's rules: by name (get_vendor_config) and by capability (the kernels' rule)
    assert vendor_profile_for_device("nvidia", arch, cc, model=device["model"]) == arch


def test_the_engine_half_repeats_no_library_figure():
    """The new engine halves hold only device_prefix, memory and precision: no hardware figure of the library's."""
    for cc, (_card, arch, _d, _p) in CARDS.items():
        if arch == "ampere":
            continue                                     # ampere's engine file is main's, unchanged
        engine_half = yaml.safe_load((ENGINE / f"{arch}.yml").read_text()) or {}
        assert set(engine_half) <= {"device_prefix", "memory", "precision"}, (arch, sorted(engine_half))
        assert "compute_capability" not in engine_half and "architecture" not in engine_half, arch


@pytest.mark.parametrize("twin", sorted(TWINS))
@pytest.mark.parametrize("cc", sorted(CARDS, key=lambda c: tuple(map(int, c.split(".")))))
def test_a_hardware_profile_is_built_from_the_detected_card(monkeypatch, tmp_path, twin, cc):
    """The profile `detect_hardware` writes for a machine holding the card (CPU, GPU and vendor detection patched),
    loaded back through the engine's own profile loader, then read by the compiled branch's readers."""
    detector = TWINS[twin]
    _card, arch, dtypes, preferred = CARDS[cc]
    device = _detected(monkeypatch, detector, cc)
    monkeypatch.setattr(detector, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(detector, "_detect_cpu", lambda os_type: {
        "model": "a desktop CPU", "cores": 8, "threads": 16, "ram_mb": 32768, "architecture": "x86_64",
        "features": ["avx2"]})
    monkeypatch.setattr(detector, "_detect_gpus", lambda os_type: ([dict(device)], "nvidia"))
    monkeypatch.setattr(detector, "_beside", lambda brand: [])
    monkeypatch.setattr(detector, "_detect_system_vendor", lambda os_type: "custom")
    data = detector.detect_hardware()
    assert data["preferred_dtype"] == preferred, data["preferred_dtype"]
    (written,) = data["devices"]
    assert (written["architecture"], written["compute_capability"]) == (arch, cc)

    (tmp_path / f"{data['id']}.yml").write_text(yaml.safe_dump(data))
    monkeypatch.setattr(profile_loader, "HARDWARE_DIR", tmp_path)
    profile = profile_loader.load_profile(data["id"])
    (dev,) = profile.devices
    assert dev.architecture == arch and dev.compute_capability == cc
    vendor = getattr(dev.brand, "value", dev.brand)
    assert get_device_prefix(vendor, dev.architecture) == "cuda"
    assert device_supports_fp64("gpu", vendor, dev.architecture) is True
    assert get_vendor_config(vendor, dev.architecture)["memory"]["compiled_allocator_settings"]
    # the Triton branch's dead-tensor queue: ampere's measured policy on 8.6; below sm_70 the branch has no code,
    # refused with the remedy; 12.x states none, refused by name as ada is
    memory = get_vendor_config(vendor, dev.architecture)["memory"]
    if arch in ("maxwell", "pascal"):
        with pytest.raises(KeyError, match=r"the PyTorch branch runs on this card \(--compiled\)"):
            deferred_drain_policy_of(memory)
    elif arch == "blackwell_consumer":
        with pytest.raises(KeyError, match="declares no deferred_drain_cliff_bytes"):
            deferred_drain_policy_of(memory)
    else:
        assert deferred_drain_policy_of(memory)[0] > 0


# ── the Triton branch on these classes (the project lead's rulings, 2026-11-24) ─────────────────────────────────────
# No card of these classes is on any NeuroBrix machine (the Dell holds V100s): card-less, the detection patched.

def _triton_profile(monkeypatch, tmp_path, cc):
    """The card's profile as the engine's own loader reads it (the Triton twin), from a patched detection."""
    from neurobrix.triton.prism import loader as triton_loader
    tmp_path.mkdir(exist_ok=True)
    detector = triton_autodetect
    device = _detected(monkeypatch, detector, cc)
    monkeypatch.setattr(detector, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(detector, "_detect_cpu", lambda os_type: {
        "model": "a desktop CPU", "cores": 8, "threads": 16, "ram_mb": 32768, "architecture": "x86_64",
        "features": ["avx2"]})
    monkeypatch.setattr(detector, "_detect_gpus", lambda os_type: ([dict(device)], "nvidia"))
    monkeypatch.setattr(detector, "_beside", lambda brand: [])
    monkeypatch.setattr(detector, "_detect_system_vendor", lambda os_type: "custom")
    data = detector.detect_hardware()
    (tmp_path / f"{data['id']}.yml").write_text(yaml.safe_dump(data))
    monkeypatch.setattr(triton_loader, "HARDWARE_DIR", tmp_path)
    return triton_loader.load_profile(data["id"])


def _triton_solver():
    """A Triton-twin Prism solver set to the triton mode, for its per-card readers."""
    from neurobrix.triton.prism.solver import PrismSolver
    solver = PrismSolver()
    solver._mode = "triton"
    return solver


def test_an_rtx30_triton_plan_reads_amperes_values(monkeypatch, tmp_path):
    """cc 8.6 resolves to ampere on the Triton branch too: the plan's dead-tensor queue and arena factor are ampere's,
    unchanged (the branch runs on every ampere card today)."""
    import types
    from neurobrix.core.config.loader import declared_arena_factor
    (spec,) = _triton_profile(monkeypatch, tmp_path, "8.6").devices
    assert spec.architecture == "ampere"
    solver, dev = _triton_solver(), types.SimpleNamespace(device_string="cuda:0", spec=spec)
    assert solver._deferred_drain_mb(dev) > 0 and solver._deferred_drain_floor_mb(dev) > 0
    assert solver._triton_arena_factor(types.SimpleNamespace(spec=spec)) == pytest.approx(
        declared_arena_factor("nvidia", "ampere", "8.6")[0])


def test_an_rtx50_triton_plan_is_refused_by_name_as_ada_is(monkeypatch, tmp_path):
    """12.x states no Triton dead-tensor queue (as nvidia/ada.yml): the plan names the missing key, and the vendor file
    names the remedy, the PyTorch branch (--compiled), in the words ada's file uses."""
    import types
    from neurobrix.triton.prism.solver import PlanNotComputable
    (spec,) = _triton_profile(monkeypatch, tmp_path, "12.0").devices
    assert spec.architecture == "blackwell_consumer"
    with pytest.raises(PlanNotComputable, match="deferred_drain_cliff_bytes"):
        _triton_solver()._deferred_drain_mb(types.SimpleNamespace(device_string="cuda:0", spec=spec))
    remedy = "The default branch where\n#     PyTorch is installed (--compiled) reads none of them."
    for arch in ("ada", "blackwell_consumer"):
        assert remedy in (ENGINE / f"{arch}.yml").read_text(), arch


@pytest.mark.parametrize("cc", ["5.2", "6.1"])
def test_a_maxwell_or_pascal_triton_plan_is_refused_with_the_remedy(monkeypatch, tmp_path, cc):
    """Below sm_70 the Triton branch has no code: the plan is refused naming the capability and the remedy."""
    import types
    from neurobrix.triton.prism.solver import PlanNotComputable
    (spec,) = _triton_profile(monkeypatch, tmp_path, cc).devices
    with pytest.raises(PlanNotComputable, match=r"below sm_70\): the PyTorch branch runs on this card \(--compiled\)"):
        _triton_solver()._deferred_drain_mb(types.SimpleNamespace(device_string="cuda:0", spec=spec))


def test_no_first_launch_probe_is_added_for_the_new_classes():
    """The new classes read the family's arena factor; the first launch measures none (the supervisor's 2026-11-07 rule,
    held by tests/unit/prism/test_the_arena_factor_is_measured_where_no_vendor_declares_it.py)."""
    from neurobrix.core.config.loader import declared_arena_factor
    for cc, (_card, arch, _d, _p) in CARDS.items():
        if arch == "ampere":
            continue                                     # ampere's own factor, unchanged (the fingerprint test)
        value, why = declared_arena_factor("nvidia", arch, cc)
        assert value == pytest.approx(1.35) and "the nvidia family" in why, (arch, why)
