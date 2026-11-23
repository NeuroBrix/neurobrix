"""Every NVIDIA architecture the detector names has a hardware profile, or is refused BY NAME (GitHub issue 6).

On 0.6.16 an RTX 4070 (Ada, compute capability 8.9) was detected as `nvidia/ada` and `neurobrix serve` refused with
"No hardware profile for nvidia/ada": the detector (pytorch/prism/autodetect.py `_NVIDIA_ARCH_MAP`, `_NVIDIA_ARCH_REFINE`)
named architectures no file in config/vendors/nvidia described — turing (T4, RTX 20xx), ampere_consumer (A10, A40,
RTX 30xx), ada (L4, L40S, RTX 40xx), blackwell (B200, B300). turing, ada and blackwell now have a profile whose
hardware values are read from NVIDIA's documentation (documentation-sourced and unmeasured, the sources written in the
file). ampere_consumer does not in 0.6.17: a profile declaring 8.6 would replace, by capability, the Ampere profile the
shipped 8.6 hardware files (architecture: ampere) plan and launch on; a bare-detected 8.6 card keeps its refusal.

What these tests would do if the profiles were missing: the loader raises UnsupportedArchitectureError for the three
names and Prism plans nothing for the three simulated cards — RED. An architecture still without a profile (kepler,
maxwell, pascal, and the "unknown" a capability of major 11 or 12 is named) keeps its refusal by name: no default
profile is ever substituted."""
import subprocess

import pytest

from neurobrix.core.config.loader import CONFIG_ROOT, UnsupportedArchitectureError, get_vendor_config
from neurobrix.pytorch.prism import InputConfig, PrismSolver
from neurobrix.pytorch.prism import autodetect
from neurobrix.pytorch.prism.solver import PlanNotComputable
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, pin_dedicated_card, pin_host,
                                              pin_no_calibration, profile)

NVIDIA = CONFIG_ROOT / "vendors" / "nvidia"
ADDED = ("turing", "ada", "blackwell")

# One card of each added architecture as nvidia-smi reports it (name, memory in MiB, compute capability): the
# capabilities and names are NVIDIA's list (developer.nvidia.com/cuda-gpus); the memory is the card's size class.
CARDS = {
    "turing": ("Tesla T4", 15360, "7.5"),
    "ada": ("NVIDIA GeForce RTX 4070", 12282, "8.9"),       # the card of issue 6
    "blackwell": ("NVIDIA B200", 183359, "10.0"),
}


def _detected(monkeypatch, name, memory_mb, cc):
    """The device the engine's own detection makes of one nvidia-smi line (autodetect._parse_nvidia_smi)."""
    line = f"0, {name}, {memory_mb}, {cc}, 00000000:01:00.0\n"
    monkeypatch.setattr(autodetect.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=line, stderr=""))
    monkeypatch.setattr(autodetect, "_detect_pcie_version_sysfs", lambda bus: "4.0")
    (device,) = autodetect._parse_nvidia_smi()
    return device


@pytest.mark.parametrize("cc", ["7.0", "7.5", "8.0", "8.7", "8.9", "9.0", "10.0", "10.3"])
def test_every_capability_of_the_current_cuda_guide_resolves_to_a_profile(cc):
    """The capabilities NVIDIA's CUDA Programming Guide lists (Tables 33-36) up to major 10, and volta's 7.0; 8.6
    (ampere_consumer) is refused below, in 0.6.17."""
    arch = autodetect._nvidia_cc_to_arch(cc)
    cfg = get_vendor_config("nvidia", arch)
    assert cfg["memory"]["max_shared_memory_per_block"] > 0 and cfg["memory"]["warp_size"] == 32, (cc, arch)


@pytest.mark.parametrize("cc", ["3.5", "5.2", "6.1", "8.6", "11.0", "12.0"])
def test_an_architecture_without_a_profile_is_refused_by_its_name(cc):
    arch = autodetect._nvidia_cc_to_arch(cc)
    assert not (NVIDIA / f"{arch}.yml").exists(), f"{arch} now has a profile: move {cc} to the test above"
    with pytest.raises(UnsupportedArchitectureError, match=f"No hardware profile for nvidia/{arch}"):
        get_vendor_config("nvidia", arch)


@pytest.mark.parametrize("arch", ADDED)
def test_an_added_profile_is_the_detectors_name_for_its_own_capability(arch):
    cfg = get_vendor_config("nvidia", arch)
    assert cfg["architecture"] == arch
    assert autodetect._nvidia_cc_to_arch(str(cfg["compute_capability"])) == arch


@pytest.mark.parametrize("arch", ADDED)
def test_an_added_profile_says_it_is_documentation_sourced_and_cites_its_sources(arch):
    text = (NVIDIA / f"{arch}.yml").read_text()
    assert "DOCUMENTATION-SOURCED AND UNMEASURED" in text and "validation_status: hand_curated_from_docs" in text
    assert "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html" in text
    # the capability is a library key since the NVIDIA files' split (nbxtensor backend/devices/nvidia/<arch>.yml): the
    # engine reads it through its merged view of the two halves
    assert get_vendor_config("nvidia", arch).get("compute_capability")


def _plan(monkeypatch, arch, mode, serve):
    name, memory_mb, cc = CARDS[arch]
    device = _detected(monkeypatch, name, memory_mb, cc)
    assert device["architecture"] == arch, device
    spec = dict(V100_16GB, id=f"scenario-{arch}", preferred_dtype="float16", cpu=dict(V100_16GB["cpu"]))
    usable = memory_mb - 300
    spec["devices"] = [{**device, "runtime_total_mb": usable, "runtime_context_mb": 306}]
    pin_host(monkeypatch, 32768, 24576, "a desktop host with 32 GB")
    pin_dedicated_card(monkeypatch, usable, 306, f"a dedicated {name}")
    pin_no_calibration(monkeypatch)
    c = NBXContainer.load(str(container_root("TinyLlama-1.1B-Chat-v1.0")))
    return PrismSolver().solve_smart(c, profile(spec), InputConfig(batch_size=1, dtype="float16"),
                                     serve_mode=serve, mode=mode)


@pytest.mark.parametrize("serve", [False, True], ids=["run", "serve"])
@pytest.mark.parametrize("arch", ADDED)
def test_prism_plans_a_small_model_on_a_detected_card_of_each_added_architecture(monkeypatch, arch, serve):
    """The default engine where PyTorch is installed (--compiled), as `neurobrix run` and `neurobrix serve` plan."""
    plan = _plan(monkeypatch, arch, "compiled", serve)
    assert {c.device for c in plan.components.values()} == {"cuda:0"}, (plan.strategy, plan.selection_reason)


def test_a_triton_plan_on_a_documentation_sourced_profile_is_refused_by_name(monkeypatch):
    """The Triton branch's dead-tensor queue is measured on a card, never read from a document: absent from the
    profile, the plan names it instead of guessing it."""
    with pytest.raises(PlanNotComputable, match="deferred_drain_cliff_bytes"):
        _plan(monkeypatch, "ada", "triton", False)


@pytest.mark.parametrize("cc,card", [("8.6", "NVIDIA A10 / RTX 3090"), ("8.7", "Jetson Orin")])
def test_a_card_between_two_profiles_of_its_major_takes_the_one_below_on_both_sides(monkeypatch, cc, card):
    """With ada (8.9) beside ampere (8.0), an 8.6 or 8.7 card resolves to ampere, as before ada existed: the engine's
    capability rule (core/config/loader.py) and the kernels' (nbxtensor ops/_configs.py, the active profile) name the
    same file, so one run never reads two profiles. Without the rule's "at or below" the engine took ada, the first file
    of major 8 in name order. The shared-memory budget is the library's key (the engine plans nothing with it): the
    kernels read the figure NVIDIA documents for the capability itself (CUDA Programming Guide, Table 34, read
    2026-11-19: 8.6 99 KB, 8.7 163 KB), not the 8.0 figure of the file."""
    import types
    from nbxtensor.backend import launcher
    from nbxtensor.ops import _configs
    from neurobrix.core.config.loader import vendor_profile_for_device
    assert vendor_profile_for_device("nvidia", "ampere", cc, model=card) == "ampere", card
    monkeypatch.setattr(launcher, "target", lambda: types.SimpleNamespace(arch=int(cc.replace(".", "")), backend="cuda"))
    monkeypatch.setattr(_configs, "_ACTIVE_PROFILE", {})
    assert _configs.arch_smem_budget() == {"8.6": 99 * 1024, "8.7": 163 * 1024}[cc], card
    assert _configs.active_vendor_profile().get("_profile") == "ampere", card
