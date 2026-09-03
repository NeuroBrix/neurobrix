"""Pins the software-pipelining budget as a DATA-DRIVEN hardware capability.

`_safe_num_stages()` clamps the `num_stages` of every autotuned Triton
config. Until 2026-09-03 it decided that by querying the CUDA driver:

    return 2 if torch.cuda.get_device_capability()[0] < 8 else n

Two defects in one line. It is a hardware parameter decided in code
(R23/R24 say those live in `config/vendors/<vendor>/<arch>.yml`), and on
ROCm `get_device_capability()` returns the *gfx* version — gfx90a reads
as (9, 0), gfx942 as (9, 4) — which is not an NVIDIA compute capability.
Compared against 8 it silently cleared unvalidated pipelining on every
CDNA card, by accident rather than by decision.

These pins cover both paths: the vendor-config path when a hardware
profile is present, and the bare-process fallback used by tests.
"""

import importlib

import pytest

wrappers = importlib.import_module("neurobrix.kernels.wrappers")
_configs = importlib.import_module("neurobrix.kernels.ops._configs")
from neurobrix.core.config.loader import get_vendor_config  # noqa: E402


class _Device:
    def __init__(self, brand, architecture):
        self.brand = brand
        self.architecture = architecture


class _Profile:
    """Duck-typed stand-in — `set_hardware_profile` accepts any object
    carrying `has_native_bf16`, and `_arch_param` reads `devices[0]`."""

    def __init__(self, brand, architecture, has_native_bf16=False):
        self.devices = [_Device(brand, architecture)]
        self.has_native_bf16 = has_native_bf16


@pytest.fixture
def restore_profile():
    saved = wrappers.get_hardware_profile()
    yield
    setattr(wrappers, "_NBX_HW_PROFILE", saved)


# --- the YAMLs carry the key at all, on every shipped architecture ----------

@pytest.mark.parametrize(
    "vendor,arch",
    [
        ("nvidia", "volta"), ("nvidia", "ampere"), ("nvidia", "hopper"),
        ("amd", "cdna"), ("amd", "cdna2"), ("amd", "cdna3"),
    ],
)
def test_every_vendor_profile_declares_a_pipelining_budget(vendor, arch):
    """R10 schema parity: a missing key would silently fall back to the
    driver query this change exists to remove."""
    cfg = get_vendor_config(vendor, arch)
    budget = cfg.get("pipelining", {}).get("max_num_stages")
    assert isinstance(budget, int) and budget >= 1, (
        f"{vendor}/{arch}.yml must declare pipelining.max_num_stages"
    )


def test_amd_architectures_resolve_at_all():
    """Prism autodetect maps gfx90a -> cdna2 and gfx940/941/942 -> cdna3.
    Before 2026-09-03 only cdna.yml existed, so the ZERO-FALLBACK loader
    raised FileNotFoundError on every MI200 and MI300 — the crash the
    roadmap recorded as the AMD bring-up seed."""
    for arch in ("cdna", "cdna2", "cdna3"):
        cfg = get_vendor_config("amd", arch)
        assert cfg["architecture"] == arch
        # 64-wide wavefronts, declared under the SAME key every other
        # profile uses. It was `wavefront_size` until 2026-09-03, so a
        # consumer asking for `memory.warp_size` got nothing on AMD and
        # would have fallen back to 32 — wrong by 2x on every CDNA card.
        assert cfg["memory"]["warp_size"] == 64


def test_cdna_generations_are_distinct():
    """The three CDNA files must not be copies: MI300 has 2.5x the LDS and
    is the only one with FP8."""
    c1, c2, c3 = (get_vendor_config("amd", a) for a in ("cdna", "cdna2", "cdna3"))
    assert c3["memory"]["max_shared_memory_per_block"] == 163840
    assert c1["memory"]["max_shared_memory_per_block"] == 65536
    assert c2["memory"]["max_shared_memory_per_block"] == 65536
    assert c3["precision"]["supports_fp8"] is True
    assert c1["precision"]["supports_fp8"] is False
    assert c2["precision"]["supports_fp8"] is False


# --- the clamp reads the profile, not the driver ----------------------------

def test_volta_profile_clamps_to_two(restore_profile):
    """sm_70 has no cp.async; >2 stages faults with
    CUDA_ERROR_MISALIGNED_ADDRESS in any tl.dot kernel. Measured, not
    conservative — this is the pin that must never regress."""
    wrappers.set_hardware_profile(_Profile("nvidia", "volta"))
    assert _configs._safe_num_stages(5) == 2
    assert _configs._safe_num_stages(2) == 2
    assert _configs._safe_num_stages(1) == 1


def test_ampere_profile_does_not_clamp(restore_profile):
    """The cap is set above the widest config space, so it never binds."""
    wrappers.set_hardware_profile(_Profile("nvidia", "ampere", has_native_bf16=True))
    assert _configs._safe_num_stages(5) == 5
    assert _configs._safe_num_stages(3) == 3


@pytest.mark.parametrize("arch", ["cdna", "cdna2", "cdna3"])
def test_cdna_profiles_clamp_pending_first_light(restore_profile, arch):
    """cp.async does not exist on CDNA and Triton's AMD backend pipelines
    by a different mechanism, so neither Volta's measured fault nor
    Ampere's clearance transfers. The budget stays 2 until an MI-series
    card measures it: clamping can only cost throughput, while allowing
    unvalidated pipelining could cost correctness."""
    wrappers.set_hardware_profile(_Profile("amd", arch, has_native_bf16=True))
    assert _configs._safe_num_stages(5) == 2


def test_clamp_never_raises_a_requested_value(restore_profile):
    """The budget is a ceiling, never a floor: a config asking for fewer
    stages than the hardware allows keeps its own value."""
    wrappers.set_hardware_profile(_Profile("nvidia", "ampere", has_native_bf16=True))
    assert _configs._safe_num_stages(1) == 1
    wrappers.set_hardware_profile(_Profile("amd", "cdna3", has_native_bf16=True))
    assert _configs._safe_num_stages(1) == 1


def test_unknown_architecture_falls_back_rather_than_crashing(restore_profile):
    """`_arch_param` swallows loader failures by contract. An architecture
    with no YAML must not take the autotune path down with it — the
    fallback decides, and the run continues."""
    wrappers.set_hardware_profile(_Profile("amd", "rdna9000"))
    assert _configs._safe_num_stages(3) in (2, 3)
