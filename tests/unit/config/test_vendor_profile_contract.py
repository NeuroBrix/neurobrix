"""Hardware profiles are the only source of hardware-dependent values.

Two failures of this contract reached users in the same day (2026-09-03):

* AMD MI200/MI300 and every Apple GPU raised a bare `FileNotFoundError`
  with a path in it, before running a single operation, because hardware
  detection emits an architecture name that had no profile file.
* The refusal, when it happened, told the user nothing about what to do.

The refusal itself is correct and must stay — guessing a block size or a
shared-memory limit for an unknown GPU produces wrong output rather than an
error. What these pins hold is that it refuses *for the right architectures*
and *says something usable*.
"""

from __future__ import annotations

import pytest

from neurobrix.core.config.loader import (
    UnsupportedArchitectureError,
    get_vendor_config,
    list_vendors,
)

# Every architecture Prism autodetect can emit and for which a profile must
# exist, because real hardware in use today maps to it.
SHIPPED = [
    ("nvidia", "volta"), ("nvidia", "ampere"), ("nvidia", "hopper"),
    ("amd", "cdna"), ("amd", "cdna2"), ("amd", "cdna3"),
    ("apple", "apple_silicon"),
]


@pytest.mark.parametrize("vendor,arch", SHIPPED)
def test_shipped_architectures_resolve(vendor, arch):
    cfg = get_vendor_config(vendor, arch)
    assert cfg["architecture"] == arch


@pytest.mark.parametrize("vendor,arch", SHIPPED)
def test_every_profile_carries_the_load_bearing_keys(vendor, arch):
    """Schema parity (R10): a key missing from one profile becomes a silent
    default on that hardware alone, which is the hardest class of bug to see."""
    cfg = get_vendor_config(vendor, arch)
    for section in ("block_sizes", "memory", "precision", "pipelining"):
        assert section in cfg, f"{vendor}/{arch}.yml lacks '{section}'"
    assert isinstance(cfg["pipelining"]["max_num_stages"], int)
    assert cfg["memory"]["max_shared_memory_per_block"] > 0
    for flag in ("supports_fp16", "supports_bf16", "supports_tf32", "supports_fp8"):
        assert isinstance(cfg["precision"][flag], bool)


def test_unknown_architecture_refuses_with_a_typed_error():
    with pytest.raises(UnsupportedArchitectureError):
        get_vendor_config("nvidia", "blackwell")


def test_the_refusal_stays_catchable_as_before():
    """Subclassing FileNotFoundError keeps every existing handler working —
    including `_arch_param`, which swallows loader failures by contract."""
    assert issubclass(UnsupportedArchitectureError, FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        get_vendor_config("amd", "rdna3")


def test_the_refusal_is_actionable():
    """A user meeting this has a working GPU and a NeuroBrix that will not
    start. The message must name their hardware, what exists, and that the
    gap is ours."""
    with pytest.raises(UnsupportedArchitectureError) as excinfo:
        get_vendor_config("nvidia", "blackwell")
    message = str(excinfo.value)
    assert "nvidia / blackwell" in message, "must name the detected hardware"
    assert "nvidia/volta" in message, "must list what does exist"
    assert "gap on our side" in message, "must not read as a hardware limit"
    assert "refuses to guess" in message, "must say why it will not continue"


def test_apple_profile_declares_no_fp64():
    """Metal GPUs have no double precision at all. The three device-scalar
    kernels that widen through f64 for bit-exactness with the host path must
    take their host-sync route there, and this flag is what tells them."""
    cfg = get_vendor_config("apple", "apple_silicon")
    assert cfg["precision"]["supports_fp64"] is False


def test_apple_threadgroup_budget_is_a_quarter_of_volta():
    """32 KB against 96 KB is the binding constraint on that platform and the
    reason its tiles are sized down; a copy-paste of the Volta numbers would
    silently overflow."""
    apple = get_vendor_config("apple", "apple_silicon")
    volta = get_vendor_config("nvidia", "volta")
    assert apple["memory"]["max_shared_memory_per_block"] == 32768
    assert volta["memory"]["max_shared_memory_per_block"] == 98304


def test_vendor_listing_covers_the_three_vendors():
    vendors = list_vendors()
    assert set(vendors) >= {"nvidia", "amd", "apple"}
