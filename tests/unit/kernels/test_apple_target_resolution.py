"""The autotune budget must resolve on a real Mac — and NOTHING else may move.

`arch_smem_budget()` asks the Triton driver for the target arch and matches it
against each profile's declared `compute_capability`. That works for NVIDIA
(which reports `7.0`) and AMD (`gfx90a`), and it could never work for Apple:
the Triton Metal target reports the Metal DEVICE NAME, lowercased and
hyphenated — measured on an M4 Pro, 2026-09-05, it says **`apple-m4-pro`**
where `apple/apple_silicon.yml` declares **`apple9`**. No string match, so the
budget resolved to `None`, so `configs_within_smem_budget` returned the space
untouched and the 32 KB filter the profile exists for **silently did not
apply**: all 17 Volta tiles were offered on a device where the largest needs
72 KB.

The fix is a profile-declared prefix list (`compute_capability_matches`), read
by the resolver. It is data, so a new chip family is a YAML line.

**This file is the proof that it costs nothing anywhere else.** The values
below were captured from the resolver BEFORE the change and are asserted
after it. If a future edit moves a single NVIDIA or AMD budget, or a single
config in the Volta or Ampere space, these fail.
"""

from __future__ import annotations

import pytest

from neurobrix.kernels.ops import _configs
from neurobrix.kernels.ops._configs import (arch_smem_budget,
                                            configs_within_smem_budget)
from neurobrix.kernels.ops.matmul import (_MATMUL_AUTOTUNE_AMPERE_PLUS,
                                          _MATMUL_AUTOTUNE_VOLTA)


class _Target:
    def __init__(self, arch):
        self.arch = arch


class _Driver:
    def __init__(self, arch):
        self._target = _Target(arch)

    def get_current_target(self):
        return self._target


def _budget_for(arch):
    """Resolve the budget as if the driver reported `arch`."""
    import triton.runtime.driver as driver_module
    saved = driver_module.__dict__.get("_active")
    driver_module._active = _Driver(arch)
    try:
        return arch_smem_budget()
    finally:
        driver_module._active = saved


# Captured from the resolver BEFORE `compute_capability_matches` existed.
# Every row here must keep its value for ever; the two Apple rows are the
# only ones the change was allowed to move, and they are asserted separately.
_BUDGETS_THAT_MUST_NOT_MOVE = {
    70: 98304,          # V100, exact match on "7.0"
    75: 98304,          # T4 — no 7.5 profile, resolved by NVIDIA major
    80: 167936,         # A100, exact on "8.0"
    86: 167936,         # 3090 — resolved by major
    90: 232448,         # H100, exact on "9.0"
    "gfx908": 65536,    # CDNA1
    "gfx90a": 65536,    # CDNA2, exact
    "gfx942": 163840,   # CDNA3
    "apple9": 32768,    # the declared family, still exact-matched
    "unknown": None,    # no profile -> no filtering, untouched
}


@pytest.mark.parametrize("arch,expected",
                         sorted(_BUDGETS_THAT_MUST_NOT_MOVE.items(), key=str))
def test_no_other_architecture_changed_budget(arch, expected):
    assert _budget_for(arch) == expected


def test_a_real_apple_target_now_resolves():
    """The defect this change exists for: before it, both were None."""
    assert _budget_for("apple-m4-pro") == 32768
    assert _budget_for("apple-m1") == 32768


def test_a_non_apple_target_cannot_reach_the_new_branch():
    """The new match only fires for profiles that DECLARE prefixes, and only
    `apple/apple_silicon.yml` does. An arch that happens to share a prefix
    with nothing must still resolve to None rather than borrow a budget."""
    assert _budget_for("apple") is None          # shorter than the prefix
    assert _budget_for("banana-m4") is None
    assert _budget_for("gfx-m4-pro") is None


def test_only_the_apple_profile_declares_matches():
    """If a second profile ever declares prefixes, the pins above stop
    covering it and this test says so rather than letting it pass."""
    import yaml
    declaring = []
    for path in sorted(_configs.Path(__file__).resolve().parents[3]
                       .glob("src/neurobrix/config/vendors/*/*.yml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        if cfg.get("compute_capability_matches"):
            declaring.append(f"{path.parent.name}/{path.stem}")
    assert declaring == ["apple/apple_silicon"], declaring


# --- the config spaces themselves, by value ---------------------------------

def _signature(configs):
    return [dict(sorted(c.kwargs.items()),
                 num_stages=c.num_stages, num_warps=c.num_warps)
            for c in configs]


_VOLTA_AT_32K = [
    {"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_N": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 2},
    {"BLOCK_K": 64, "BLOCK_M": 32, "BLOCK_N": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 2},
    {"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_N": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 2},
    {"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_N": 64, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
    {"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_N": 128, "GROUP_M": 8, "num_stages": 3, "num_warps": 4},
    {"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 32, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
    {"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8, "num_stages": 2, "num_warps": 4},
    {"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8, "num_stages": 3, "num_warps": 4},
    {"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_K": 32, "BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8, "num_stages": 2, "num_warps": 4},
]


def test_the_volta_space_is_unchanged_at_its_own_budget():
    """17 of 17 survive 96 KB — the pin `test_smem_budget.py` opens with,
    restated here so this change is covered by its own file too."""
    assert len(_signature(configs_within_smem_budget(
        _MATMUL_AUTOTUNE_VOLTA, 96 * 1024))) == 17


def test_the_volta_space_is_unchanged_with_no_budget():
    assert (_signature(configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA, None))
            == _signature(_MATMUL_AUTOTUNE_VOLTA))


def test_the_apple_selection_is_exactly_the_ten_that_fit():
    """The 10 of 17 the Dell chantier computed, now actually reachable on a
    Mac. Pinned by VALUE: a count alone would not catch a swap."""
    got = _signature(configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA, 32768))
    assert got == _VOLTA_AT_32K


def test_the_ampere_space_is_unchanged():
    assert len(configs_within_smem_budget(
        _MATMUL_AUTOTUNE_AMPERE_PLUS, 164 * 1024)) == 25
