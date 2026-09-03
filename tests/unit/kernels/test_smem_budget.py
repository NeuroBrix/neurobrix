"""Autotune tiles are filtered by the hardware's real shared-memory budget.

Every vendor profile has declared `memory.max_shared_memory_per_block` since
the profiles were written, and until 2026-09-03 nothing read it. The autotune
space was picked by architecture NAME, with the stated reasoning that anything
unrecognised should take "the Volta subset, which fits the smallest budget".

That reasoning is false in both directions:

* **Apple** has 32 KB of threadgroup memory against Volta's 96 KB. The largest
  Volta tile wants 72 KB and cannot run on an Apple GPU at all.
* **CDNA1/2** declare 64 KB, so the same 72 KB tile is out of reach there too —
  and CDNA reached the Volta space by accident in the first place (a string
  arch failing an `isinstance(cap, int)` test), which the selection docstring
  already flagged as a first-light task.

The binding constraint of this file is the FIRST test: filtering must not move
a single config on the hardware the engine is validated on. A "portability"
change that quietly narrows Volta's measured space would be a regression
wearing a feature's clothes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurobrix.kernels.ops._configs import (
    arch_smem_budget,
    configs_within_smem_budget,
    smem_bytes_for_config,
)
from neurobrix.kernels.ops.matmul import (
    _MATMUL_AUTOTUNE_AMPERE_PLUS,
    _MATMUL_AUTOTUNE_VOLTA,
)

VENDORS = (Path(__file__).resolve().parents[3]
           / "src" / "neurobrix" / "config" / "vendors")


def _budget(vendor: str, arch: str) -> int:
    cfg = yaml.safe_load((VENDORS / vendor / f"{arch}.yml").read_text())
    return int(cfg["memory"]["max_shared_memory_per_block"])


# --- the change must be invisible where the engine is validated -------------

def test_volta_keeps_every_config_it_had():
    """The measured Phase 1.5 space is untouched on the hardware that measured
    it. If this ever fails, the filter is deleting validated configs."""
    kept = configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA,
                                      _budget("nvidia", "volta"))
    assert len(kept) == len(_MATMUL_AUTOTUNE_VOLTA)


def test_ampere_keeps_every_config_it_had():
    kept = configs_within_smem_budget(_MATMUL_AUTOTUNE_AMPERE_PLUS,
                                      _budget("nvidia", "ampere"))
    assert len(kept) == len(_MATMUL_AUTOTUNE_AMPERE_PLUS)


# --- the physics ------------------------------------------------------------

def test_the_working_set_is_both_operand_slabs_times_the_stage_count():
    """A blocked matmul stages [BLOCK_M,BLOCK_K] of A and [BLOCK_K,BLOCK_N] of
    B per pipeline stage. That quantity is what spilled on Volta at 98-145 ms
    in the Phase 1.5 measurement."""
    import triton

    cfg = triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64,
                         'GROUP_M': 8}, num_stages=3, num_warps=4)
    assert smem_bytes_for_config(cfg, dtype_bytes=2) == (64 * 64 + 64 * 128) * 2 * 3
    assert smem_bytes_for_config(cfg, dtype_bytes=2) == 73728          # 72 KB


def test_a_config_with_no_tile_declares_no_constraint():
    """Element-wise spaces name BLOCK_SIZE, not a tile. They must pass the
    filter untouched rather than be measured with a formula that does not
    describe them — which is also why conv2d's BLOCK_BHW space is not
    filtered here."""
    import triton

    assert smem_bytes_for_config(triton.Config({'BLOCK_SIZE': 1024})) == 0


# --- what the budget actually excludes --------------------------------------

def test_the_largest_volta_tile_cannot_run_on_apple():
    """32 KB against a 72 KB tile. This is the concrete reason 'send Apple to
    the Volta subset' was wrong."""
    biggest = max(_MATMUL_AUTOTUNE_VOLTA, key=smem_bytes_for_config)
    assert smem_bytes_for_config(biggest) > _budget("apple", "apple_silicon")
    kept = configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA,
                                      _budget("apple", "apple_silicon"))
    assert biggest not in kept
    assert 0 < len(kept) < len(_MATMUL_AUTOTUNE_VOLTA)


def test_cdna_excludes_the_tile_it_used_to_inherit():
    """CDNA1/2 declare 64 KB, so the 72 KB tile was never viable there — it was
    offered only because the space was chosen by name."""
    kept = configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA,
                                      _budget("amd", "cdna2"))
    assert len(kept) < len(_MATMUL_AUTOTUNE_VOLTA)
    assert all(smem_bytes_for_config(c) <= 65536 for c in kept)


# --- refusing to guess ------------------------------------------------------

def test_no_profile_means_no_filtering():
    """Filtering on an invented budget would silently delete working configs.
    A config never offered can never be chosen; one that spills is measured by
    the autotuner and rejected. So the safe direction is to leave it alone."""
    assert configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA, None) is _MATMUL_AUTOTUNE_VOLTA
    assert configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA, 0) is _MATMUL_AUTOTUNE_VOLTA


def test_the_space_is_never_emptied():
    """An empty autotune list is an import-time crash. A single too-large
    config is a slow kernel. The second is survivable."""
    kept = configs_within_smem_budget(_MATMUL_AUTOTUNE_VOLTA, 1)
    assert len(kept) == 1


# --- the profiles are the source of truth -----------------------------------

@pytest.mark.parametrize("path", sorted(VENDORS.glob("*/*.yml")),
                         ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_profile_declares_the_budget(path):
    """A new architecture that forgets this key gets no filtering at all, and
    would silently inherit tiles it cannot run."""
    cfg = yaml.safe_load(path.read_text())
    assert (cfg.get("memory") or {}).get("max_shared_memory_per_block"), (
        f"{path.parent.name}/{path.name} declares no "
        f"memory.max_shared_memory_per_block")


@pytest.mark.parametrize("path", sorted(VENDORS.glob("*/*.yml")),
                         ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_profile_is_findable_by_its_compute_capability(path):
    """Resolution is data, not a mapping table: the budget is found by matching
    the Triton target against `compute_capability`. A profile that omits it is
    unreachable, and the failure would be invisible — no filtering, no error."""
    cfg = yaml.safe_load(path.read_text())
    assert str(cfg.get("compute_capability", "")).strip(), (
        f"{path.parent.name}/{path.name} declares no compute_capability, so no "
        f"Triton target can ever resolve to it")


def test_the_running_machine_resolves_to_its_own_profile():
    budget = arch_smem_budget()
    if budget is None:
        pytest.skip("no GPU / no matching profile on this host")
    declared = {_budget(p.parent.name, p.stem) for p in VENDORS.glob("*/*.yml")}
    assert budget in declared, (
        "the resolved budget came from somewhere other than a vendor profile")


# --- the schema is uniform, or consumers must branch on vendor --------------

_REFERENCE = "nvidia/volta"

# Keys the reference declares that another profile may legitimately omit,
# each with the reason. Anything NOT on this list is a schema gap.
_MAY_OMIT = {
    "memory.sdpa_math_max_chunks":
        "the deterministic-SDPA routing budget is a Volta-SIMT remedy "
        "(P-TRITON-MOE-DETERMINISM-RESIDUAL); the race does not transfer to "
        "other backends, and every other profile records that as 0 or absent",
    "memory.sdpa_math_scores_device_fraction": "same routing budget",
}


def _flatten(d, prefix=""):
    out = {}
    for key, value in (d or {}).items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, full + "."))
        else:
            out[full] = value
    return out


@pytest.mark.parametrize("path", sorted(VENDORS.glob("*/*.yml")),
                         ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_profile_declares_what_the_reference_declares(path):
    """One hardware concept, one key name, in every profile.

    The AMD profiles declared 64-wide wavefronts as `memory.wavefront_size`
    while every other profile called the same concept `memory.warp_size`. No
    consumer existed yet, which is exactly what made it dangerous: the first
    reader of `memory.warp_size` would have got nothing on AMD and fallen back
    to 32 — wrong by 2x, silently, on every CDNA card.

    A schema whose key names depend on the vendor forces every consumer to
    branch on vendor to find a value, which is the anti-pattern the data-driven
    hardware engine exists to remove.
    """
    name = f"{path.parent.name}/{path.stem}"
    if name == _REFERENCE:
        return
    reference = set(_flatten(yaml.safe_load(
        (VENDORS / "nvidia" / "volta.yml").read_text())))
    mine = set(_flatten(yaml.safe_load(path.read_text())))
    missing = sorted(reference - mine - set(_MAY_OMIT))
    assert not missing, (
        f"{name} does not declare, and has no recorded reason to omit:\n  "
        + "\n  ".join(missing)
        + f"\n\nEither add the key, or add it to _MAY_OMIT in this file with "
          f"the reason it does not apply to this architecture."
    )
