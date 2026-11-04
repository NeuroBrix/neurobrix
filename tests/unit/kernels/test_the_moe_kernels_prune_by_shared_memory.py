"""The MoE decode band's config spaces are pruned to the blocks the device's shared memory holds (`_configs.
fits_shared_memory`, `memory.max_shared_memory_per_block` of the profile in force) BEFORE any compile: on Metal (32 KB)
the half-precision decode band's 256 x 64 fp32 block asked 65 536 bytes and the MSL emitter aborted the whole certifier
(the Mac's RUN 24d, 2026-11-04 14:59) — an abort no candidate exclusion can catch.

What this test would do if the code were wrong: a band kernel without its prune, or a prune that reads another block
than the one the kernel stages — a 64 KB config kept at a 32 KB budget, red; a prune too eager — a fitting config lost
at the V100's 96 KB, red; a profile with no budget answered by a guess — red at the refusal. Card-less."""
from __future__ import annotations

import pytest

from nbxtensor.ops import _configs


def _pruned(tuner, budget, monkeypatch, **named):
    monkeypatch.setattr(_configs, "arch_smem_budget", lambda: budget)
    prune = tuner.early_config_prune
    return prune(tuner.configs, named)


@pytest.mark.parametrize("name", ["moe_gateup_vec_half_kernel", "moe_down_split_vec_half_kernel"])
def test_the_half_band_keeps_only_what_fits(monkeypatch, name):
    from nbxtensor.ops import moe_decode_vec as MV
    tuner = getattr(MV, name)
    every = tuner.configs
    metal = _pruned(tuner, 32768, monkeypatch)
    assert metal and all(4 * c.kwargs["BLOCK_K"] * c.kwargs["BLOCK_N"] <= 32768 for c in metal)
    assert len(metal) == sum(4 * c.kwargs["BLOCK_K"] * c.kwargs["BLOCK_N"] <= 32768 for c in every) < len(every)
    assert len(_pruned(tuner, 98304, monkeypatch)) == len(every)          # the V100: every candidate fits


def test_a_profile_without_a_budget_is_refused(monkeypatch):
    monkeypatch.setattr(_configs, "arch_smem_budget", lambda: None)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: a config space is pruned by memory.max_shared_memory"):
        _configs.fits_shared_memory(1024)
