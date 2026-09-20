"""The MEMORY BUDGET is what gets quantised — onto a ladder — never the tile.

Measured 2026-09-20, one request (`real-esrgan-x8` at 1024x1024), four ambient
states: the sizing read 0.40 of the LIVE free figure (2.53 / 2.82 / 3.18 /
4.13 GB) and the tile followed it through four conv-extent families (405 / 429
/ 510 / 520). Every family demands ~7 fresh keys and one-shot runs persist no
sweeps, so the certified directory chased a set it could not enumerate.

The law (owner, 2026-09-20): the planner reads free memory as it always did,
rounds that figure DOWN onto a standard ladder of whole gigabytes, and
everything downstream — tile, overlap, band count — is derived from the RUNG,
exactly, with nothing downstream rounded. Three properties are the acceptance
criteria, not the motivation:

  * the tile is deterministic for a request on a machine, because the only
    noisy input was the reading and the ladder removes the noise at its source;
  * the plan never asks for everything it sees — rounding down leaves the
    difference between reading and rung free by construction, which is the
    headroom the autotuner's transients at (8 x tile)^2 had been taking from
    nowhere;
  * the ladder is vendor-neutral: unified memory and CUDA VRAM round by the
    same rule, so both backends tile by one law.

The ladder is fine at the bottom and coarse at the top — a uniform 2 GB step
discards half of an 8 GB machine and gives a 128 GB machine sixty-four key
families for nothing. The 4096 MB floor is the lowest usable rung. Validation
(does the plan FIT) stays against the LIVE figure: the rung sizes, the truth
validates.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neurobrix.core.paths import cache_dir
from neurobrix.core.prism.solver import (
    ComponentMemory, PrismSolver, memory_ladder_rung_mb,
)

MB = 1024 * 1024
CACHE = cache_dir()
MODEL = "real-esrgan-x8"


class _Container:
    def __init__(self, path):
        self._cache_path = Path(path)


def _spec(budget_mb):
    if not (CACHE / MODEL).exists():
        pytest.skip(f"{MODEL} is not in this machine's cache")
    s = PrismSolver.__new__(PrismSolver)
    from neurobrix.core.prism.profiler import InputConfig
    s._input_config = InputConfig(batch_size=1, height=1024, width=1024)
    mem = ComponentMemory("model", 32 * MB, 16384 * MB, 821 * MB)
    return s._spatial_component_tiling(
        _Container(CACHE / MODEL), "model", mem, budget_mb * MB)


def test_the_ladder_rounds_free_down_and_never_up():
    assert memory_ladder_rung_mb(6333) == 6 * 1024
    assert memory_ladder_rung_mb(7950) == 6 * 1024
    assert memory_ladder_rung_mb(8192) == 8 * 1024
    assert memory_ladder_rung_mb(10330) == 8 * 1024
    assert memory_ladder_rung_mb(130 * 1024) == 128 * 1024
    # the 4096 MB floor is the lowest USABLE rung: below it there is nothing
    # to size against, and the reading passes through untouched for the floor
    # machinery to refuse in its own words
    assert memory_ladder_rung_mb(3900) == 3900


def test_two_readings_on_one_rung_size_one_tile():
    """Readings 8.2 and 10.3 GB both sit on the 8 GB rung: same tile, same
    demanded conv extents — the key family closes."""
    a = _spec(8300)
    b = _spec(10330)
    assert a is not None and b is not None
    assert a["tile_size"] == b["tile_size"], (
        f"one rung, two tiles ({a['tile_size']} vs {b['tile_size']}): the "
        f"sizing is still reading the weather")


def test_readings_on_different_rungs_may_differ_and_the_rung_is_named():
    a = _spec(6333)
    b = _spec(10330)
    assert a is not None and b is not None
    assert a.get("budget_rung_mb") == 6 * 1024, a
    assert b.get("budget_rung_mb") == 8 * 1024, b
    assert a["tile_size"] <= b["tile_size"]


def test_the_rung_never_exceeds_the_reading():
    """Property two: the plan asks for less than it sees, by construction."""
    for reading in (4100, 5000, 6333, 7950, 10330, 13000):
        assert memory_ladder_rung_mb(reading) <= reading
