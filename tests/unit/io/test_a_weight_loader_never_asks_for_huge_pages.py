"""Neither weight loader lets numpy ask for transparent huge pages (`io.host_huge_pages: false`).

numpy madvises MADV_HUGEPAGE on every array over 4 MiB; with the kernel's THP defrag at `madvise` a first touch
may compact memory synchronously, and under a page cache full of weight shards Qwen3-VL-30B's Triton load sat
30 min in `compact_zone` (V100 rack, 2026-10-23). Both engines' loaders apply the configured policy before
their first buffer.

What this test would do if the code were wrong: a loader that does not apply the policy leaves numpy's switch
on — red at the switch read-back of that loader (inject: drop `apply_host_huge_pages()` from either loader).
"""
from __future__ import annotations

import pytest
from numpy._core import multiarray as _ma

from neurobrix.core import workspace


@pytest.fixture
def numpy_asks():
    """numpy's switch on (its default) and the once-per-process latch open; both restored."""
    before = _ma._set_madvise_hugepage(True)
    workspace._huge_pages_set = False
    yield
    workspace._huge_pages_set = False
    _ma._set_madvise_hugepage(before)


def _asks_now() -> bool:
    on = _ma._set_madvise_hugepage(True)
    _ma._set_madvise_hugepage(on)
    return on


def test_the_configured_policy_reaches_numpy(numpy_asks):
    workspace.apply_host_huge_pages()
    assert _asks_now() is False


def test_the_triton_loader_applies_it_before_its_first_buffer(numpy_asks, tmp_path):
    from neurobrix.triton import weight_loader as TWL
    assert TWL.load_component_weights(str(tmp_path), "model", 0) == {}   # a component that stores no weights
    assert _asks_now() is False


def test_the_compiled_loader_applies_it_before_its_first_buffer(numpy_asks, tmp_path):
    from neurobrix.core.io.weight_loader import WeightLoader
    with pytest.raises(FileNotFoundError):
        WeightLoader(tmp_path / "absent.nbx")
    assert _asks_now() is False
