"""The headroom a cell must leave free is a share of the host it runs on, not a constant.

`HOST_HEADROOM` was 16 GiB, measured as a margin on the rack's 251 GB host. On the Mac (24 GiB of
unified memory, 5.5-13 GiB available depending on the owner's VM) a cell is admitted only when the
available memory covers its need, the owed growth and that headroom — 16 GiB alone is more than the
Mac ever has available, so `run_cell` slept in its admission loop for ever, every cell, whatever its
plan (read from the code 2026-10-03 20:50; the earlier harness never got that far, it died on
/proc/meminfo). The headroom is now `HOST_HEADROOM_SHARE` of the host: 1/16, which is the same
~16 GiB on the rack and 1.5 GiB on the Mac. Before this branch the first test fails (not admitted).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GiB = 1 << 30


def _host(monkeypatch, total, available):
    monkeypatch.setattr(R, "_host_bytes", lambda: total)
    monkeypatch.setattr(R, "_mem_available", lambda: available)


def test_a_small_host_admits_a_cell_its_available_memory_covers(tmp_path, monkeypatch):
    _host(monkeypatch, 24 * GiB, 10 * GiB)
    assert R.reserve_host(tmp_path, 4 * GiB) is True


def test_a_small_host_still_keeps_its_headroom(tmp_path, monkeypatch):
    _host(monkeypatch, 24 * GiB, 10 * GiB)
    headroom = int(24 * GiB * R.HOST_HEADROOM_SHARE)
    assert R.reserve_host(tmp_path, 10 * GiB - headroom + 1) is False


def test_the_rack_keeps_its_sixteen_gibibytes(tmp_path, monkeypatch):
    _host(monkeypatch, 256 * GiB, 100 * GiB)
    assert int(256 * GiB * R.HOST_HEADROOM_SHARE) == 16 * GiB
    assert R.reserve_host(tmp_path, 84 * GiB + 1) is False
    assert R.reserve_host(tmp_path, 84 * GiB) is True
