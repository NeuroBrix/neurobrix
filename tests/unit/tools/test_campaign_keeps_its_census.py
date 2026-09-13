"""A cold campaign keeps the shapes it discovered.

`clear_caches()` empties the three caches before every arm so each is measured
COLD. One of the three is the autotune config store — which is also the census
the certified directory is built from (`autotune_certify.census` reads it by
default).

Measured 2026-09-11: a four-model campaign left a census of ZERO shapes. The
shapes had passed through the store one arm at a time and each clearing threw
the previous arm's away. Two correct requirements, one destroying the other.

They are reconciled by keeping the store aside before clearing it: the engine
still starts every launch from an empty store, and the shapes the catalogue
demanded survive the campaign that demanded them. Certifying shapes nobody
asked for is the alternative, and it is the wrong one.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_campaign_keeps_its_census.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import hub_family_sweep as sweep  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fake replay cache and a campaign directory."""
    home = tmp_path / "home"
    (home / ".neurobrix" / "replay_cache").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(sweep, "CACHES", ())        # nothing else to remove here
    out = tmp_path / "campaign"
    out.mkdir()
    monkeypatch.setattr(sweep, "CENSUS_KEEP", out)
    return home / ".neurobrix" / "replay_cache", out


def _write(rc: Path, entries: dict) -> None:
    (rc / "autotune_configs_metal-apple-m4-pro.json").write_text(
        json.dumps({"entries": entries}))


def _kept(out: Path) -> dict:
    f = out / "autotune_configs_metal-apple-m4-pro.json"
    return json.loads(f.read_text())["entries"] if f.exists() else {}


def test_the_store_is_emptied_so_the_next_arm_is_cold(store):
    rc, out = store
    _write(rc, {"k::a": 1})
    sweep.clear_caches()
    assert not list(rc.glob("autotune_configs_*.json")), (
        "the next arm would start warm, and a cold measurement is the point")


def test_what_the_arm_discovered_survives_the_clearing(store):
    rc, out = store
    _write(rc, {"kernel::(1, 2)": {"BLOCK": 64}})
    sweep.clear_caches()
    assert _kept(out) == {"kernel::(1, 2)": {"BLOCK": 64}}


def test_arms_accumulate_instead_of_replacing(store):
    """The failure that made this necessary: each clearing threw the previous
    arm's shapes away, so a four-model campaign ended with none."""
    rc, out = store
    _write(rc, {"kernel::(1, 2)": 1})
    sweep.clear_caches()
    _write(rc, {"kernel::(3, 4)": 2})
    sweep.clear_caches()
    _write(rc, {"other::(5,)": 3})
    sweep.clear_caches()
    assert _kept(out) == {"kernel::(1, 2)": 1, "kernel::(3, 4)": 2, "other::(5,)": 3}


def test_an_arm_that_discovered_nothing_does_not_erase_what_came_before(store):
    rc, out = store
    _write(rc, {"kernel::(1, 2)": 1})
    sweep.clear_caches()
    sweep.clear_caches()                    # no store at all this time
    assert _kept(out) == {"kernel::(1, 2)": 1}


def test_without_a_campaign_directory_nothing_is_kept_and_nothing_breaks(store, monkeypatch):
    rc, out = store
    monkeypatch.setattr(sweep, "CENSUS_KEEP", None)
    _write(rc, {"kernel::(1, 2)": 1})
    sweep.clear_caches()
    assert not list(rc.glob("autotune_configs_*.json"))
    assert _kept(out) == {}
