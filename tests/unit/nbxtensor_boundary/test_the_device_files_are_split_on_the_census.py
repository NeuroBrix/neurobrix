"""A vendor file is split once, on the device-key census (docs/nbxtensor/device_keys.txt; tools/split_vendor_files.py):
every key the library reads lives in the library's device file, the engine's vendor file keeps the rest, and no key
is in both. Checked for every vendor the library already describes — a vendor not yet split names itself here."""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "nbxtensor" / "src" / "nbxtensor" / "backend" / "devices"
ENG = REPO / "src" / "neurobrix" / "config" / "vendors"
KEYS = set((REPO / "docs" / "nbxtensor" / "device_keys.txt").read_text().split())
SPLIT = sorted(p.name for p in LIB.iterdir() if p.is_dir())


def _paths(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for k2 in v:
                yield f"{k}.{k2}"
        else:
            yield k


def _census_names(path):
    """The census names a 2-level key path through itself, a deeper path, or its section taken whole."""
    top = path.split(".")[0]
    whole = top in KEYS and not any(k.startswith(top + ".") for k in KEYS)
    return whole or path in KEYS or any(k.startswith(path + ".") for k in KEYS)


def test_the_split_vendors_are_named():
    assert "apple" in SPLIT, SPLIT


@pytest.mark.parametrize("vendor", SPLIT)
def test_no_census_key_is_left_in_the_engine_and_none_is_in_both(vendor):
    from nbxtensor.backend import device_files as D
    for eng in sorted((ENG / vendor).glob("*.yml")):
        engine_half = yaml.safe_load(eng.read_text()) or {}
        left = [p for p in _paths(engine_half) if _census_names(p)]
        assert not left, f"{eng.name}: keys the library reads are still in the engine's file: {left}"
        D.merge(D.read(vendor, eng.stem), engine_half, f"{vendor}/{eng.stem}: ")      # refuses a key in both


@pytest.mark.parametrize("vendor", SPLIT)
def test_every_library_device_has_its_engine_file(vendor):
    lib = {p.stem for p in (LIB / vendor).glob("*.yml")}
    eng = {p.stem for p in (ENG / vendor).glob("*.yml")}
    assert lib <= eng, f"library device files with no engine vendor file: {sorted(lib - eng)}"
