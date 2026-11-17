"""A vendor file is split once, on the device-key census (the library repository's docs/device_keys.txt;
tools/split_vendor_files.py): every key the library reads lives in the library's device file (the installed package's),
the engine's vendor file keeps the rest, and no key is in both. Checked for every vendor the library already describes
— a vendor not yet split names itself here. The census list is not shipped with the library: that the engine's half
holds no census key is the library repository's check; here, that no key is in both halves."""
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
LIB = Path(importlib.util.find_spec("nbxtensor").origin).parent / "backend" / "devices"
ENG = REPO / "src" / "neurobrix" / "config" / "vendors"
SPLIT = sorted(p.name for p in LIB.iterdir() if p.is_dir())


def test_the_split_vendors_are_named():
    assert "apple" in SPLIT, SPLIT


@pytest.mark.parametrize("vendor", SPLIT)
def test_no_key_is_in_both_halves(vendor):
    from nbxtensor.backend import device_files as D
    for eng in sorted((ENG / vendor).glob("*.yml")):
        engine_half = yaml.safe_load(eng.read_text()) or {}
        D.merge(D.read(vendor, eng.stem), engine_half, f"{vendor}/{eng.stem}: ")      # refuses a key in both


@pytest.mark.parametrize("vendor", SPLIT)
def test_every_library_device_has_its_engine_file(vendor):
    lib = {p.stem for p in (LIB / vendor).glob("*.yml")}
    eng = {p.stem for p in (ENG / vendor).glob("*.yml")}
    assert lib <= eng, f"library device files with no engine vendor file: {sorted(lib - eng)}"
