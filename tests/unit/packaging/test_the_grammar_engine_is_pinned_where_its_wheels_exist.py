"""The grammar engine's dependency (pyproject.toml: nbxgrammar, pinned exactly, with platform markers) holds on EXACTLY
the platforms the pinned release has a wheel for on PyPI: a platform the markers name without a wheel would get a pin
pip cannot satisfy (the sdist needs Rust) — the whole install of neurobrix would fail there; a platform with a wheel
the markers miss would lose constrained decoding for nothing (decided between the Dell and the Mac, 2026-11-10 10:30).
The wheels of each release are recorded below as published (github.com/NeuroBrix/nbxgrammar, its wheels workflow; read
on PyPI at the upload): raising the pin means adding the new release's line. What this test would do on a drift: red."""
import re
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]
# release -> the (sys_platform, platform_machine) of every wheel PyPI holds for it
PUBLISHED = {
    # 0.1.0, 2026-11-10: manylinux2014 x86_64 + aarch64, macosx_11_0_arm64, macosx_10_12_x86_64, win_amd64
    "0.1.0": {("linux", "x86_64"), ("linux", "aarch64"), ("darwin", "arm64"), ("darwin", "x86_64"), ("win32", "AMD64")},
}


def _requirement():
    # the dependency's line read as text: the TOML reader of the standard library is Python 3.11+, and the rack's engine
    # Python is 3.10 — one quoted requirement naming nbxgrammar in pyproject.toml's dependencies
    text = (ROOT / "pyproject.toml").read_text()
    found = [Requirement(m) for m in re.findall(r'^\s*"(nbxgrammar[^"]*)",', text, re.M)]
    assert len(found) == 1, found
    return found[0]


def _pinned():
    spec = str(_requirement().specifier)
    assert spec.startswith("=="), spec
    return spec[2:]


def test_the_pin_is_a_release_whose_wheels_are_recorded():
    assert _pinned() in PUBLISHED, f"nbxgrammar {_pinned()}: record its published wheels in PUBLISHED"


def test_the_markers_name_exactly_the_platforms_with_a_wheel():
    marker = _requirement().marker
    assert isinstance(marker, Marker)
    wheels = PUBLISHED[_pinned()]
    others = {("linux", "ppc64le"), ("linux", "s390x"), ("linux", "armv7l"), ("win32", "ARM64"), ("win32", "x86"),
              ("freebsd14", "amd64"), ("darwin", "ppc"), ("linux", "x86"), ("linux", "64"), ("darwin", "arm")}
    held = {(p, m) for p, m in wheels | others if marker.evaluate({"sys_platform": p, "platform_machine": m})}
    assert held == wheels, (sorted(held - wheels), sorted(wheels - held))


def test_the_installed_grammar_engine_is_the_pinned_one():
    import importlib.metadata as md
    import pytest
    try:
        installed = md.version("nbxgrammar")
    except md.PackageNotFoundError:
        pytest.skip("nbxgrammar is not installed here")
    assert installed == _pinned()
