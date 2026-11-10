"""The grammar engine's dependency (pyproject.toml: nbxgrammar, pinned exactly, with platform markers) holds on EXACTLY
the platforms its CI builds a wheel for (rust/nbxgrammar/.github/workflows/wheels.yml): a platform the markers name and
the CI does not would get a pin pip cannot satisfy (no wheel, and the sdist needs Rust) — the whole install would fail;
a platform with a wheel the markers miss would lose constrained decoding for nothing (decided between the Dell and the
Mac, 2026-11-10 10:30). The pin is the package's own version. What this test would do if they drifted: red."""
import re
from pathlib import Path

import yaml
from packaging.markers import Marker
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]
OS = {"ubuntu": "linux", "macos": "darwin", "windows": "win32"}


def _requirement():
    # the dependency's line read as text: the TOML reader of the standard library is Python 3.11+, and the rack's engine
    # Python is 3.10 — one quoted requirement naming nbxgrammar in pyproject.toml's dependencies
    text = (ROOT / "pyproject.toml").read_text()
    found = [Requirement(m) for m in re.findall(r'^\s*"(nbxgrammar[^"]*)",', text, re.M)]
    assert len(found) == 1, found
    return found[0]


def _ci_platforms():
    jobs = yaml.safe_load((ROOT / "rust/nbxgrammar/.github/workflows/wheels.yml").read_text())["jobs"]["wheels"]
    return {(OS[e["os"].split("-")[0]], e["arch"]) for e in jobs["strategy"]["matrix"]["include"]}


def test_the_markers_name_exactly_the_platforms_with_a_wheel():
    marker = _requirement().marker
    assert isinstance(marker, Marker)
    ci = _ci_platforms()
    others = {("linux", "ppc64le"), ("linux", "s390x"), ("linux", "armv7l"), ("win32", "ARM64"), ("win32", "x86"),
              ("freebsd14", "amd64"), ("darwin", "ppc"), ("linux", "x86"), ("linux", "64"), ("darwin", "arm")}
    held = {(p, m) for p, m in ci | others if marker.evaluate({"sys_platform": p, "platform_machine": m})}
    assert held == ci, (sorted(held - ci), sorted(ci - held))


def test_the_pin_is_the_package_s_own_version():
    req = _requirement()
    version = re.search(r'^version = "([^"]+)"', (ROOT / "rust/nbxgrammar/pyproject.toml").read_text(), re.M).group(1)
    assert str(req.specifier) == f"=={version}"
