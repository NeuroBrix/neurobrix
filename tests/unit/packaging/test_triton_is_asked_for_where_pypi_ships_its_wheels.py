"""neurobrix's triton dependency (pyproject.toml, unpinned) holds on EXACTLY the platforms PyPI publishes a triton
wheel for: a platform the marker names without a wheel has no solution, and the whole install of neurobrix fails
there (native Windows, 0.6.16: GitHub issue 6; triton ships no sdist that builds there); a platform with a wheel the
marker misses loses the Triton engine for nothing. Darwin has none on PyPI: a Mac's triton is the Metal backend's.

Recorded as read on PyPI (pypi.org/pypi/triton/json, 2026-11-18 03:5x): latest 3.9.0; every wheel of every release is
manylinux x86_64 (from the first) or manylinux aarch64 (from 3.7.1); none for win, none for macOS. A drift: red."""
import re
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]
#: the (sys_platform, platform_machine) of every wheel PyPI holds for triton, any release
PUBLISHED = {("linux", "x86_64"), ("linux", "aarch64")}


def _requirement():
    # read as text: the standard library's TOML reader is Python 3.11+, the rack's engine Python is 3.10
    text = (ROOT / "pyproject.toml").read_text()
    found = [Requirement(m) for m in re.findall(r'^\s*"(triton[^"]*)",', text, re.M)]
    assert len(found) == 1, found
    return found[0]


def test_the_triton_marker_names_exactly_the_platforms_with_a_wheel():
    marker = _requirement().marker
    assert isinstance(marker, Marker)
    others = {("win32", "AMD64"), ("win32", "ARM64"), ("win32", "x86"), ("darwin", "arm64"), ("darwin", "x86_64"),
              ("linux", "ppc64le"), ("linux", "s390x"), ("linux", "armv7l"), ("freebsd14", "amd64"), ("linux", "x86"),
              ("linux", "64"), ("cygwin", "x86_64")}
    held = {(p, m) for p, m in PUBLISHED | others
            if marker.evaluate({"sys_platform": p, "platform_machine": m})}
    assert held == PUBLISHED, (sorted(held - PUBLISHED), sorted(PUBLISHED - held))


def test_native_windows_and_macos_ask_for_no_triton():
    marker = _requirement().marker
    for env in ({"sys_platform": "win32", "platform_machine": "AMD64", "platform_system": "Windows"},
                {"sys_platform": "darwin", "platform_machine": "arm64", "platform_system": "Darwin"},
                {"sys_platform": "darwin", "platform_machine": "x86_64", "platform_system": "Darwin"}):
        assert not marker.evaluate(env), env
