"""`python -m nbxtensor install-metal` installs Triton for the Apple GPU from the release of the project's fork
(github.com/NeuroBrix/triton-ext), each wheel by the URL and sha256 written in the code, through pip with --no-deps --require-hashes --no-index — never an extra index,
never a package name pip would resolve on PyPI (anyone can publish "triton-apple-backend" there: dependency
confusion). Off an Apple-silicon Mac, or on a Python the table does not cover, it refuses by name. `python -m nbxtensor
doctor` on a Mac without Triton names the command (the owner via the supervisor, 2026-11-07 00:45).

What this test would do if the code were wrong: a run on Linux that tries to install (red), a covered Python missing
from the table that installs anyway (red), a pip command reading an index or resolving dependencies (red), an entry
whose URL leaves the release or whose hash is not a sha256 (red), the table changed without this file (red)."""
from __future__ import annotations

import re
import sys

import pytest

from nbxtensor.backend import install_metal as M

#: the table as published — this file changes with it, by hand, from the files the release carries
EXPECTED_METAL_WHEELS: dict = {}


def _mac(monkeypatch):
    monkeypatch.setattr(M.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(M.platform, "machine", lambda: "arm64")


def test_the_table_is_the_published_one():
    assert M.METAL_WHEELS == EXPECTED_METAL_WHEELS


def test_every_entry_is_a_release_url_of_the_fork_with_a_sha256():
    for tag, entries in M.METAL_WHEELS.items():
        assert re.fullmatch(r"cp3\d+", tag), tag
        names = [url.rsplit("/", 1)[1] for url, _ in entries]
        assert [n.split("-")[0] for n in names] == ["triton", "nbx_triton_apple"], names   # Triton first, then the backend
        for (url, sha), name in zip(entries, names):
            assert url.startswith(M.RELEASES) and url.count("/") == M.RELEASES.count("/") + 1, url
            assert re.fullmatch(r"[A-Za-z0-9_.]+-[^-]+-%s-%s-macosx_\d+_\d+_arm64\.whl" % (tag, tag), name), name
            assert re.fullmatch(r"[0-9a-f]{64}", sha), (name, sha)


@pytest.mark.parametrize("system, machine", [("Linux", "x86_64"), ("Darwin", "x86_64"), ("Windows", "AMD64")])
def test_off_an_apple_silicon_mac_it_refuses_by_name(monkeypatch, system, machine):
    monkeypatch.setattr(M.platform, "system", lambda: system)
    monkeypatch.setattr(M.platform, "machine", lambda: machine)
    with pytest.raises(M.InstallMetalRefused, match="Apple-silicon Mac only"):
        M.wheels("0.0.4")
    calls = []
    monkeypatch.setattr(M.subprocess, "call", lambda *a, **k: calls.append(a) or 0)
    assert M.main("0.0.4") == 1 and not calls


def test_a_python_the_table_does_not_cover_is_refused_by_name(monkeypatch):
    _mac(monkeypatch)
    monkeypatch.setattr(M, "METAL_WHEELS", {"cp399": (("triton-3.8.0-cp399-cp399-macosx_15_0_arm64.whl", "0" * 64),)})
    with pytest.raises(M.InstallMetalRefused, match=r"no Metal wheels for Python .*covered: cp399"):
        M.wheels("0.0.4")


def test_a_covered_python_installs_by_url_and_hash_only(monkeypatch):
    _mac(monkeypatch)
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    base = M.RELEASES + "apple-v0.1.0+triton3.8.0/"
    t = f"{base}triton-3.8.0-{tag}-{tag}-macosx_15_0_arm64.whl"
    b = f"{base}nbx_triton_apple-0.1.0-{tag}-{tag}-macosx_15_0_arm64.whl"
    monkeypatch.setattr(M, "METAL_WHEELS", {tag: ((t, "a" * 64), (b, "b" * 64))})
    seen = {}

    def call(cmd):
        seen["cmd"] = cmd
        seen["req"] = open(cmd[-1]).read()
        return 0
    monkeypatch.setattr(M.subprocess, "call", call)
    assert M.main("0.0.4") == 0
    assert seen["req"] == f"{t} --hash=sha256:{'a' * 64}\n{b} --hash=sha256:{'b' * 64}\n"
    cmd = seen["cmd"]
    assert cmd[:4] == [sys.executable, "-m", "pip", "install"]
    assert {"--no-deps", "--require-hashes", "--no-index"} <= set(cmd), cmd
    assert not any(a.startswith(("--index-url", "--extra-index-url", "-i", "--find-links", "-f")) for a in cmd), cmd


def test_doctor_on_a_mac_without_triton_names_the_command(monkeypatch, capsys):
    import nbxtensor.__main__ as main_mod
    from nbxtensor.backend import metal_backend
    monkeypatch.setattr(metal_backend, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(metal_backend, "triton_metal_available", lambda: False)
    assert main_mod.main(["doctor"]) == 0
    assert capsys.readouterr().out.strip() == (
        "Apple GPU found; Triton for Metal not installed — run: python -m nbxtensor install-metal")


def test_the_command_line_names_install_metal(capsys):
    import nbxtensor.__main__ as main_mod
    assert main_mod.main(["nothing"]) == 2
    assert "install-metal" in capsys.readouterr().err
