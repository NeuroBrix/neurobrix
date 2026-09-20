"""Xcode 26/27's broken toolchain delegation is detected and fixed, once.

macOS 26 / Xcode 26+ ship the Metal compiler as a separate on-demand toolchain.
Once installed it runs, but Xcode's DEFAULT-toolchain shim does not delegate to
it, so a bare `xcrun metal` fails with "missing Metal Toolchain" even though the
compiler is present. `TOOLCHAINS=Metal` selects it. This is a recognised Apple
defect since Xcode 26; every user on 26/27 meets it, so the backend fixes it
rather than leaving each user to find `TOOLCHAINS` alone.

Backend-agnostic: mocks the `xcrun metal` probe, so it runs on any host.
"""
from __future__ import annotations

import pytest

from neurobrix.triton import metal_backend as MB


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(MB, "_DELEGATION_ANNOUNCED", False)
    monkeypatch.delenv("TOOLCHAINS", raising=False)
    monkeypatch.setattr(MB, "is_apple_silicon", lambda: True)
    yield


def test_broken_delegation_is_selected_and_announced_once(monkeypatch, capsys):
    # Bare metal fails; TOOLCHAINS=Metal makes it run — the delegation defect.
    def runs(env_extra=None):
        return bool(env_extra) and env_extra.get("TOOLCHAINS") == "Metal"
    monkeypatch.setattr(MB, "_metal_runs", runs)

    MB.ensure_metal_toolchain_selected()
    import os
    assert os.environ.get("TOOLCHAINS") == "Metal"
    out = capsys.readouterr().out
    assert "does not delegate" in out

    # Announced ONCE: a second call is silent (and a no-op — TOOLCHAINS is set).
    MB.ensure_metal_toolchain_selected()
    assert "does not delegate" not in capsys.readouterr().out


def test_a_working_delegation_is_left_alone(monkeypatch, capsys):
    monkeypatch.setattr(MB, "_metal_runs", lambda env_extra=None: True)   # bare runs
    MB.ensure_metal_toolchain_selected()
    import os
    assert os.environ.get("TOOLCHAINS") is None       # nothing forced
    assert capsys.readouterr().out == ""              # nothing announced


def test_an_explicit_toolchain_is_respected(monkeypatch):
    monkeypatch.setenv("TOOLCHAINS", "Something")
    monkeypatch.setattr(MB, "_metal_runs",
                        lambda env_extra=None: pytest.fail("must not probe when the caller chose"))
    MB.ensure_metal_toolchain_selected()
    import os
    assert os.environ["TOOLCHAINS"] == "Something"    # the caller's choice stands


def test_a_genuinely_absent_compiler_is_not_masked(monkeypatch, capsys):
    monkeypatch.setattr(MB, "_metal_runs", lambda env_extra=None: False)   # nothing runs
    MB.ensure_metal_toolchain_selected()
    import os
    assert os.environ.get("TOOLCHAINS") is None       # not set to a lie
    assert capsys.readouterr().out == ""              # no false "fixed" claim
