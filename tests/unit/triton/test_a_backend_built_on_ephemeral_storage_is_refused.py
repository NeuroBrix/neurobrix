"""A backend is refused when ANY leg of its installation sits on storage the
machine will wipe.

Written after the third loss (2026-09-22). The first two were a PACKAGE resolving
into a session scratchpad; a guard was said to exist for that and does not exist
in this tree at all. The third was wider: the package was installed into a venv
that is gone, from a git WORKTREE on `/private/tmp`, and the only surviving
artefacts were wheels from a different commit. Nothing refused, because nothing
looked.

Three legs, because losing any one of them loses the install:
  the package itself, the environment it is installed into, and the tree it was
  built from (pip records that in `direct_url.json`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobrix.core.paths import ephemeral_reason, installation_refusals


def test_a_session_scratchpad_is_ephemeral():
    why = ephemeral_reason("/private/tmp/claude-502/x/scratchpad/agpu-b9d5c06")
    assert why is not None
    assert "/private/tmp" in why or "tmp" in why


def test_the_macos_per_user_temp_is_ephemeral():
    assert ephemeral_reason("/var/folders/cn/bwr_xyjd/T/pip-ephem-wheel-cache/x") is not None


def test_a_durable_path_is_not_refused(tmp_path_factory):
    # A real durable location on this machine: the repo itself.
    assert ephemeral_reason(Path(__file__).resolve()) is None


def test_a_name_that_merely_contains_tmp_is_not_ephemeral():
    # `/Users/x/tmpwork` is NOT under a temp root; matching on the substring
    # would refuse a durable directory and teach everyone to disable the guard.
    assert ephemeral_reason("/Users/hocine/Workspace/tmpwork/pkg") is None


def test_every_leg_is_reported_not_just_the_first(monkeypatch, tmp_path):
    """A package that is durable but built from an ephemeral tree is still lost."""
    pkg = tmp_path / "site-packages" / "fake_backend"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    dist = tmp_path / "site-packages" / "fake_backend-0.1.0.dist-info"
    dist.mkdir()
    (dist / "direct_url.json").write_text(json.dumps(
        {"url": "file:///private/tmp/claude-502/dead/scratchpad/agpu-b9d5c06/backend/AppleGPU"}))

    reasons = installation_refusals("fake_backend", search_root=tmp_path / "site-packages")
    assert reasons, "a build tree on /private/tmp must be refused"
    assert any("built from" in r for r in reasons), reasons


def test_the_refusal_names_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.prefix", "/private/tmp/claude-502/dead/venv")
    reasons = installation_refusals("neurobrix", search_root=tmp_path)
    assert any("environment" in r for r in reasons), reasons


def test_the_engine_refuses_the_selected_backend_rather_than_warns(monkeypatch):
    """The predicate is not a guard until the engine raises on it — on the backend the library SELECTED, whose module
    the library names (`metal_backend.backend_module`); the library itself asks nothing (API.md section 11)."""
    from nbxtensor.backend import metal_backend as mb
    from neurobrix.core import paths
    seen = []
    monkeypatch.setattr(mb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mb, "selected_metal_backend", lambda: "triton_ext")
    monkeypatch.setattr(paths, "installation_refusals", lambda name, search_root=None: seen.append(name) or [
        f"the package {name} is installed at /private/tmp/dead/x, under /private/tmp, which the machine clears"])
    with pytest.raises(paths.BackendOnClearedStorage) as excinfo:
        paths.refuse_a_backend_on_cleared_storage()
    assert seen == [mb.backend_module("triton_ext")]
    assert "/private/tmp" in str(excinfo.value) and "durable" in str(excinfo.value)


def test_a_durable_backend_passes_and_another_machine_is_not_asked(monkeypatch):
    from nbxtensor.backend import metal_backend as mb
    from neurobrix.core import paths
    monkeypatch.setattr(mb, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mb, "selected_metal_backend", lambda: "triton_ext")
    monkeypatch.setattr(paths, "installation_refusals", lambda name, search_root=None: [])
    paths.refuse_a_backend_on_cleared_storage()
    monkeypatch.setattr(mb, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(mb, "selected_metal_backend", lambda: (_ for _ in ()).throw(AssertionError("asked")))
    paths.refuse_a_backend_on_cleared_storage()
