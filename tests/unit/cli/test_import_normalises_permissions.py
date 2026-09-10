"""An imported container has the same permissions whoever imported it.

`zipfile.extractall` does not apply the permission bits stored in the archive:
every extracted file takes the umask of the process that happened to run the
import. Measured on this machine — two models imported in May carried their
whole content owner-only (manifest, profile, weights index, topology, twelve
files) while a third imported in August was world-readable. Same engine, same
archives, different shell.

The state on disk of an artefact must not depend on the environment of whoever
imported it, in an engine that sells determinism.

Note on the neighbouring question: `extractall` here IS preceded by a member
path check — every member is resolved and refused if it escapes the cache
directory. Path traversal is covered; this is only about the mode bits.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_import_normalises_permissions.py
"""
from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

import pytest


def _archive(tmp_path: Path) -> Path:
    p = tmp_path / "model.nbx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("topology.json", "{}")
        zf.writestr("components/transformer/graph.json", "{}")
        zf.writestr("weights/index.json", "{}")
    return p


def _modes(root: Path):
    files, dirs = {}, {}
    for p in root.rglob("*"):
        m = stat.S_IMODE(p.stat().st_mode)
        (dirs if p.is_dir() else files)[str(p.relative_to(root))] = m
    return files, dirs


@pytest.mark.parametrize("umask", [0o077, 0o022, 0o000])
def test_the_mode_does_not_depend_on_the_importer_s_umask(tmp_path, umask):
    from neurobrix.cli.commands.registry import extract_container

    src = _archive(tmp_path)
    dest = tmp_path / f"cache_{umask:03o}"
    old = os.umask(umask)
    try:
        extract_container(src, dest)
    finally:
        os.umask(old)

    files, dirs = _modes(dest)
    assert files, "nothing was extracted"
    for name, mode in files.items():
        assert mode == 0o644, (
            f"under umask {umask:03o}, {name} came out {mode:o} — an artefact's "
            f"mode must not depend on the shell that imported it")
    for name, mode in dirs.items():
        assert mode == 0o755, f"under umask {umask:03o}, dir {name} came out {mode:o}"


def test_every_umask_gives_the_same_tree(tmp_path):
    """The property stated directly: two imports, two environments, one result."""
    from neurobrix.cli.commands.registry import extract_container

    src = _archive(tmp_path)
    out = {}
    for umask in (0o077, 0o022):
        dest = tmp_path / f"t{umask:03o}"
        old = os.umask(umask)
        try:
            extract_container(src, dest)
        finally:
            os.umask(old)
        out[umask] = _modes(dest)
    assert out[0o077] == out[0o022]


def test_a_member_escaping_the_cache_is_refused(tmp_path):
    """The check that already existed, kept under test so it stays."""
    from neurobrix.cli.commands.registry import extract_container

    p = tmp_path / "evil.nbx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("../escaped.json", "{}")
    with pytest.raises(ValueError, match="traversal"):
        extract_container(p, tmp_path / "cache")
