"""The engine's package installs on a machine other than the one that built it.

The stable's first reading (2026-11-05, wheels built from 30383f54): neurobrix's METADATA said
`Requires-Dist: nbxtensor @ file:///home/mlops/nbx/worktrees/release-notes/nbxtensor` — pyproject's
`nbxtensor @ {root:uri}/nbxtensor` bakes the building machine's path into the wheel, so no other machine can install
it, and PyPI refuses a direct-URL dependency. The same wheel carried 45 files of 0 bytes (kernels/cache/) no code read.

What this test does if the packaging were wrong: a dependency named by a path or URL, or the hatch switch that allows
one, is back — red at the first test; an empty file other than a package marker is shipped — red at the second."""
from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]


def test_every_dependency_is_named_by_version_never_by_path():
    for project in (ROOT / "pyproject.toml", ROOT / "nbxtensor" / "pyproject.toml"):
        meta = tomllib.loads(project.read_text())
        deps = list(meta["project"].get("dependencies", []))
        for extra in (meta["project"].get("optional-dependencies") or {}).values():
            deps += list(extra)
        direct = [d for d in deps if re.search(r"\s@\s|file:|://", d)]
        assert not direct, f"{project.relative_to(ROOT)} names a dependency by path or URL: {direct}"
        hatch = ((meta.get("tool") or {}).get("hatch") or {}).get("metadata") or {}
        assert not hatch.get("allow-direct-references"), f"{project.relative_to(ROOT)} allows direct references"


def test_the_packages_ship_no_empty_file():
    empty = [str(p.relative_to(ROOT)) for base in (ROOT / "src" / "neurobrix", ROOT / "nbxtensor" / "src" / "nbxtensor")
             for p in base.rglob("*") if p.is_file() and p.stat().st_size == 0
             and p.name not in ("__init__.py", "py.typed") and "__pycache__" not in p.parts]
    assert not empty, empty
