"""The engine's package installs on a machine other than the one that built it.

The stable's first reading (2026-11-05, wheels built from 30383f54): neurobrix's METADATA said
`Requires-Dist: nbxtensor @ file://~/nbx/worktrees/release-notes/nbxtensor` — pyproject's
`nbxtensor @ {root:uri}/nbxtensor` bakes the building machine's path into the wheel, so no other machine can install
it, and PyPI refuses a direct-URL dependency. The same wheel carried 45 files of 0 bytes (kernels/cache/) no code read.

What this test does if the packaging were wrong: a dependency named by a path or URL, or the hatch switch that allows
one, is back — red at the first test; an empty file other than a package marker is shipped — red at the second."""
from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]


def test_every_dependency_is_named_by_version_never_by_path():
    for project in (ROOT / "pyproject.toml",):     # the library's packaging is its own repository's
        meta = tomllib.loads(project.read_text())
        deps = list(meta["project"].get("dependencies", []))
        for extra in (meta["project"].get("optional-dependencies") or {}).values():
            deps += list(extra)
        direct = [d for d in deps if re.search(r"\s@\s|file:|://", d)]
        assert not direct, f"{project.relative_to(ROOT)} names a dependency by path or URL: {direct}"
        hatch = ((meta.get("tool") or {}).get("hatch") or {}).get("metadata") or {}
        assert not hatch.get("allow-direct-references"), f"{project.relative_to(ROOT)} allows direct references"


def test_the_packages_ship_no_empty_file():
    empty = [str(p.relative_to(ROOT)) for base in (ROOT / "src" / "neurobrix",)
             for p in base.rglob("*") if p.is_file() and p.stat().st_size == 0
             and p.name not in ("__init__.py", "py.typed") and "__pycache__" not in p.parts]
    assert not empty, empty


def test_the_engine_depends_on_one_exact_release_of_the_library():
    # 0.0.3 published, the engine still asked for nbxtensor==0.0.1 — which names CUDA on an AMD or Intel machine (the
    # supervisor 2026-11-06 22:10). The library lives in its own repository and is released first: the engine names ONE
    # release of it, exactly, and moves that pin when the library publishes the next one (docs/reference/RELEASING.md)
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pins = [d.replace(" ", "") for d in deps if re.match(r"nbxtensor\b", d)]
    assert len(pins) == 1 and re.fullmatch(r"nbxtensor==\d+\.\d+\.\d+", pins[0]), pins


def test_the_library_the_engine_imports_is_the_exact_release_it_pins():
    # A python holding another release of the library graded the engine against code it does not ship with (for example
    # a venv with 0.1.6 under a pin of 0.1.10): the version installed is the pinned one, read from pyproject, exactly
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pinned = next(d.replace(" ", "").split("==", 1)[1] for d in deps if re.match(r"nbxtensor\b", d))
    installed = importlib.metadata.version("nbxtensor")
    assert installed == pinned, (f"this python imports nbxtensor {installed}, pyproject.toml pins {pinned}: install "
                                 f"nbxtensor=={pinned}")


def test_the_pytorch_extra_asks_for_the_first_torch_with_the_allocator_setter():
    # the compiled engine applies its vendor profile's allocator settings through a setter first released in torch
    # 2.10.0 (core/runtime/torch_allocator.py): an extra that admits an older torch installs an engine that refuses
    from neurobrix.core.runtime.torch_allocator import SETTER_TORCH
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["pytorch"]
    assert extra == [f"torch>={SETTER_TORCH}"], extra
