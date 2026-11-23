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


def _library_pin():
    """tools/library_pin.py, the one reading of the engine's nbxtensor pin (loaded by path: tools is no package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("library_pin", ROOT / "tools" / "library_pin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_dependency_is_named_by_version_never_by_path():
    for project in (ROOT / "pyproject.toml",):     # the library's packaging is its own repository's
        meta = tomllib.loads(project.read_text())
        deps = list(meta["project"].get("dependencies", []))
        for extra in (meta["project"].get("optional-dependencies") or {}).values():
            deps += list(extra)
        # the ONE exception between releases: the library pinned by its commit (supervisor 23/11 01:01), read by
        # `library_pin.pin_of`; at a release it is the published version and hatch allows no direct reference
        pin = _library_pin().pin_of(meta["project"].get("dependencies", []))
        direct = [d for d in deps if re.search(r"\s@\s|file:|://", d) and not re.match(r"nbxtensor\b", d.strip())]
        assert not direct, f"{project.relative_to(ROOT)} names a dependency by path or URL: {direct}"
        hatch = ((meta.get("tool") or {}).get("hatch") or {}).get("metadata") or {}
        assert bool(hatch.get("allow-direct-references")) == (pin.commit is not None), (
            f"{project.relative_to(ROOT)}: direct references allowed {hatch.get('allow-direct-references')!r} under "
            f"the pin {pin}: allowed only while the library is pinned by commit")


def test_the_packages_ship_no_empty_file():
    empty = [str(p.relative_to(ROOT)) for base in (ROOT / "src" / "neurobrix",)
             for p in base.rglob("*") if p.is_file() and p.stat().st_size == 0
             and p.name not in ("__init__.py", "py.typed") and "__pycache__" not in p.parts]
    assert not empty, empty


def test_the_engine_depends_on_one_exact_release_of_the_library():
    # 0.0.3 published, the engine still asked for nbxtensor==0.0.1 — which names CUDA on an AMD or Intel machine (the
    # supervisor 2026-11-06 22:10). The library lives in its own repository and is released first: the engine names ONE
    # release of it, exactly, and moves that pin when the library publishes the next one (docs/reference/RELEASING.md)
    # Between releases that one is a COMMIT of the library's repository (supervisor 23/11 01:01), at a release its
    # published version: `library_pin.pin_of` refuses anything else by name.
    pin = _library_pin().pin_of(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
    assert (pin.commit is None) != (pin.release is None), pin


def test_the_library_the_engine_imports_is_the_exact_release_it_pins():
    # A python holding another release of the library graded the engine against code it does not ship with (for example
    # a venv with 0.1.6 under a pin of 0.1.10): the version installed is the pinned one, read from pyproject, exactly
    # exactly — by commit between releases: 0.1.11 named both the release and the commit main imported
    # `one_device_env` from, which the release lacks (2026-11-23)
    lp = _library_pin()
    pin = lp.pin_of(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
    import nbxtensor
    assert lp.installed_matches(pin), (f"this python imports nbxtensor {nbxtensor.__version__} at commit "
                                    f"{lp.installed_commit()}, pyproject.toml pins {pin.commit or pin.release}")


def test_the_pytorch_extra_asks_for_the_first_torch_with_every_api_the_compiled_engine_calls():
    # torch 2.12.0 first ships torch.accelerator.empty_host_cache (the loader calls it after every cuda/hip load) beside
    # the allocator setter (2.10.0): an extra that admits an older torch installs an engine that refuses
    # (core/runtime/torch_allocator.py COMPILED_TORCH_APIS reads this line back from the package metadata). macOS reaches
    # neither call and PyPI has no torch 2.12 for an Intel Mac or a macOS older than 14: there it asks what 0.6.16 asked
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["pytorch"]
    assert extra == ["torch>=2.12.0; sys_platform != 'darwin'", "torch>=2.1.0; sys_platform == 'darwin'"], extra
