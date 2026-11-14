"""Suite-wide conftest.

A test that cannot run must SAY so, not fail: on a host with no device to give it (no card visible to the process,
or a GPU runtime installed without its driver), a card test skips by name. The condition and the conversion live in
ONE place, the library suite's `nbxtensor/tests/_no_device.py` (the library owns the device question; its door,
`nbxtensor.device.count()`, is what is asked), and this file registers the same hooks for the engine's suite. Its
docstring says how "no driver" is told from "a broken driver": a host whose driver is there and refuses is never
armed, and its failures stay failures. (Measured on 2026-09-16 under `CUDA_VISIBLE_DEVICES=`: 50 failures, 42 of them
the absence of a card; on 2026-11-13 on a host whose runtime has no driver: 169.)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _no_device_module():
    name = "_no_device"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "nbxtensor" / "tests" / "_no_device.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_ND = _no_device_module()
pytest_configure = _ND.pytest_configure
pytest_runtest_setup = _ND.pytest_runtest_setup
pytest_runtest_call = _ND.pytest_runtest_call
pytest_make_collect_report = _ND.pytest_make_collect_report


@pytest.fixture
def tl_dot_gemms(monkeypatch):
    """The bound profile with its matrix unit undeclared: every GEMM keeps its tl.dot kernel and its key.

    A cell that asserts a GEMM's or a flash call's autotune KEY speaks of that path. On a profile that declares
    `matrix_unit` (Volta's m8n8k4, its fp32 split included) the unit takes those launches with no key
    (`launch_keys.matrix_unit_takes`), so such a cell bound to the V100 profile went red the day the unit
    was declared — the key it asserts is still the one an arch without the unit launches. The ONE decision
    point, `_configs.matrix_unit`, answers `{}` (what it answers for a profile that declares none); the
    unit's own placement is asserted by the cells that leave it declared."""
    from nbxtensor.ops import _configs
    monkeypatch.setattr(_configs, "matrix_unit", lambda: {})


def installed_container(name: str) -> Path:
    """The container `name` in the cache the ENGINE resolves, or a skip that names it and the path.

    THE ONE PLACE a test learns whether a real container is on this machine. The cache root is
    `neurobrix.core.paths.cache_dir()` — `NEUROBRIX_CACHE`, then `~/.neurobrix/paths.json`, then the
    default — never a literal in a test: a literal is blind to where another machine keeps its
    models, and blind to the run that hides the cache to prove this very skip.

    Installed means extracted: its `manifest.json` is there, the first file every reader opens.
    Absent is not a result, so it is a SKIP, by name — before 2026-10-12 a machine without a model
    (the M4, 140 failed + 14 errors under tests/unit/{prism,runtime}) reported one red per test that
    plans it, and a real red drowned in them. Everything PAST the manifest — a missing component, a
    graph that will not load — is the container's defect and fails, so this helper is called first
    and never wrapped in a try."""
    from neurobrix.core import paths
    root = paths.cache_dir() / name
    if not (root / "manifest.json").is_file():
        pytest.skip(f"container {name} not installed at {root}")
    return root

# The engine provides its tensor library with what the library does not own (src/neurobrix/_nbxtensor_host.py): a test
# that imports nbxtensor alone runs under the engine all the same, whatever test ran before it.
import neurobrix  # noqa: E402,F401


@pytest.fixture
def engine_log(caplog):
    """The engine's lines as a test reads them: `engine_log()` returns the text of every record the `neurobrix.*` and
    `nbxtensor.*` loggers emitted since the last call (every level — what `-v` / NBX_LOG=debug prints), and consumes
    them, as `capsys.readouterr()` does. The engine logs; it does not print (nbxtensor/terminal.py): a test that read its
    lines from stdout reads them here."""
    import logging
    caplog.set_level(logging.DEBUG, logger="neurobrix")
    caplog.set_level(logging.DEBUG, logger="nbxtensor")

    def text() -> str:
        out = "\n".join(r.getMessage() for r in caplog.records)
        caplog.clear()
        return out
    return text
