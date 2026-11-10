"""tools/intel_spirv_census.py measures Intel's compiler: run with a Triton that has no Intel backend (the CUDA build of
the engine's environment) it must refuse by name, never census the wrong compiler; an unknown device file is refused
with the list of those that exist. What this would do if the code were wrong: a census of CUDA's compiler written as
Intel's — RED here (no refusal)."""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def _tool():
    sys.path.insert(0, str(TOOLS))
    try:
        return importlib.import_module("intel_spirv_census")
    finally:
        sys.path.remove(str(TOOLS))


def test_a_triton_without_the_intel_backend_is_refused_by_name():
    from triton import knobs
    if hasattr(knobs, "intel"):
        pytest.skip("this environment holds Intel's Triton build: the census runs here")
    with pytest.raises(SystemExit, match="no Intel backend"):
        _tool().build_target("mtl", "none")


def test_an_unknown_device_file_is_refused_with_those_that_exist(monkeypatch):
    from triton import knobs
    monkeypatch.setattr(knobs, "intel", object(), raising=False)
    with pytest.raises(SystemExit, match="no Intel device file nope.yml .*mtl"):
        _tool().build_target("nope", "none")
