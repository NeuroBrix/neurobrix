"""A test that needs the workshop configuration is skipped by name on a machine without it, and only there
(`tests/unit/workshop_configuration.py`).

On a clean host, 78 test modules that import a tool reading the configuration at import stopped their folder's
collection with `AtelierUndeclared` (tools, prism, kernels, census, regression: 0 tests ran without
`--continue-on-collection-errors`). What this test does if the helper were wrong: an absent configuration not turned
into a skip, or a skip that does not name the file and its variable -> red; an EXISTING configuration that lacks a role
skipped instead of failing -> red; a half-run tool left in `sys.modules` (the next `from tool import name` then fails
with "cannot import name") -> red."""
from __future__ import annotations

import json
import sys

import pytest

from tests.unit.workshop_configuration import off_the_workshop, the_workshop_configuration

import atelier  # noqa: E402  (tools/ is on sys.path: tests/unit/conftest.py)


def _skip_reason(block) -> str:
    with pytest.raises(pytest.skip.Exception) as info:
        with the_workshop_configuration():
            block()
    return str(info.value)


def test_an_absent_configuration_skips_naming_the_file_and_its_variable(monkeypatch, tmp_path):
    absent = tmp_path / "absent" / "atelier.json"
    monkeypatch.setenv(atelier.CONFIG_ENV, str(absent))
    monkeypatch.delenv("NBX_ATELIER_CAMPAIGNS", raising=False)
    why = _skip_reason(lambda: atelier.root("campaigns"))
    assert str(absent) in why and atelier.CONFIG_ENV in why, why


def test_a_configuration_that_lacks_the_role_still_fails(monkeypatch, tmp_path):
    cfg = tmp_path / "atelier.json"
    cfg.write_text(json.dumps({"tmp": str(tmp_path / "t")}))
    monkeypatch.setenv(atelier.CONFIG_ENV, str(cfg))
    monkeypatch.delenv("NBX_ATELIER_CAMPAIGNS", raising=False)
    try:
        with pytest.raises(atelier.AtelierUndeclared, match="campaigns"):
            with the_workshop_configuration():
                atelier.root("campaigns")
    except pytest.skip.Exception as skipped:
        pytest.fail(f"a broken workshop was skipped as an absent one: {skipped}")


def test_another_error_is_never_a_skip(monkeypatch, tmp_path):
    monkeypatch.setenv(atelier.CONFIG_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(ZeroDivisionError):
        with the_workshop_configuration():
            1 / 0


def test_a_tool_whose_import_stopped_is_not_left_half_run(monkeypatch, tmp_path):
    monkeypatch.setenv(atelier.CONFIG_ENV, str(tmp_path / "absent.json"))
    monkeypatch.delenv("NBX_ATELIER_CAMPAIGNS", raising=False)
    tool = tmp_path / "a_tool_reading_the_workshop.py"
    tool.write_text("import atelier\nBEFORE = 1\nOUT = atelier.root('campaigns')\nAFTER = 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib.util
    spec = importlib.util.spec_from_file_location("a_tool_reading_the_workshop", tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module               # the loaders' pattern: registered before it runs
    try:
        _skip_reason(lambda: spec.loader.exec_module(module))
        assert spec.name not in sys.modules
    finally:
        sys.modules.pop(spec.name, None)


def test_a_missing_catalogue_file_is_a_skip_only_off_the_workshop(monkeypatch, tmp_path):
    monkeypatch.setenv(atelier.CONFIG_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(pytest.skip.Exception, match="M's container"):
        off_the_workshop("M's container")
    cfg = tmp_path / "atelier.json"
    cfg.write_text("{}")
    monkeypatch.setenv(atelier.CONFIG_ENV, str(cfg))
    try:
        off_the_workshop("M's container")              # a workshop machine: the test's own refusal stands
    except pytest.skip.Exception as skipped:
        pytest.fail(f"a workshop machine was skipped: {skipped}")
