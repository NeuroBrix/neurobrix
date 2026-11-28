"""Engine 0.7.0 strictly requires manifest["nbx_container_version"] == "0.2".

Every container carries its format version in manifest.json under
"nbx_container_version". A container with nbx_container_version == "0.2" loads
normally; an older container (such as legacy nbx_version: "0.1" without
nbx_container_version, nbx_version: "0.2" without nbx_container_version, or
nbx_container_version: "0.1") or missing/unreadable version is refused with a
clear, actionable user-facing message explaining it must be updated/re-downloaded
for NeuroBrix 0.7.0; a newer incompatible format version (e.g. "0.3") is refused
cleanly with an upgrade instruction.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.core.validators.nbx_validator import NBXValidator, ValidationLevel
from neurobrix.nbx.container import NBXContainer
from neurobrix.nbx.spec import (
    MAX_SUPPORTED_NBX_CONTAINER_VERSION,
    MIN_SUPPORTED_NBX_CONTAINER_VERSION,
    REQUIRED_NBX_CONTAINER_VERSION,
    parse_nbx_container_version,
    refuse_unsupported_nbx_container_version,
)


def _make_dir_container(tmp_path, manifest, name="Demo-Model"):
    d = tmp_path / name
    (d / "runtime").mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (d / "manifest.json").write_text(json.dumps(manifest))
    for rel in ("topology.json", "runtime/variables.json", "runtime/defaults.json"):
        (d / rel).write_text("{}")
    return d


def _make_zip_container(tmp_path, manifest, name="demo.nbx"):
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    return archive


# ---------------------------------------------------------------------------
# parse_nbx_container_version & refuse_unsupported_nbx_container_version
# ---------------------------------------------------------------------------

def test_parse_nbx_container_version_compares_numerically():
    assert parse_nbx_container_version("0.2") == (0, 2)
    assert parse_nbx_container_version("0.2.0") == (0, 2)
    assert parse_nbx_container_version("0.10") > parse_nbx_container_version("0.2")
    assert parse_nbx_container_version("0.2.1") > parse_nbx_container_version("0.2")


@pytest.mark.parametrize("bad", [None, "", " ", 0.2, 2, "0", "abc", "0.2-beta", "-0.2"])
def test_parse_nbx_container_version_rejects_malformed_values(bad):
    with pytest.raises(ValueError):
        parse_nbx_container_version(bad)


def test_target_version_is_accepted():
    assert refuse_unsupported_nbx_container_version(
        {"model_name": "Demo-Model", "nbx_container_version": REQUIRED_NBX_CONTAINER_VERSION},
        "Demo-Model",
    ) == REQUIRED_NBX_CONTAINER_VERSION


@pytest.mark.parametrize("old_ver", ["0.1", "0.0", "0.1.9"])
def test_old_container_version_is_refused_with_update_command(old_ver):
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        refuse_unsupported_nbx_container_version(
            {"model_name": "Demo-Model", "nbx_container_version": old_ver},
            "/cache/Demo-Model",
        )
    msg = str(exc_info.value)
    assert old_ver in msg
    assert MIN_SUPPORTED_NBX_CONTAINER_VERSION in msg
    assert "0.7.0" in msg
    assert "neurobrix remove Demo-Model && neurobrix import <org>/Demo-Model" in msg


@pytest.mark.parametrize("legacy_ver", ["0.1", "0.2"])
def test_legacy_nbx_version_without_nbx_container_version_is_refused_cleanly(legacy_ver):
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        refuse_unsupported_nbx_container_version(
            {"model_name": "Demo-Model", "nbx_version": legacy_ver},
            "/cache/Demo-Model",
        )
    msg = str(exc_info.value)
    assert "nbx_version" in msg
    assert legacy_ver in msg
    assert "nbx_container_version" in msg
    assert REQUIRED_NBX_CONTAINER_VERSION in msg
    assert "0.7.0" in msg
    assert "neurobrix remove Demo-Model && neurobrix import <org>/Demo-Model" in msg


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        {},
        {"model_name": "Demo-Model"},
        {"model_name": "Demo-Model", "nbx_container_version": None},
        {"model_name": "Demo-Model", "nbx_container_version": ""},
    ],
)
def test_missing_or_none_version_is_refused_with_update_command(manifest):
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        refuse_unsupported_nbx_container_version(manifest, "/cache/Demo-Model")
    msg = str(exc_info.value)
    assert "nbx_container_version" in msg
    assert MIN_SUPPORTED_NBX_CONTAINER_VERSION in msg
    assert "neurobrix remove " in msg
    assert "neurobrix import <org>/" in msg


@pytest.mark.parametrize("future_ver", ["0.3", "0.10", "1.0", "2.0"])
def test_future_incompatible_version_is_refused_cleanly(future_ver):
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        refuse_unsupported_nbx_container_version(
            {"model_name": "Demo-Model", "nbx_container_version": future_ver},
            "/cache/Demo-Model",
        )
    msg = str(exc_info.value)
    assert future_ver in msg
    assert MAX_SUPPORTED_NBX_CONTAINER_VERSION in msg
    assert "pip install -U neurobrix" in msg


# ---------------------------------------------------------------------------
# NBXContainer.load gate
# ---------------------------------------------------------------------------

def test_nbx_container_load_accepts_target_version(tmp_path):
    d = _make_dir_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": REQUIRED_NBX_CONTAINER_VERSION, "components": {}},
    )
    container = NBXContainer.load(str(d))
    assert container.get_manifest()["nbx_container_version"] == REQUIRED_NBX_CONTAINER_VERSION
    assert container.info()["nbx_container_version"] == REQUIRED_NBX_CONTAINER_VERSION


@pytest.mark.parametrize("manifest", [
    {"model_name": "Demo-Model", "nbx_version": "0.1"},
    {"model_name": "Demo-Model", "nbx_version": "0.2"},
    {"model_name": "Demo-Model", "nbx_container_version": "0.1"},
    {"model_name": "Demo-Model", "nbx_container_version": None},
    {"model_name": "Demo-Model"},
])
def test_nbx_container_load_refuses_old_legacy_or_missing_version(tmp_path, manifest):
    d = _make_dir_container(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        NBXContainer.load(str(d))
    msg = str(exc_info.value)
    assert MIN_SUPPORTED_NBX_CONTAINER_VERSION in msg
    assert "neurobrix remove Demo-Model && neurobrix import <org>/Demo-Model" in msg


def test_nbx_container_load_refuses_future_version(tmp_path):
    d = _make_dir_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": "0.3"},
    )
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        NBXContainer.load(str(d))
    msg = str(exc_info.value)
    assert "0.3" in msg
    assert "pip install -U neurobrix" in msg


# ---------------------------------------------------------------------------
# NBXRuntimeLoader.load gate
# ---------------------------------------------------------------------------

def test_runtime_loader_passes_target_version(tmp_path):
    d = _make_dir_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": REQUIRED_NBX_CONTAINER_VERSION},
    )
    try:
        NBXRuntimeLoader().load(str(d))
    except RuntimeError as exc:
        assert "NBX CONTAINER VERSION" not in str(exc)
    except Exception:
        pass


@pytest.mark.parametrize("manifest", [
    {"model_name": "Demo-Model", "nbx_version": "0.1"},
    {"model_name": "Demo-Model", "nbx_version": "0.2"},
    {"model_name": "Demo-Model", "nbx_container_version": "0.1"},
    {"model_name": "Demo-Model", "nbx_container_version": None},
    {"model_name": "Demo-Model"},
])
def test_runtime_loader_refuses_old_legacy_or_missing_version(tmp_path, manifest):
    d = _make_dir_container(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        NBXRuntimeLoader().load(str(d))
    msg = str(exc_info.value)
    assert MIN_SUPPORTED_NBX_CONTAINER_VERSION in msg
    assert "neurobrix remove Demo-Model && neurobrix import <org>/Demo-Model" in msg


def test_runtime_loader_refuses_future_version(tmp_path):
    d = _make_dir_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": "0.3"},
    )
    with pytest.raises(RuntimeError, match="NBX CONTAINER VERSION") as exc_info:
        NBXRuntimeLoader().load(str(d))
    msg = str(exc_info.value)
    assert "0.3" in msg
    assert "pip install -U neurobrix" in msg


# ---------------------------------------------------------------------------
# NBXValidator gate
# ---------------------------------------------------------------------------

def test_validator_accepts_target_version_and_refuses_old_legacy_or_future(tmp_path):
    validator = NBXValidator(level=ValidationLevel.SCHEMA)

    ok_zip = _make_zip_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": REQUIRED_NBX_CONTAINER_VERSION, "components": {"tok": {"type": "tokenizer"}}},
        "ok.nbx",
    )
    res_ok = validator.validate(ok_zip)
    assert res_ok.is_valid, res_ok.errors

    legacy_01_zip = _make_zip_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_version": "0.1", "components": {"tok": {"type": "tokenizer"}}},
        "legacy01.nbx",
    )
    res_legacy_01 = validator.validate(legacy_01_zip)
    assert not res_legacy_01.is_valid
    assert any("NBX CONTAINER VERSION" in e.message and "0.1" in e.message for e in res_legacy_01.errors)

    legacy_02_zip = _make_zip_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_version": "0.2", "components": {"tok": {"type": "tokenizer"}}},
        "legacy02.nbx",
    )
    res_legacy_02 = validator.validate(legacy_02_zip)
    assert not res_legacy_02.is_valid
    assert any("NBX CONTAINER VERSION" in e.message and "0.2" in e.message for e in res_legacy_02.errors)

    old_zip = _make_zip_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": "0.1", "components": {"tok": {"type": "tokenizer"}}},
        "old.nbx",
    )
    res_old = validator.validate(old_zip)
    assert not res_old.is_valid
    assert any("NBX CONTAINER VERSION" in e.message and "0.1" in e.message for e in res_old.errors)

    future_zip = _make_zip_container(
        tmp_path,
        {"model_name": "Demo-Model", "nbx_container_version": "0.3", "components": {"tok": {"type": "tokenizer"}}},
        "future.nbx",
    )
    res_future = validator.validate(future_zip)
    assert not res_future.is_valid
    assert any("NBX CONTAINER VERSION" in e.message and "0.3" in e.message for e in res_future.errors)
