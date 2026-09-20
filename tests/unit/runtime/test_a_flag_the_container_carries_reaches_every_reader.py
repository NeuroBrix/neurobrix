"""A runtime flag the container carries is read without the build toolchain.

Measured 2026-09-20: `zero_pad_embeddings` was read only through the toolchain's
registry (a gitignored `.nbx_registry` pointer), so every worktree and every
installed engine ran Wan2.1-T2V-1.3B with the flag at its default and rendered a
lattice of 16-px cells; the developer's checkout rendered the sailboat. Six flags
are read that way. The build now writes each into the container's extracted
values and the container records them when opened; the reader consults them after
the developer's registry and before the default.

What would this file do if the code were wrong? With the container lookup absent,
the first cell returns the default and fails; with the registry not winning, the
second fails; with the env override not winning, the third fails; with the loader
not registering, the fourth fails on a container written to disk with no registry.
"""
from __future__ import annotations

import json

import pytest

import neurobrix.core.runtime.registry_flags as rf
from neurobrix.nbx import component_flags


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    # No registry: the deployed-install state, which is the state under test.
    monkeypatch.setattr(rf, "_REGISTRY_CACHE", None)
    monkeypatch.setattr(rf, "_find_registry_yaml", lambda: None)
    component_flags.clear()
    yield
    component_flags.clear()
    rf._REGISTRY_CACHE = None


def test_a_flag_the_container_declares_is_read_with_no_registry():
    component_flags.register("wan", {
        "text_encoder": {"zero_pad_embeddings": True, "max_position_embeddings": 512},
        "transformer": {"i2v_latent_conditioning": {"style": "wan"}},
        "tokenizer": {"zero_pad_embeddings": True},
    })
    assert rf.get_component_flag("wan", "text_encoder", "zero_pad_embeddings", default=False) is True
    assert rf.get_component_flag("wan", "transformer", "i2v_latent_conditioning", default=None) == {"style": "wan"}
    # A value that is not a runtime flag is not carried, and an undeclared flag is the default.
    assert component_flags.registered("wan")["text_encoder"] == {"zero_pad_embeddings": True}
    assert rf.get_component_flag("wan", "transformer", "vace_control_conditioning", default=None) is None
    assert rf.get_component_flag("other", "text_encoder", "zero_pad_embeddings", default=False) is False


def test_the_registry_stays_the_developer_s_override(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text("video:\n  wan:\n    components:\n      text_encoder:\n        zero_pad_embeddings: false\n")
    monkeypatch.setattr(rf, "_find_registry_yaml", lambda: p)
    component_flags.register("wan", {"text_encoder": {"zero_pad_embeddings": True}})
    assert rf.get_component_flag("wan", "text_encoder", "zero_pad_embeddings", default=None) is False
    # A component the registry knows but a flag it does not declare falls through to the container.
    component_flags.register("wan", {"text_encoder": {"requires_fp32_compute": True}})
    assert rf.get_component_flag("wan", "text_encoder", "requires_fp32_compute", default=False) is True


def test_the_env_override_wins_over_the_container(monkeypatch):
    component_flags.register("m", {"vae": {"requires_fp32_compute": True}})
    monkeypatch.setenv("NBX_FORCE_FP32_COMPUTE", "0")
    assert rf.get_component_flag("m", "vae", "requires_fp32_compute", default=False,
                                 env_override="NBX_FORCE_FP32_COMPUTE") is False


def test_opening_a_container_records_what_it_carries(monkeypatch, tmp_path):
    from neurobrix.nbx import container as container_mod
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"model_name": "wan", "family": "video"}))
    (cache / "topology.json").write_text(json.dumps({
        "components": {}, "flow": {},
        "extracted_values": {"text_encoder": {"zero_pad_embeddings": True},
                             "transformer": {"vace_control_conditioning": {"vace_layers": 15}}}}))
    nbx = tmp_path / "wan.nbx"
    nbx.write_bytes(b"")
    monkeypatch.setattr(container_mod, "ensure_extracted", lambda _p: cache)
    container_mod.NBXContainer.load(str(nbx))
    assert component_flags.registered("wan") == {
        "text_encoder": {"zero_pad_embeddings": True},
        "transformer": {"vace_control_conditioning": {"vace_layers": 15}},
    }
    assert rf.get_component_flag("wan", "transformer", "vace_control_conditioning", default=None) == {"vace_layers": 15}
