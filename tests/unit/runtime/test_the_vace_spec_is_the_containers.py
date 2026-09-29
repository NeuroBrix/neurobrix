"""The VACE conditioning spec is the container's, in both engines — never a model's constants.

`conditioning_spec` filled a bare `vace_control_conditioning` flag with Wan2.1-VACE-1.3B's own
numbers (15 control layers, 64 mask channels, z_dim 16, "vae_encoder"); the retraced container
carries the whole spec, and a container that does not is refused by name. Injection: the
refusal removed from either engine -> RED."""
import types

import pytest

import neurobrix.core.runtime.resolution.vace_control_conditioning as CV
import neurobrix.triton.vace_control_conditioning as TV


def _ctx():
    return types.SimpleNamespace(pkg=types.SimpleNamespace(manifest={"model_name": "m"}))


@pytest.mark.parametrize("mod", [CV, TV], ids=["compiled", "triton"])
def test_a_full_spec_is_read_as_declared(mod, monkeypatch):
    spec = {"condition_component": "enc", "mask_channels": 16, "vace_layers": 8, "z_dim": 48}
    monkeypatch.setattr(mod, "get_component_flag", lambda *a, **k: dict(spec))
    assert mod.conditioning_spec(_ctx(), "transformer") == spec


@pytest.mark.parametrize("mod", [CV, TV], ids=["compiled", "triton"])
def test_a_bare_flag_is_refused_by_name(mod, monkeypatch):
    monkeypatch.setattr(mod, "get_component_flag", lambda *a, **k: True)
    with pytest.raises(RuntimeError, match="lacks condition_component, mask_channels, vace_layers, z_dim"):
        mod.conditioning_spec(_ctx(), "transformer")


@pytest.mark.parametrize("mod", [CV, TV], ids=["compiled", "triton"])
def test_no_flag_is_inert(mod, monkeypatch):
    monkeypatch.setattr(mod, "get_component_flag", lambda *a, **k: None)
    assert mod.conditioning_spec(_ctx(), "transformer") is None
