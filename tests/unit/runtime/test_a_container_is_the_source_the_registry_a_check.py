"""A runtime flag is the CONTAINER's; the build toolchain's registry only checks it.

The supervisor's R18 decision of 2026-09-27 02:57: a container carries every flag the engine
reads, and the developer registry serves the build only. Before it, the registry was read FIRST
at runtime, so a model ran differently wherever it was not reachable: fifteen flags of eight
video containers existed only there (2026-09-27 02:55), and Wan2.1-T2V-1.3B without
`zero_pad_embeddings` rendered a lattice of 16-px cells (2026-09-20). Now the rack with its
registry runs exactly what the Mac runs without one — or refuses by name.
"""
from __future__ import annotations

import pytest

import neurobrix.core.runtime.registry_flags as rf
from neurobrix.nbx import component_flags

REG = ("video:\n"
       "  Wan2.1-VACE-1.3B-diffusers:\n"
       "    components:\n"
       "      text_encoder:\n"
       "        zero_pad_embeddings: true\n"
       "      vae_encoder:\n"
       "        requires_fp32_compute: false\n")


@pytest.fixture
def registry(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text(REG)
    monkeypatch.setattr(rf, "_REGISTRY_CACHE", None)
    monkeypatch.setattr(rf, "_find_registry_yaml", lambda: p)
    yield p
    component_flags.clear()


@pytest.fixture
def no_registry(monkeypatch):
    monkeypatch.setattr(rf, "_REGISTRY_CACHE", None)
    monkeypatch.setattr(rf, "_find_registry_yaml", lambda: None)
    yield
    component_flags.clear()


def test_without_a_registry_the_container_answers(no_registry):
    component_flags.register("Wan2.1-VACE-1.3B-diffusers", {"text_encoder": {"zero_pad_embeddings": True}})
    assert rf.get_component_flag("Wan2.1-VACE-1.3B-diffusers", "text_encoder",
                                 "zero_pad_embeddings", default=False) is True


def test_with_the_registry_the_same_container_gives_the_same_answer(registry):
    component_flags.register("Wan2.1-VACE-1.3B-diffusers", {"text_encoder": {"zero_pad_embeddings": True}})
    assert rf.get_component_flag("Wan2.1-VACE-1.3B-diffusers", "text_encoder",
                                 "zero_pad_embeddings", default=False) is True


def test_a_container_that_lacks_a_declared_flag_is_refused_by_name(registry):
    component_flags.register("Wan2.1-VACE-1.3B-diffusers", {"text_encoder": {}})
    with pytest.raises(RuntimeError, match=r"text_encoder\.zero_pad_embeddings.*does not carry it"):
        rf.get_component_flag("Wan2.1-VACE-1.3B-diffusers", "text_encoder", "zero_pad_embeddings", default=False)


def test_a_container_that_carries_another_value_is_refused(registry):
    component_flags.register("Wan2.1-VACE-1.3B-diffusers", {"text_encoder": {"zero_pad_embeddings": "yes"}})
    with pytest.raises(RuntimeError, match="carries 'yes'"):
        rf.get_component_flag("Wan2.1-VACE-1.3B-diffusers", "text_encoder", "zero_pad_embeddings", default=False)


def test_a_flag_declared_false_is_the_default_and_needs_no_carriage(registry):
    component_flags.register("Wan2.1-VACE-1.3B-diffusers", {})
    assert rf.get_component_flag("Wan2.1-VACE-1.3B-diffusers", "vae_encoder",
                                 "requires_fp32_compute", default=False) is False



def test_every_reader_asks_by_the_container_s_own_name():
    """The container's flags are registered under its MANIFEST's model_name. A reader that asks
    under the REQUESTED name (`--model` may be a path or an alias; the daemon's model_name is what
    the client typed) finds nothing, and with the registry reachable the check refuses a container
    that carries the flag. Two readers did (run.py's image inputs and VACE all-generate; the
    daemon's image inputs); every reader now names the container by its manifest."""
    import re
    from pathlib import Path
    src = Path(rf.__file__).resolve().parents[2]
    bad = []
    for p in src.rglob("*.py"):
        text = p.read_text()
        for m in re.finditer(r"(?:get_component_flag|_gcf|prepare_image_inputs)\(\s*([^,]*),\s*([^,]*),", text):
            args = m.group(1) + "," + m.group(2)
            if "getattr(args" in args or "self.model_name" in args:
                bad.append(f"{p.relative_to(src)}:{text[:m.start()].count(chr(10)) + 1}")
    assert not bad, f"flag readers asking by the requested name, not the container's: {bad}"
