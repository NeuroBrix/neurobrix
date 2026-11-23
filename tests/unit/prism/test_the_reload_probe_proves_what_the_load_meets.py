"""The reload probe of a first launch proves what the load meets (register 23/11 21:12): its child ran certified-only,
so in a fresh home a bf16 container's load (bf16_to_fp16_kernel, an element-wise variant no table holds yet) was
refused before the serve was ready (DeepSeek-Coder-V2-Lite, deepseek-moe-16b-chat on --triton). The reload child now
runs as the run does; only the arena measure keeps its certified-only bound; a confirmation run stays one.

What it would do if the claim were wrong: the reload child's environment forces NBX_AUTOTUNE_CERTIFIED_ONLY (red, both
twins); the arena measure loses its bound (red); a --certified-only parent no longer passes its refusal on (red)."""
import importlib

import pytest

TWINS = ("neurobrix.pytorch.prism.arena_probe", "neurobrix.triton.prism.arena_probe")


def _spy(monkeypatch, module):
    seen = {}

    def run(cmd, capture_output, text, env):
        seen["env"] = env
        raise RuntimeError("stop after the environment is seen")
    monkeypatch.setattr(module.subprocess, "run", run)
    return seen


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("engine", ["triton", "compiled"])
def test_the_reload_child_is_not_a_confirmation_run(twin, engine, monkeypatch):
    A = importlib.import_module(twin)
    monkeypatch.delenv("NBX_AUTOTUNE_CERTIFIED_ONLY", raising=False)
    seen = _spy(monkeypatch, A)
    with pytest.raises(RuntimeError, match="stop after"):
        A.child("/nowhere/c", "lm_head", "hw", engine)
    assert "NBX_AUTOTUNE_CERTIFIED_ONLY" not in seen["env"]


@pytest.mark.parametrize("twin", TWINS)
def test_the_arena_measure_keeps_its_certified_only_bound(twin, monkeypatch):
    A = importlib.import_module(twin)
    monkeypatch.delenv("NBX_AUTOTUNE_CERTIFIED_ONLY", raising=False)
    seen = _spy(monkeypatch, A)
    with pytest.raises(RuntimeError, match="stop after"):
        A.child("/nowhere/c", "model", "hw", 2)
    assert seen["env"]["NBX_AUTOTUNE_CERTIFIED_ONLY"] == "1"


@pytest.mark.parametrize("twin", TWINS)
def test_a_confirmation_parent_passes_its_refusal_on_to_the_reload_child(twin, monkeypatch):
    A = importlib.import_module(twin)
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    seen = _spy(monkeypatch, A)
    with pytest.raises(RuntimeError, match="stop after"):
        A.child("/nowhere/c", "lm_head", "hw", "triton")
    assert seen["env"]["NBX_AUTOTUNE_CERTIFIED_ONLY"] == "1"


class TileUnproven(RuntimeError):
    """Named as the library names it (nbxtensor.ops.ew_proof.TileUnproven): the classification reads the type's name."""


class KeyNotCertified(RuntimeError):
    pass


@pytest.mark.parametrize("twin", TWINS)
def test_an_unproven_variant_in_the_arena_measure_bounds_it_and_stops_nothing(twin):
    """The register 23/11 21:27: in the certified-only arena child an element-wise variant no table holds fell through
    to a plain RuntimeError and stopped the serve; it is the store serving nothing, like a missing key: it bounds."""
    A = importlib.import_module(twin)
    met = TileUnproven("ZERO FALLBACK: a confirmation run met element-wise variant bf16_to_fp16_kernel at tile "
                       "(256, 8, 3), which no proof table holds (stamp 93281ee799e5cd85)")
    assert A.serves_nothing_certified(met)
    assert A.serves_nothing_certified(KeyNotCertified("addmm M_BUCKET=9216"))
    assert not A.serves_nothing_certified(TileUnproven("ZERO FALLBACK: element-wise class c: no candidate passes"))
    assert not A.serves_nothing_certified(RuntimeError("CUDA error: an illegal memory access"))
