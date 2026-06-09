"""Ch8 P-LAYER7-AUTO-FP32-FAMILY-AWARE — auto-detect pins.

Pins the data-driven structural auto-detect at
`PrismSolver._auto_fp32_components` against the empirical
per-component matrix that motivated Ch8. Each row is a real cached
model whose graph op counts are reproduced as test data; the rule
must fire on exactly the VAE-class components the Layer-7 follow-up
names and on nothing else.

Also pins:
- env bypass (`NBX_DISABLE_AUTO_FP32=1`) silences the auto-detect
  without affecting the manual `requires_fp32_compute` flag (manual
  ⊕ auto is set union, manual wins by construction).
- family-gate: auto-detect is gated by
  `config/families/<family>.yml dtype_policy.auto_fp32_on_overflow_risk`
  (default-absent ⇒ disabled).
- hw-gate: auto-detect is skipped when the hardware supports
  bfloat16 (bf16 exponent range = fp32, no conv-output saturation).
- conv-dominance gate: `conv2d ≥ 20` floor AND `conv2d ≥ 10·sdpa`
  ratio. Both must hold.

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_auto_fp32_family_aware.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from neurobrix.core.prism.solver import PrismSolver


# ─────────────────────────────────────────────────────────────────────
# Minimal stubs — match the surface PrismSolver._auto_fp32_components
# reads: container.get_manifest(), container.get_neural_components(),
# component.name, component.graph, profile.devices_support_dtype.
# ─────────────────────────────────────────────────────────────────────

class _StubComponent:
    def __init__(self, name: str, graph: Dict):
        self.name = name
        self.graph = graph


class _StubContainer:
    def __init__(self, family: str, components: List[_StubComponent]):
        self._family = family
        self._components = components

    def get_manifest(self) -> Dict:
        return {"family": self._family}

    def get_neural_components(self) -> List[_StubComponent]:
        return self._components


class _StubProfile:
    def __init__(self, supports_bf16: bool, preferred_dtype: str = "float16"):
        self._bf16 = supports_bf16
        self.preferred_dtype = preferred_dtype

    def devices_support_dtype(self, dtype: str) -> bool:
        if dtype == "bfloat16":
            return self._bf16
        return True


def _graph(torch_dtype: str, conv2d: int = 0, sdpa: int = 0,
           layer_norm: int = 0) -> Dict:
    """Build a minimal graph.json-like dict with the op counts the
    auto-detect cares about."""
    ops: Dict[str, Dict] = {}
    uid = 0
    for _ in range(conv2d):
        ops[f"op_{uid}"] = {"op_type": "aten::conv2d"}
        uid += 1
    for _ in range(sdpa):
        ops[f"op_{uid}"] = {"op_type": "aten::scaled_dot_product_attention"}
        uid += 1
    for _ in range(layer_norm):
        ops[f"op_{uid}"] = {"op_type": "aten::layer_norm"}
        uid += 1
    return {"torch_dtype": torch_dtype, "ops": ops}


# Empirical matrix (Ch8 audit Section 3) — one row per cached
# component, op counts taken from the audit table. The stubs are
# typed as Any so the duck-typed call sites pass static analysis
# without requiring a full PrismProfile dataclass instantiation.
_FP16_HW: Any = _StubProfile(supports_bf16=False, preferred_dtype="float16")
_BF16_HW: Any = _StubProfile(supports_bf16=True,  preferred_dtype="bfloat16")


# (family, comp_name, graph_dtype, conv2d, sdpa, layer_norm, expected_fire)
_MATRIX: List[Tuple[str, str, str, int, int, int, bool]] = [
    # PixArt-Sigma (image)
    ("image", "vae",          "float32", 36, 1,  0,  True),
    ("image", "transformer",  "float32", 1,  56, 57, False),
    ("image", "text_encoder", "float32", 0,  0,  0,  False),
    # Sana 1024 (image)
    ("image", "vae",          "float32", 70, 0,  0,  True),
    ("image", "transformer",  "float32", 61, 20, 41, False),
    ("image", "text_encoder", "float16", 0,  26, 0,  False),
    # Janus-Pro-7B (multimodal — family gate excludes)
    ("multimodal", "gen_vision_model", "bfloat16", 59, 0,  0,  False),
    ("multimodal", "language_model",   "bfloat16", 0,  30, 0,  False),
    # TinyLlama (llm)
    ("llm", "model",          "bfloat16", 0,  22, 0,  False),
    # openaudio (audio)
    ("audio", "codec.decoder","float32", 30, 0,  0,  False),
    # Kokoro (audio)
    ("audio", "decoder",      "float32", 47, 0,  0,  False),
    # swin2SR (upscaler — family gate excludes; SwinIR manual flag
    # is what handles its fp16-unsafety, separate surface)
    ("upscaler", "swin2sr",   "float32", 5,  0,  0,  False),
]


@pytest.mark.parametrize("family,comp,gdtype,conv,sdpa,ln,expected",
                         _MATRIX)
def test_auto_detect_on_empirical_matrix(family, comp, gdtype, conv,
                                         sdpa, ln, expected):
    """Row-by-row: the rule must fire on PixArt-α/σ vae + Sana vae,
    and on nothing else, on V100-class hw."""
    c = _StubContainer(family, [_StubComponent(comp,
                                               _graph(gdtype, conv, sdpa, ln))])
    auto = PrismSolver()._auto_fp32_components(c, _FP16_HW, family)
    fires = comp in auto
    assert fires is expected, (
        f"{family}/{comp} dtype={gdtype} conv2d={conv} sdpa={sdpa} "
        f"layer_norm={ln}: expected fire={expected}, got fire={fires}, "
        f"auto={sorted(auto)}"
    )


def test_hw_gate_skips_on_bf16_hardware():
    """A100/H100-class hw (bf16 native): conv saturation impossible
    because bf16 exponent = fp32. Auto-detect must not fire."""
    c = _StubContainer("image", [
        _StubComponent("vae", _graph("float32", conv2d=36, sdpa=1)),
    ])
    auto = PrismSolver()._auto_fp32_components(c, _BF16_HW, "image")
    assert auto == set(), (
        f"bf16 hw must skip auto-detect; got {auto}"
    )


def test_family_gate_omitted_means_disabled():
    """Families whose YAML omits the dtype_policy section get the
    default-disabled treatment. The image and video families enable;
    llm/audio/audio_llm/multimodal/upscaler/stt/tts/vlm do not."""
    c = _StubContainer("llm", [
        _StubComponent("vae", _graph("float32", conv2d=100, sdpa=0)),
    ])
    auto = PrismSolver()._auto_fp32_components(c, _FP16_HW, "llm")
    assert auto == set(), (
        f"llm family must be auto-detect-disabled; got {auto}"
    )


def test_env_bypass_silences_auto_only_not_manual(monkeypatch):
    """NBX_DISABLE_AUTO_FP32=1 must short-circuit
    _components_force_fp32 to the manual set only. _auto_fp32_components
    itself is NOT gated by the env (it's a pure structural query the
    bypass operates one level above)."""
    monkeypatch.setenv("NBX_DISABLE_AUTO_FP32", "1")
    c = _StubContainer("image", [
        _StubComponent("vae", _graph("float32", conv2d=36, sdpa=1)),
    ])
    forced = PrismSolver()._components_force_fp32(c, _FP16_HW)
    assert forced == set(), (
        f"NBX_DISABLE_AUTO_FP32=1 must zero the auto set when manual "
        f"is empty; got {forced}"
    )
    # The auto-detect query itself is unchanged — it's the public
    # query the verdict R29 baseline uses.
    auto = PrismSolver()._auto_fp32_components(c, _FP16_HW, "image")
    assert auto == {"vae"}, (
        f"_auto_fp32_components must compute the same answer "
        f"regardless of env (bypass is in the consumer); got {auto}"
    )


def test_conv_dominance_floor_excludes_minor_conv_components():
    """conv2d_count < 20 floor: a component with a few conv2d ops
    (e.g. a DiT patch-embed with one conv2d at the input) is NOT
    auto-pinned."""
    for n in [0, 1, 5, 19]:
        c = _StubContainer("image", [
            _StubComponent("transformer", _graph("float32", conv2d=n, sdpa=56)),
        ])
        auto = PrismSolver()._auto_fp32_components(c, _FP16_HW, "image")
        assert auto == set(), (
            f"conv2d={n} below floor must not fire; got {auto}"
        )


def test_conv_dominance_ratio_excludes_hybrid_attention():
    """conv2d_count >= 10 × sdpa_count: a component above the floor
    but with substantial attention (Sana DiT: 61 conv / 20 sdpa = 3)
    is NOT auto-pinned."""
    c = _StubContainer("image", [
        _StubComponent("transformer", _graph("float32", conv2d=61, sdpa=20)),
    ])
    auto = PrismSolver()._auto_fp32_components(c, _FP16_HW, "image")
    assert auto == set(), (
        f"61 conv2d / 20 sdpa (ratio 3) must not fire; got {auto}"
    )


def test_graph_dtype_gate_excludes_bf16_graph():
    """Component whose build-time graph dtype is bf16 / fp16 must not
    fire even if it is conv-dominant and in an enabled family. The
    Janus gen_vision_model row (bf16, 59 conv2d) is the canonical
    test case — it would over-fire without the dtype gate."""
    c = _StubContainer("image", [   # forced into image family to
                                    # isolate the dtype gate
        _StubComponent("gen_vision_model",
                       _graph("bfloat16", conv2d=59, sdpa=0)),
    ])
    auto = PrismSolver()._auto_fp32_components(c, _FP16_HW, "image")
    assert auto == set(), (
        f"bf16 graph dtype must not fire; got {auto}"
    )


def test_manual_and_auto_compose_by_union(monkeypatch):
    """The two sources compose by set union — manual entries are never
    lost regardless of what the auto-detect returns."""
    monkeypatch.delenv("NBX_DISABLE_AUTO_FP32", raising=False)
    monkeypatch.delenv("NBX_FORCE_FP32_COMPUTE", raising=False)
    # Synthesise a container with two components: one auto (VAE) and
    # one only-manual (a hypothetical "post_processor" the registry
    # marks fp16-unsafe).
    c = _StubContainer("image", [
        _StubComponent("vae",            _graph("float32", conv2d=36, sdpa=1)),
        _StubComponent("post_processor", _graph("float32", conv2d=0,  sdpa=0)),
    ])

    # Manual entry via monkeypatching the registry_flags lookup; both
    # _components_force_fp32 and the auto-detect are then called.
    from neurobrix.core.runtime import registry_flags
    def _fake_flag(model_name, comp_name, flag, default=False, env_override=None):
        return flag == "requires_fp32_compute" and comp_name == "post_processor"
    monkeypatch.setattr(registry_flags, "get_component_flag", _fake_flag)

    forced = PrismSolver()._components_force_fp32(c, _FP16_HW)
    assert forced == {"vae", "post_processor"}, (
        f"manual ∪ auto = union; got {forced}"
    )
