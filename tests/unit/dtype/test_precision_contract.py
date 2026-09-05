"""Precision contract — DtypeEngine.activations_fp16_safe + fp32_op_uids.

The conservative default (fp32 matmul on fp16 hardware) is unchanged for
every component without a declared contract; a declared-safe component
takes PyTorch AMP's fp16 path; a vendor keep-in-fp32 pin wins over both.
Lever record: validation_outputs/image_fp16_2026_09_04.
"""
import pytest
import torch

from neurobrix.core.config.loader import get_backbone_dtype_contract
from neurobrix.core.dtype.engine import DtypeEngine, _FP16_GEMM_OPS, _FP16_NEED_FP32


def _mm_out_dtype(engine: DtypeEngine, op_uid=None) -> torch.dtype:
    fn = engine.compile_op("aten::mm", torch.mm, {}, op_uid=op_uid)
    a = torch.randn(4, 4, dtype=torch.float16)
    b = torch.randn(4, 4, dtype=torch.float16)
    return fn(a, b).dtype


def test_gemm_ops_are_a_subset_of_the_frozen_need_fp32_set():
    assert _FP16_GEMM_OPS <= _FP16_NEED_FP32
    assert "div" not in _FP16_GEMM_OPS  # epsilon protection is never lifted


def test_default_keeps_the_fp32_upcast_on_fp16_compute():
    eng = DtypeEngine(torch.float16)
    assert eng.activations_fp16_safe is False
    assert _mm_out_dtype(eng) == torch.float32


def test_declared_safe_component_runs_matmul_in_fp16():
    eng = DtypeEngine(torch.float16, activations_fp16_safe=True)
    assert _mm_out_dtype(eng) == torch.float16


def test_vendor_pin_wins_over_the_contract():
    eng = DtypeEngine(torch.float16, activations_fp16_safe=True,
                      fp32_op_uids=frozenset({"aten.mm::7"}))
    assert _mm_out_dtype(eng, op_uid="aten.mm::7") == torch.float32
    assert _mm_out_dtype(eng, op_uid="aten.mm::8") == torch.float16


def test_div_computes_fp32_and_stores_per_contract():
    """`div` is never in _FP16_GEMM_OPS: its inputs are always upcast
    (epsilon protection); the contract only decides the STORE dtype."""
    x = torch.randn(4, dtype=torch.float16)
    seen = {}

    def spy_div(a, b):
        seen["in"] = a.dtype
        return torch.div(a, b)

    safe = DtypeEngine(torch.float16, activations_fp16_safe=True,
                       narrow_op_uids=frozenset({"aten.div::0"}))
    assert safe.compile_op("aten::div", spy_div, {}, op_uid="aten.div::0")(x, x).dtype == torch.float16
    assert seen["in"] == torch.float32
    default = DtypeEngine(torch.float16)
    assert default.compile_op("aten::div", spy_div, {})(x, x).dtype == torch.float32


def test_amp_cast_inputs_mirrors_compile_op():
    a = torch.randn(4, 4, dtype=torch.float16)
    safe = DtypeEngine(torch.float16, activations_fp16_safe=True,
                       fp32_op_uids=frozenset({"aten.mm::1"}))
    assert safe.amp_cast_inputs("aten::mm", [a, a])[0].dtype == torch.float16
    assert safe.amp_cast_inputs("aten::mm", [a, a], op_uid="aten.mm::1")[0].dtype == torch.float32
    default = DtypeEngine(torch.float16)
    assert default.amp_cast_inputs("aten::mm", [a, a])[0].dtype == torch.float32


def test_bf16_compute_is_untouched_by_the_contract():
    for safe in (False, True):
        eng = DtypeEngine(torch.bfloat16, activations_fp16_safe=safe)
        fn = eng.compile_op("aten::mm", torch.mm, {})
        a = torch.randn(4, 4, dtype=torch.bfloat16)
        assert fn(a, a).dtype == torch.bfloat16


def test_contract_file_carries_the_t5_keep_in_fp32_list():
    t5 = get_backbone_dtype_contract("text_encoder_t5")
    assert t5["activations_fp16_safe"] is True
    assert "ffn.down" in t5["keep_in_fp32_modules"]
    assert get_backbone_dtype_contract("no_such_backbone") == {}
    assert get_backbone_dtype_contract(None) == {}


def test_executor_resolver_pins_matmuls_by_parent_module_suffix(tmp_path):
    """The executor's resolver reads manifest + contract and pins by
    parent_module suffix — exercised on a synthetic component."""
    import json
    from neurobrix.core.runtime.graph_executor import GraphExecutor

    cache = tmp_path / "ModelX"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({
        "model_name": "ModelX",
        "components": {"text_encoder": {"backbone": "text_encoder_t5"}}}))
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._cache_path = str(cache)
    ex._component_name = "text_encoder"
    ex._dag = {"ops": {
        "aten.mm::0": {"op_type": "aten::mm", "parent_module": "encoder.block.0.block.1.ffn.down"},
        "aten.mm::1": {"op_type": "aten::mm", "parent_module": "encoder.block.0.block.1.ffn.up_0"},
        "aten.mm::2": {"op_type": "aten::mm", "parent_module": "encoder.block.0.block.0.attn.down"},
        "aten.add::0": {"op_type": "aten::add", "parent_module": "encoder.block.0.block.1.ffn.down"},
        "aten.view::0": {"op_type": "aten::view", "parent_module": "encoder.block.0.block.1.ffn.down"},
        "aten.add_::0": {"op_type": "aten::add_", "parent_module": "encoder.block.0.block.1.ffn.down"},
    }}
    safe, pinned, narrow = ex._resolve_fp16_activation_policy(torch.float16)
    assert safe is True
    # every COMPUTE op of the module; never a view (no kernel) nor an in-place op
    assert pinned == frozenset({"aten.mm::0", "aten.add::0"})
    # a per-model registry list unions with the contract (monkeypatched flag reader)
    import neurobrix.core.runtime.graph_executor as GE
    from neurobrix.core.runtime import registry_flags as RF
    orig = RF.get_component_flag
    def fake(model, comp, flag, default=None, env_override=None):
        if flag == "keep_in_fp32_modules":
            return ["attn.down"]
        if flag == "activations_fp16_safe":
            return None
        return default
    RF.get_component_flag = fake
    try:
        safe, pinned, _ = ex._resolve_fp16_activation_policy(torch.float16)
    finally:
        RF.get_component_flag = orig
    assert pinned == frozenset({"aten.mm::0", "aten.add::0", "aten.mm::2"})
    # bf16 / fp32 compute: contract is moot, nothing is read
    assert ex._resolve_fp16_activation_policy(torch.float32) == (False, frozenset(), frozenset())


def test_vendor_pin_survives_amp_off():
    """AMP off = the vendor's plain fp16 forward; the vendor still keeps its
    keep-in-fp32 modules in fp32, so the pin must apply with AMP off too."""
    eng = DtypeEngine(torch.float16, amp_enabled=False,
                      fp32_op_uids=frozenset({"aten.mm::3"}))
    assert _mm_out_dtype(eng, op_uid="aten.mm::3") == torch.float32
    assert _mm_out_dtype(eng, op_uid="aten.mm::4") == torch.float16


def test_contract_norms_take_fp16_io_and_fp32_ops_store_fp16():
    eng = DtypeEngine(torch.float16, activations_fp16_safe=True,
                      narrow_op_uids=frozenset({"aten.exp::1", "aten.div::1"}))
    x = torch.randn(2, 8, dtype=torch.float32)
    ln = eng.compile_op("aten::native_layer_norm", torch.native_layer_norm, {})
    out = ln(x, [8], None, None, 1e-5)
    assert out[0].dtype == torch.float16          # the vendor's fp16-IO kernel
    ex = eng.compile_op("aten::exp", torch.exp, {}, op_uid="aten.exp::1")
    assert ex(x.half()).dtype == torch.float16    # fp32 compute, fp16 store (narrowable)
    ex2 = eng.compile_op("aten::exp", torch.exp, {}, op_uid="aten.exp::2")
    assert ex2(x.half()).dtype == torch.float32   # feeds a precision consumer: stays fp32
    dv = eng.compile_op("aten::div", torch.div, {}, op_uid="aten.div::1")
    assert dv(x.half(), x.half()).dtype == torch.float16
    # default: unchanged fp32 outputs
    default = DtypeEngine(torch.float16)
    assert default.compile_op("aten::exp", torch.exp, {})(x.half()).dtype == torch.float32
    assert default.compile_op("aten::native_layer_norm", torch.native_layer_norm, {})(
        x.half(), [8], None, None, 1e-5)[0].dtype == torch.float32


def test_sequential_mirror_of_the_contract():
    eng = DtypeEngine(torch.float16, activations_fp16_safe=True,
                      fp32_op_uids=frozenset({"aten.exp::9"}),
                      narrow_op_uids=frozenset({"aten.exp::1"}))
    x = torch.randn(2, 8, dtype=torch.float32)
    assert eng.amp_cast_inputs("aten::native_layer_norm", [x])[0].dtype == torch.float16
    assert eng.amp_cast_result("aten::exp", x, op_uid="aten.exp::1").dtype == torch.float16
    assert eng.amp_cast_result("aten::exp", x, op_uid="aten.exp::2").dtype == torch.float32
    assert eng.amp_cast_result("aten::exp", x, op_uid="aten.exp::9").dtype == torch.float32
    assert eng.amp_cast_result("aten::mul", x).dtype == torch.float32   # never narrowed
    assert DtypeEngine(torch.float16).amp_cast_result("aten::exp", x).dtype == torch.float32


def test_guarded_square_keeps_its_fp32_output_under_the_contract():
    eng = DtypeEngine(torch.float16, activations_fp16_safe=True)
    mul = eng.compile_op("aten::mul", torch.mul, {})
    x = torch.full((4,), 300.0, dtype=torch.float16)      # x*x = 90000 > 65504
    out = mul(x, x)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    y = torch.randn(4, dtype=torch.float16)
    assert mul(x, y).dtype == torch.float16                 # plain mul: passthrough


def test_fnmatch_pattern_pins_block_level_ops_only(tmp_path):
    import json
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.core.runtime import registry_flags as RF
    cache = tmp_path / "ModelY"; cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({
        "model_name": "ModelY", "components": {"transformer": {"backbone": "unknown"}}}))
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._cache_path = str(cache); ex._component_name = "transformer"
    ex._dag = {"ops": {
        "aten.add::0": {"op_type": "aten::add", "parent_module": "block.0"},
        "aten.add::1": {"op_type": "aten::add", "parent_module": "block.12"},
        "aten.mm::0": {"op_type": "aten::mm", "parent_module": "block.0.self_attn.query"},
        "aten.bmm::0": {"op_type": "aten::bmm", "parent_module": "block.0.self_attn"},
    }}
    orig = RF.get_component_flag
    RF.get_component_flag = lambda m, c, flag, default=None, env_override=None: (
        ["self_attn", "block.[0-9]", "block.[0-9][0-9]"] if flag == "keep_in_fp32_modules"
        else (True if flag == "activations_fp16_safe" else default))
    try:
        safe, pinned, _ = ex._resolve_fp16_activation_policy(torch.float16)
    finally:
        RF.get_component_flag = orig
    assert safe is True
    assert pinned == frozenset({"aten.add::0", "aten.add::1", "aten.bmm::0"})


def test_pin_applies_to_pass_through_op_classes():
    eng = DtypeEngine(torch.float16, activations_fp16_safe=True,
                      fp32_op_uids=frozenset({"aten.add::7"}))
    x = torch.randn(4, dtype=torch.float16)
    assert eng.compile_op("aten::add", torch.add, {}, op_uid="aten.add::7")(x, x).dtype == torch.float32
    assert eng.compile_op("aten::add", torch.add, {}, op_uid="aten.add::8")(x, x).dtype == torch.float16


def test_broken_manifest_raises_instead_of_defaulting(tmp_path):
    from neurobrix.core.runtime.precision_contract import registry_model_name
    cache = tmp_path / "ModelZ"; cache.mkdir()
    (cache / "manifest.json").write_text("{not json")
    with pytest.raises(ValueError):
        registry_model_name(str(cache))
    assert registry_model_name(None) is None


def test_narrowable_set_follows_the_graph_consumers():
    """The timestep sinusoid (div → exp → mul → sin/cos) stays fp32 — the
    vendor's island; the attention rescale div consumed by mul/add narrows."""
    from neurobrix.core.runtime.precision_contract import narrowable_op_uids
    dag = {"ops": {
        "aten.div::0": {"op_type": "aten::div", "input_tensor_ids": ["a"], "output_tensor_ids": ["d0"]},
        "aten.exp::0": {"op_type": "aten::exp", "input_tensor_ids": ["d0"], "output_tensor_ids": ["e0"]},
        "aten.mul::0": {"op_type": "aten::mul", "input_tensor_ids": ["e0", "t"], "output_tensor_ids": ["m0"]},
        "aten.sin::0": {"op_type": "aten::sin", "input_tensor_ids": ["m0"], "output_tensor_ids": ["s0"]},
        "aten.div::1": {"op_type": "aten::div", "input_tensor_ids": ["attn"], "output_tensor_ids": ["d1"]},
        "aten.mul::1": {"op_type": "aten::mul", "input_tensor_ids": ["gate", "d1"], "output_tensor_ids": ["m1"]},
        "aten.add::0": {"op_type": "aten::add", "input_tensor_ids": ["x", "m1"], "output_tensor_ids": ["out"]},
        "aten.addmm::0": {"op_type": "aten::addmm", "input_tensor_ids": ["b", "out", "w"], "output_tensor_ids": ["y"]},
        "aten.add::1": {"op_type": "aten::add", "input_tensor_ids": ["y", "x"], "output_tensor_ids": ["final"]},
    }}
    n = narrowable_op_uids(dag)
    assert "aten.div::0" not in n          # feeds exp
    assert "aten.exp::0" not in n          # feeds mul → sin: the island propagates back
    assert "aten.mul::0" not in n          # feeds sin
    assert "aten.div::1" in n              # feeds mul / add → addmm (casts anyway)
    assert "aten.add::0" in n              # ends in a casting consumer
    assert "aten.add::1" not in n          # component output: no consumer shown


def test_scalar_div_outside_an_island_runs_fp16_under_the_contract():
    from neurobrix.core.dtype.engine import _scalar_divisor_is_plain
    plain = {"args": [{"type": "tensor"}, {"type": "scalar", "value": 1.0}]}
    eps = {"args": [{"type": "tensor"}, {"type": "scalar", "value": 1e-15}]}
    tensor = {"args": [{"type": "tensor"}, {"type": "tensor", "tensor_id": "x"}]}
    assert _scalar_divisor_is_plain(plain) and not _scalar_divisor_is_plain(eps) \
        and not _scalar_divisor_is_plain(tensor)
    x = torch.randn(4, dtype=torch.float16)
    seen = {}

    def spy_div(a, b):
        seen["in"] = a.dtype
        return torch.div(a, b)

    eng = DtypeEngine(torch.float16, activations_fp16_safe=True,
                      narrow_op_uids=frozenset({"aten.div::1"}))
    assert eng.compile_op("aten::div", spy_div, plain, op_uid="aten.div::1")(x, 1.0).dtype == torch.float16
    assert seen["in"] == torch.float16                      # the vendor's fp16 divide
    eng.compile_op("aten::div", spy_div, plain, op_uid="aten.div::0")(x, 128)   # island: fp32 compute
    assert seen["in"] == torch.float32
    eng.compile_op("aten::div", spy_div, eps, op_uid="aten.div::1")(x, 1e-15)   # epsilon: fp32 compute
    assert seen["in"] == torch.float32
