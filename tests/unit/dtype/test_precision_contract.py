"""Precision contract — DtypeEngine.activations_fp16_safe + fp32_op_uids.

The conservative default (fp32 matmul on fp16 hardware) is unchanged for
every component without a calibration record; a calibrated component takes
PyTorch AMP's fp16 path with the record's fp32 islands pinned. No model
name, module name or hand-written list takes part (2026-09-05 reframe).
Lever records: validation_outputs/image_fp16_2026_09_04 (the contract),
validation_outputs/precision_census_2026_09_05 (the calibration).
"""
import pytest
import torch

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


def _executor(tmp_path, model, component, dag):
    import json
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    cache = tmp_path / model
    cache.mkdir(exist_ok=True)
    (cache / "manifest.json").write_text(json.dumps({"model_name": model, "components": {component: {}}}))
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._cache_path = str(cache)
    ex._component_name = component
    ex._dag = dag
    return ex


def _dag_with_over_range_down_projection():
    return {"component_name": "text_encoder", "ops": {
        "aten.mm::0": {"op_type": "aten::mm", "input_tensor_ids": ["x", "w0"], "output_tensor_ids": ["m0"]},
        "aten.view::0": {"op_type": "aten::view", "input_tensor_ids": ["m0"], "output_tensor_ids": ["v0"]},
        "aten.add::0": {"op_type": "aten::add", "input_tensor_ids": ["v0", "x"], "output_tensor_ids": ["r0"]},
        "aten.mm::1": {"op_type": "aten::mm", "input_tensor_ids": ["r0", "w1"], "output_tensor_ids": ["m1"]},
        "aten.mm::2": {"op_type": "aten::mm", "input_tensor_ids": ["x", "w2"], "output_tensor_ids": ["m2"]},
    }, "execution_order": ["aten.mm::0", "aten.view::0", "aten.add::0", "aten.mm::1", "aten.mm::2"]}


def test_executor_resolver_islands_from_the_calibration_record(tmp_path, monkeypatch):
    """No record → conservative. A record measured on this graph → the
    contract with the ops above the bound (and every reader of such a
    value) pinned fp32; a view never; bf16 / fp32 compute reads nothing."""
    from neurobrix.core.dtype import calibration as cal
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path / "store")
    monkeypatch.delenv("NBX_ACTIVATIONS_FP16_SAFE", raising=False)
    dag = _dag_with_over_range_down_projection()
    ex = _executor(tmp_path, "ModelX", "text_encoder", dag)
    assert ex._resolve_fp16_activation_policy(torch.float16) == (False, frozenset(), frozenset())
    rec = cal.CalibrationRecord.build("ModelX", "text_encoder", dag,
                                      {"aten.mm::0": 1.41e5, "aten.view::0": 1.41e5, "aten.add::0": 2.3e5,
                                       "aten.mm::1": 40.0, "aten.mm::2": 9.0},
                                      stimulus={"prompt": "p"}, passes=2, reference="conservative")
    rec.save(cal.store_path("ModelX", "text_encoder"))
    safe, pinned, narrow = ex._resolve_fp16_activation_policy(torch.float16)
    assert safe is True
    assert pinned == frozenset({"aten.mm::0", "aten.add::0", "aten.mm::1"})
    assert "aten.mm::2" not in pinned and "aten.view::0" not in pinned
    assert ex._resolve_fp16_activation_policy(torch.float32) == (False, frozenset(), frozenset())
    assert ex._resolve_fp16_activation_policy(torch.bfloat16) == (False, frozenset(), frozenset())


def test_a_record_from_another_trace_is_ignored_and_the_env_switch_rules(tmp_path, monkeypatch):
    from neurobrix.core.dtype import calibration as cal
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path / "store")
    dag = _dag_with_over_range_down_projection()
    other = _dag_with_over_range_down_projection()
    other["ops"]["aten.mm::2"]["op_type"] = "aten::addmm"
    cal.CalibrationRecord.build("ModelX", "text_encoder", other, {"aten.mm::0": 1e6},
                                stimulus={}, passes=1, reference="conservative"
                                ).save(cal.store_path("ModelX", "text_encoder"))
    ex = _executor(tmp_path, "ModelX", "text_encoder", dag)
    monkeypatch.delenv("NBX_ACTIVATIONS_FP16_SAFE", raising=False)
    assert ex._resolve_fp16_activation_policy(torch.float16) == (False, frozenset(), frozenset())
    monkeypatch.setenv("NBX_ACTIVATIONS_FP16_SAFE", "1")   # forced, no island
    safe, pinned, narrow = ex._resolve_fp16_activation_policy(torch.float16)
    assert safe is True and pinned == frozenset() and narrow
    monkeypatch.setenv("NBX_ACTIVATIONS_FP16_SAFE", "0")   # the reference path
    cal.CalibrationRecord.build("ModelX", "text_encoder", dag, {"aten.mm::0": 1e6},
                                stimulus={}, passes=1, reference="conservative"
                                ).save(cal.store_path("ModelX", "text_encoder"))
    assert ex._resolve_fp16_activation_policy(torch.float16) == (False, frozenset(), frozenset())


def test_an_engine_without_per_op_islands_takes_the_contract_only_when_none_is_needed(tmp_path, monkeypatch):
    from neurobrix.core.dtype import calibration as cal
    from neurobrix.core.runtime.precision_contract import resolve
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path / "store")
    monkeypatch.delenv("NBX_ACTIVATIONS_FP16_SAFE", raising=False)
    dag = _dag_with_over_range_down_projection()
    ex = _executor(tmp_path, "ModelX", "text_encoder", dag)
    cal.CalibrationRecord.build("ModelX", "text_encoder", dag, {"aten.mm::0": 1e6},
                                stimulus={}, passes=1, reference="conservative"
                                ).save(cal.store_path("ModelX", "text_encoder"))
    assert resolve(ex._cache_path, "text_encoder", dag, compute_dtype=torch.float16,
                   supports_op_pins=False)[0] is False
    cal.CalibrationRecord.build("ModelX", "text_encoder", dag, {"aten.mm::0": 10.0},
                                stimulus={}, passes=1, reference="conservative"
                                ).save(cal.store_path("ModelX", "text_encoder"))
    assert resolve(ex._cache_path, "text_encoder", dag, compute_dtype=torch.float16,
                   supports_op_pins=False)[0] is True


def test_an_unrecognised_env_switch_value_raises(monkeypatch):
    from neurobrix.core.runtime.precision_contract import resolve
    monkeypatch.setenv("NBX_ACTIVATIONS_FP16_SAFE", "maybe")
    with pytest.raises(ValueError):
        resolve(None, "c", {"ops": {}}, compute_dtype=torch.float16)


def test_a_corrupt_existing_record_raises_instead_of_defaulting(tmp_path, monkeypatch):
    from neurobrix.core.dtype import calibration as cal
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path / "store")
    p = cal.store_path("ModelX", "text_encoder")
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    with pytest.raises(ValueError):
        cal.load_record("ModelX", "text_encoder")
    p.write_text('{"format": "something-else"}')
    with pytest.raises(ValueError):
        cal.load_record("ModelX", "text_encoder")


def test_a_stale_record_is_refused_loudly_and_never_applied(tmp_path, monkeypatch, capsys):
    from neurobrix.core.dtype import calibration as cal
    monkeypatch.setattr(cal, "STORE_ROOT", tmp_path / "store")
    monkeypatch.delenv("NBX_ACTIVATIONS_FP16_SAFE", raising=False)
    dag = _dag_with_over_range_down_projection()
    other = _dag_with_over_range_down_projection()
    other["ops"]["aten.mm::2"]["op_type"] = "aten::addmm"
    cal.CalibrationRecord.build("ModelX", "text_encoder", other, {"aten.mm::0": 1e6},
                                stimulus={}, passes=1, reference="conservative"
                                ).save(cal.store_path("ModelX", "text_encoder"))
    ex = _executor(tmp_path, "ModelX", "text_encoder", dag)
    assert ex._resolve_fp16_activation_policy(torch.float16) == (False, frozenset(), frozenset())
    err = capsys.readouterr().err
    assert "REFUSED RECORD" in err and "neurobrix calibrate --model ModelX" in err


def test_calibrate_refuses_the_triton_engines_upfront(monkeypatch):
    import types
    from neurobrix.cli.commands import calibrate as C
    monkeypatch.setattr(C, "_identity_of", lambda m: ("image", m))
    from neurobrix.serving import client as SC
    monkeypatch.setattr(SC.DaemonClient, "is_running", staticmethod(lambda: False))
    args = types.SimpleNamespace(model="M", triton=True, triton_sequential=False, output=None, mode=None)
    assert C.cmd_calibrate(args) == 2


def test_the_contract_can_be_set_after_the_engine_exists():
    """The executor builds the engine before the graph-rewriting passes
    (constants convert through it) and installs the contract after them."""
    eng = DtypeEngine(torch.float16)
    assert _mm_out_dtype(eng, op_uid="aten.mm::1") == torch.float32
    eng.set_precision_contract(True, frozenset({"aten.mm::0"}), frozenset())
    assert _mm_out_dtype(eng, op_uid="aten.mm::1") == torch.float16
    assert _mm_out_dtype(eng, op_uid="aten.mm::0") == torch.float32
    eng.set_precision_contract(False)
    assert _mm_out_dtype(eng, op_uid="aten.mm::1") == torch.float32
