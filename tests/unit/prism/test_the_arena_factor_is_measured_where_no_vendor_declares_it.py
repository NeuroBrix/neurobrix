"""The Triton arena's factor is a measured fact (the owner, 2026-11-02 15:01): a vendor file's declared measurement
stands; else its family's (vendors/family_facts.yml, each with its source — the supervisor 2026-11-07 12:00: a first
launch never probes it any more, a probe on whatever a user imported gave 1.4 to 28 and swept uncertified kernels);
with neither the plan is refused by name. The probe (a tool now; bounded to census shapes in 0.6.2) doubles its request while the component still
fits the card and keeps the ratio at the largest that fits, three runs within the storage measure's steadiness (on a
V100-16GB, HAT-S_SRx4: 3.50 at x1, 2.39 at x2, 1.48 at x4 — a fixed hold a small walk over-weights).

What this test would do if the code were wrong: a measured fact read over a vendor's declaration (every V100 plan
moves), a measured fact ignored on an unknown card (the refusal stays), a refusal that does not name the probe, a probe
that keeps the small walk's ratio or writes disagreeing runs, a probe run where the vendor declares the factor — RED.
No card."""
from __future__ import annotations

import types

import pytest
from pathlib import Path

from neurobrix.core.prism import solver as S


def _dev(measured=None):
    return types.SimpleNamespace(brand="nvidia", architecture="volta", compute_capability="7.0", name="V100",
                                 measured=measured, memory_mb=16384, device_string="cuda:0")


def _factor(monkeypatch, vendor_value, measured):
    monkeypatch.setattr("neurobrix.core.config.loader.vendor_config_for_device",
                        lambda *a, **k: {"memory": {"triton_arena_activation_factor": vendor_value}})
    return S.PrismSolver()._triton_arena_factor(types.SimpleNamespace(spec=_dev(measured)))


def test_a_vendor_declaration_stands_over_a_measured_fact(monkeypatch):
    assert _factor(monkeypatch, 1.35, {"triton_arena_activation_factor": 1.476}) == 1.35


def test_an_undeclared_card_reads_its_family_never_a_first_launch_measure(monkeypatch):
    # the supervisor 2026-11-07 12:00: a measure taken at a user's first run on whatever was imported came out 1.4 to
    # 28 (a fresh Mac: 28.27, then 2.37) — the family's declaration (vendors/family_facts.yml) is read instead
    assert _factor(monkeypatch, None, {"triton_arena_activation_factor": 28.266}) == 1.35      # nvidia: Volta's


def test_a_family_nobody_declares_is_refused_by_name(monkeypatch):
    monkeypatch.setattr("neurobrix.core.config.loader.vendor_config_for_device",
                        lambda *a, **k: {"memory": {"triton_arena_activation_factor": None}})
    dev = types.SimpleNamespace(brand="unknownvendor", architecture="x", compute_capability="0.0", name="X",
                                measured={"triton_arena_activation_factor": 1.4}, memory_mb=1, device_string="cuda:0")
    with pytest.raises(S.PlanNotComputable, match="nor the unknownvendor family declares it"):
        S.PrismSolver()._triton_arena_factor(types.SimpleNamespace(spec=dev))


@pytest.mark.parametrize("brand, arch, cc, model, expect", [
    ("apple", "apple-m2", "0.0", "Apple M2", 1.477),            # an Apple chip whose file declares none: the family
    ("nvidia", "ampere", "8.6", "NVIDIA RTX A6000", 1.35),
    ("amd", "rdna3", "0.0", "AMD Radeon RX 7900", 1.35),
    ("nvidia", "volta", "7.0", "Tesla V100-SXM2-32GB", 1.35),   # Volta's own file
])
def test_every_family_has_a_declared_factor(brand, arch, cc, model, expect):
    from neurobrix.core.config.loader import declared_arena_factor
    value, why = declared_arena_factor(brand, arch, cc, model=model)
    assert value == expect, why


def _runs(hold_walk):
    calls = []

    def run(k):
        calls.append(k)
        h, w = hold_walk[k]
        return {"hold_bytes": h, "walked_bytes": w}
    return run, calls


def test_the_probe_keeps_the_largest_request_that_fits():
    from neurobrix.core.prism import arena_probe as A
    GB = 1 << 30
    run, calls = _runs({1: (1.0 * GB, 0.29 * GB), 2: (2.7 * GB, 1.14 * GB), 4: (6.7 * GB, 4.55 * GB)})
    factor, runs = A.factor_on_card(15 * GB, 0, run, 3, 0.05)
    assert calls == [1, 2, 4, 4, 4] and factor == pytest.approx(6.7 / 4.55)


def test_disagreeing_runs_write_nothing():
    from neurobrix.core.prism import arena_probe as A
    GB = 1 << 30
    seq = iter([(1.0 * GB, 1.0 * GB), (1.2 * GB, 1.0 * GB), (1.5 * GB, 1.0 * GB)])
    factor, _ = A.factor_on_card(3 * GB, 0, lambda k: dict(zip(("hold_bytes", "walked_bytes"), next(seq))), 3, 0.05)
    assert factor is None


def test_a_first_launch_never_probes_the_arena(monkeypatch, tmp_path):
    # 0.6.1: no probe at a user's first run, declared or not (the bounded probe on census shapes is 0.6.2's)
    from neurobrix.core.prism import autodetect as AD
    from neurobrix.core.prism import facts_probe as P
    import yaml
    spec = {"id": "default-x", "devices": [{"index": 0, "brand": "nvidia", "architecture": "volta", "model": "V100",
                                            "measured": {"source": "s", "vmm_chunk_bytes": 1, "gemv_gbps": {"triton": {}},
                                                         "expert_copy": {"block": 4096, "warps": 4, "gbps": 3.4}},
                                            "runtime_total_mb": 16151, "runtime_context_mb": 306}],
            "host_measured": {"host_matmul_gflops": 1}}
    monkeypatch.setattr(AD, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr("neurobrix.core.paths.cache_dir", lambda: tmp_path / "no-cache")   # no reload probe here
    (tmp_path / "default-x.yml").write_text(yaml.safe_dump(spec))
    probed = []
    monkeypatch.setattr(AD, "_measure_arena_factor", lambda hw, d: probed.append(hw) or 1.5)
    for declared, expect in ((1.35, []), (None, [])):
        monkeypatch.setattr("neurobrix.core.config.loader.vendor_config_for_device",
                            lambda *a, _v=declared, **k: {"memory": {"triton_arena_activation_factor": _v}})
        probed.clear()
        AD.ensure_measured_facts("default-x", "triton")
        assert probed == expect


def test_the_probe_sizes_a_unified_device_by_its_memory_and_a_card_by_its_runtime(monkeypatch, tmp_path):
    # a unified device (Apple) carries no CUDA runtime total / context: the planner reads its memory_mb (the census
    # shadow's capacity), and so does the probe — it was refused by name on every Apple machine (03/11)
    from neurobrix.core.prism import autodetect as AD, arena_probe as A
    (tmp_path / "components" / "c").mkdir(parents=True)
    (tmp_path / "components" / "c" / "graph.json").write_text('{"symbolic_context": {"symbols": {}}}')
    monkeypatch.setattr(A, "smallest_container", lambda cache: (tmp_path, "c"))
    seen = []
    monkeypatch.setattr(A, "factor_on_card", lambda usable, w, run, reps, steady, grows: seen.append(usable) or (1.2, []))
    AD._measure_arena_factor("hw", {"unified_memory": True, "memory_mb": 18186})
    AD._measure_arena_factor("hw", {"runtime_total_mb": 16151, "runtime_context_mb": 306})
    assert seen == [18186 << 20, 15845 << 20]
    with pytest.raises(RuntimeError, match="memory_mb on a unified device"):
        AD._measure_arena_factor("hw", {"memory_mb": 16384})


def test_a_probe_the_device_cannot_hold_bounds_the_measure(monkeypatch):
    # a fresh Mac home (2026-11-07): the 4x prediction let x32 run, the child ran out of memory and the first
    # `neurobrix run` ended in a Traceback — the device's refusal is the measure's bound: the runs stay at x16
    from neurobrix.core.prism import arena_probe as A
    GB = 1 << 30
    calls = []

    def run(k):
        calls.append(k)
        if k >= 32:
            raise A.ProbeOutOfMemory("the arena probe of m:c x32 failed: DeviceOOMError: GPU malloc failed (error 2)")
        return {"hold_bytes": 0.01 * GB * k, "walked_bytes": 0.008 * GB * k}
    factor, runs = A.factor_on_card(18 * GB, 0, run, 3, 0.05)
    assert calls == [1, 2, 4, 8, 16, 32, 16, 16] and factor == pytest.approx(0.01 / 0.008)


def test_the_child_says_its_own_error(monkeypatch, tmp_path):
    from neurobrix.core.prism import arena_probe as A
    import json
    import subprocess

    def fake(args, **k):
        payload = {"error": "DeviceOOMError: GPU malloc failed (error 2) for 3932160000 bytes", "oom": True}
        return subprocess.CompletedProcess(args, 3, stdout=A.MARK + json.dumps(payload) + "\n",
                                           stderr="Exception ignored in: <function NBXTensor.__del__> sys.meta_path is None")
    monkeypatch.setattr(A.subprocess, "run", fake)
    with pytest.raises(A.ProbeOutOfMemory, match="GPU malloc failed .* 3932160000 bytes"):
        A.child(str(tmp_path), "c", "hw", 32)
    monkeypatch.setattr(A.subprocess, "run", lambda args, **k: subprocess.CompletedProcess(
        args, 3, stdout=A.MARK + json.dumps({"error": "KeyError: 'x'", "oom": False}) + "\n", stderr="noise"))
    with pytest.raises(RuntimeError, match="KeyError: 'x'") as e:
        A.child(str(tmp_path), "c", "hw", 1)
    assert not isinstance(e.value, A.ProbeOutOfMemory)


def _graph(inputs, symbols):
    return {"input_tensor_ids": list(inputs), "tensors": inputs,
            "symbolic_context": {"symbols": symbols}}


def test_the_probe_scales_only_what_the_priced_request_scales():
    # a fresh Mac home measured whisper-large-v3-turbo's decoder at 28.27 (the lab: 1.477): every input's two last axes
    # were scaled — the batch and the hidden width — while the request Prism priced kept them at x1
    from neurobrix.core.prism import arena_probe as A
    dec = _graph({"input::input_ids": {"shape": [1, 7], "symbolic_shape": {"dims": [{"type": "symbol", "id": "s0"},
                                                                                     {"type": "symbol", "id": "s1"}]}},
                  "input::encoder_hidden_states": {"shape": [1, 1500, 1280], "symbolic_shape": {"dims": [
                      {"type": "symbol", "id": "s2"}, {"type": "symbol", "id": "s3"}, 1280]}}},
                 {"s0": {"name": "batch", "trace_value": 1, "source": "input::input_ids::dim_0"},
                  "s1": {"name": "seq_len", "trace_value": 7, "source": "input::input_ids::dim_1"},
                  "s2": {"name": "batch", "trace_value": 1, "source": "input::encoder_hidden_states::dim_0"},
                  "s3": {"name": "seq_len", "trace_value": 1500, "source": "input::encoder_hidden_states::dim_1"}})
    assert A.scaled_names(dec) == ()             # two sequence lengths, one priced: nothing scaled, the probe stays x1
    img = _graph({"input::x": {"shape": [1, 3, 64, 64], "symbolic_shape": {"dims": [
                      {"type": "symbol", "id": "s0"}, 3, {"type": "symbol", "id": "s1"}, {"type": "symbol", "id": "s2"}]}}},
                 {"s0": {"name": "batch", "trace_value": 1, "source": "input::x::dim_0"},
                  "s1": {"name": "height", "trace_value": 64, "source": "input::x::dim_2"},
                  "s2": {"name": "width", "trace_value": 64, "source": "input::x::dim_3"}})
    assert A.scaled_names(img) == ("height", "width")
    llm = _graph({"input::input_ids": {"shape": [1, 23], "symbolic_shape": {"dims": [
                      {"type": "symbol", "id": "s0"}, {"type": "symbol", "id": "s1"}]}}},
                 {"s0": {"name": "batch", "trace_value": 1, "source": "input::input_ids::dim_0"},
                  "s1": {"name": "seq_len", "trace_value": 23, "source": "input::input_ids::dim_1"}})
    assert A.scaled_names(llm) == ("seq_len",)


def test_a_graph_with_nothing_to_scale_is_probed_at_its_traced_extents():
    from neurobrix.core.prism import arena_probe as A
    calls = []

    def run(k):
        calls.append(k)
        return {"hold_bytes": 1.0, "walked_bytes": 1.0}
    factor, _ = A.factor_on_card(1 << 40, 0, run, 3, 0.05, grows=False)
    assert calls == [1, 1, 1] and factor == 1.0


def test_the_probe_is_taken_on_a_spatial_component_else_on_a_models_whole_walk(monkeypatch, tmp_path):
    import json
    from neurobrix.core.prism import arena_probe as A
    from neurobrix.nbx import container as NC

    def model(name, comps):     # comps: {component: (weight bytes, graph)}
        d = tmp_path / name
        (d / "components").mkdir(parents=True)
        (d / "manifest.json").write_text("{}")
        for c, (n, g) in comps.items():
            (d / "components" / c / "weights").mkdir(parents=True)
            (d / "components" / c / "weights" / "w.safetensors").write_bytes(b"x" * n)
            (d / "components" / c / "graph.json").write_text(json.dumps(g))
        return d
    seq = lambda nops: {"symbolic_context": {"symbols": {"s1": {"name": "seq_len", "trace_value": 7,
                                                                "source": "input::ids::dim_1"}}},
                        "ops": {f"op{i}": {} for i in range(nops)}}
    spatial = {"symbolic_context": {"symbols": {"s1": {"name": "height", "trace_value": 64, "source": "input::x::dim_2"}}}}
    llm = model("llm", {"lm_head": (10, seq(1)), "model": (100, seq(500))})
    monkeypatch.setattr(NC.NBXContainer, "load", staticmethod(lambda p: type("C", (), {"get_neural_components": lambda self: [
        type("N", (), {"name": c.name})() for c in sorted((Path(p) / "components").iterdir())]})()))
    assert A.smallest_container(tmp_path) == (llm, "model")          # the whole decoder, not the lone projection
    up = model("upscaler", {"model": (1000, spatial)})
    assert A.smallest_container(tmp_path) == (up, "model")           # a spatial component wins wherever it is
