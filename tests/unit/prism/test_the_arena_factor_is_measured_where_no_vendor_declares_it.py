"""The Triton arena's factor is a measured fact (the owner, 2026-11-02 15:01): a vendor file's declared measurement
stands; a card the vendor files do not know measures itself at its first launch (core/prism/arena_probe) and Prism
reads that; with neither the plan is refused naming the probe. The probe doubles its request while the component still
fits the card and keeps the ratio at the largest that fits, three runs within the storage measure's steadiness (on a
V100-16GB, HAT-S_SRx4: 3.50 at x1, 2.39 at x2, 1.48 at x4 — a fixed hold a small walk over-weights).

What this test would do if the code were wrong: a measured fact read over a vendor's declaration (every V100 plan
moves), a measured fact ignored on an unknown card (the refusal stays), a refusal that does not name the probe, a probe
that keeps the small walk's ratio or writes disagreeing runs, a probe run where the vendor declares the factor — RED.
No card."""
from __future__ import annotations

import types

import pytest

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


def test_an_unknown_card_reads_its_measured_fact(monkeypatch):
    assert _factor(monkeypatch, None, {"triton_arena_activation_factor": 1.476}) == 1.476


def test_with_neither_the_plan_is_refused_naming_the_probe(monkeypatch):
    with pytest.raises(S.PlanNotComputable, match="arena_probe"):
        _factor(monkeypatch, None, {})


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


def test_the_first_launch_probes_only_where_no_vendor_declares(monkeypatch, tmp_path):
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
    for declared, expect in ((1.35, []), (None, ["default-x"])):
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
    monkeypatch.setattr(A, "smallest_container", lambda cache: (tmp_path, "c"))
    seen = []
    monkeypatch.setattr(A, "factor_on_card", lambda usable, w, run, reps, steady: seen.append(usable) or (1.2, []))
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
