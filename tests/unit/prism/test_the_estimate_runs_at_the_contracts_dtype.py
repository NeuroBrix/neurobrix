"""Prism sizes a component's ACTIVATIONS at the dtype they will hold, read
from the precision contract's own decision — not from the profile's preferred
dtype. On the conservative path (no calibration record for this graph, or a
record measured "conservative" on this arch) the matmul results are stored in
fp32 and their consumers compute in fp32: Mochi's VAE decoder was planned at
3.2 GB and held 26 GB (2026-09-14), half of the gap being this dtype.

The weights and the executor's per-component dtype map are untouched: the
first wiring fed the contract's dtype into the component map itself and the
56-container explain-plan sweep (2026-09-16) doubled thirty weight bills and
moved nine strategies — the last test pins that boundary.

Injection: `contract_taken` forced True made the first test RED; restored,
green. The last test was RED on the first wiring (the map returned float32).
"""
from types import SimpleNamespace

from neurobrix.core.prism.solver import PrismSolver
from neurobrix.core.runtime import precision_contract as PC


def _comp(name="vae"):
    return SimpleNamespace(name=name, graph={"ops": []})


def _container(path="/tmp/x"):
    return SimpleNamespace(cache_path=path)


def test_no_record_means_the_conservative_path_is_estimated_at_fp32(monkeypatch):
    monkeypatch.setattr(PC, "load_calibration", lambda *a, **k: None)
    monkeypatch.setattr(PC, "_env_force", lambda: None)
    assert PrismSolver._activation_dtype_under_the_contract("float16", _comp(), _container()) == "float32"


def test_a_record_the_contract_takes_keeps_fp16(monkeypatch):
    rec = SimpleNamespace(prefer={}, timing={})
    monkeypatch.setattr(PC, "load_calibration", lambda *a, **k: rec)
    monkeypatch.setattr(PC, "_env_force", lambda: None)
    assert PrismSolver._activation_dtype_under_the_contract("float16", _comp(), _container()) == "float16"


def test_a_record_measured_conservative_on_this_arch_is_fp32(monkeypatch):
    rec = SimpleNamespace(prefer={"cuda-70": "conservative"}, timing={})
    monkeypatch.setattr(PC, "load_calibration", lambda *a, **k: rec)
    monkeypatch.setattr(PC, "_env_force", lambda: None)
    assert PC.contract_taken("/tmp/x", "vae", {}, compute_dtype="float16", arch="cuda-70") is False
    assert PC.contract_taken("/tmp/x", "vae", {}, compute_dtype="float16", arch="cuda-80") is True


def test_other_dtypes_and_a_container_without_a_path_are_untouched(monkeypatch):
    monkeypatch.setattr(PC, "load_calibration", lambda *a, **k: None)
    assert PrismSolver._activation_dtype_under_the_contract("float32", _comp(), _container()) == "float32"
    assert PrismSolver._activation_dtype_under_the_contract("bfloat16", _comp(), _container()) == "bfloat16"
    assert PrismSolver._activation_dtype_under_the_contract("float16", _comp(), SimpleNamespace(cache_path=None)) == "float16"


def test_the_component_map_the_executor_uses_keeps_the_weights_dtype(monkeypatch):
    """The contract sizes activations only; the weights are loaded as they are
    and the executor's per-component dtype is not the estimator's to change."""
    monkeypatch.setattr(PC, "load_calibration", lambda *a, **k: None)
    monkeypatch.setattr(PC, "_env_force", lambda: None)
    solver = PrismSolver.__new__(PrismSolver)
    comp = SimpleNamespace(name="transformer", graph={"ops": []}, get_dominant_dtype=lambda: "float16")
    profile = SimpleNamespace(preferred_dtype="float16", devices_support_dtype=lambda d: True)
    monkeypatch.setattr(PrismSolver, "_components_force_fp32", lambda self, c, p: set())
    assert solver._resolve_component_dtypes([comp], profile, _container()) == {"transformer": "float16"}
    assert PrismSolver._activation_dtype_under_the_contract("float16", comp, _container()) == "float32"
