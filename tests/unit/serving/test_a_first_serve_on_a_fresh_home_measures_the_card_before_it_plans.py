"""The first `neurobrix serve` on a home where no `neurobrix run` was ever made plans on a profile whose card carries
its runtime total and context: serve asks the first-launch measurements through the one path `run` asks them.

Before (gate cell L3, 18/11/2026): the daemon's engine loaded the generated profile as written, without the
first-launch measurements (`autodetect.ensure_measured_facts`, which completes `runtime_card_facts`), and the plan
refused: "ProfileFactMissing: ZERO FALLBACK: the hardware profile declares no runtime_context_mb (nor
runtime_total_mb) for cuda:0", with --compiled and with --triton.

What this stands on: `InferenceEngine.load` — what the serve daemon runs in its own process once
`ServingDaemon._resolve_hardware` has generated the default profile — on an empty profile folder holding one freshly
generated profile whose card states every first-launch fact except the two runtime figures (so no GEMM or copy probe
runs here). The one stub of the device is `facts_probe.card_facts_in_a_child`, the child process that reads
`autodetect.runtime_card_facts(index)` on the real card: it answers the Dell's reading (16151 MB, 306 MB). The model
lookup and the container are stand-ins that stop the load right after the profile is read. If the code is wrong (serve
never asks the measurements), the card is never read and the profile the plan gets has no runtime context: red.
"""
import pytest
import yaml
from types import SimpleNamespace as NS

DELL = {"runtime_total_mb": 16151, "runtime_context_mb": 306}


class _Stop(Exception):
    pass


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    from neurobrix.core.prism import autodetect as A
    from neurobrix.core.prism import facts_probe as P
    from neurobrix.core.prism import loader as L
    hw = tmp_path / "hardware"
    hw.mkdir()
    monkeypatch.setattr(A, "HARDWARE_DIR", hw)
    monkeypatch.setattr(L, "GENERATED_DIR", hw)
    read = []
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda index: read.append(index) or dict(DELL))
    monkeypatch.setattr("nbxtensor.device.count", lambda: 2)   # several cards seen: the child reads them
    monkeypatch.setattr(P, "vmm_first_replay", lambda *a, **k: None)
    measured = {"source": "measured at generation", "matmul_tflops": {"float16": 100.0}, "h2d_gbps": 12.0,
                "d2h_gbps": 12.0, "d2d_gbps": 800.0, "vmm_chunk_bytes": 2 << 20,
                "gemv_gbps": {"compiled": {"float16": 700.0}, "triton": {"float16": 700.0}},
                "expert_copy": {"block": 1024, "warps": 4}}
    (hw / "default-fresh.yml").write_text(yaml.safe_dump({
        "id": "default-fresh",
        "devices": [{"index": 0, "name": "a discrete card", "memory_mb": 16384, "supports_dtypes": ["float16"],
                     "measured": measured}],
        "host_measured": {"source": "measured at generation", "host_matmul_gflops": 300, "host_memory_gbps": 20}}))
    container_dir = tmp_path / "container"
    (container_dir / "components").mkdir(parents=True)
    import neurobrix.cli.utils as U
    import neurobrix.nbx as N
    import neurobrix.processors as PR
    monkeypatch.setattr(U, "find_model", lambda name: container_dir)
    monkeypatch.setattr(N.NBXContainer, "load", staticmethod(lambda p: NS(
        _cache_path=container_dir, get_manifest=lambda: {"family": "llm"}, get_neural_components=lambda: [])))
    planned = []
    real = L.load_profile

    def load_profile(hardware_id):
        planned.append(real(hardware_id))
        return planned[-1]
    import neurobrix.core.prism as PRISM
    monkeypatch.setattr(L, "load_profile", load_profile)
    monkeypatch.setattr(PRISM, "load_profile", load_profile)
    monkeypatch.setattr(PR, "open_declared", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    return NS(hw=hw, read=read, planned=planned)


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_the_first_serve_reads_the_card_s_runtime_figures_before_it_plans(fresh_home, mode):
    from neurobrix.serving.engine import InferenceEngine
    with pytest.raises(_Stop):
        InferenceEngine("a model", "default-fresh", mode).load()
    assert fresh_home.read == [0], "serve never asked the first-launch measurements: the card was not read"
    card = fresh_home.planned[-1].devices[0]
    assert (card.runtime_total_mb, card.runtime_context_mb) == (16151, 306)
    written = yaml.safe_load((fresh_home.hw / "default-fresh.yml").read_text())["devices"][0]
    assert (written["runtime_total_mb"], written["runtime_context_mb"]) == (16151, 306)


def test_the_served_launch_names_its_profile_for_the_loader_s_own_rates(fresh_home, monkeypatch):
    from neurobrix.core.prism import autodetect as A
    from neurobrix.serving.engine import InferenceEngine
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", None)
    with pytest.raises(_Stop):
        InferenceEngine("a model", "default-fresh", "compiled").load()
    assert A.ACTIVE_HARDWARE_ID == "default-fresh"
