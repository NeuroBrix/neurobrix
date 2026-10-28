"""The compiled engine's [Peak] line is the whole RUN's peak: each component restarts torch's peak counter at its load
(`_reset_cuda_peak`, so its release line says its own worst op), and the run's line read the counter after the last
one only — Flex.1-alpha native said 792 MB (its VAE) while its transformer held 12 466 MB (the lot gate's registers,
2026-10-27). What the test would do if the code were wrong: the last component's peak — red."""
import pytest

from neurobrix.core.runtime import graph_executor as G

torch = pytest.importorskip("torch")


def test_the_run_peak_survives_the_components_resets(monkeypatch):
    state = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda idx: state["peak"])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda idx: state.update(peak=0))
    monkeypatch.setattr(G, "_RESERVED_PEAK_BEFORE_RESET", {})
    state["peak"] = 9_000
    G._reset_cuda_peak("cuda:0")          # the text encoder ran, the transformer loads
    state["peak"] = 12_000
    G._reset_cuda_peak("cuda:0")          # the transformer ran, the VAE loads
    state["peak"] = 600
    assert G.process_peak_reserved(0) == 12_000
