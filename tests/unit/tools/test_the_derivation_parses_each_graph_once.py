"""The derivation parses a component's graph.json ONCE per process, however many extent probes ask for it.

Each probe re-read and re-parsed it: openaudio's LM graph is 189 MB and its decode extents ask hundreds
of derivations each, so the tts table ran past its hour after two models (2026-10-04); with one parse
orpheus's 16 GB table took 119 s and its 1 512 rows were identical to the re-parsing run's. The red
line: a census takes minutes per model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derived_census as D  # noqa: E402


def test_a_graph_is_read_from_disk_once(tmp_path, monkeypatch):
    comp = tmp_path / "M" / "components" / "c"
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps({"tensors": {}, "ops": {}}))
    monkeypatch.setattr(D, "CACHE", tmp_path)
    monkeypatch.setattr(D, "_RAW_GRAPHS", {})
    reads = []
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self.name), real(self, *a, **k))[1])
    first = D.raw_graph("M", "c")
    for _ in range(50):
        assert D.raw_graph("M", "c") is first
    assert reads.count("graph.json") == 1, reads


def test_the_precision_contract_is_computed_once_per_component(monkeypatch):
    """The plan-time precision contract reads the calibration record and the graph, never the request's
    extents; an extent probe recomputed it over the whole graph each time (openaudio: 490 s with it
    computed once, against a 2 400 s timeout before)."""
    calls = []
    from neurobrix.core.prism import runtime_widths as RW
    monkeypatch.setattr(RW, "plan_time_contract", lambda *a, **k: calls.append(a) or object())
    monkeypatch.setattr(D, "_CONTRACTS", {})
    g = {}
    first = D._contract("M", "c", g, "float16")
    for _ in range(20):
        assert D._contract("M", "c", g, "float16") is first
    assert len(calls) == 1, calls
    D._contract("M", "c", g, "bfloat16")          # another compute dtype is another contract
    assert len(calls) == 2, calls
