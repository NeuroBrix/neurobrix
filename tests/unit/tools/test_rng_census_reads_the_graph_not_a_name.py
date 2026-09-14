"""The RNG census is a brick, read from graphs and topologies, never from a
model name (D-RNG-DRAW-UNARMED-IN-A-FLOW: 4 of 56 containers draw inside
their graph; the guard that runs each twice must read the same list).

Injection (2026-09-14): with `RNG_OPS` emptied, the first test went RED
(no container found); with `request_reaching` returning "default" for every
component, the speech-leg test went RED. Restored, green.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import rng_census as RC  # noqa: E402


def _container(root, name, comps, flow):
    d = root / name; (d / "components").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"family": "tts"}))
    (d / "topology.json").write_text(json.dumps({"flow": flow}))
    for comp, ops in comps.items():
        (d / "components" / comp).mkdir()
        (d / "components" / comp / "graph.json").write_text(json.dumps(
            {"ops": [{"op_uid": f"{o}::{i}", "op_type": o} for i, o in enumerate(ops)]}))
    return d


def test_the_census_finds_a_draw_in_a_nested_component_and_nothing_elsewhere(tmp_path):
    _container(tmp_path, "draws", {"decoder": ["aten::conv1d", "aten::rand", "aten::randn_like", "aten::randn_like"],
                                   "encoder": ["aten::mm"]},
               {"order": ["encoder", "decoder"]})
    _container(tmp_path, "pure", {"model": ["aten::mm", "aten::multinomial"]}, {"order": ["model"]})
    (tmp_path / "draws-backup").mkdir()
    census = RC.containers_with_rng_ops(tmp_path)
    assert census == {"draws": {"decoder": {"aten::rand": 1, "aten::randn_like": 2}}}


def test_a_speech_leg_component_is_reached_by_the_speech_request_not_the_default(tmp_path):
    topo = {"flow": {"order": ["vpm", "llm.model"],
                     "speech": {"components": {"backbone": "tts.model", "vocoder": "hift"}}}}
    assert RC.request_reaching(topo, "llm.model") == "default"
    assert RC.request_reaching(topo, "hift") == "speech"
    assert RC.request_reaching(topo, "nowhere") is None
    audio = {"flow": {"order": [], "audio": {"stages": [{"component": "codec.decoder"}]}}}
    assert RC.request_reaching(audio, "codec.decoder") == "default"
