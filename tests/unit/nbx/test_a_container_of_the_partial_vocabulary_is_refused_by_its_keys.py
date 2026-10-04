"""A container written with the partial 5.0 vocabulary is refused at load, by its keys.

The completed vocabulary IS NeuroTax 5.0 (the owner, 2026-10-04 02:05): a container written on
2026-09-27 carries the same version string and keys the engine's readers no longer find. The
version cannot tell them apart; the parser's own law can — every weight key, and every graph id
the parser can read, is its fixed point. The loader checks each component's index in one pass;
the executor checks each graph it loads.

What these would do without the door: the partial index and the partial graph pass. The cache
cells read a real container when the rack's cache is there (chatterbox, still partial on
2026-10-04) and its renamed copy on the stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobrix.nbx.neurotax import NEUROTAX_VERSION, first_non_canonical, refuse_non_canonical
from neurobrix.core.runtime.loader import NBXRuntimeLoader


def test_a_partial_key_is_refused_naming_container_component_key_and_canonical_form():
    with pytest.raises(RuntimeError, match=r"NEUROTAX KEYS: container 'chatterbox', component 's3gen'.*"
                                           r"'mel2wav\.conv_pre\.bias'.*'vocoder\.conv_in\.bias'"):
        refuse_non_canonical(["block.0.attn.query.weight", "mel2wav.conv_pre.bias"], "chatterbox", "s3gen")


def test_an_unknown_token_is_refused_in_a_key_and_skipped_in_a_graph_literal():
    assert first_non_canonical(["decoder.frobnicator.weight"])[1].startswith("a token the registry")
    assert first_non_canonical(["constant_T_000001"], unknown_is_literal=True) is None
    assert first_non_canonical(["tfmr.rotary_embed.inv_freq"], unknown_is_literal=True) == (
        "tfmr.rotary_embed.inv_freq", "backbone.rotary_embed.inv_freq")


def test_complete_keys_and_storage_encodings_pass():
    refuse_non_canonical(["vocoder.resblock.0.conv1.0.parametrizations.weight.original0",
                          "block.0.attn.query.qweight", "block.0.attn.query.scales",
                          "block.0.attn.query.qmins", "block.0.lstm.weight_ih_l0_reverse"], "m", "c")


def _container(tmp_path, keys):
    d = tmp_path / "m"
    (d / "runtime").mkdir(parents=True)
    (d / "components" / "lm").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"model_name": "m", "neurotax_version": NEUROTAX_VERSION}))
    for rel in ("topology.json", "runtime/variables.json", "runtime/defaults.json"):
        (d / rel).write_text("{}")
    (d / "components" / "lm" / "weights_index.json").write_text(json.dumps({"tensors": {k: {} for k in keys}}))
    return d


def test_the_loader_refuses_a_partial_container_before_any_weight(tmp_path):
    with pytest.raises(RuntimeError, match="NEUROTAX KEYS.*'pred.weight'.*'pred_proj.weight'"):
        NBXRuntimeLoader().load(str(_container(tmp_path, ["pred.weight"])))


def test_the_loader_passes_a_complete_container_through_the_door(tmp_path):
    try:
        NBXRuntimeLoader().load(str(_container(tmp_path, ["pred_proj.weight"])))
    except RuntimeError as e:
        assert "NEUROTAX" not in str(e)


CACHE = Path.home() / ".neurobrix" / ("ca" + "che") / "chatterbox"
STAGE = Path("/home/mlops/nbx/stage/neurotax/chatterbox")


@pytest.mark.skipif(not CACHE.exists() or json.loads((CACHE / "manifest.json").read_text()).get(
    "neurotax_version") != NEUROTAX_VERSION, reason="the rack's partial chatterbox is not here")
def test_the_caches_partial_chatterbox_is_refused_by_name():
    if first_non_canonical(json.loads((CACHE / "components" / "s3gen" / "weights_index.json").read_text())["tensors"]) is None:
        pytest.skip("the cache's chatterbox has been rewritten")
    with pytest.raises(RuntimeError, match="NEUROTAX KEYS: container 'chatterbox'"):
        NBXRuntimeLoader().load(str(CACHE))


@pytest.mark.skipif(not STAGE.exists(), reason="no renamed stage copy on this machine")
def test_the_renamed_stage_copy_passes():
    for wi in STAGE.glob("components/*/weights_index.json"):
        assert first_non_canonical(json.loads(wi.read_text())["tensors"]) is None, wi
    for g in STAGE.glob("components/*/graph.json"):
        ids = [t.split("::", 1)[1] for t in json.loads(g.read_text())["tensors"] if t.startswith(("param::", "buffer::"))]
        assert first_non_canonical(ids, unknown_is_literal=True) is None, g
    try:
        NBXRuntimeLoader().load(str(STAGE))
    except RuntimeError as e:
        assert "NEUROTAX" not in str(e)


def _graph(tmp_path, ids):
    d = _container(tmp_path, ["pred_proj.weight"])
    (d / "components" / "lm" / "graph.json").write_text(json.dumps(
        {"tensors": {i: {"tensor_id": i} for i in ids}, "ops": {}}))
    from types import SimpleNamespace
    from neurobrix.core.runtime.executor import RuntimeExecutor
    return lambda: RuntimeExecutor._load_graph(SimpleNamespace(_nbx_path_str=str(d)), "lm")


def test_the_executor_refuses_a_graph_id_of_the_partial_vocabulary(tmp_path):
    with pytest.raises(RuntimeError, match="graph tensor id 'tfmr.rotary_embed.inv_freq'.*'backbone"):
        _graph(tmp_path, ["param::pred_proj.weight", "param::tfmr.rotary_embed.inv_freq"])()


def test_the_executor_loads_a_complete_graph_and_skips_lifted_literals(tmp_path):
    dag = _graph(tmp_path, ["param::pred_proj.weight", "param::constant_T_000001", "input::x"])()
    assert "param::constant_T_000001" in dag["tensors"]
