"""Each NeuroTax twin reads the tie a container declares (topology.json), names only, frozen; a key the container does
not carry is NOT DECLARED (never guessed); a malformed declaration is refused by name, with its remedy.

The declarations mirror the three containers the builder corrected for the tie (whisper-large-v3-turbo,
granite-3.1-1b-a400m-instruct, canary-qwen-2.5b): a token table, a tie inside one component, a tie across two
components, a hidden output, the words "self" and "tied". One test reads the real corrected copies and the shipped
containers when the share is mounted, and is skipped when it is not.

TWO FORMS: a topology with none of the new keys (token_embedding, tied_weights, hidden_output) is of the OLD
form: not declared and never refused, whatever its logits_source word (a shipped container of each old word:
lm_head, embed_weight_tied, self). One new key anywhere makes it of the tie form, read strictly.

What these would do if the code were wrong: a container without the keys gets a guessed table or word, or a shipped
container of an old word is refused (or its word mapped); a tie missing
its source_weight, a source outside the topology, a word outside head/self/tied, "tied" without a tie or without a
hidden output, "head" without a head component, or a hidden output a "self" stage never reads, loads silently or is
refused without its name and remedy; the two twins answer differently.
Run: python -m pytest tests/unit/nbx/test_each_neurotax_twin_reads_the_tie_a_container_declares.py
"""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

import pytest

from neurobrix.pytorch.neurotax import tie as pt_tie
from neurobrix.triton.neurotax import tie as tr_tie

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
TWINS = (pt_tie, tr_tie)
COPIES = [Path(p).expanduser() for p in (
    os.environ.get("NBX_TIE_COPIES", ""),
    "~/Mounts/Super-Models/_agents/transfer/builder_copies/tie_1124",
    "/mnt/agents_transfer/builder_copies/tie_1124") if p]
CACHES = [Path(p).expanduser() for p in ("~/Mounts/Super-NeuroBrix-Cache", "~/.neurobrix/cache")]


def _stage(component, **extra):
    return {"component": component, "execution": "autoregressive", **extra}


def whisper_turbo():
    """The whisper copy's declarations, with the ruled word "tied" on its decoder stage."""
    dec = {"type": "neural_component", "token_embedding": "token_embed.weight",
           "tied_weights": [{"param": "proj_out.weight", "source_component": "model.decoder",
                             "source_weight": "token_embed.weight"}],
           "hidden_output": "aten.native_layer_norm::12::out_0"}
    stages = [{"component": "model.encoder", "execution": "forward"},
              _stage("model.decoder", cross_attention_from="model.encoder", logits_source="tied")]
    return {"model_name": "whisper-large-v3-turbo",
            "components": {"model.decoder": dec, "model.encoder": {"type": "neural_component"}},
            "flow": {"type": "encoder_decoder", "audio": {"stages": copy.deepcopy(stages)}, "stages": stages}}


def granite():
    return {"model_name": "granite-3.1-1b-a400m-instruct",
            "components": {"model": {"type": "language_model", "token_embedding": "token_embed.weight"},
                           "lm_head": {"type": "neural_component", "tied_weights": [
                               {"param": "weight", "source_component": "model", "source_weight": "token_embed.weight"}]}},
            "flow": {"type": "autoregressive_generation", "generation": {
                "type": "autoregressive_text", "lm_component": "model", "head_component": "lm_head"}}}


def canary():
    stages = [{"component": "perception", "execution": "forward"}, _stage("llm", logits_source="self")]
    return {"model_name": "canary-qwen-2.5b",
            "components": {"embed_tokens": {"token_embedding": "weight"},
                           "llm": {"tied_weights": [{"param": "head.weight", "source_component": "embed_tokens",
                                                     "source_weight": "weight"}]},
                           "perception": {}},
            "flow": {"type": "audio_llm", "audio": {"stages": copy.deepcopy(stages)}, "stages": stages}}


def tinyllama_shipped():
    """A shipped container of today: no tie key anywhere."""
    return {"model_name": "TinyLlama-1.1B-Chat-v1.0",
            "components": {"model": {"type": "language_model"}, "lm_head": {"type": "neural_component"}},
            "flow": {"type": "autoregressive_generation", "generation": {
                "type": "autoregressive_text", "lm_component": "model", "head_component": "lm_head"}}}


@pytest.mark.parametrize("twin", TWINS)
def test_the_whisper_declaration_reads_its_table_its_tie_its_hidden_and_the_word_tied(twin):
    r = twin.read_tie(whisper_turbo())
    dec = r.component("model.decoder")
    assert dec.token_embedding == "token_embed.weight"
    assert dec.tied_weights == (twin.TiedWeight("proj_out.weight", "model.decoder", "token_embed.weight"),)
    assert dec.hidden_output == "aten.native_layer_norm::12::out_0"
    assert not r.component("model.encoder").declared
    assert [(s.where, s.component, s.logits_source) for s in r.stages] == [
        ("flow.stages[0]", "model.encoder", None), ("flow.stages[1]", "model.decoder", "tied"),
        ("flow.audio.stages[0]", "model.encoder", None), ("flow.audio.stages[1]", "model.decoder", "tied")]
    with pytest.raises(Exception):
        dec.token_embedding = "other"          # frozen


@pytest.mark.parametrize("twin", TWINS)
def test_the_granite_declaration_ties_the_head_to_the_models_table(twin):
    r = twin.read_tie(granite())
    assert r.component("model").token_embedding == "token_embed.weight"
    assert r.component("lm_head").tied_weights == (twin.TiedWeight("weight", "model", "token_embed.weight"),)
    assert r.stages == (twin.StageLogits("flow.generation", "model", None, "lm_head"),)


@pytest.mark.parametrize("twin", TWINS)
def test_the_canary_declaration_ties_across_two_components_and_says_self(twin):
    r = twin.read_tie(canary())
    assert r.component("embed_tokens").token_embedding == "weight"
    assert r.component("llm").tied_weights == (twin.TiedWeight("head.weight", "embed_tokens", "weight"),)
    assert r.component("llm").hidden_output is None
    assert {s.logits_source for s in r.stages if s.component == "llm"} == {"self"}


@pytest.mark.parametrize("twin", TWINS)
def test_a_container_without_the_keys_is_not_declared(twin):
    r = twin.read_tie(tinyllama_shipped())
    assert not r.declared and r.old_form and not twin.carries_the_new_keys(tinyllama_shipped())
    assert all(c.token_embedding is None and c.tied_weights == () and c.hidden_output is None for c in r.components)
    assert all(s.logits_source is None for s in r.stages)


def _break_tie_key(t):
    del t["components"]["model.decoder"]["tied_weights"][0]["source_weight"]


def _break_source(t):
    t["components"]["model.decoder"]["tied_weights"][0]["source_component"] = "model.text_decoder"


def _break_word(t):
    t["flow"]["stages"][1]["logits_source"] = "embed_weight_tied"


def _drop_tie(t):
    del t["components"]["model.decoder"]["tied_weights"]


def _drop_hidden(t):
    del t["components"]["model.decoder"]["hidden_output"]


def _head_without_head(t):
    t["flow"]["stages"][1]["logits_source"] = "head"


def _self_with_unread_hidden(t):
    for stages in (t["flow"]["stages"], t["flow"]["audio"]["stages"]):    # no 'tied' stage is left to read it
        stages[1]["logits_source"] = "self"


def _tie_not_a_list(t):
    t["components"]["model.decoder"]["tied_weights"] = {"param": "proj_out.weight"}


def _empty_table(t):
    t["components"]["model.decoder"]["token_embedding"] = ""


def _tie_twice(t):
    ties = t["components"]["model.decoder"]["tied_weights"]
    ties.append(dict(ties[0]))


MALFORMED = [
    (_break_tie_key, "component 'model.decoder'", "tied_weights[0].source_weight", "missing"),
    (_break_source, "component 'model.decoder'", "tied_weights[0].source_component", "not a component of the topology"),
    (_break_word, "stage flow.stages[1]", "logits_source", "'embed_weight_tied' is not one of"),
    (_drop_tie, "stage flow.stages[1]", "logits_source", "declares no tied_weights"),
    (_drop_hidden, "stage flow.stages[1]", "logits_source", "declares no hidden_output"),
    (_head_without_head, "stage flow.stages[1]", "logits_source", "names no head component"),
    (_self_with_unread_hidden, "component 'model.decoder'", "hidden_output", "nothing reads it"),
    (_tie_not_a_list, "component 'model.decoder'", "tied_weights", "not a non-empty list"),
    (_empty_table, "component 'model.decoder'", "token_embedding", "not a non-empty name"),
    (_tie_twice, "component 'model.decoder'", "tied_weights", "tied more than once"),
]


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("breaker,place,key,wrong", MALFORMED, ids=[m[0].__name__ for m in MALFORMED])
def test_a_malformed_declaration_is_refused_by_name_with_its_remedy(twin, breaker, place, key, wrong):
    t = whisper_turbo()
    breaker(t)
    with pytest.raises(twin.TieDeclarationError) as e:
        twin.read_tie(t)
    msg = str(e.value)
    assert "container 'whisper-large-v3-turbo'" in msg and place in msg and f"key '{key}'" in msg and wrong in msg
    assert "FIX: rebuild the container with the builder so that" in msg and "is declared" in msg


@pytest.mark.parametrize("twin", TWINS)
def test_a_head_stage_with_its_head_component_is_read(twin):
    t = canary()
    t["components"]["lm_head"] = {}
    t["flow"]["stages"][1].update(logits_source="head", head_component="lm_head")
    r = twin.read_tie(t)
    assert r.stages[1] == twin.StageLogits("flow.stages[1]", "llm", "head", "lm_head")


def _as_data(reading):
    return repr(reading).replace("neurobrix.triton.", "").replace("neurobrix.pytorch.", "")


def test_both_twins_give_identical_results():
    assert (SRC / "pytorch" / "neurotax" / "tie.py").read_bytes() == (SRC / "triton" / "neurotax" / "tie.py").read_bytes()
    for make in (whisper_turbo, granite, canary, tinyllama_shipped):
        assert _as_data(pt_tie.read_tie(make())) == _as_data(tr_tie.read_tie(make()))
    for breaker, *_ in MALFORMED:
        msgs = []
        for twin in TWINS:
            t = whisper_turbo()
            breaker(t)
            with pytest.raises(twin.TieDeclarationError) as e:
                twin.read_tie(t)
            msgs.append(str(e.value))
        assert msgs[0] == msgs[1]


def test_the_triton_tie_reader_imports_only_the_standard_library():
    tree = ast.parse((SRC / "triton" / "neurotax" / "tie.py").read_text())
    imported = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__", "typing", "dataclasses"}, imported


def _first(paths):
    return next((p for p in paths if p.is_dir()), None)


def test_the_corrected_copies_on_the_share_read_as_declared():
    root = _first(COPIES)
    if root is None or not (root / "canary-qwen-2.5b" / "topology.json").is_file():
        pytest.skip("the corrected copies are not on this machine")
    load = lambda m: json.loads((root / m / "topology.json").read_text())
    for twin in TWINS:
        g = twin.read_tie(load("granite-3.1-1b-a400m-instruct"))
        assert g.component("lm_head").tied_weights == (twin.TiedWeight("weight", "model", "token_embed.weight"),)
        c = twin.read_tie(load("canary-qwen-2.5b"))
        assert c.component("llm").tied_weights == (twin.TiedWeight("head.weight", "embed_tokens", "weight"),)
        w = load("whisper-large-v3-turbo")
        assert twin.carries_the_new_keys(w)        # the new keys + (today) the old word: refused, never old form
        words = {s.get("logits_source") for s in w["flow"]["stages"]} - {None}
        if words <= set(twin.LOGITS_SOURCES):
            assert twin.read_tie(w).component("model.decoder").tied_weights
        else:                                  # a copy that still carries a word outside the three
            with pytest.raises(twin.TieDeclarationError, match="logits_source"):
                twin.read_tie(w)


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("word", ["lm_head", "embed_weight_tied", "self", "a word of no form"])
def test_an_old_form_topology_is_never_refused_whatever_its_word(twin, word):
    t = vlm(word)
    del t["components"]["model.language_model"]["token_embedding"]
    r = twin.read_tie(t)
    assert r.old_form and not r.declared and r.stages == ()


@pytest.mark.parametrize("twin", TWINS)
def test_one_new_key_anywhere_makes_the_topology_of_the_tie_form(twin):
    assert twin.carries_the_new_keys(whisper_turbo()) and twin.carries_the_new_keys(granite())
    t = tinyllama_shipped()
    t["components"]["model"]["token_embedding"] = "token_embed.weight"
    assert twin.carries_the_new_keys(t) and not twin.read_tie(t).old_form


#: a shipped container of each old word (the cache as it is, before the tie form)
OLD_WORDS = {"lm_head": ["GLM-4.1V-9B-Thinking", "Voxtral-Mini-3B-2507", "granite-speech-3.3-8b"],
             "embed_weight_tied": ["whisper-large-v3-turbo"], "self": ["canary-qwen-2.5b", "s1-mini"]}


@pytest.mark.parametrize("word,models", OLD_WORDS.items(), ids=list(OLD_WORDS))
def test_a_shipped_container_of_each_old_word_reads_as_old_form(word, models):
    root = _first(CACHES)
    paths = [root / m / "topology.json" for m in models] if root else []
    paths = [p for p in paths if p.is_file()]
    if not paths:
        pytest.skip("no shipped container of this word on this machine")
    for path in paths:
        t = json.loads(path.read_text())
        assert word in json.dumps(t["flow"]), path
        for twin in TWINS:
            r = twin.read_tie(t)
            assert r.old_form and not r.declared, path


def test_shipped_containers_without_the_keys_read_as_not_declared():
    root = _first(CACHES)
    shipped = [root / m / "topology.json" for m in ("TinyLlama-1.1B-Chat-v1.0", "granite-3.1-1b-a400m-instruct")] \
        if root else []
    shipped = [p for p in shipped if p.is_file()]
    if not shipped:
        pytest.skip("no shipped container on this machine")
    for path in shipped:
        for twin in TWINS:
            assert not twin.read_tie(json.loads(path.read_text())).declared, path



def vlm(word):
    """A vision-language container of the shipped shape (GLM-4.1V-9B-Thinking's topology.json, flow.vlm): the section
    names its vision, language-model and head components, its stages name their component by `component`; it carries
    a new key (token_embedding), so it is of the tie form and read strictly."""
    return {"model_name": "GLM-4.1V-9B-Thinking",
            "components": {"model.visual": {}, "model.language_model": {"token_embedding": "embed_tokens.weight"},
                           "lm_head": {}},
            "flow": {"type": "vlm", "vlm": {
                "vision_component": "model.visual", "lm_component": "model.language_model",
                "head_component": "lm_head", "stages": [
                    {"component": "model.visual", "execution": "forward"},
                    {"component": "model.language_model", "execution": "autoregressive", "logits_source": word}]}}}


@pytest.mark.parametrize("twin", TWINS)
def test_a_vlm_stage_of_the_shipped_shape_is_read_with_its_section_s_head(twin):
    r = twin.read_tie(vlm("head"))
    assert r.stages[1] == twin.StageLogits("flow.vlm.stages[1]", "model.language_model", "head", "lm_head")


@pytest.mark.parametrize("twin", TWINS)
def test_the_shipped_vlm_word_lm_head_is_refused_by_name(twin):
    with pytest.raises(twin.TieDeclarationError) as e:
        twin.read_tie(vlm("lm_head"))
    msg = str(e.value)
    assert "stage flow.vlm.stages[1]" in msg and "'lm_head' is not one of" in msg and "FIX: rebuild" in msg


@pytest.mark.parametrize("twin", TWINS)
def test_a_stage_declaring_logits_source_without_its_component_is_refused(twin):
    t = vlm("head")
    del t["flow"]["vlm"]["stages"][1]["component"]
    with pytest.raises(twin.TieDeclarationError) as e:
        twin.read_tie(t)
    msg = str(e.value)
    assert "stage flow.vlm.stages[1]" in msg and "key 'component'" in msg and "names no component" in msg
    assert "FIX: rebuild the container with the builder so that the stage's component is declared" in msg


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("where", ["section", "nested"])
def test_a_logits_source_outside_a_stage_is_refused_never_ignored(twin, where):
    t = vlm("head")
    if where == "section":
        t["flow"]["vlm"]["logits_source"] = "head"
        path = "flow path flow.vlm,"
    else:
        t["flow"]["vlm"]["output"] = {"modality": "text", "logits_source": "self"}
        path = "flow path flow.vlm.output,"
    with pytest.raises(twin.TieDeclarationError) as e:
        twin.read_tie(t)
    msg = str(e.value)
    assert path in msg and "key 'logits_source'" in msg and "outside a stage" in msg and "FIX: rebuild" in msg


@pytest.mark.parametrize("twin", TWINS)
def test_a_new_key_off_a_component_of_an_old_topology_is_refused_as_misplaced(twin):
    t = vlm("lm_head")                         # qa's probe: an old topology plus an unrelated hidden_output
    del t["components"]["model.language_model"]["token_embedding"]
    t["metadata"] = {"hidden_output": "x"}
    assert not twin.carries_the_new_keys(t)
    with pytest.raises(twin.TieDeclarationError) as e:
        twin.read_tie(t)
    msg = str(e.value)
    assert "topology path metadata.hidden_output" in msg and "misplaced" in msg and "key 'hidden_output'" in msg
    assert "FIX: rebuild the container with the builder so that hidden_output on its component is declared" in msg
    assert "'lm_head' is not one of" not in msg


@pytest.mark.parametrize("twin", TWINS)
def test_a_new_key_nested_inside_a_component_is_misplaced(twin):
    t = whisper_turbo()
    t["components"]["model.encoder"]["shapes"] = {"token_embedding": "token_embed.weight"}
    with pytest.raises(twin.TieDeclarationError, match=r"topology path components\.model\.encoder\.shapes"
                                                       r"\.token_embedding.*misplaced"):
        twin.read_tie(t)


@pytest.mark.parametrize("twin", TWINS)
def test_a_hidden_output_no_stage_reads_is_refused(twin):
    t = granite()                              # qa's probe: components.<c>.hidden_output, no 'tied'/'head' stage
    t["components"]["model"]["hidden_output"] = "aten.mm::0::out_0"
    with pytest.raises(twin.TieDeclarationError) as e:
        twin.read_tie(t)
    msg = str(e.value)
    assert "component 'model'" in msg and "key 'hidden_output'" in msg and "nothing reads it" in msg
    assert "FIX: rebuild the container with the builder so that" in msg


@pytest.mark.parametrize("twin", TWINS)
def test_a_hidden_output_a_head_stage_reads_is_accepted(twin):
    t = granite()                              # the head graph reads a hidden the language model does not return
    t["components"]["model"]["hidden_output"] = "aten.mm::0::out_0"
    t["flow"]["generation"]["logits_source"] = "head"
    r = twin.read_tie(t)
    assert r.component("model").hidden_output == "aten.mm::0::out_0"
    assert r.stages[-1] == twin.StageLogits("flow.generation", "model", "head", "lm_head")
