"""The plan prices every decode cache the flow opens, from the facts the session builds it from.

`core.runtime.lm_facts` is the one reader: the LM facts (the package's `lm_config`, else the LM
component's extracted values), the component a flow decodes with, and how many sequences the
cache holds at once. Before it, Prism read `lm_config` alone and priced no cache for VibeVoice's
next-token diffusion, whose session opened two (the prompt context and the CFG negative one):
714 MiB unplanned (2026-09-29, the derived census). Janus's image generation decodes
[cond, uncond] as a batch of two and was priced for one.

Injections, each seen RED: `_read_lm_config` back to `defaults["lm_config"]` only -> the
VibeVoice plan case; `decode_sequences` returning 1 -> both pricing cases; the compiled session
reading `lm_config` directly -> the one-reader case; the solver reading the flow through
`container.get_topology()` (the deprecated alias of the graph, None on a real plan) -> both
Prism cases.
"""
import ast
import inspect
import json
import textwrap

from neurobrix.core.runtime import lm_facts as F

VIBEVOICE_TOPO = {
    "components": {"model.language_model": {}, "model.prediction_head": {}},
    "flow": {"type": "next_token_diffusion",
             "stages": [{"component": "model.language_model", "execution": "forward"},
                        {"component": "model.prediction_head", "execution": "diffusion",
                         "diffusion": {"condition_from": "model.language_model"}}]},
    "extracted_values": {"model.language_model": {"num_layers": 28, "num_heads": 12,
                                                  "hidden_size": 1536}},
}
JANUS_TOPO = {"components": {"language_model": {}, "gen_head": {}},
              "flow": {"type": "autoregressive_generation",
                       "generation": {"type": "autoregressive_image",
                                      "lm_component": "language_model"}}}


def test_the_lm_facts_are_the_packages_else_the_components_extracted_values():
    assert F.lm_config_of({"lm_config": {"num_layers": 3}}, VIBEVOICE_TOPO, "model.language_model") \
        == {"num_layers": 3}
    got = F.lm_config_of({}, VIBEVOICE_TOPO, "model.language_model")
    assert (got["num_layers"], got["num_heads"], got["hidden_size"], got["num_kv_heads"]) \
        == (28, 12, 1536, None)


def test_the_decoder_is_named_by_the_flow():
    assert F.decode_lm_component(VIBEVOICE_TOPO, VIBEVOICE_TOPO["components"]) == "model.language_model"
    assert F.decode_lm_component(JANUS_TOPO, JANUS_TOPO["components"]) == "language_model"
    assert F.decode_lm_component({"flow": {"type": "iterative_process"}}, []) is None


def test_the_cache_holds_every_sequence_the_decoder_runs():
    assert F.decode_sequences(VIBEVOICE_TOPO, {"cfg_scale": 1.3}) == 2
    assert F.decode_sequences(VIBEVOICE_TOPO, {"cfg_scale": 1.0}) == 1
    assert F.decode_sequences(VIBEVOICE_TOPO, {"cfg_scale": 1.3}, {"global.guidance_scale": 1.0}) == 1
    assert F.decode_sequences(JANUS_TOPO, {"guidance_scale": 5.0}) == 2
    assert F.decode_sequences({"flow": {"type": "autoregressive_generation",
                                        "generation": {"type": "autoregressive"}}}, {}) == 1


class _Container:
    """As the plan's NBXContainer is: its topology on disk in the cache, and a `get_topology()` that
    is the deprecated alias of `get_graph()` — None here. The first form of these tests gave the
    stub a working get_topology, and the solver's reads through it passed here while every real
    plan saw None (98/98 plans unchanged, 2026-09-29)."""
    def __init__(self, root, topology, defaults):
        self.cache_path = root
        (root / "runtime").mkdir(parents=True, exist_ok=True)
        (root / "runtime" / "defaults.json").write_text(json.dumps(defaults))
        (root / "topology.json").write_text(json.dumps(topology))

    def get_topology(self):
        return None


def test_prism_prices_the_next_token_diffusion_cache_for_both_contexts(tmp_path):
    from neurobrix.core.prism.solver import PrismSolver
    solver = PrismSolver.__new__(PrismSolver)
    c = _Container(tmp_path, VIBEVOICE_TOPO, {"cfg_scale": 1.3, "max_tokens": 2048})
    lmc = solver._read_lm_config(c)
    assert lmc and lmc["component_name"] == "model.language_model" and lmc["num_layers"] == 28
    one = PrismSolver._kv_per_token_bytes(lmc, "float16")
    assert one == 28 * 12 * (128 + 128) * 2
    assert solver._kv_token_bytes(c, lmc, "float16") == 2 * one


def test_prism_prices_the_image_generation_cache_at_the_guidance_batch(tmp_path):
    from neurobrix.core.prism.solver import PrismSolver
    solver = PrismSolver.__new__(PrismSolver)
    lmc = {"num_layers": 30, "num_heads": 32, "hidden_size": 4096}
    c = _Container(tmp_path, JANUS_TOPO, {"guidance_scale": 5.0, "lm_config": lmc})
    assert solver._kv_token_bytes(c, lmc, "float16") == 2 * PrismSolver._kv_per_token_bytes(lmc, "float16")


def test_both_sessions_read_the_one_brick():
    from neurobrix.core.flow import autoregressive as CA
    from neurobrix.triton.flow import autoregressive as TA
    assert TA.session_lm_config is F.lm_config_of
    src = textwrap.dedent(inspect.getsource(CA.AutoregressiveHandler._create_session)) \
        if hasattr(CA, "AutoregressiveHandler") else inspect.getsource(CA)
    calls = {n.func.id for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "lm_config_of" in calls


def test_both_sessions_refuse_a_plan_without_the_cache():
    """The constant-sized cache both sessions fell back on (22 layers, 32 heads, float16 in the
    Triton session; the request's budget in the compiled one) ran VibeVoice unplanned; a plan
    without a cache is refused by name in both engines now."""
    import pytest
    from types import SimpleNamespace
    from neurobrix.core.module.cache.factory import StateCacheFactory
    from neurobrix.triton.flow.autoregressive import session_kv_params
    lmc = {"num_layers": 2, "num_heads": 2, "hidden_size": 16}
    with pytest.raises(RuntimeError, match="no KV cache"):
        session_kv_params(None, 0, 64)
    ctx = SimpleNamespace(plan=SimpleNamespace(kv_cache_plan=None))
    with pytest.raises(RuntimeError, match="no KV cache"):
        StateCacheFactory.create(ctx, lmc, "cuda", "float16")


def test_a_decoder_without_its_facts_is_refused_by_name():
    """No lm_config and no extracted values for the decoding component: refused, naming what is
    missing — never a dict of Nones that every `if not lm_config` gate lets through (review
    2026-09-29)."""
    import pytest
    with pytest.raises(RuntimeError, match="num_layers, num_heads, hidden_size"):
        F.lm_config_of({}, {"extracted_values": {}}, "model.language_model")


def test_serving_a_decoder_without_a_declared_window_is_refused_by_that_name(tmp_path):
    """Serve plans the full context window; VibeVoice's decoder declares none (its extracted
    values carry no max_position_embeddings). The plan said 'no strategy can fit model + KV cache'
    — a budget — where the cause is the missing window."""
    import pytest
    from neurobrix.core.prism.solver import PrismSolver
    solver = PrismSolver.__new__(PrismSolver)
    solver._serve_mode = True
    solver._serve_requested = True
    c = _Container(tmp_path, VIBEVOICE_TOPO, {"cfg_scale": 1.3, "max_tokens": 2048})
    with pytest.raises(RuntimeError, match="declares none for 'model.language_model'"):
        solver._compute_kv_cache_plan(c, "float16", 16 * 2 ** 30)
