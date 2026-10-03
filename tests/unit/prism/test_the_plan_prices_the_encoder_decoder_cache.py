"""The plan prices the encoder-decoder decoder's KV cache, and both engines build it from the plan.

The encoder-decoder flow (whisper) built its decoder cache from the decoder's graph and max_tokens, and
the plan priced none (`kv_cache: null`) — VibeVoice's class, small here (whisper-large-v3-turbo: 11.8 MB).
Now `core.runtime.lm_facts.decode_lm_component` names the flow's autoregressive stage, so Prism prices
its cache with the one LM-facts reader (`lm_config_of`, the decoder's extracted values — equal to the
graph's self-attention geometry on both whispers, measured), and each flow takes the plan's cache
(`encoder_decoder_cache_from_plan`), refusing a plan without one, of another geometry than its graph's
(`decoder_cache_facts`, the flow's own reader), or shorter than the window it decodes.

Injections, each seen RED: decode_lm_component's encoder-decoder case removed -> the plan case; the
refusal on a missing cache removed -> the refusal case; a flow sizing from max_tokens again -> the
both-flows case.
"""
import ast
import inspect
from types import SimpleNamespace

import pytest

from neurobrix.core.runtime import lm_facts as F

MODEL = "whisper-large-v3-turbo"


def test_the_decoder_is_the_flows_autoregressive_stage():
    topo = {"flow": {"type": "encoder_decoder", "stages": [
        {"component": "model.encoder", "execution": "forward"},
        {"component": "model.decoder", "execution": "autoregressive", "cross_attention_from": "model.encoder"}]}}
    assert F.decode_lm_component(topo, ["model.encoder", "model.decoder"]) == "model.decoder"


def test_the_plan_prices_the_decoders_cache_at_its_graphs_geometry(monkeypatch):
    from neurobrix.core.prism.profiler import InputConfig
    from neurobrix.core.prism.solver import PrismSolver
    from neurobrix.nbx.container import NBXContainer
    from tests.unit.prism._pinned_machine import V100_16GB, container_root, impose_rung, pin_dedicated_card, profile
    pin_dedicated_card(monkeypatch, 16151, 267, "the rack's card 0, 2026-09-29")
    impose_rung(monkeypatch, 16384)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    c = NBXContainer.load(str(container_root(MODEL)))
    plan = PrismSolver().solve_smart(c, profile(V100_16GB), InputConfig(batch_size=1), mode="triton")
    kv = plan.kv_cache_plan
    assert kv is not None, "the plan priced no cache for the decoder its flow decodes with"
    dec = next(x for x in c.get_neural_components() if x.name == "model.decoder")
    facts = F.decoder_cache_facts(dec.graph)
    assert (kv.num_layers, kv.num_kv_heads, kv.k_head_dim, kv.v_head_dim) == \
        (facts["num_layers"], facts["num_heads"], facts["head_dim"], facts["head_dim"])
    assert F.encoder_decoder_cache_from_plan(kv, facts, 448, "model.decoder") is kv


def test_a_plan_without_the_cache_or_of_another_geometry_is_refused():
    facts = {"num_layers": 4, "num_heads": 20, "head_dim": 64}
    with pytest.raises(RuntimeError, match="carries no KV cache"):
        F.encoder_decoder_cache_from_plan(None, facts, 448, "model.decoder")
    other = SimpleNamespace(num_layers=4, num_kv_heads=8, k_head_dim=64, v_head_dim=64, max_cache_len=576)
    with pytest.raises(RuntimeError, match="priced another decoder"):
        F.encoder_decoder_cache_from_plan(other, facts, 448, "model.decoder")
    short = SimpleNamespace(num_layers=4, num_kv_heads=20, k_head_dim=64, v_head_dim=64, max_cache_len=100)
    with pytest.raises(RuntimeError, match="holds 100 positions"):
        F.encoder_decoder_cache_from_plan(short, facts, 448, "model.decoder")


def test_both_engines_build_the_cache_from_the_plan():
    from neurobrix.core.flow import encoder_decoder as CE
    from neurobrix.triton.flow import encoder_decoder as TE
    for mod, fn in ((CE, "_decoder_kv_wrapper"), (TE, "_decoder_kv_interceptor")):
        cls = next(v for v in vars(mod).values() if inspect.isclass(v) and hasattr(v, fn))
        src = inspect.getsource(getattr(cls, fn))
        calls = {n.func.id for n in ast.walk(ast.parse(src.replace("\n    ", "\n", 0) if False else __import__("textwrap").dedent(src)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "encoder_decoder_cache_from_plan" in calls, f"{mod.__name__}.{fn} sizes its cache itself"
        assert "max_cache_len=int(max_tokens)" not in src, f"{mod.__name__}.{fn} sizes from max_tokens"
