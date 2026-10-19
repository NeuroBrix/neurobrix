"""Every decoding flow decodes on the KV cache through ONE opener per engine (state engine step 3).

`core/flow/decoder_cache.open_decoder_cache` and `triton/flow/decoder_cache.open_decoder_cache` read the
decoder's self-attentions and positional mechanism from its graph (`core/flow/decoder_kv`), size the cache
from the plan and register the interceptors by op id. `DecodeSteps` is the loop protocol: the whole context
at the first run (the prefill), the one new token afterwards. The VLM (three contracts), audio-LLM, TTS-LLM
and dual-AR flows re-forwarded their whole context at every token until 2026-10-18.

What this file would do if the code were wrong: a loop that re-forwards (no `open_decoder_cache`, no
`before_run`/`after_run`, no `feed`) is red at the source door; a step protocol that hands the whole
context at a decode step, or forgets the position offset, is red at the protocol test; an opener that
offsets an arange of a decoder whose positions enter as an input, or refuses a cumsum-positioned decoder,
is red at the plan tests; a torch import in the Triton opener is red at R33.
"""
from __future__ import annotations

import ast
import inspect
import types

import pytest

from neurobrix.core.flow import decoder_kv as DK
from tests.unit.flow.test_decoder_kv_plan import _dag

SDPA = "aten._scaled_dot_product_efficient_attention"   # the op id prefix of the test DAG


# ── the source door: every decode loop asks the opener and walks the protocol ─────────────────────────
def _loops():
    from neurobrix.core.flow import audio_llm as CA, dual_ar as CDA, tts_llm as CT, vlm as CV
    from neurobrix.triton.flow import audio_llm as TA, dual_ar as TDA, tts_llm as TT, vlm as TV
    for mod, methods in ((CV, ("execute", "_execute_staged_splice", "_execute_mrope_masked_splice")),
                         (TV, ("execute", "_execute_staged_splice", "_execute_mrope_masked_splice"))):
        cls = next(v for v in vars(mod).values() if inspect.isclass(v) and all(hasattr(v, m) for m in methods))
        for m in methods:
            yield f"{mod.__name__}.{cls.__name__}.{m}", inspect.getsource(getattr(cls, m))
    for mod in (CA, TA, CT, TT, CDA, TDA):
        yield mod.__name__, inspect.getsource(mod)


def test_every_decode_loop_opens_the_one_cache_and_walks_its_steps():
    for name, src in _loops():
        for needle in ("open_decoder_cache(", "DecodeSteps(", ".before_run()", ".after_run()", ".feed("):
            assert needle in src, f"{name} decodes without {needle}: it re-forwards its whole context"


def test_the_triton_opener_imports_no_torch():
    from neurobrix.triton.flow import decoder_cache as TD
    tree = ast.parse(inspect.getsource(TD))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(n == "torch" or n.startswith("torch.") for n in names), names


# ── the step protocol ─────────────────────────────────────────────────────────────────────────────────
class _Wrapper:
    def __init__(self):
        self.calls = []
    def set_decode_mode(self, actual_seq_len=None):
        self.calls.append(("decode_mode", actual_seq_len))
    def update_position_offset(self):
        self.calls.append(("offset",))
    def use_branch(self, st):
        self.calls.append(("branch", st))


@pytest.mark.parametrize("engine", ["core", "triton"])
def test_the_protocol_prefills_once_then_hands_one_token_at_its_offset(engine):
    """The first decode step is placed at the cache's length like every later one (openaudio Triton, 2026-10-19:
    the first form moved the offset from the second step on and the first cached token sat at position 0)."""
    mod = __import__(f"neurobrix.{engine}.flow.decoder_cache", fromlist=["DecodeSteps"])
    w = _Wrapper()
    steps = mod.DecodeSteps(w)
    assert steps.cached and not steps.one_token
    assert steps.feed("whole", "last") == "whole"          # the prefill sees the context
    steps.before_run(); steps.after_run()
    assert w.calls[-1][0] == "decode_mode"                 # decode mode right after the prefill
    assert steps.one_token and steps.feed("whole", "last") == "last"
    steps.before_run()                                     # EVERY decode step, the first included: the offset moves first
    assert w.calls[-1] == ("offset",)
    steps.after_run()
    steps.before_run(); steps.after_run()
    assert [c[0] for c in w.calls] == ["decode_mode", "offset", "offset"]


@pytest.mark.parametrize("engine", ["core", "triton"])
def test_without_a_cache_every_run_sees_the_whole_context(engine):
    mod = __import__(f"neurobrix.{engine}.flow.decoder_cache", fromlist=["DecodeSteps"])
    steps = mod.DecodeSteps(None)
    for _ in range(3):
        assert steps.feed("whole", "last") == "whole"
        steps.before_run(); steps.after_run()
    assert not steps.cached and not steps.one_token


@pytest.mark.parametrize("engine", ["core", "triton"])
def test_a_branch_is_selected_before_each_of_its_runs(engine):
    mod = __import__(f"neurobrix.{engine}.flow.decoder_cache", fromlist=["DecodeSteps"])
    w = _Wrapper()
    cond, uncond = mod.DecodeSteps(w, "cond"), mod.DecodeSteps(w, "uncond")
    for st in (cond, uncond, cond, uncond, cond):
        st.before_run(); st.after_run()
    assert [c[1] for c in w.calls if c[0] == "branch"] == ["cond", "uncond", "cond", "uncond", "cond"]
    assert uncond.one_token and cond.runs == 3


# ── the plan: the four positional forms ───────────────────────────────────────────────────────────────
def _decoder_dag(positions):
    d = _dag()
    d["input_tensor_ids"] = ["input::input_ids"]
    for uid in ("aten.mm::4", "aten.mm::5"):                 # the second attention is a self-attention too
        d["ops"][uid]["input_tensor_ids"][0] = "h1"
    del d["ops"]["aten.arange::0"]
    if positions == "input":
        d["input_tensor_ids"].append("input::position_ids")
    elif positions == "arange":
        d["ops"]["aten.arange::0"] = {"op_type": "aten::arange", "input_tensor_ids": [], "output_tensor_ids": ["pos"]}
    elif positions == "cumsum":
        d["ops"]["aten.ones_like::0"] = {"op_type": "aten::ones_like", "input_tensor_ids": ["h0"], "output_tensor_ids": ["ones"]}
        d["ops"]["aten.cumsum::0"] = {"op_type": "aten::cumsum", "input_tensor_ids": ["ones"], "output_tensor_ids": ["count"]}
        d["ops"]["aten.sub::0"] = {"op_type": "aten::sub", "input_tensor_ids": ["count"], "output_tensor_ids": ["pos"]}
        d["ops"]["aten.index_select::0"] = {"op_type": "aten::index_select", "input_tensor_ids": ["param::freqs", "pos"],
                                            "output_tensor_ids": ["rope"]}
    return d


def test_positions_handed_by_the_caller_offset_nothing_in_the_graph():
    plan = DK.decoder_self_attention_plan(_decoder_dag("input"))
    assert plan["positions_input"] == "input::position_ids"
    assert plan["arange_uids"] == [] and plan["position_cumsum_uids"] == []
    assert DK.positions_are_offset_by(plan)


def test_positions_counted_by_a_cumsum_into_a_table_are_offset_at_the_count():
    plan = DK.decoder_self_attention_plan(_decoder_dag("cumsum"))
    assert plan["positions_input"] is None
    assert plan["position_cumsum_uids"] == ["aten.cumsum::0"]
    assert DK.positions_are_offset_by(plan)


def test_a_decoder_with_no_positional_mechanism_cannot_be_offset():
    plan = DK.decoder_self_attention_plan(_decoder_dag("none"))
    assert not DK.positions_are_offset_by(plan)


# ── the compiled opener registers what the plan names ────────────────────────────────────────────────
class _Executor:
    dtype = "float16"
    vendor, arch = "nvidia", "volta"
    def __init__(self, dag):
        self._dag = dag
        self.registered = {}
    def register_op_uid_interceptors(self, interceptors):
        self.registered.update(interceptors)


def _ctx(ex):
    from neurobrix.core.runtime.lm_facts import decoder_cache_facts
    f = decoder_cache_facts(ex._dag)
    plan = types.SimpleNamespace(num_layers=f["num_layers"], num_kv_heads=f["num_heads"], k_head_dim=f["head_dim"],
                                 v_head_dim=f["head_dim"], max_cache_len=448, memory_bytes=1 << 30)
    return types.SimpleNamespace(executors={"dec": ex}, plan=types.SimpleNamespace(kv_cache_plan=plan))


def test_the_opener_registers_the_self_attentions_and_the_positional_op_the_plan_names(monkeypatch):
    monkeypatch.delenv("NBX_KV_RECOMPUTE", raising=False)
    from neurobrix.core.flow.decoder_cache import open_decoder_cache
    ex = _Executor(_decoder_dag("input"))
    kv = open_decoder_cache(_ctx(ex), "dec", 64)
    assert kv is not None
    assert set(ex.registered) == {f"{SDPA}::0", f"{SDPA}::1", f"{SDPA}::2"}      # no arange, no count
    ex = _Executor(_decoder_dag("cumsum"))
    kv = open_decoder_cache(_ctx(ex), "dec", 64)
    assert ex.registered["aten.cumsum::0"] == kv.intercept_position_cumsum
    ex = _Executor(_decoder_dag("none"))
    ctx = types.SimpleNamespace(executors={"dec": ex}, plan=types.SimpleNamespace(kv_cache_plan=None))
    assert open_decoder_cache(ctx, "dec", 64) is None                            # refused, recompute path
    assert ex.registered == {}


def test_the_count_interceptor_adds_the_cache_length_at_decode_only():
    import torch
    from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper
    w = KVCacheAttentionWrapper.__new__(KVCacheAttentionWrapper)
    ones = torch.ones(1, 3, dtype=torch.long)
    w._is_prefill, w._position_offset = True, 0
    assert w.intercept_position_cumsum(ones, 1).tolist() == [[1, 2, 3]]
    w._is_prefill, w._position_offset = False, 7
    assert w.intercept_position_cumsum(torch.ones(1, 1, dtype=torch.long), 1).tolist() == [[8]]


# ── the measuring tool of the identity proof: every judged flow writes the per-step top-k record ─────────────
def test_every_staged_decode_loop_writes_the_topk_record():
    from neurobrix.core.flow import audio_llm as CA, dual_ar as CDA, tts_llm as CT
    from neurobrix.triton.flow import audio_llm as TA, dual_ar as TDA, tts_llm as TT
    for mod in (CA, TA, CT, TT, CDA, TDA):
        assert "log_topk(" in inspect.getsource(mod), f"{mod.__name__} writes no NBX_DECODE_TOPK record"


def test_the_topk_record_is_one_format_for_both_engines(tmp_path, monkeypatch):
    import json
    import numpy as np
    import torch
    from neurobrix.core.flow.decode_topk import log_topk as torch_log
    from neurobrix.triton.flow.decode_topk import log_topk as np_log
    path = tmp_path / "topk.jsonl"
    monkeypatch.setenv("NBX_DECODE_TOPK", str(path))
    logits = [0.5, 3.0, 2.9, -1.0, 7.25, 0.0]
    torch_log(torch.tensor(logits).view(1, 1, -1), 3)
    np_log(np.asarray(logits, dtype=np.float32).reshape(1, 1, -1), 3)
    a, b = [json.loads(l) for l in path.read_text().splitlines()]
    assert a["ids"] == b["ids"] == [4, 1, 2, 0] and a["margin12"] == b["margin12"] == 4.25
    assert (a["engine"], b["engine"], a["step"]) == ("torch", "triton", 3)


# ── the gate of 2026-10-18 22:56 (12 red of 98): grouped keys ──────────────────────────────────────────────────
def test_a_plan_with_the_declared_key_value_heads_fits_a_graph_whose_keys_are_expanded_to_the_query_heads():
    """GLM-4.1V: 2 key-value heads declared, the graph hands the op keys expanded to its 32 query heads; MiniCPM-o,
    Voxtral, Ming, Qwen3-Omni, openaudio alike. The plan is the declared cache; the check accepts a multiple."""
    from neurobrix.core.runtime.lm_facts import decoder_cache_facts, decoder_cache_from_plan
    d = _decoder_dag("input")
    for uid in (f"{SDPA}::0", f"{SDPA}::1", f"{SDPA}::2"):
        d["ops"][uid]["input_shapes"] = [[1, 20, 7, 64], [1, 20, 64, 7], [1, 20, 7, 64]]
        d["ops"][uid]["attributes"] = {"nbx_k_pre_transposed": True}
    facts = decoder_cache_facts(d)
    assert (facts["num_heads"], facts["kv_heads_at_op"], facts["head_dim"], facts["v_head_dim"]) == (20, 20, 64, 64)
    plan = types.SimpleNamespace(num_layers=3, num_kv_heads=4, k_head_dim=64, v_head_dim=64, max_cache_len=448)
    assert decoder_cache_from_plan(plan, facts, 64, "dec") is plan          # 4 divides 20: expanded keys
    plan.num_kv_heads = 20
    assert decoder_cache_from_plan(plan, facts, 64, "dec") is plan          # the op's own count
    plan.num_kv_heads = 3
    with pytest.raises(RuntimeError, match="priced another decoder"):
        decoder_cache_from_plan(plan, facts, 64, "dec")
    plan.num_kv_heads, plan.v_head_dim = 4, 32
    with pytest.raises(RuntimeError, match="priced another decoder"):
        decoder_cache_from_plan(plan, facts, 64, "dec")


def test_both_interceptors_accept_the_graphs_enable_gqa_kwarg():
    """chatterbox's t3 backbone traces its SDPA with enable_gqa=True; the interceptors are called with the op's kwargs."""
    from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper
    assert "enable_gqa" in inspect.signature(KVCacheAttentionWrapper.intercept_attention).parameters
    pytest.importorskip("triton")
    from neurobrix.triton.kv_cache import TritonAttentionInterceptor
    assert "enable_gqa" in inspect.signature(TritonAttentionInterceptor.intercept).parameters
