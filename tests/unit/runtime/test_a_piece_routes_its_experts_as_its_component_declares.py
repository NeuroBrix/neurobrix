"""A streamed piece routes its experts under the MoE declaration its COMPONENT's flow made.

Measured 2026-10-12 on the M4 Pro, DeepSeek-Coder-V2-Lite-Instruct (`lm_config.norm_topk_prob`
false), greedy, layer_streaming, against the vendor's own greedy ids (429 9539 6266 5501 2320):
native gave 429 9539 6 at step 2 — the vendor's third choice there (p 1.5e-5), +11.1 behind.
An env-gated probe printed every piece's fused ops at compile: `norm_topk_prob` True in all of
them, and no declaration on any piece. An `llm` graph fuses its experts when it loads, before any
flow declares, at the default True; the pieces are cut from that graph; the flow then declares
False on the component's executor (`set_moe_config`), whose `run` the strategy has replaced and
which compiles nothing. Every piece renormalised the top-6. Forced False on the pieces, native
gave the vendor's 429 9539 6266 5501 and then 3266, the vendor's fp32 choice at its step-4
near-tie (results/deepseek_v2_native_probe_2026_10_12).

A piece reads its component's state from the base as it is now (`_component_from`), at every
load; the declaration is read the same way, at every run, so a sliced stretch's passes — built
at their first run — take it too.

The routing is judged only by the vendor's greedy ids (a gate cell judges rc and certified misses,
so a wrong renormalisation passes it). Both catalogue DeepSeek MoE containers declare
`norm_topk_prob: false`, 6 experts per token (vendor config.json); their vendor references, bf16
on CPU, 32 tokens, recorded by the Dell on 2026-10-12 (_agents/archives/dell.vendor_greedy_2026_10_12_*.json):
  DeepSeek-Coder-V2-Lite-Instruct  429 9539 6266 5501 2320 254 1492 1373 ...
  deepseek-moe-16b-chat            429 9539 6266 5501 2320 254 1492 3266 ...

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_piece_routes_its_experts_as_its_component_declares.py
"""
import pytest

from neurobrix.core.runtime.graph_executor import GraphExecutor


class _Ran(Exception):
    pass


def _fused_dag(norm):
    return {"ops": {"moe_fused::block.1": {"op_type": "custom::moe_fused",
                                          "attributes": {"norm_topk_prob": norm}}},
            "execution_order": ["moe_fused::block.1"], "tensors": {}}


def _executor(mode, dag, **attrs):
    ex = GraphExecutor.__new__(GraphExecutor)
    ex.mode = mode
    ex.family = "llm"
    ex._dag = dag
    ex._persistent_pre_op_callback = None
    ex._post_run_hook = None
    for k, v in attrs.items():
        setattr(ex, k, v)
    return ex


def _norm_seen_at_run(ex, monkeypatch):
    seen = {}

    def stop(*a, **k):
        seen["norm"] = ex._dag["ops"]["moe_fused::block.1"]["attributes"]["norm_topk_prob"]
        raise _Ran()
    monkeypatch.setattr(ex, "_run_triton", stop, raising=False)
    monkeypatch.setattr(ex, "_prepare_execution", stop, raising=False)
    with pytest.raises(_Ran):
        ex.run({})
    return seen["norm"]


@pytest.mark.parametrize("mode", ["triton", "compiled"])
def test_a_piece_runs_the_declaration_its_component_received_after_the_cut(mode, monkeypatch):
    base = _executor(mode, _fused_dag(True))
    piece = _executor(mode, _fused_dag(True), _component_from=base)
    base.set_moe_config(norm_topk_prob=False)        # the flow declares after the pieces exist
    assert _norm_seen_at_run(piece, monkeypatch) is False


def test_a_piece_whose_component_declared_nothing_keeps_its_graph(monkeypatch):
    base = _executor("triton", _fused_dag(True))
    piece = _executor("triton", _fused_dag(True), _component_from=base)
    assert _norm_seen_at_run(piece, monkeypatch) is True

