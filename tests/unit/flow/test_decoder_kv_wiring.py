"""The encoder-decoder flow registers the KV brick on the decoder's
self-attentions and positional arange only — never on a cross-attention —
and honours the recompute oracle switch."""
import types

import pytest
import torch

from neurobrix.core.flow import encoder_decoder as ED
from tests.unit.flow.test_decoder_kv_plan import _dag


class _Executor:
    dtype = "float16"

    def __init__(self):
        self._dag = _dag()
        self.registered = {}

    def register_op_uid_interceptors(self, interceptors):
        self.registered.update(interceptors)


def _handler(executor, with_plan=True, max_cache_len=448):
    h = ED.EncoderDecoderHandler.__new__(ED.EncoderDecoderHandler) \
        if hasattr(ED, "EncoderDecoderHandler") else None
    if h is None:
        cls = next(c for c in vars(ED).values() if isinstance(c, type) and hasattr(c, "_decoder_kv_wrapper"))
        h = cls.__new__(cls)
    # The flow takes the decoder's cache from the PLAN and checks it against the graph (lm_facts.
    # encoder_decoder_cache_from_plan, 2026-09-29: Prism prices the cache of every decoding flow): the
    # stand-in context carries the plan Prism would hand it, its geometry read by the flow's own reader.
    from neurobrix.core.runtime.lm_facts import decoder_cache_facts
    facts = decoder_cache_facts(executor._dag)
    kv_plan = None if not with_plan or facts is None else types.SimpleNamespace(
        num_layers=facts["num_layers"], num_kv_heads=facts["num_heads"], k_head_dim=facts["head_dim"],
        v_head_dim=facts["head_dim"], max_cache_len=max_cache_len)
    h.ctx = types.SimpleNamespace(executors={"dec": executor}, plan=types.SimpleNamespace(kv_cache_plan=kv_plan))
    return h


def test_kv_is_registered_on_self_attentions_and_arange_only(monkeypatch):
    monkeypatch.delenv("NBX_KV_RECOMPUTE", raising=False)
    ex = _Executor()
    kv = _handler(ex)._decoder_kv_wrapper("dec", max_tokens=64)
    assert kv is not None
    assert set(ex.registered) == {"aten._scaled_dot_product_efficient_attention::0",
                                  "aten._scaled_dot_product_efficient_attention::2",
                                  "aten.arange::0"}
    assert "aten._scaled_dot_product_efficient_attention::1" not in ex.registered   # cross-attention stays native
    # a second window reuses the registered wrapper and resets it
    assert _handler(ex)._decoder_kv_wrapper("dec", max_tokens=64) is kv


def test_recompute_oracle_switch_disables_the_cache(monkeypatch):
    monkeypatch.setenv("NBX_KV_RECOMPUTE", "1")
    ex = _Executor()
    assert _handler(ex)._decoder_kv_wrapper("dec", max_tokens=64) is None
    assert ex.registered == {}


def test_a_positional_table_slice_is_registered_like_the_arange(monkeypatch):
    from tests.unit.flow.test_decoder_kv_plan import _dag_with_positional_table_slice
    monkeypatch.delenv("NBX_KV_RECOMPUTE", raising=False)
    ex = _Executor()
    ex._dag = _dag_with_positional_table_slice()
    kv = _handler(ex)._decoder_kv_wrapper("dec", max_tokens=64)
    assert kv is not None
    assert "aten.slice::0" in ex.registered and ex.registered["aten.slice::0"] == kv.intercept_position_slice
    assert "aten.slice::1" not in ex.registered


def test_a_decoder_with_no_positional_mechanism_refuses_the_cache(monkeypatch, capsys):
    """Neither an arange nor a positional-table slice: one token per step
    would sit at position 0 every step (whisper-large, 2026-09-05) — the
    cache is refused loudly and the recompute path keeps the transcript right."""
    monkeypatch.delenv("NBX_KV_RECOMPUTE", raising=False)
    ex = _Executor()
    del ex._dag["ops"]["aten.arange::0"]
    assert _handler(ex)._decoder_kv_wrapper("dec", max_tokens=64) is None
    assert ex.registered == {}
    assert "KV cache REFUSED" in capsys.readouterr().err


def test_a_plan_without_the_decoders_cache_is_refused_by_name(monkeypatch):
    """The flow never sizes the cache itself: a plan that carries none, or one shorter than the window
    it decodes, is refused naming the decoder."""
    import pytest
    monkeypatch.delenv("NBX_KV_RECOMPUTE", raising=False)
    with pytest.raises(RuntimeError, match="the plan carries no KV cache for the decoder 'dec'"):
        _handler(_Executor(), with_plan=False)._decoder_kv_wrapper("dec", max_tokens=64)
    with pytest.raises(RuntimeError, match="holds 8 positions"):
        _handler(_Executor(), max_cache_len=8)._decoder_kv_wrapper("dec", max_tokens=64)
