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


def _handler(executor):
    h = ED.EncoderDecoderHandler.__new__(ED.EncoderDecoderHandler) \
        if hasattr(ED, "EncoderDecoderHandler") else None
    if h is None:
        cls = next(c for c in vars(ED).values() if isinstance(c, type) and hasattr(c, "_decoder_kv_wrapper"))
        h = cls.__new__(cls)
    h.ctx = types.SimpleNamespace(executors={"dec": executor})
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
