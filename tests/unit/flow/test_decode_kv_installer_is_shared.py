"""One installer decides whether a decoder can be advanced one token at a time.

The encoder_decoder flow carried this code and the audio_llm flow carried none, so a listening
model re-ran its whole context at every token — quadratic in the generated length, and the reason
those rows decode at about one token a second. What a KV cache may hold is a property of the
GRAPH, so the analysis and the installation are shared, and both engines have the same one (R30).

The third positional mechanism is what a decoder-only LM has: the CALLER supplies the position as
a graph input, so there is nothing for the cache to offset — and an internal arange must then NOT
be intercepted, or every token would be placed twice.
"""
from __future__ import annotations

import pytest


def _executor(dag):
    class _E:
        dtype = "float16"

        def __init__(self):
            self._dag = dag
            self.registered = {}

        def register_op_uid_interceptors(self, interceptors):
            self.registered.update(interceptors)
    return _E()


def _positions_from_input(dag):
    """The same graph, but with the positions handed in rather than computed inside."""
    d = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
         for k, v in dag.items()}
    d["input_tensor_ids"] = list(dag["input_tensor_ids"]) + ["input::position_ids"]
    d["uses_absolute_position"] = True
    return d


def test_the_plan_reports_the_third_positional_mechanism(dag_no_positions):
    from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan
    plain = decoder_self_attention_plan(dag_no_positions)
    assert plain is not None
    assert not plain["arange_uids"] and not plain["position_slice_uids"]
    assert plain["uses_absolute_position"] is False

    given = decoder_self_attention_plan(_positions_from_input(dag_no_positions))
    assert given["uses_absolute_position"] is True


@pytest.mark.parametrize("engine", ["triton", "compiled"])
def test_a_decoder_with_no_positional_mechanism_is_refused_by_both(engine, dag_no_positions, capsys):
    install = _installer(engine)
    assert install(_executor(dag_no_positions), max_tokens=16, label="dec") is None
    assert "KV cache REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize("engine", ["triton", "compiled"])
def test_positions_from_an_input_are_accepted_and_the_arange_is_left_alone(engine, dag, capsys):
    """`dag` has an arange. With the caller supplying positions, the cache must not touch it."""
    install = _installer(engine)
    ex = _executor(_positions_from_input(dag))
    wrapper = install(ex, max_tokens=16, label="dec")
    assert wrapper is not None, "a decoder whose caller places the token was refused"
    assert "aten.arange::0" not in ex.registered, "the arange was intercepted as well as the input"
    assert ex.registered, "no self-attention was intercepted"


def _installer(engine):
    if engine == "triton":
        from neurobrix.triton.decode_kv import install_self_attention_kv
    else:
        from neurobrix.core.runtime.graph.kv_cache_wrapper import install_self_attention_kv
    return install_self_attention_kv
