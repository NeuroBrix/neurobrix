"""The consumed-weight filter is applied in the loader's key space, and it
knows BOTH readers of the weight dict.

2026-09-13: Wan2.2's text encoder rendered in compiled mode and died in triton
at aten.embedding::0 with a None weight. The graph names the embedding
`encoder.token_embed.weight`; the index stores `token_embed.weight`; the
`only=` filter compared the two by exact membership BEFORE the suffix
reconciliation that would have joined them, so the weight was never loaded.

Same day, the full suite on the fixed tree: ten triton cells (four VLMs, two
audio-LLMs, a TTS backbone, two int4 builds, one warm serve) failed with
"requires embed_tokens weight" or a None weight at `aten.mm::0`. The graph is
not the only reader of the weight dict. A language model whose graph takes
`inputs_embeds` never consumes its token embedding — the FLOW handler reads it
by name to build the context and the tied logits — and an int4 build stores a
consumed `X.weight` as three keys the graph never names, each of which the
index says it `encodes`. Register entry 50.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_consumed_weights_reach_the_loader_key_space.py
"""
from neurobrix.core.runtime.graph_executor import GraphExecutor

filt = GraphExecutor.consumed_in_loader_space
bind = GraphExecutor.bind_weight_keys


def test_a_prefix_difference_does_not_drop_the_weight():
    # Wan2.2's text encoder: every graph name carries `encoder.`, no index
    # key does. The graph names two blocks and consumes one; the binding is
    # computed over EVERY graph param (as the reconcile does), which is what
    # keeps `attn.key.weight` ambiguous and block 1 bound to its own name.
    consumed = {"encoder.token_embed.weight", "encoder.block.0.attn.key.weight"}
    graph_params = consumed | {"encoder.block.1.attn.key.weight"}
    index = ["token_embed.weight", "block.0.attn.key.weight", "block.1.attn.key.weight"]
    assert filt(consumed, index, graph_params) == {"token_embed.weight", "block.0.attn.key.weight"}


def test_none_means_everything_as_before():
    assert filt(None, ["a", "b"], {"a"}) is None


def test_a_weight_the_flow_reads_outside_the_graph_is_loaded():
    # A VLM / audio-LLM language model: the graph takes inputs_embeds, so no
    # op consumes the token embedding; the flow handler reads it by name.
    # The same holds for the head a flow projects logits with. Neither is a
    # block weight, and every non-block weight is loaded, whatever the graph
    # says; an unrouted expert is a block weight no op consumes and stays out.
    consumed = {"block.0.attn.key.weight", "block.0.attn.out.weight"}
    index = ["token_embed.weight", "norm.weight", "lm_head.weight",
             "block.0.attn.key.weight", "block.0.attn.out.weight",
             "block.0.mlp.experts.7.down.weight"]
    graph_params = consumed | {"block.0.mlp.experts.3.down.weight"}
    wanted = filt(consumed, index, graph_params)
    assert {"token_embed.weight", "norm.weight", "lm_head.weight"} <= wanted, wanted
    assert "block.0.mlp.experts.7.down.weight" not in wanted, wanted


def test_an_encoded_weight_is_wanted_through_the_name_the_index_says_it_encodes():
    # int4-g128-asym: the graph names `X.weight`; the index holds three keys
    # per encoded weight, each carrying `encodes: X.weight`. The triplet of a
    # consumed name is wanted; the triplet of an unconsumed block name is not;
    # and a dense block parameter whose leaf happens to be `scales` is judged
    # by its own name, not by a leaf rule.
    consumed = {"block.0.attn.key.weight"}
    graph_params = consumed | {"block.3.attn.key.weight", "block.3.norm.scales"}
    index = ["block.0.attn.key.qweight", "block.0.attn.key.scales", "block.0.attn.key.qmins",
             "block.3.attn.key.qweight", "block.3.attn.key.scales", "block.3.attn.key.qmins",
             "block.3.norm.scales"]
    encodes = {k: k.rsplit(".", 1)[0] + ".weight" for k in index if "attn.key" in k}
    wanted = filt(consumed, index, graph_params, encodes)
    assert wanted == {"block.0.attn.key.qweight", "block.0.attn.key.scales",
                      "block.0.attn.key.qmins"}, wanted


def test_exact_names_are_the_identity_binding():
    # The common container: every index key IS a graph name. The reconcile
    # leaves the dict alone (bind returns None) and the filter must read
    # that as the identity — seen failing 2026-09-13 on the first run of
    # the regression cells: `'NoneType' object has no attribute 'get'`.
    graph_params = {"block.0.attn.key.weight", "block.1.attn.key.weight", "token_embed.weight"}
    consumed = {"block.0.attn.key.weight"}
    assert filt(consumed, sorted(graph_params), graph_params) == \
        {"block.0.attn.key.weight", "token_embed.weight"}


def test_one_loader_key_can_fill_two_graph_names():
    # Pass 0: the same loader key is the exact match of one graph name and
    # the unique prefix-strip match of another. The result is keyed by NAME
    # so neither vanishes — a key-keyed result kept one (review, 2026-09-13).
    # (With every index key already a graph name the fast path leaves the
    # dict alone — old and new — so the index carries one foreign key here.)
    graph_params = {"x.y.weight", "y.weight"}
    b = bind(graph_params, ["x.y.weight", "extra"])
    assert b == {"x.y.weight": "x.y.weight", "y.weight": "x.y.weight"}, b
    assert filt({"y.weight"}, ["x.y.weight", "extra"], graph_params) == {"x.y.weight", "extra"}


def test_the_filter_and_the_reconcile_use_one_function():
    graph_params = {"encoder.token_embed.weight", "encoder.block.0.attn.key.weight",
                    "encoder.block.1.attn.key.weight"}
    index = ["token_embed.weight", "block.0.attn.key.weight", "block.1.attn.key.weight",
             "encoder_extra.weight"]
    b = bind(graph_params, index)
    assert b == {"encoder.token_embed.weight": "token_embed.weight",
                 "encoder.block.0.attn.key.weight": "block.0.attn.key.weight",
                 "encoder.block.1.attn.key.weight": "block.1.attn.key.weight",
                 "encoder_extra.weight": "encoder_extra.weight"}, b
    assert bind({"a.weight"}, ["a.weight"]) is None, \
        "every key already a graph name: the dict is left alone"


def test_every_consumed_name_binds_after_the_filtered_load():
    # The property that matters: what the filter keeps, the reconcile over
    # that subset binds to every consumed name — on the three index shapes
    # above (prefix-different, exact, encoded). Not a proof for every shape.
    cases = [
        ({"encoder.token_embed.weight", "encoder.block.0.attn.key.weight"},
         {"encoder.block.1.attn.key.weight"},
         ["token_embed.weight", "block.0.attn.key.weight", "block.1.attn.key.weight"], {}),
        ({"block.0.attn.key.weight"}, {"block.1.attn.key.weight", "token_embed.weight"},
         ["token_embed.weight", "block.0.attn.key.weight", "block.1.attn.key.weight"], {}),
        ({"block.0.attn.key.weight"}, {"block.3.attn.key.weight"},
         ["block.0.attn.key.qweight", "block.0.attn.key.scales", "block.0.attn.key.qmins",
          "block.3.attn.key.qweight"],
         {"block.0.attn.key.qweight": "block.0.attn.key.weight",
          "block.0.attn.key.scales": "block.0.attn.key.weight",
          "block.0.attn.key.qmins": "block.0.attn.key.weight",
          "block.3.attn.key.qweight": "block.3.attn.key.weight"}),
    ]
    for consumed, others, index, encodes in cases:
        graph_params = consumed | others
        loaded = [k for k in index if k in filt(consumed, index, graph_params, encodes)]
        # after the load an encoded triplet is folded under the dense name
        folded = list(dict.fromkeys(encodes.get(k, k) for k in loaded))
        b = bind(graph_params, folded) or {k: k for k in folded}
        assert consumed <= set(b.keys()), (consumed, b)
