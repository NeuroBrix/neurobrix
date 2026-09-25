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


def _executor_with(graph_params, tmp_path, index_keys, consumed):
    import json
    ex = GraphExecutor.__new__(GraphExecutor)
    tensors = {f"param::{n}": {"is_parameter": True, "weight_name": n} for n in graph_params}
    ops = {"op0": {"input_tensor_ids": [f"param::{n}" for n in consumed]}}
    ex._dag = {"tensors": tensors, "ops": ops, "execution_order": ["op0"]}
    comp = tmp_path / "components" / "c"; comp.mkdir(parents=True)
    (comp / "weights_index.json").write_text(json.dumps({"tensors": {k: {} for k in index_keys}}))
    return ex


def test_the_post_load_reconcile_applies_the_pre_load_binding(tmp_path):
    # Review, 2026-09-13: recomputed over the FILTERED dict, pass 0's coverage
    # test fails (the unconsumed block params are gone on purpose) and the
    # reconcile falls to the suffix heuristics. The binding computed over the
    # whole index before the load is what the reconcile applies.
    consumed = {"encoder.block.0.attn.key.weight"}
    graph_params = consumed | {"encoder.block.1.attn.key.weight", "encoder.token_embed.weight"}
    index = ["token_embed.weight", "block.0.attn.key.weight", "block.1.attn.key.weight"]
    ex = _executor_with(graph_params, tmp_path, index, consumed)
    wanted = ex._consumed_in_loader_space(ex.consumed_weight_names(), tmp_path, "c")
    assert wanted == {"token_embed.weight", "block.0.attn.key.weight"}, wanted
    ex._weights = {k: object() for k in wanted}          # what the loader returns
    loaded = dict(ex._weights)
    ex._reconcile_weight_keys()
    assert set(ex._weights) == {"encoder.token_embed.weight", "encoder.block.0.attn.key.weight"}, set(ex._weights)
    assert ex._weights["encoder.block.0.attn.key.weight"] is loaded["block.0.attn.key.weight"]
    assert getattr(ex, "_pending_weight_binding", None) is None, "applied once"


def test_an_unreadable_index_loads_everything(tmp_path):
    # The graph-space set handed to a loader that filters index keys by exact
    # membership is the Wan2.2 failure; without an index the answer is None.
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._dag = {"tensors": {}, "ops": {}, "execution_order": []}
    assert ex._consumed_in_loader_space({"a.weight"}, tmp_path, "absent") is None


# ─────────── 2026-09-25: a layer_streaming piece is not a reader for the flow ───────────
#
# The flow reads its component's BASE executor by name; under `layer_streaming` the base holds
# every non-block weight resident (`load_flow_read_weights`) and a piece loads only what its own
# ops consume. Before, every piece loaded every non-block key with every run — granite-speech's
# 8 pieces loaded 48-53 weights each for ~46 they read, the embedding among them, in no plan's
# budget — while the base the flow reads held none (the Mac's 30 "requires embed_tokens weight").

def test_a_piece_loads_only_what_its_ops_consume():
    # A piece holding block 0 and the final norm; the embedding and the head are elsewhere.
    consumed = {"block.0.attn.key.weight", "norm.weight"}
    index = ["token_embed.weight", "norm.weight", "lm_head.weight",
             "block.0.attn.key.weight", "block.1.attn.key.weight"]
    graph_params = consumed
    wanted = filt(consumed, index, graph_params, flow_reads=False)
    assert wanted == {"block.0.attn.key.weight", "norm.weight"}, wanted


def test_a_whole_executor_still_serves_the_flow():
    consumed = {"block.0.attn.key.weight"}
    index = ["token_embed.weight", "norm.weight", "block.0.attn.key.weight"]
    assert filt(consumed, index, consumed) == {"token_embed.weight", "norm.weight",
                                                "block.0.attn.key.weight"}


def test_the_binding_follows_the_same_rule():
    consumed = {"block.0.attn.key.weight"}
    index = ["token_embed.weight", "block.0.attn.key.weight"]
    loaded = GraphExecutor.binding_of_the_loaded(consumed, index, consumed, flow_reads=False)
    assert set(loaded.values()) == {"block.0.attn.key.weight"}, loaded


def test_an_encoded_weight_is_held_and_borrowed_as_the_tensor_it_was_assembled_into():
    """An int4 build stores `head.weight` as `head.qweight`/`.scales`/`.qmins`, and the loader
    assembles them into ONE tensor under `head.weight` (`assemble_quantized`). A base that holds
    it must count the triplet as held (else every call reloads it, and the joined arenas grow),
    and a piece that consumes it must take the assembled tensor, not load a second copy — the
    guardian's review of 2026-09-25."""
    assembled = object()
    lender = {"head.weight": assembled, "norm.weight": object()}
    for leaf in ("qweight", "scales", "qmins"):
        assert GraphExecutor._held_as(f"head.{leaf}", lender) == "head.weight"
    assert GraphExecutor._held_as("block.0.w.qweight", lender) is None
    piece = GraphExecutor.__new__(GraphExecutor)
    piece._borrow_from = type("Base", (), {"_weights": lender})()
    keep, taken = piece._borrow({"head.qweight", "head.scales", "head.qmins",
                                 "norm.weight", "block.0.w.weight"})
    assert keep == {"block.0.w.weight"}, keep
    assert taken == {"head.weight": assembled, "norm.weight": lender["norm.weight"]}, taken
