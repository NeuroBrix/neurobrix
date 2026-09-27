"""A head dimension is fixed by the weights, not by the request (2026-09-23, Sana).

Sana_1600M_1024px_MultiLing has 70 attention heads of 32, and at its traced 1024 px the
latent is 32x32 — so the head dim and the latent height are both 32. The tracer bound the
head dim to the HEIGHT symbol:

    aten.view::12  (2, 2240, 1024) -> [s_batch, 70, s6(height), tokens]

At the traced size s6 resolves to 32 and the split is right. At 4096 px s6 is 128, the
head dim becomes 128, the element count no longer matches, and `metadata_ops._reshape`
invents a shape — the run dies at the first attention op with

    bmm shape mismatch: (140, 33, 16384) @ (35, 16384, 128)

where 35 is an invented batch and 128 is the image height standing in for the head dim.

The discriminator is structural, not a value: 70 * 32 = 2240 is the extent of the mm
weight that produced the tensor. A group of target entries that reconstructs a WEIGHT
extent cannot contain a request-dependent symbol — the weights do not change shape with
the request. The same view's token entry, which reconstructs a request-dependent input
dim, must stay symbolic.

Runnable: PYTHONPATH=src python3 -m pytest \
  tests/unit/runtime/test_a_head_dimension_is_not_the_image_height.py -v
"""
from __future__ import annotations

import pytest

from neurobrix.triton.promotion import promote_seq_len_scalars
from neurobrix.triton.symbols import SymbolResolver

HEADS, HEAD_DIM = 70, 32
HIDDEN = HEADS * HEAD_DIM          # 2240 — a weight extent
TRACE_LATENT = 32                  # 1024 px / vae scale 32
RUN_LATENT = 128                   # a 4096 px request
TRACE_TOKENS = TRACE_LATENT * TRACE_LATENT
RUN_TOKENS = RUN_LATENT * RUN_LATENT


def _tok(trace):
    return {"type": "mul", "trace": trace,
            "left": {"type": "symbol", "id": "s6", "trace": TRACE_LATENT},
            "right": {"type": "symbol", "id": "s7", "trace": TRACE_LATENT}}


def _dag():
    t = lambda tid: {"type": "tensor", "tensor_id": tid}
    lst = lambda v: {"type": "list", "value": v}
    batch = {"type": "symbol", "id": "s5", "trace": 2}
    height = {"type": "symbol", "id": "s6", "trace": TRACE_LATENT}
    return {
        "symbolic_context": {"symbols": {
            "s5": {"name": "batch", "trace_value": 2,
                   "source": "input::hidden_states::dim_0"},
            "s6": {"name": "height", "trace_value": TRACE_LATENT,
                   "source": "input::hidden_states::dim_2"},
            "s7": {"name": "width", "trace_value": TRACE_LATENT,
                   "source": "input::hidden_states::dim_3"},
        }},
        "tensors": {
            "input::hidden_states": {"shape": [2, 32, TRACE_LATENT, TRACE_LATENT]},
            "param::qkv": {"shape": [HIDDEN, HIDDEN], "weight_name": "qkv"},
        },
        "ops": {
            "aten.mm::2": {
                "op_type": "aten::mm",
                "input_tensor_ids": ["input::hidden_states", "param::qkv"],
                "output_tensor_ids": ["aten.mm::2::out_0"],
                "input_shapes": [[2 * TRACE_TOKENS, HIDDEN], [HIDDEN, HIDDEN]],
                "output_shapes": [[2 * TRACE_TOKENS, HIDDEN]],
                "attributes": {"args": [t("input::hidden_states"), t("param::qkv")]}},
            # the split into heads: entry 2 is the HEAD DIM, bound to height by coincidence
            "aten.view::12": {
                "op_type": "aten::view",
                "input_tensor_ids": ["aten.mm::2::out_0"],
                "output_tensor_ids": ["aten.view::12::out_0"],
                "input_shapes": [[2, HIDDEN, TRACE_TOKENS]],
                "output_shapes": [[2, HEADS, HEAD_DIM, TRACE_TOKENS]],
                "attributes": {"args": [t("aten.mm::2::out_0"),
                                        lst([batch, HEADS, height, _tok(TRACE_TOKENS)])]}},
        },
        "execution_order": ["aten.mm::2", "aten.view::12"],
    }


@pytest.fixture()
def resolved():
    dag = _dag()
    promote_seq_len_scalars(dag, dag["tensors"], dag["ops"], config_constants=None)
    r = SymbolResolver(dag["symbolic_context"])
    r._bind("s5", 2)
    r._bind("s6", RUN_LATENT)
    r._bind("s7", RUN_LATENT)

    def target():
        out = []
        for a in dag["ops"]["aten.view::12"]["attributes"]["args"][1]["value"]:
            if isinstance(a, dict) and a.get("type") == "scalar":
                out.append(a["value"])
            elif isinstance(a, dict):
                out.append(r.resolve(a))
            else:
                out.append(a)
        return out

    return target


def test_the_head_dimension_stays_fixed(resolved):
    got = resolved()
    assert got[2] == HEAD_DIM, (
        f"the head dim resolved to {got[2]} — it followed the image height, so the "
        f"reshape no longer matches the element count and a shape gets invented")


def test_the_token_dimension_still_follows_the_request(resolved):
    """The correction must not freeze the entry that really is request-dependent."""
    assert resolved()[3] == RUN_TOKENS


def test_the_whole_view_keeps_the_element_count(resolved):
    got = resolved()
    assert got[0] * got[1] * got[2] * got[3] == 2 * HIDDEN * RUN_TOKENS, got


# --------------------------------------------------------------------------
# The same coincidence at a position that maps 1:1 to an input dim, where the
# alignment gives no split to reason about. Sana carries 124 of these. The
# discriminator there is the PARTNER: a genuine spatial use names height and
# width as bare entries together (the VAE keeps them as two adjacent dims),
# while a misattributed one is a lone height with no width beside it. The
# partner must be looked for at TOP LEVEL only -- Sana's token entry is
# `mul(s6, s7)`, so a nested search would find s7 inside it and keep every
# misattribution.
# --------------------------------------------------------------------------

def _lone_vs_pair_dag():
    t = lambda tid: {"type": "tensor", "tensor_id": tid}
    lst = lambda v: {"type": "list", "value": v}
    h = {"type": "symbol", "id": "s6", "trace": TRACE_LATENT}
    w = {"type": "symbol", "id": "s7", "trace": TRACE_LATENT}
    return {
        "symbolic_context": {"symbols": {
            "s5": {"name": "batch", "trace_value": 1,
                   "source": "input::sample::dim_0"},
            "s6": {"name": "height", "trace_value": TRACE_LATENT,
                   "source": "input::sample::dim_2"},
            "s7": {"name": "width", "trace_value": TRACE_LATENT,
                   "source": "input::sample::dim_3"},
        }},
        "tensors": {
            "input::sample": {"shape": [1, 32, TRACE_LATENT, TRACE_LATENT]},
            # HEAD_DIM is also a weight extent, which is the whole trap
            "param::attn": {"shape": [HEAD_DIM, HEAD_DIM], "weight_name": "attn"},
        },
        "ops": {
            # GENUINE: height and width kept as two bare adjacent entries
            "aten.view::0": {
                "op_type": "aten::view", "input_tensor_ids": ["input::sample"],
                "output_tensor_ids": ["aten.view::0::out_0"],
                "input_shapes": [[1, 32, 32, TRACE_LATENT, TRACE_LATENT]],
                "output_shapes": [[1, TRACE_TOKENS, TRACE_LATENT, TRACE_LATENT]],
                "attributes": {"args": [t("input::sample"),
                                        lst([1, _tok(TRACE_TOKENS), h, w])]}},
            # MISATTRIBUTED: a lone height standing in for the head dim
            "aten.view::6": {
                "op_type": "aten::view", "input_tensor_ids": ["input::sample"],
                "output_tensor_ids": ["aten.view::6::out_0"],
                "input_shapes": [[1, 64, TRACE_TOKENS, HEAD_DIM]],
                "output_shapes": [[64, TRACE_TOKENS, HEAD_DIM]],
                "attributes": {"args": [t("input::sample"),
                                        lst([64, _tok(TRACE_TOKENS), dict(h)])]}},
        },
        "execution_order": ["aten.view::0", "aten.view::6"],
    }


@pytest.fixture()
def lone_vs_pair():
    dag = _lone_vs_pair_dag()
    promote_seq_len_scalars(dag, dag["tensors"], dag["ops"], config_constants=None)
    r = SymbolResolver(dag["symbolic_context"])
    r._bind("s5", 1)
    r._bind("s6", RUN_LATENT)
    r._bind("s7", RUN_LATENT)

    def target(uid):
        out = []
        for a in dag["ops"][uid]["attributes"]["args"][1]["value"]:
            if isinstance(a, dict) and a.get("type") == "scalar":
                out.append(a["value"])
            elif isinstance(a, dict):
                out.append(r.resolve(a))
            else:
                out.append(a)
        return out

    return target


def test_a_lone_height_at_a_weight_extent_is_literalised(lone_vs_pair):
    got = lone_vs_pair("aten.view::6")
    assert got[2] == HEAD_DIM, (
        f"the head dim resolved to {got[2]} — a lone height symbol standing in for "
        "an extent the weights fix")


def test_a_genuine_height_width_pair_is_left_alone(lone_vs_pair):
    """The VAE really does keep H and W as two dims; freezing them breaks it."""
    got = lone_vs_pair("aten.view::0")
    assert got[2] == RUN_LATENT and got[3] == RUN_LATENT, got
