"""A patch-embedded transformer's token count must follow the request (2026-09-23, Apple).

`_spatial_promotion_pass` was written for VAE decoders, whose graphs reference H, W and
H*W and their UPSCALED multiples. A patch-embedded diffusion transformer divides instead:
its token grid is (H/p)*(W/p). PixArt-XL-1024 traced at 1024 (latent 128) carries 84 pairs
of

    aten.view::21  [2, 4096, 1152] -> target [8192, 1152]        (8192 = 2 * 4096)
    aten.view::22  [8192, 1152]    -> target [2, 4096, 1152]     (4096 = 64 * 64)

left ALL-LITERAL, because neither 64 nor 4096 nor 8192 matches H, W or H*W. Nothing
refuses them: `metadata_ops._reshape` invents a numel-preserving shape, so at 2048 the
hidden state became (8, 4096, 1152) and at 4096 (32, 4096, 1152) — the tile count folded
into the batch — and the run died 11 ops later on
`Cannot broadcast (2, 1, 1152) and (32, 4096, 1152)`.

Measured on the census shadow: 1024 (= the traced size) clean, 2048 and 4096 broken.
1536 came out CORRECT because its ratio 2.25 is not an integer and the invention could not
scale the batch — a clean integer ratio is the dangerous one.

The collision this must not cause: the T5 stream is (2, 120, 4096), so 4096 is ALSO the
text feature width. Promoting it there would break the cross-attention while still looking
bit-perfect at the traced size, where trace == runtime makes every wrong promotion invisible.

Runnable: PYTHONPATH=src python3 -m pytest \
  tests/unit/runtime/test_a_patchified_token_count_is_not_left_literal.py -v
"""
from __future__ import annotations

import math

import pytest

from neurobrix.triton.promotion import promote_seq_len_scalars
from neurobrix.triton.symbols import SymbolResolver

TRACE_LATENT = 128          # 1024 px / vae scale 8
TRACE_TOKENS = 64 * 64      # patch 2 -> 4096
RUN_LATENT = 512            # a 4096 px request
RUN_TOKENS = 256 * 256      # 65536
BATCH = 2                   # CFG


def _h_expr(sid, trace):
    """((s - 2) // 2 + 1) — exactly the form the container carries at view::0."""
    return {"type": "add", "right": 1, "trace": trace // 2,
            "left": {"type": "floordiv", "right": 2, "trace": trace // 2 - 1,
                     "left": {"type": "add", "right": -2, "trace": trace - 2,
                              "left": {"type": "symbol", "id": sid, "trace": trace}}}}


def _dag():
    tok = {"type": "mul", "trace": TRACE_TOKENS,
           "left": _h_expr("s4", TRACE_LATENT), "right": _h_expr("s5", TRACE_LATENT)}
    t = lambda tid: {"type": "tensor", "tensor_id": tid}
    lst = lambda v: {"type": "list", "value": v}
    return {
        "symbolic_context": {"symbols": {
            "s3": {"name": "batch", "trace_value": BATCH,
                   "source": "input::hidden_states::dim_0"},
            "s4": {"name": "height", "trace_value": TRACE_LATENT,
                   "source": "input::hidden_states::dim_2"},
            "s5": {"name": "width", "trace_value": TRACE_LATENT,
                   "source": "input::hidden_states::dim_3"},
        }},
        "tensors": {
            "input::hidden_states": {"shape": [BATCH, 4, TRACE_LATENT, TRACE_LATENT]},
            "input::encoder_hidden_states": {"shape": [BATCH, 120, 4096]},
            "param::proj": {"shape": [1152, 4, 2, 2], "weight_name": "proj"},
        },
        "ops": {
            # the correctly-symbolised sibling the container already carries
            "aten.view::0": {"op_type": "aten::view",
                             "input_shapes": [[BATCH, 1152, TRACE_LATENT // 2, TRACE_LATENT // 2]],
                             "attributes": {"args": [t("aten.convolution::0::out_0"),
                                                     lst([{"type": "symbol", "id": "s3",
                                                           "trace": BATCH}, 1152, tok])]}},
            # flatten / unflatten around the attention out-projection — the defect
            "aten.view::21": {"op_type": "aten::view",
                              "input_shapes": [[BATCH, TRACE_TOKENS, 1152]],
                              "attributes": {"args": [t("aten.view::20::out_0"),
                                                      lst([BATCH * TRACE_TOKENS, 1152])]}},
            "aten.view::22": {"op_type": "aten::view",
                              "input_shapes": [[BATCH * TRACE_TOKENS, 1152]],
                              "attributes": {"args": [t("aten.addmm::12::out_0"),
                                                      lst([BATCH, TRACE_TOKENS, 1152])]}},
            # the unpatchify: already carries a -1, so the two 64s must promote themselves
            "aten.view::880": {"op_type": "aten::view",
                               "input_shapes": [[BATCH, TRACE_TOKENS, 32]],
                               "attributes": {"args": [t("aten.view::879::out_0"),
                                                       lst([-1, 64, 64, 2, 2, 8])]}},
            # the T5 stream: 4096 here is a FEATURE width and must stay literal
            "aten.view::5": {"op_type": "aten::view",
                             "input_shapes": [[BATCH, 120, 4096]],
                             "attributes": {"args": [t("input::encoder_hidden_states"),
                                                     lst([-1, 4096])]}},
        },
        "execution_order": ["aten.view::0", "aten.view::21", "aten.view::22",
                            "aten.view::880", "aten.view::5"],
    }


def _resolved_target(dag, uid, resolver):
    arg = dag["ops"][uid]["attributes"]["args"][1]
    return [resolver.resolve(v) if isinstance(v, dict) else v for v in arg["value"]]


def _applied(target, in_shape):
    """The shape a view actually produces, -1 inferred exactly as NBXTensor.view does."""
    numel = math.prod(in_shape)
    known, neg = 1, None
    for i, d in enumerate(target):
        if d == -1:
            assert neg is None, f"two -1 in {target}"
            neg = i
        else:
            known *= d
    out = list(target)
    if neg is not None:
        out[neg] = numel // known
    assert math.prod(out) == numel, (
        f"{target} against an input of {numel} elements gives {out} "
        f"({math.prod(out)} elements) — the view silently changed the element count")
    return tuple(out)


@pytest.fixture()
def promoted():
    dag = _dag()
    promote_seq_len_scalars(dag, dag["tensors"], dag["ops"], config_constants=None)
    r = SymbolResolver(dag["symbolic_context"])
    r._bind("s3", BATCH)
    r._bind("s4", RUN_LATENT)
    r._bind("s5", RUN_LATENT)
    return dag, r


def test_the_flatten_follows_the_request(promoted):
    dag, r = promoted
    got = _applied(_resolved_target(dag, "aten.view::21", r), (BATCH, RUN_TOKENS, 1152))
    assert got == (BATCH * RUN_TOKENS, 1152), got


def test_the_unflatten_keeps_the_batch_and_follows_the_request(promoted):
    dag, r = promoted
    got = _applied(_resolved_target(dag, "aten.view::22", r), (BATCH * RUN_TOKENS, 1152))
    assert got == (BATCH, RUN_TOKENS, 1152), (
        f"{got} — the token count stayed at its trace value and the overflow "
        "was absorbed by the batch, which is the 2026-09-23 defect")


def test_the_unpatchify_grid_follows_the_request(promoted):
    dag, r = promoted
    got = _applied(_resolved_target(dag, "aten.view::880", r), (BATCH, RUN_TOKENS, 32))
    assert got == (BATCH, RUN_LATENT // 2, RUN_LATENT // 2, 2, 2, 8), got


def test_the_text_feature_width_is_not_promoted(promoted):
    """4096 is the T5 hidden size here; a value-only rule would destroy cross-attention."""
    dag, r = promoted
    target = _resolved_target(dag, "aten.view::5", r)
    assert target == [-1, 4096], target
    assert _applied(target, (BATCH, 120, 4096)) == (BATCH * 120, 4096)
