"""A timestep embedding is not an image (2026-09-23, PixArt -MS on Apple).

PixArt-XL-2-1024-MS traced at 1024 px has a latent of 128 and a sinusoidal timestep
embedding of 256 that is split in half and swapped — `cat([emb[:, 128:], emb[:, :128]])`.
Both halves are 128, and so is the latent height, so the tracer bound the slice END of
the embedding to the HEIGHT symbol:

    aten.slice::5   (2, 256) dim 1  start 0  end {"symbol": "s4" (height), "trace": 128}

At the traced size this is invisible: s4 resolves to 128 and the slice is right. At 4096 px
s4 is 512, `emb[:, :512]` clamps to the full 256, the concatenation yields 384 instead of
256, and the first op of the transformer dies

    addmm shape mismatch: (2, 384) @ (256, 1152)

which the census received as a bare `AssertionError:` with nothing in it.

The symbol is misattributed in the container, so the engine corrects it at load time the
way `_override_misattributed_arith` already corrects the opposite mistake. The
discriminator is whether the sliced tensor carries a spatial extent at all: this one
descends from `input::timestep`, never from `input::hidden_states`.

Runnable: PYTHONPATH=src python3 -m pytest \
  tests/unit/runtime/test_a_slice_end_is_not_a_spatial_symbol_by_coincidence.py -v
"""
from __future__ import annotations

import pytest

from neurobrix.triton.promotion import promote_seq_len_scalars
from neurobrix.triton.symbols import SymbolResolver

TRACE_LATENT = 128
RUN_LATENT = 512


def _dag():
    t = lambda tid: {"type": "tensor", "tensor_id": tid}
    sc = lambda v: {"type": "scalar", "value": v}
    h = {"type": "symbol", "id": "s4", "trace": TRACE_LATENT}
    return {
        "symbolic_context": {"symbols": {
            "s4": {"name": "height", "trace_value": TRACE_LATENT,
                   "source": "input::hidden_states::dim_2"},
            "s5": {"name": "width", "trace_value": TRACE_LATENT,
                   "source": "input::hidden_states::dim_3"},
        }},
        "tensors": {
            "input::hidden_states": {"shape": [2, 4, TRACE_LATENT, TRACE_LATENT]},
            "input::timestep": {"shape": [2]},
        },
        "ops": {
            # the timestep branch — never touches hidden_states
            "aten.cat::0": {
                "op_type": "aten::cat", "input_tensor_ids": ["input::timestep"],
                "output_tensor_ids": ["aten.cat::0::out_0"],
                "input_shapes": [[2, 128], [2, 128]], "output_shapes": [[2, 256]],
                "attributes": {"args": [t("input::timestep"), sc(-1)]}},
            # `emb[:, :128]` — the end was bound to HEIGHT by coincidence
            "aten.slice::5": {
                "op_type": "aten::slice", "input_tensor_ids": ["aten.cat::0::out_0"],
                "output_tensor_ids": ["aten.slice::5::out_0"],
                "input_shapes": [[2, 256]], "output_shapes": [[2, 128]],
                "attributes": {"args": [t("aten.cat::0::out_0"), sc(1), sc(0), h]}},
            # a GENUINE spatial slice, on a tensor descended from hidden_states
            "aten.convolution::0": {
                "op_type": "aten::convolution",
                "input_tensor_ids": ["input::hidden_states"],
                "output_tensor_ids": ["aten.convolution::0::out_0"],
                "input_shapes": [[2, 4, TRACE_LATENT, TRACE_LATENT]],
                "output_shapes": [[2, 1152, TRACE_LATENT, TRACE_LATENT]],
                "attributes": {"args": [t("input::hidden_states")]}},
            "aten.slice::9": {
                "op_type": "aten::slice",
                "input_tensor_ids": ["aten.convolution::0::out_0"],
                "output_tensor_ids": ["aten.slice::9::out_0"],
                "input_shapes": [[2, 1152, TRACE_LATENT, TRACE_LATENT]],
                "output_shapes": [[2, 1152, TRACE_LATENT, TRACE_LATENT]],
                "attributes": {"args": [t("aten.convolution::0::out_0"), sc(2), sc(0),
                                        dict(h)]}},
        },
        "execution_order": ["aten.cat::0", "aten.slice::5",
                            "aten.convolution::0", "aten.slice::9"],
    }


@pytest.fixture()
def resolved():
    dag = _dag()
    promote_seq_len_scalars(dag, dag["tensors"], dag["ops"], config_constants=None)
    r = SymbolResolver(dag["symbolic_context"])
    r._bind("s4", RUN_LATENT)
    r._bind("s5", RUN_LATENT)

    def end_of(uid):
        arg = dag["ops"][uid]["attributes"]["args"][3]
        # Mirror of graph_executor._resolve_sequential_arg: a literal reaches the
        # dispatcher as {"type": "scalar"} — the container's own form — and only a
        # symbol or an arithmetic node goes to the symbol resolver.
        if isinstance(arg, dict) and arg.get("type") == "scalar":
            return arg["value"]
        return r.resolve(arg)

    return end_of


def test_the_timestep_slice_keeps_its_own_extent(resolved):
    """The half-split of a sinusoidal embedding does not grow with the image."""
    got = resolved("aten.slice::5")
    assert got == TRACE_LATENT, (
        f"the embedding half was resolved to {got} — it followed the image height, "
        "so cat() returns 384 instead of 256 and addmm::0 dies")


def test_a_genuine_spatial_slice_still_follows_the_request(resolved):
    """The correction must not literalise a slice that really is spatial."""
    assert resolved("aten.slice::9") == RUN_LATENT
