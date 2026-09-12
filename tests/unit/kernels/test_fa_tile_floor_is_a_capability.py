"""The FlashAttention tile floor is a backend capability, not a constant.

whisper reached `aten._scaled_dot_product_efficient_attention::0` and was
refused:

    FlashAttention with a block tile dimension < 32 (smallest dot tile
    dimension is 16) is not supported by the GENERIC attention lowering: it
    silently mis-computes for BLOCK_M/BLOCK_N below 32, for any head_dim.

The wrapper chooses `BLOCK_M = 16` when `seqlen_q <= 16`, which is right on a
backend whose `tl.dot` floor is 16 and produces a REFUSED kernel on one whose
generic attention lowering needs 32. Sixteen wastes half a Q tile at
seqlen_q=1; a refusal wastes the model.

So the floor is a row in a capability table, like every other backend
difference in this tree: adding a backend is adding a row, not guessing. On
cuda and hip it is 16 and nothing changes -- this must stay numerically and
structurally inert there.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_fa_tile_floor_is_a_capability.py -v
"""
from __future__ import annotations

import pytest


def _table():
    from neurobrix.kernels.nbx_tensor import _BACKEND_FA_MIN_TILE
    return _BACKEND_FA_MIN_TILE


def test_every_backend_this_tree_knows_has_a_row():
    from neurobrix.kernels.nbx_tensor import _BACKEND_TRAPS_ON_DEVICE_ASSERT

    known = set(_BACKEND_TRAPS_ON_DEVICE_ASSERT)
    missing = known - set(_table())
    assert not missing, (
        f"these backends have a row in another capability table and none "
        f"here: {sorted(missing)}. A backend without a row must refuse, not "
        f"inherit someone else's floor.")


def test_the_floor_is_unchanged_where_it_was_sixteen():
    """CUDA must not move. A shared-engine change is numerically inert there
    unless it fixes a bug, and this fixes one only on Metal."""
    assert _table()["cuda"] == 16
    assert _table()["hip"] == 16


def test_metal_asks_for_thirty_two():
    assert _table()["metal"] == 32, (
        "the generic attention lowering mis-computes below 32 and says so")


def test_a_backend_without_a_row_refuses():
    """The row is the answer; its absence is a refusal, never a default."""
    from neurobrix.kernels.nbx_tensor import _backend_capability

    with pytest.raises(RuntimeError) as exc:
        _backend_capability({"cuda": 16}, "_TEST_TABLE", "its FA tile floor")
    assert "ZERO FALLBACK" in str(exc.value) or "does not say" in str(exc.value)


def test_the_wrapper_reads_the_table_rather_than_a_literal():
    """A capability nobody consults is the vacuous form.

    The floor exists to be applied; asserting the table alone would pass with
    the wrapper still writing 16.
    """
    from pathlib import Path
    import neurobrix.kernels.wrappers as W

    src = Path(W.__file__).read_text()
    assert "fa_min_tile" in src or "_BACKEND_FA_MIN_TILE" in src, (
        "the attention wrapper must consult the capability, not carry its own "
        "literal floor")
