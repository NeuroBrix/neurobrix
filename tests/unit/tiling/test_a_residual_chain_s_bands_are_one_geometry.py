"""A band-streamed residual chain's bands come from ONE function (`tiling_sizes.residual_chain_bands`, both twins): the
chain runs them, the census records the conv keys they form. Sana 4K's VAE at 2048x1408 on 32 GB met no certified key
for a conv on a 418-row band (2026-10-17): the census derived its convs at the full extent only.

What it would do if the code were wrong: a geometry that differs from the loop's gives other heights (red); the census
deriving a chain conv at one height gives one launch instead of one per band height (red)."""
from __future__ import annotations

import sys
from pathlib import Path

from nbxtensor.ops import tiling_sizes as TS
from nbxtensor.ops import tiling_sizes as TTS

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))


def test_the_bands_are_the_loop_s_and_the_twins_agree():
    bands = TS.residual_chain_bands(1408, 4, 33)
    assert bands == [(0, 352, 0, 385), (352, 704, 319, 737), (704, 1056, 671, 1089), (1056, 1408, 1023, 1408)]
    assert sorted({b[3] - b[2] for b in bands}) == [385, 418]            # the edge bands and the middle ones
    assert TTS.residual_chain_bands(1408, 4, 33) == bands
    assert TS.residual_chain_bands(10, 4, 1) == [(0, 3, 0, 4), (3, 6, 2, 7), (6, 9, 5, 10), (9, 10, 8, 10)]


def test_the_census_derives_a_chain_conv_at_every_band_height():
    import collections
    import derived_census as D
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    shapes = {"x": [1, 512, 1408, 1216], "w": [512, 512, 3, 3]}
    op = {"op_type": "aten::convolution", "input_tensor_ids": ["x", "w"], "output_tensor_ids": ["y"],
          "attributes": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}}
    dt = lambda t: NBXDtype.float32
    contract = type("C", (), {"fp32_op_uids": frozenset(), "narrow_op_uids": frozenset(), "safe": False})()
    kw = dict(kind="aten::convolution", uid="conv::49", o=op, ins=["x", "w"], shape=lambda t: shapes[t], dt=dt, LK=LK,
              contract=contract, cdtype="float32", has_native_bf16=False, sdpa_budget_bytes=0, sdpa_min_rows=1,
              sdpa_max_chunks=1, conv_band_bytes=TS.conv2d_band_bytes(), unhandled=collections.Counter())
    whole = D._op_launches(**kw, tile_factor=None)
    chain = D._op_launches(**kw, tile_factor=("chain", 4, 33))
    heights = sorted({int(k[2]) for _, k in chain})
    assert heights == [385, 418], heights
    assert len(chain) > len(whole)
    assert all(int(k[2]) == 1408 for _, k in whole)
