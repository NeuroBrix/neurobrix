"""The spatial promotion pass must not turn a CHANNEL literal into a spatial expression because the two numbers
coincide at the traced size.

Sana_1600M_4Kpx_BF16 is traced square: the VAE's latent is 128 x 128, and its upsampling blocks carry literal
channel counts that are multiples of 128. c76a4620 ("the misattributed symbol also hides inside the arithmetic")
made the pass rewrite aten.view::94's channel entry 1024 as mul(8, height): 8 x 128 = 1024 at the trace, 8 x 96 =
768 at a 3072 x 4096 request. The compiled engine then died in the VAE's op-tiled residual chain, "The size of
tensor a (1536) must match the size of tensor b (2048) at non-singleton dimension 2" (aten.add::76) — measured on
Metal 2026-09-27 and bisected to that commit; the rack found the same cell red on main after the merge.

Checked on the real container, which is the only place the coincidence exists: skipped where it is not cached.
Red on c76a4620 (the channel entry of view::94 becomes an expression over a spatial symbol), green on its revert.
"""
import copy
import json
import pathlib

import pytest

CONTAINER = pathlib.Path.home() / ".neurobrix" / "cache" / "Sana_1600M_4Kpx_BF16"
VAE = CONTAINER / "components" / "vae" / "graph.json"


@pytest.mark.skipif(not VAE.exists(), reason="Sana_1600M_4Kpx_BF16 is not in the local cache")
def test_the_vae_channel_counts_stay_literal():
    from neurobrix.triton.promotion import _spatial_promotion_pass
    dag = json.loads(VAE.read_text())
    ops = copy.deepcopy(dag["ops"])
    symbols = (dag.get("symbolic_context") or {}).get("symbols") or {}
    before = copy.deepcopy(ops["aten.view::94"]["attributes"]["args"][1])
    _spatial_promotion_pass(dag, dag["tensors"], ops, symbols, set(), set())
    after = ops["aten.view::94"]["attributes"]["args"][1]
    val_b = before.get("value", before) if isinstance(before, dict) else before
    val_a = after.get("value", after) if isinstance(after, dict) else after
    assert val_b[1] == 1024, f"the fixture moved: view::94 target was {val_b}"
    assert val_a[1] == 1024, (
        f"view::94's channel entry 1024 became {json.dumps(val_a[1])}: a channel count rewritten as a spatial "
        f"expression because 8 x 128 = 1024 at the square trace; it is 768 at a 3072 x 4096 request")
