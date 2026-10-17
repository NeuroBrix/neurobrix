"""A vendor tiling declared `when: memory` is the vendor's memory path, not its reference: Prism takes it only when no
exact strategy keeps the component on the accelerator.

Measured 2026-10-17: Sana_1600M_4Kpx_BF16 on a 32 GB V100 planned its VAE through DC-AE's `tiled_decode` (64 px
cross-fades, `vendor_tiling.json`, when: memory) and the image carried a faint grid along the tile overlaps; before
the declaration the same card kept the VAE whole with op-level tiling — the vendor's own 4K path (`patch_conv`), exact
— and the background was smooth. On a 16 GB card no exact strategy holds it: the vendor's tiles stay the way down.

What it would do if the rule were gone: the 32 GB plan is `lazy_sequential` with the vendor's tiles — red.
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.paths import cache_dir

MODEL = "Sana_1600M_4Kpx_BF16"


def _plan(hardware):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands import run as R
    from neurobrix.core.prism import PrismSolver, load_profile
    from neurobrix.nbx import NBXContainer
    c = cache_dir() / MODEL
    if not (c / "components" / "vae" / "vendor_tiling.json").exists():
        pytest.skip(f"{MODEL}: not in this machine's cache, or no vendor tiling declared ({c})")
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "a red apple on a wooden table",
                                       "--hardware", hardware])
    manifest = json.loads((c / "manifest.json").read_text())
    ic = R.request_input_config(args, manifest, manifest.get("family"), c)
    return PrismSolver().solve_smart(NBXContainer.load(str(c)), load_profile(hardware), ic, mode="compiled")


def test_where_an_exact_cut_holds_the_vae_the_vendor_s_memory_tiles_are_not_taken():
    p = _plan("v100-32g")
    assert str(p.components["vae"].device).startswith("cuda"), p.components["vae"].device
    tiling = (p.component_tiling or {}).get("vae")
    assert not (isinstance(tiling, dict) and tiling.get("when") == "memory"), (p.strategy, tiling)
    assert "vae" in (p.runtime_op_tiling or {}), (p.strategy, p.runtime_op_tiling)


def test_where_nothing_exact_holds_it_on_the_card_the_vendor_s_tiles_are_the_way_down():
    p = _plan("v100-16g")
    assert str(p.components["vae"].device).startswith("cuda"), p.components["vae"].device
    assert (p.component_tiling or {}).get("vae", {}).get("when") == "memory", (p.strategy, p.component_tiling)
