"""Certification tries the pipelining depths the profile allows, for every tile shape the kernel offers.

The Apple profile restricts certification to num_stages 1 and 2 (owner, 2026-09-26; the triton-ext AppleGPU
author's recommendation on PR #126). Filtering the configuration list by depth would silently drop every tile the
kernel only listed at depth 3 or 4; the depth is its own axis, so each tile is re-offered at each allowed depth.
"""
from __future__ import annotations

import triton

from neurobrix.kernels.autotune_certify import _num_stages_space, restrict_num_stages


def _cfg(bm, warps, ns):
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": 64}, num_warps=warps, num_stages=ns)


def test_every_tile_is_offered_at_every_allowed_depth():
    configs = [_cfg(64, 4, 4), _cfg(128, 4, 3), _cfg(64, 4, 2), _cfg(32, 2, 5)]
    out = restrict_num_stages(configs, (1, 2))
    got = [(c.kwargs["BLOCK_M"], c.num_warps, c.num_stages) for c in out]
    assert got == [(64, 4, 1), (64, 4, 2), (128, 4, 1), (128, 4, 2), (32, 2, 1), (32, 2, 2)]


def test_no_space_leaves_the_list_as_it_was():
    configs = [_cfg(64, 4, 4), _cfg(128, 4, 3)]
    assert restrict_num_stages(configs, None) is configs


def test_the_apple_profile_declares_depths_one_and_two():
    assert _num_stages_space("apple", "apple_m4_pro") == (1, 2)


def test_a_profile_without_the_key_declares_nothing():
    assert _num_stages_space("nvidia", "volta") is None
