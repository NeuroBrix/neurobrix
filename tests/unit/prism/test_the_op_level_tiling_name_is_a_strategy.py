"""A name Prism can emit and the strategy registry cannot build is a crash at selection (`cpu_streaming` was exactly
that on 2026-09-03): `op_level_tiling` builds the single-GPU executor — the placement is single-GPU, the overflowing
ops are cut, which rides in `runtime_op_tiling`. The registry is the PyTorch branch's executors (`core/strategies`,
torch at import): skipped by name where torch is not installed. Moved out of test_op_level_tiling_is_a_cascade_rung.py
so that file runs whole in the torch-free proof.

What this test would do if the code were wrong: the name absent from the registry, or bound to another executor — red."""
import pytest

pytest.importorskip("torch", reason="the strategy registry is the PyTorch branch's")

from neurobrix.core.strategies import STRATEGY_REGISTRY  # noqa: E402


def test_the_name_is_buildable():
    assert "op_level_tiling" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["op_level_tiling"] is STRATEGY_REGISTRY["single_gpu"]
