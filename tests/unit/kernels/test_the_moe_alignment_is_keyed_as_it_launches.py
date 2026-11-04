"""The MoE alignment's three stages (`ops/moe_align.py`, launched by `triton/moe.moe_align_block_size`) run their
certified configuration: the key each forms at the launch (`autotune_cache.key_of`, the launcher's own) IS the key
`launch_keys.moe_align_launches` computes for the layer — the function the derived census calls — at the FMA kernel's
block and at a unit tile's rows, and a card's share of a layer split across cards is keyed by the layer's pairs; and
the certifier's synthesis forms each key and meets the host sort EXACTLY. The chunk widths were a per-backend branch
(128 / the experts' count on CUDA, 32-64 on Metal) at a literal 4 warps before 2026-11-04.

What this test would do if the code were wrong: the key's length class taken from the share's own ids, the expert
count unpadded, the unit tile's block left out of the census — red at the keys; a synthesis that forms another key or
an oracle that is not the host sort (one index off) — red at the certification. Needs a device."""
from __future__ import annotations

import numpy as np
import pytest


def _card():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.device_count() >= 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _card(), reason="needs a device")


class _Launched(Exception):
    pass


def _record(monkeypatch):
    """Every stage's key, the launch itself then run with the configuration the test passes (the first)."""
    from nbxtensor.ops.autotune_cache import key_of
    from nbxtensor.ops import moe_align as MA
    seen = []
    for i in (1, 2, 3):
        tuner, prune = getattr(MA, f"moe_align_stage{i}_kernel"), getattr(MA, f"_prune_stage{i}")

        def run(*args, _t=tuner, _p=prune, _i=i, **kwargs):
            seen.append((f"nbxtensor.ops.moe_align.moe_align_stage{_i}_kernel", tuple(key_of(_t, args, kwargs))))
            cfg = _p(_t.configs, {**dict(zip(_t.arg_names, args)), **kwargs})[0]
            return _t.fn.run(*args, **{**kwargs, **cfg.all_kwargs()})
        monkeypatch.setattr(tuner, "run", run)
    return seen


def _ids(n, experts):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.random.default_rng(n).integers(0, experts, size=n).astype(np.int64))


@pytest.mark.parametrize("tokens,top_k,experts", [(1, 8, 128), (20, 6, 64), (300, 8, 16), (37, 2, 8)])
def test_every_stage_is_its_key(monkeypatch, tokens, top_k, experts):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops import launch_keys as LK
    from neurobrix.kernels.autotune_certify import _bind_hardware_profile
    from neurobrix.triton.moe import moe_align_block_size, _BLOCK_SIZE_M
    _bind_hardware_profile()
    want = LK.moe_align_launches(tokens, top_k, experts, NBXDtype.float16)
    blocks = sorted({k[0] for q, k in want if q != LK.MOE_ALIGN[2]})
    assert _BLOCK_SIZE_M in blocks and len(blocks) > 1, blocks        # the V100 profile declares the unit
    for bs in blocks:                     # the FMA block and every unit tile's rows the census names
        seen = _record(monkeypatch)
        moe_align_block_size(_ids(tokens * top_k, experts), bs, experts, DeviceAllocator.get_device())
        assert set(seen) <= set(want) and len(seen) == 3, (bs, seen, want)


def test_a_cards_share_is_keyed_by_the_layers_pairs(monkeypatch):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops import launch_keys as LK
    from neurobrix.kernels.autotune_certify import _bind_hardware_profile
    from neurobrix.triton.moe import moe_align_block_size, _BLOCK_SIZE_M
    _bind_hardware_profile()
    tokens, top_k, experts = 300, 8, 64
    want = LK.moe_align_launches(tokens, top_k, experts, NBXDtype.float16)
    seen = _record(monkeypatch)
    moe_align_block_size(_ids(777, experts), _BLOCK_SIZE_M, experts, DeviceAllocator.get_device(),
                         pairs=tokens * top_k)
    assert set(seen) <= set(want) and len(seen) == 3, (seen, want)


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("tokens,top_k,experts", [(20, 6, 64), (300, 8, 128)])
def test_the_certifier_certifies_every_stage(stage, tokens, top_k, experts):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.ops import moe_align as MA
    from neurobrix.kernels.autotune_certify import _bind_hardware_profile, certify_key
    _bind_hardware_profile()
    qual = LK.MOE_ALIGN[stage]
    key = next(k for q, k in LK.moe_align_launches(tokens, top_k, experts, NBXDtype.float16) if q == qual)
    entry = certify_key(qual, getattr(MA, qual.rsplit(".", 1)[1]), key, 1e-6, np.random.default_rng(2),
                        bench=lambda fn: (fn(), 1.0)[1])
    assert entry["excluded"] == [] and entry["proof"]["accepted"] >= 1, entry
    assert entry["proof"]["deviation"] == 0.0 and entry["proof"]["shape"] == list(key)


def test_every_device_states_the_integer_tolerance_the_alignment_is_certified_at():
    """The certifier takes a key's tolerance from its device's `autotune_screen_rtol` by output dtype and invents
    none: the alignment's int64 tables went unproven on every key (lot-cert 18:41, 14 failed a card: "declares no
    autotune_screen_rtol for int64"). Every device that states tolerances states int64 at 0 — met exactly."""
    import yaml
    from pathlib import Path
    import nbxtensor
    root = Path(nbxtensor.__file__).resolve().parent / "backend" / "devices"
    files = sorted(root.glob("*/*.yml"))
    assert files
    for f in files:
        rtol = (yaml.safe_load(f.read_text()) or {}).get("autotune_screen_rtol")
        if rtol is not None:
            assert rtol.get("int64") == 0.0, f"{f.name}: autotune_screen_rtol.int64 = {rtol.get('int64')!r}"
