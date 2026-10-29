"""Roadmap C3: the tl.dot flash on an arch whose profile MEASURED it at every head dim (`flash.any_head_dim_correct`)
takes a head dim that is not a power of two like a power of two — on the device's scores budget — instead of routing it
on `sdpa.non_pow2_head_scores_bytes` to the math attention, which writes the scores (Allegro's 96: 28 min of its
transformer on Apple); and its tile is certified per key where the profile declares no hand table (`sdpa_thresholds`).
What this test would do if the code were wrong: a measured arch routing 96 to math, an unmeasured one (every NVIDIA
profile) routing it to the flash, a key where the hand table chooses the tile or none where it is certified -> red."""
import pytest


@pytest.fixture
def prof(monkeypatch):
    from nbxtensor.ops import _configs as CF
    state = {}
    monkeypatch.setattr(CF, "active_vendor_profile", lambda: state)
    return state


def _route(D, Tq=4096, Tk=4096):
    from nbxtensor.ops import launch_keys as LK
    # an arch with no scores budget (every Apple and Ampere+ profile: memory.sdpa_math_max_scores_bytes 0)
    return LK.sdpa_route(2, 24, Tq, Tk, D, D, 0, 128, 64)[0]


def test_an_unmeasured_arch_keeps_a_non_pow2_head_on_the_math_route(prof):
    assert _route(96) == "chunked"       # 2 x 24 x 4096 x 4096 x 4 = 3.2 GB over 2 GiB: chunked, never flash
    assert _route(96, 1024, 1024) == "math"
    assert _route(64) == "flash"


def test_a_measured_arch_takes_the_flash_at_every_head_dim(prof):
    prof["flash"] = {"any_head_dim_correct": True}
    assert _route(96) == "flash" and _route(96, 1024, 1024) == "flash" and _route(72, 77, 77) == "flash"
    assert _route(64) == "flash"


def test_the_measurement_must_be_stated_true(prof):
    prof["flash"] = {"any_head_dim_correct": "yes"}
    assert _route(96, 1024, 1024) == "math"


def test_the_flash_is_keyed_where_no_hand_table_chooses_its_tile(prof):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    (kern, key), = LK.flash_launches(4680, 4680, 96, D.bfloat16, D.bfloat16, D.bfloat16)
    assert kern == LK.FLASH
    assert key == (LK.bucket_of("M", 4680), LK.bucket_of("N", 4680), 96,
                   "bf16", "bf16", "bf16", "bf16", "bf16", "fp32", "fp32")
    (_k, masked), = LK.flash_launches(77, 300, 64, D.float16, D.float16, D.float16, D.float32)
    assert masked[3:] == ("fp16", "fp16", "fp16", "fp32", "fp16", "fp32", "fp32")


def test_a_hand_table_keeps_its_tiles_unkeyed(prof):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    prof["sdpa_thresholds"] = [{"head_dim_lt": 128, "block_m": 64, "block_n": 64, "num_warps": 4}]
    assert LK.flash_launches(4096, 4096, 64, D.float16, D.float16, D.float16) == []


def test_the_candidate_tiles_respect_the_backends_bounds(monkeypatch):
    from nbxtensor import tensor as NT
    import nbxtensor.backend.capabilities as _nbx_capabilities
    from nbxtensor.ops import flash_attention as F
    monkeypatch.setattr(_nbx_capabilities, "fa_min_tile", lambda: 32)
    monkeypatch.setattr(_nbx_capabilities, "reduce_tile_max", lambda: 1024)
    keep = F._prune_flash(F.flash_configs(), {})
    assert {(c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"]) for c in keep} == {(32, 32)}
    assert {c.num_warps for c in keep} == {2, 4, 8}
    monkeypatch.setattr(_nbx_capabilities, "fa_min_tile", lambda: 64)
    with pytest.raises(ValueError):
        F._prune_flash(F.flash_configs(), {})


def test_every_apple_profile_retired_its_hand_table():
    from pathlib import Path
    import yaml
    apple = Path(__file__).resolve().parents[3] / "src/neurobrix/config/vendors/apple"
    assert not [y.name for y in sorted(apple.glob("*.yml")) if yaml.safe_load(y.read_text()).get("sdpa_thresholds")]
