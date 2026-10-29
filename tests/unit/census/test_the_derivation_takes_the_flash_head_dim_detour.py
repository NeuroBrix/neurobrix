"""The derived census follows the attention wrapper through its zero-pad detour.

A power-of-two head dim >= 128 never reaches the flash kernel as is: the wrapper pads Q, K and V
by one and re-enters itself (`flash_headdim_detour`), routed afresh at D + 1 — on an arch whose
profile declares no scores budget, a non-power-of-two dim takes the MATH route under its 2 GiB
bound. The derivation stopped at the first "flash" and formed no key: PixArt-XL's VAE attention
(3072 tokens, head dim 512) missed baddbmm (3072, 3072, 544) and (3072, 544, 3072) on Apple
(measured by the Mac, 2026-09-29 14:08), while the certified-only run served them.

What would this file do if the code were wrong? The derivation's re-route removed -> the
no-budget case derives no launch, RED; the detour answering for a non-power-of-two dim, or not
for 512 -> the function case, RED; V's head dim left unpadded -> the second key's N, RED.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derived_census as D  # noqa: E402
from nbxtensor.ops import launch_keys as LK  # noqa: E402
from neurobrix.kernels import census as _census  # noqa: E402
from nbxtensor.dtype.nbx_dtype import NBXDtype  # noqa: E402

import pytest  # noqa: E402

F32 = NBXDtype.float32


def _arch(monkeypatch, correct: bool):
    """The arch profile's own declaration decides the detour (flash.pow2_head_dim_correct, every profile
    declares it): these cells state which arch they speak of. They ran on whatever profile the process had
    bound — on a V100 rack, Volta, which declares true since 2026-10-03 — and two of them went red there."""
    from nbxtensor.ops import _configs
    real = _configs.active_vendor_profile
    # the bound profile as it is (its ladders, its thresholds), with this one declaration stated
    monkeypatch.setattr(_configs, "active_vendor_profile",
                        lambda: {**(real() or {}), "flash": {"pow2_head_dim_correct": correct}})


@pytest.fixture
def unmeasured_arch(monkeypatch):
    _arch(monkeypatch, False)


def test_the_detour_is_the_power_of_two_head_dims_from_128(unmeasured_arch):
    assert [LK.flash_headdim_detour(d) for d in (64, 72, 127, 128, 129, 256, 512)] == \
        [64, 72, 127, 129, 129, 257, 513]
    assert LK.flash_headdim_detour(512, enabled=False) == 512


def test_an_arch_that_measured_the_kernel_correct_takes_no_detour(monkeypatch):
    _arch(monkeypatch, True)
    assert [LK.flash_headdim_detour(d) for d in (64, 127, 128, 256, 512)] == [64, 127, 128, 256, 512]
    # and the derivation then stays on the flash route at the true head dim: no launch, nothing unhandled
    launches, unhandled = _sdpa_launches(512, 0)
    assert launches == [] and not unhandled


def _sdpa_launches(D_, budget):
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)  # a committed V100 profile: its ladders
    o = {"op_type": "aten::_scaled_dot_product_efficient_attention", "attributes": {},
         "input_tensor_ids": ["q", "k", "v"]}
    shape = lambda t: [1, 1, 3072, D_]
    unhandled = collections.Counter()
    out = D._op_launches(o["op_type"], "aten._scaled_dot_product_efficient_attention::0", o,
                         ["q", "k", "v"], shape, lambda t: F32, LK, None, "float32", False,
                         budget, 128, 0, 1 << 30, unhandled)
    return out or [], unhandled


def test_no_scores_budget_routes_the_padded_call_to_math(unmeasured_arch, tl_dot_gemms):
    launches, unhandled = _sdpa_launches(512, 0)
    # the math route's two products, and its softmax over the scores' rows (an autotuned kernel since the
    # softmax certification, keyed by its padded row)
    keys = [k for q, k in launches if q.endswith("baddbmm_kernel")]
    soft = [k for q, k in launches if "softmax" in q]
    assert len(keys) == 2 and len(soft) == 1 and len(launches) == 3, launches
    assert soft[0][:2] == (3072, LK.softmax_feat_bucket(3072)), soft
    b = LK.bucket_of
    assert keys[0][:3] == (b("M", 3072), b("N", 3072), b("K", 513))
    assert keys[1][:3] == (b("M", 3072), b("N", 513), b("K", 3072))
    assert not D.unplaced(unhandled)


def test_a_budget_that_admits_the_scores_keeps_the_true_head_dim(tl_dot_gemms):
    launches, _ = _sdpa_launches(512, 2 << 30)
    assert [k[:3] for _, k in launches][0] == (LK.bucket_of("M", 3072), LK.bucket_of("N", 3072),
                                               LK.bucket_of("K", 512))


def test_a_dim_the_detour_leaves_alone_stays_flash_without_a_budget_over_its_bound():
    # head dim 64 (power of two, under 128): no detour; no budget -> flash: the matrix-unit flash's one key
    # (certified since workstream 1), no product of the math route
    launches, _ = _sdpa_launches(64, 0)
    assert len(launches) == 1 and launches[0][0].endswith("flash_attention_m8n8k4_kernel"), launches
    assert launches[0][1][2] == 64, launches
