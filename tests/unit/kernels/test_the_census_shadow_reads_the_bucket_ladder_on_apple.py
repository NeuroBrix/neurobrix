"""The census shadow must read the bucket ladder on Apple, as it already does on CUDA.

Behind the door (`CUDA_VISIBLE_DEVICES=`) no driver answers which vendor profile applies, so
`_bind_target` binds one from the hardware profile the census names. It did that for nvidia
and amd only — a brand whose compute capability is not a CUDA-style number returned early —
and on Apple the profile therefore resolved EMPTY, the ladder went unread, and every recorded
key was composed in the EXACT form.

Measured 2026-09-22 on the first bucketed Apple census: 825 of 2 955 harvested keys carried a
first dimension off the ladder (matmul M_BUCKET 8 664 where the launcher keys 8 704, addmm
158 400 where it keys 163 840). A census of exact keys cannot serve a bucketed launcher, and
"zero miss at verification" could never be true against it.

This is the same defect the rack fixed for itself on 2026-09-21 (matmul M = 226 and 3 136
where the served launcher keys 240 and 3 200), left open for every non-CUDA brand.
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import census
from neurobrix.kernels.ops import _configs
from neurobrix.kernels.autotune_bucket import bucket_of, ladder_for


APPLE_HW = {
    "id": "auto-apple-m4-pro-18g",
    "vendor": "apple",
    "devices": [{"index": 0, "brand": "apple", "model": "Apple M4 Pro",
                 "memory_mb": 18186, "compute_capability": "0.0"}],
}


@pytest.fixture(autouse=True)
def _clear_profile():
    _configs._ACTIVE_PROFILE.clear()
    yield
    _configs._ACTIVE_PROFILE.clear()


def test_without_a_bind_the_ladder_is_unread(monkeypatch):
    """The state this test exists to forbid.

    The driver is silenced the way the shadow silences it — `arch_smem_budget` is what
    `active_vendor_profile` calls to resolve, and behind the door it answers nothing. This
    models that rather than depending on a machine with no device.
    """
    monkeypatch.setattr(_configs, "arch_smem_budget", lambda: None)
    assert list(ladder_for("M")) == [(None, 1)]
    assert bucket_of("M", 8664) == 8664


def test_the_bind_does_not_need_the_driver(monkeypatch):
    """The fix must work with the driver silenced, which is the only case that matters."""
    monkeypatch.setattr(_configs, "arch_smem_budget", lambda: None)
    census._bind_target(None, APPLE_HW)
    assert bucket_of("M", 8664) == 8704


def test_a_device_with_no_model_is_refused():
    with pytest.raises(RuntimeError, match="names no model"):
        census._bind_target(None, {"devices": [{"brand": "apple", "compute_capability": "0.0"}]})


def test_a_device_no_vendor_profile_matches_is_refused():
    """A name nothing covers is refused rather than censused under no profile.

    Note what this does NOT test: an unknown APPLE variant is not unmatched — it falls back
    to `apple_silicon` by declared prefix, on purpose ("correctly, not optimally"). That
    profile declares no ladder, so such a machine censuses EXACT keys, which is the honest
    answer for a chip whose ladder nobody has measured.
    """
    with pytest.raises(RuntimeError, match="no vendor profile matches"):
        census._bind_target(None, {"devices": [{"brand": "acme", "model": "Widget 9000",
                                                "compute_capability": "0.0"}]})


def test_the_shadow_binds_the_apple_vendor_profile_from_its_hardware_profile():
    census._bind_target(None, APPLE_HW)
    rows = list(ladder_for("M"))
    assert len(rows) > 1, f"the ladder is still unread behind the door: {rows}"
    assert bucket_of("M", 8664) == 8704, "M must key on its bucket's top, as the launcher does"
    assert bucket_of("M", 158400) == 163840


def test_a_profile_naming_no_device_is_still_refused():
    with pytest.raises(RuntimeError, match="names no device"):
        census._bind_target(None, {"devices": []})
