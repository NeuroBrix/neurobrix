"""An "out of memory" that does not say what would have fit costs the diagnosis.

Addition 4 of `docs/reference/adaptive-memory-a-runtime-controller.md`. The
allocator knows what it asked for and what the driver had; between those two
numbers is a concrete answer — how many pieces the work must be cut into, and
whether any cut can reach it at all.

The reproducer this design is written against:

    Failed at aten.convolution::349 (aten::convolution):
    GPU malloc failed (error 2) for 8589934592 bytes
    [device cuda:0 live_tracked=8242MB pool_cached=0MB driver_free=7598MB]

8.59 GB wanted, 7.60 GB free. The reader is left to do the division.

Injection that turns these red: make `annotate` return its message unchanged.
"""

from __future__ import annotations


from neurobrix.kernels.nbx_tensor import DeviceOOMError
from neurobrix.kernels.oom_advice import annotate, what_would_have_fit

_GB = 1024 ** 3


def _oom(requested, free, total=16 * _GB):
    return DeviceOOMError("GPU malloc failed", requested=requested, device_idx=0,
                          live=8242 * 1024 * 1024, pool_cached=0, pool_blocks=0,
                          driver_free=free, driver_total=total)


def test_the_real_reproducer_is_told_it_needs_two_bands():
    """8.59 GB against 7.60 GB free: two bands, and the shortfall named."""
    oom = _oom(8589934592, 7598 * 1024 * 1024)
    said = what_would_have_fit(oom)
    assert "2 bands" in said, said
    assert "short by" in said
    # The shortfall is the real one, not the whole request.
    assert oom.shortfall == 8589934592 - 7598 * 1024 * 1024


def test_the_band_size_fits_beside_what_is_already_live():
    """A band must fit the free figure AT THE MOMENT OF FAILURE.

    Dividing by the card's total would produce a band that only fits an empty
    card — advice that fails the moment it is followed.
    """
    oom = _oom(10 * _GB, 3 * _GB)
    said = what_would_have_fit(oom)
    assert "4 bands" in said, said          # ceil(10/3)
    band = 10 * _GB / 4
    assert band <= 3 * _GB


def test_a_request_larger_than_the_whole_card_says_no_split_reaches_it():
    """Four bands of a 40 GB request on a 16 GB card is still bad advice.

    Without this the sentence would cheerfully propose a split for work the card
    cannot hold under any division, which is worse than saying nothing.
    """
    said = what_would_have_fit(_oom(40 * _GB, 2 * _GB, total=16 * _GB))
    assert "no band split alone reaches it" in said, said
    assert "entire" in said


def test_the_leading_axis_is_named_only_when_a_shape_is_given():
    oom = _oom(8 * _GB, 3 * _GB)
    assert "leading axis" not in what_would_have_fit(oom)
    said = what_would_have_fit(oom, shape=(1, 64, 512, 512))
    assert "leading axis" not in said, "a leading axis of 1 cannot be cut into 3"
    said = what_would_have_fit(oom, shape=(12, 64, 512, 512))
    assert "leading axis" in said and "12 to 4" in said, said


def test_a_driver_that_reported_nothing_produces_no_sentence():
    """Silence beats a sentence built on a figure the driver never gave."""
    oom = DeviceOOMError("GPU malloc failed", requested=8 * _GB, driver_free=None)
    assert what_would_have_fit(oom) == ""
    assert annotate("Failed at op x", oom) == "Failed at op x"


def test_a_card_reporting_zero_free_is_told_no_split_helps():
    """0 free is a real reading, and dividing by it would raise."""
    said = what_would_have_fit(_oom(8 * _GB, 0))
    assert "nothing free" in said and "no band split" in said, said


def test_annotate_actually_appends_it_to_an_op_failure():
    """The one that catches a no-op wiring, and it was missing.

    Every other cell here calls `what_would_have_fit` directly. With `annotate`
    stubbed to `return message` — which is what a broken or reverted wiring looks
    like from the outside — all eight of them stayed GREEN, because none of them
    asked whether the advice ever reaches a message. The seams in both sequences
    call `annotate`, not `what_would_have_fit`, so this is the assertion that
    covers what actually runs.
    """
    oom = _oom(8589934592, 7598 * 1024 * 1024)
    out = annotate("Failed at op aten.convolution::349 (aten::convolution)", oom)
    assert out.startswith("Failed at op aten.convolution::349")
    assert "what would have fit" in out, out
    assert "2 bands" in out, out


def test_an_unrelated_failure_is_not_dressed_up_as_a_memory_problem():
    """The seam wraps EVERY op failure, so the guard has to be on the cause.

    Without it, an illegal memory access or a shape mismatch would arrive with a
    sentence about bands, and the reader would spend the next hour on memory.
    """
    for cause in (ValueError("shape mismatch"),
                  RuntimeError("CUDA error: an illegal memory access was encountered")):
        assert annotate("Failed at op x", cause) == "Failed at op x"


def test_the_seams_that_use_it_pass_the_cause_and_not_the_message():
    """Both modes wire it, and neither parses the allocator's prose.

    A controller built on a regex over a diagnostic breaks the first time the
    diagnostic is reworded; this pins that neither seam does that.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
    triton = (root / "triton" / "sequence.py").read_text()
    compiled = (root / "core" / "runtime" / "graph" / "compiled_sequence.py").read_text()
    assert "oom_advice" in triton and "oom_advice" in compiled
    for text in (triton, compiled):
        assert "GPU malloc failed" not in text, (
            "a seam is matching on the allocator's message text")
