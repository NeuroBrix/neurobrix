"""The vendor's statement of its size is HELP, not a gate (owner 13:48, project lead 13:51, 23/11): runtime/defaults.json
may declare `size_note` (one sentence), `size_static` (the vendor documents one size) and `size_range` ({min, max,
multiple}). The engine shows the note on the user's INFO line (`cli run`: `note_the_size`) when the request names a size
other than the declared default, not when it names the default, and never refuses on any of the three.

What would turn it red: the note silent for another size, shown for the default or for no size, the static or range
statement missing from the line, or a size outside the documented one/range refused."""
import logging
from types import SimpleNamespace

import pytest

from neurobrix.cli.commands import run
from neurobrix.core.runtime.size_policy import resolve_size, size_notice

NOTE = "Trained at 480x720; other sizes run outside the vendor documentation."
STATIC = {"size_policy": "default", "height": 480, "width": 720, "size_note": NOTE, "size_static": True}
RANGED = {"size_policy": "from_image", "max_area": 399360, "size_multiple": 16, "size_note": "Wan sizes.",
          "size_range": {"min": 256, "max": 1280, "multiple": 16}}


@pytest.mark.parametrize("h,w", [(368, 640), (480, 640), (240, None), (None, 360)])
def test_the_note_is_shown_when_the_size_is_not_the_default(caplog, h, w):
    with caplog.at_level(logging.INFO, logger=run.log.name):
        run.note_the_size(STATIC, SimpleNamespace(height=h, width=w))
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [f"[size] {NOTE} The vendor documents one size: 480x720."]


@pytest.mark.parametrize("h,w", [(480, 720), (480, None), (None, 720), (None, None)])
def test_the_note_is_not_shown_for_the_default_or_no_size(caplog, h, w):
    with caplog.at_level(logging.INFO, logger=run.log.name):
        run.note_the_size(STATIC, SimpleNamespace(height=h, width=w))
    assert not caplog.records


def test_a_from_image_container_shows_its_note_and_range_for_any_requested_size():
    assert size_notice(RANGED, 480, 832) == "[size] Wan sizes. The vendor documents 256 to 1280, a multiple of 16."
    assert size_notice(RANGED, None, None) is None
    assert size_notice({"size_policy": "default", "height": 480, "width": 720}, 368, 640) is None


@pytest.mark.parametrize("h,w", [(368, 640), (720, 1280), (16, 16)])
def test_size_static_never_refuses(h, w):
    assert resolve_size(STATIC, (512, 512), h, w) == (h, w)
    assert resolve_size(STATIC, (512, 512), None, None) == (480, 720)


def test_size_range_never_refuses():
    assert resolve_size(RANGED, (448, 448), 2048, 2048) == (2048, 2048)
    assert resolve_size(RANGED, (448, 448), 64, 64) == (64, 64)
