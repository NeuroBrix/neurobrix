"""An import that cannot fit refuses before it starts, with both numbers.

A user on an A40 lost a 30.6 GB import at **98 %** on a full disk. Nothing had
checked. And `--no-keep` does not lower the peak: the archive is only deleted
AFTER extraction, so the moment of maximum pressure is store + cache — about
twice the model — whatever that flag says.

It is the disk twin of the memory defect being worked on elsewhere: a plan
accepted without measuring what is actually available, then a death in the
middle. The check goes BEFORE, declares both numbers — what it needs and what
there is — and refuses loudly rather than beginning.

Resume is a separate matter and is NOT this: a check that refuses cleanly is not
a resume, and pretending otherwise would leave the 98 % loss unaddressed for
whoever reads this later. Filed as D-IMPORT-RESUMABLE-DOWNLOAD.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_import_checks_the_disk_first.py
"""
from __future__ import annotations

import pytest

GB = 1024 ** 3


def test_an_import_that_cannot_fit_is_refused():
    from neurobrix.cli.commands.registry import disk_refusal

    reason = disk_refusal(needed_bytes=30 * GB, free_bytes=20 * GB)
    assert reason is not None
    assert "30" in reason and "20" in reason, (
        f"the refusal must carry BOTH numbers, what it needs and what there "
        f"is: {reason}")


def test_the_peak_is_twice_the_archive_even_with_no_keep():
    """The archive is deleted after extraction, so the peak is store + cache."""
    from neurobrix.cli.commands.registry import import_peak_bytes

    assert import_peak_bytes(10 * GB, keep=True) == 20 * GB
    assert import_peak_bytes(10 * GB, keep=False) == 20 * GB, (
        "--no-keep saves space AFTER the extraction, not during it")


def test_the_live_shape_is_refused():
    """30.6 GB on the disk that was full."""
    from neurobrix.cli.commands.registry import disk_refusal, import_peak_bytes

    need = import_peak_bytes(int(30.6 * GB), keep=False)
    assert disk_refusal(needed_bytes=need, free_bytes=int(35 * GB)) is not None, (
        "30.6 GB needs ~61 GB at peak; 35 GB free must refuse")


def test_an_import_that_fits_is_admitted():
    from neurobrix.cli.commands.registry import disk_refusal

    assert disk_refusal(needed_bytes=10 * GB, free_bytes=200 * GB) is None


def test_an_unknown_size_does_not_refuse():
    """The hub does not always publish a size. Refusing on a number nobody has
    would block imports that would have worked — the same rule as every other
    guard here: an unknown fact refuses nothing."""
    from neurobrix.cli.commands.registry import disk_refusal

    assert disk_refusal(needed_bytes=0, free_bytes=1 * GB) is None
    assert disk_refusal(needed_bytes=None, free_bytes=1 * GB) is None


def test_the_margin_is_declared_not_hidden():
    """A check that refuses at exactly 100 % leaves a disk with nothing on it.
    Whatever margin is kept must be visible in the refusal."""
    from neurobrix.cli.commands.registry import disk_refusal

    reason = disk_refusal(needed_bytes=100 * GB, free_bytes=101 * GB)
    assert reason is not None, "one GB of headroom on a 100 GB import is not room"
    assert "margin" in reason.lower() or "marge" in reason.lower()
