"""A Prism test plans on the container and the pinned facts, never on the machine's own calibration store.

The store (`core/dtype/calibration.STORE_ROOT`, `~/.neurobrix/calibration`) holds what this machine's runs measured —
precision records and, since 2026-11-08, every compiled run's runtime growth (`core/runtime/runtime_growth`, read from
the same root). A record there moves a plan: the Mac's mochi-1-preview growth record of RUN 82 (10 033 MB) turned
`test_a_stream_chunk_is_priced_at_what_the_driver_holds` red on the Mac only. `tests/unit/prism/conftest.py` pins the
store empty for every test of this directory; a test that wants a record writes its own.

What this test would do if the code were wrong: with no pin, the store is the engine home's — red at the first assert;
a growth record planted in the real home's layout would be read — red at the second.
"""
from __future__ import annotations

from pathlib import Path


def test_the_store_a_prism_test_reads_is_not_the_machine_s():
    import neurobrix.core.dtype.calibration as calibration
    from neurobrix.core.runtime import runtime_growth
    home = Path.home() / ".neurobrix" / "calibration"
    assert Path(calibration.STORE_ROOT) != home
    assert not Path(runtime_growth.path("mochi-1-preview")).is_relative_to(home)
