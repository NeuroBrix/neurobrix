"""The first launch measures the loaders' parallel reads on THIS machine's cache (`nbxtensor.io.read_rate.workers_for`,
called by autodetect). Its contract: an integer between 1 and the machine's threads, or the NAMED refusal when no measure
is stable (another reader on the storage) — never None, never a default.

The measure reads the real storage for minutes and is red whenever anything else reads it (a parallel unit suite: the
refusal after 475 s, the Dell 2026-10-31 21:02), so it runs only on request: NBX_STORAGE_MEASURE=1, on a quiet machine
— the first launch IS that measure. The contract with a stubbed measure is held in nbxtensor/tests/test_io_read_rate.py."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("NBX_STORAGE_MEASURE") != "1",
                                reason="reads the real storage for minutes: NBX_STORAGE_MEASURE=1 on a quiet machine")


def test_the_cache_s_read_count_is_measured_or_refused_by_name():
    from neurobrix.core.paths import cache_dir
    from nbxtensor.io.read_rate import workers_for
    root = cache_dir()
    if not any(root.glob("*/components/*/weights/*.safetensors")):
        pytest.skip(f"no weight file in {root}")
    try:
        n = workers_for(root)["workers"]
    except RuntimeError as e:
        assert "no stable read-rate measure" in str(e), e
        return
    assert isinstance(n, int) and 1 <= n <= (os.cpu_count() or 1), n
