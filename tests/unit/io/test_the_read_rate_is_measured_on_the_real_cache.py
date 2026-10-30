"""The first launch measures the loaders' parallel reads on THIS machine's cache (`nbxtensor.io.read_rate.workers_for`,
called by autodetect): an integer between 1 and the machine's threads — never None, never a default. Runs on the real
cache when it holds weight files (skipped elsewhere).

Seen red 2026-10-30 on the rack's RDMA cache with 256 MiB trials (every draw read as unsteady: None, and every run on
the generated 32 GB profile refused): the trials now grow while the measure is not stable (the supervisor, 12:05)."""
import os

import pytest


def test_the_cache_s_read_count_is_measured():
    from neurobrix.core.paths import cache_dir
    from nbxtensor.io.read_rate import workers_for
    root = cache_dir()
    if not any(root.glob("*/components/*/weights/*.safetensors")):
        pytest.skip(f"no weight file in {root}")
    n = workers_for(root)
    assert isinstance(n, int) and 1 <= n <= (os.cpu_count() or 1), n
