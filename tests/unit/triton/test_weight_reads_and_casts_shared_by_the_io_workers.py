"""A weight is read and cast by the loader's workers, one tensor at a time — the same bytes.

The triton loader read each tensor with one `f.read` and cast it with one numpy loop: 0.40 GB/s on
Wan2.2-I2V-A14B's fp32 shards over the RDMA cache, 4.7 min of a 2-step gate cell spent loading one
expert (native: 68 s). `_read_bytes` splits the read into page-aligned spans and `_cast` the cast into
element spans, across `io_workers()` threads. What this holds: the bytes are exactly the ones a single
read and a single `astype` give, at every split boundary (an empty tensor, under one page per worker,
an odd tail); a short read is refused by name, never returned shorter; and the split really happens
(more than one span) — a pool of one would pass every equality while sharing nothing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from neurobrix.triton import weight_loader as W
from nbxtensor.io import safetensors as SF   # the primitives (M2); the pool is the loader's


@pytest.fixture
def blob(tmp_path):
    def write(arr, lead=b"hdr"):
        p = tmp_path / "shard.bin"
        p.write_bytes(lead + arr.tobytes())
        fd = os.open(p, os.O_RDONLY)
        return fd, len(lead)
    return write


@pytest.mark.parametrize("n", [0, 1, 7, 1023, 4096 * 8 + 3, 3_000_001])
def test_the_read_and_the_cast_are_the_single_pass_bytes(blob, n):
    a = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    fd, off = blob(a)
    try:
        r = SF.read_bytes(fd, off, a.nbytes, W._io_pool())
    finally:
        os.close(fd)
    assert r.tobytes() == a.tobytes()
    c = SF.cast(np.frombuffer(r, np.float32), np.float16, W._io_pool())
    assert c.flags.c_contiguous and c.tobytes() == a.astype(np.float16).tobytes()


def test_a_strided_or_widening_cast_keeps_shape_and_bytes():
    rng = np.random.default_rng(1)
    t = rng.standard_normal((300, 513)).astype(np.float32)[:, ::2]
    assert SF.cast(t, np.float16, W._io_pool()).tobytes() == t.astype(np.float16).tobytes()
    h = rng.standard_normal((37, 1001)).astype(np.float16)
    out = SF.cast(h, np.float32, W._io_pool())
    assert out.shape == h.shape and out.tobytes() == h.astype(np.float32).tobytes()


def test_a_short_read_is_refused_by_name(blob):
    fd, off = blob(np.zeros(1000, np.float32))
    try:
        with pytest.raises(RuntimeError, match="short"):
            SF.read_bytes(fd, off, 8_000_000, W._io_pool())
    finally:
        os.close(fd)


def test_a_large_tensor_is_split_across_the_workers():
    workers = W._io_pool()[0]
    if workers < 2:
        pytest.skip("one I/O worker configured: nothing to share")
    spans = SF.spans(10_000_000, 4, 1, workers)
    assert len(spans) == workers
    assert spans[0][0] == 0 and spans[-1][1] == 10_000_000
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    page_spans = SF.spans(10_000_000, 1, SF.PAGE, workers)
    assert all(a % SF.PAGE == 0 for a, _ in page_spans)
