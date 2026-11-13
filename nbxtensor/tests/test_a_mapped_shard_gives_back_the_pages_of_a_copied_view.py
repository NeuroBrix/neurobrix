"""nbxtensor.io.mapped_shards: a shard is mapped once, a host view's span is noted, and after the copy the span's
page-aligned range is given back; a tensor over no noted span is refused by name.

What this test does if the code is wrong: the mapping cell fails when a shard is mapped twice; the give-back cell fails
when the range advised is not the page-aligned one around the span, or when the view is not found by its address
through another object over the same bytes; the refusal cell fails when a tensor over no noted span gives back
anything.
"""
from __future__ import annotations

import mmap

import numpy as np
import pytest

from nbxtensor.io import mapped_shards as MS


class _Host:
    """A host tensor stand-in: the address of its first byte, as every tensor of the library and of torch answers."""

    def __init__(self, address: int):
        self._address = address

    def data_ptr(self) -> int:
        return self._address


def _shard(tmp_path, head: int, nbytes: int) -> str:
    path = tmp_path / "shard.bin"
    path.write_bytes(b"\x00" * head + bytes(range(256)) * (nbytes // 256) + b"\x00" * 64)
    return str(path)


def test_a_shard_is_mapped_once_per_path(tmp_path):
    shard = _shard(tmp_path, 1000, 1024)
    maps = MS.ShardMaps(mmap.ACCESS_READ)
    assert maps.map(shard) is maps.map(shard) and len(maps) == 1


def test_the_span_s_page_aligned_range_is_given_back_through_any_object_over_its_bytes(tmp_path, monkeypatch):
    head, nbytes = 1000, 1024                                   # the span starts inside the first page
    shard = _shard(tmp_path, head, nbytes)
    maps = MS.ShardMaps(mmap.ACCESS_COPY)
    m = maps.map(shard)
    view = np.frombuffer(m, dtype=np.uint8, count=nbytes, offset=head)
    address = view.ctypes.data
    maps.note(_Host(address), shard, head, nbytes)
    advised = []
    real = MS.drop_pages
    monkeypatch.setattr(MS, "drop_pages", lambda mm, off, n: advised.append((mm is m, off, n)) or real(mm, off, n))
    start, length = maps.give_back(_Host(address))             # another object, the same first byte
    assert advised == [(True, head, nbytes)] and (start, length) == (0, head + nbytes)
    assert bytes(view[:4]) == bytes(range(4))                   # a private, never written mapping re-reads the file


def test_drop_pages_advises_the_page_aligned_range(tmp_path):
    shard = _shard(tmp_path, 2 * mmap.PAGESIZE, 1024)              # past the first page: the range starts at one
    with open(shard, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    page = mmap.PAGESIZE
    assert MS.drop_pages(m, page + 10, 100) == (page, 110)


def test_a_tensor_over_no_noted_span_is_refused_by_name():
    maps = MS.ShardMaps(mmap.ACCESS_READ)
    with pytest.raises(RuntimeError, match="no view of a mapped weight file"):
        maps.give_back(_Host(12345))
