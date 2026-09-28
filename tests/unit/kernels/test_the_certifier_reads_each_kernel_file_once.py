"""The certify loop reads each kernel file ONCE per pass, not once per key.

`per_dtype.setdefault(dtype, _read_file(path))` evaluated its default on every key: each of a
pass's thousands of census keys re-read and re-parsed a kernel file of up to 21 MB (0.25 s) and
discarded it — the 16 and 32 GB GEMM certifiers sat 8-10 minutes in `json.loads` before touching
their card (py-spy, 2026-09-29 00:55), the cards idle between passes.

What the test would do if the loop re-read per key: the read count would be the key count, not
one (seen red with the setdefault form restored in `_file_entries`).
"""
from __future__ import annotations

from neurobrix.kernels import autotune_certify as CF


def test_a_thousand_keys_read_the_file_once(tmp_path, monkeypatch):
    reads = []
    monkeypatch.setattr(CF, "_read_file", lambda p: reads.append(p) or {"k": {}})
    cache: dict = {}
    for _ in range(1000):
        entries = CF._file_entries(cache, "fp32", tmp_path / "matmul_kernel.fp32.json")
    assert len(reads) == 1, f"{len(reads)} reads for 1000 keys"
    assert entries is cache["fp32"]


def test_the_certify_loop_takes_its_view_through_the_once_reader():
    import inspect
    src = inspect.getsource(CF.certify)
    assert "_file_entries(per_dtype, dtype, path)" in src
    assert "setdefault(dtype, _read_file" not in src
