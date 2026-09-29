"""The conv2d key's `fp16` flag names an fp16 input, and the certified entries made under the wrong
flag are re-keyed in place (the owner's decision, 2026-09-29 02:49).

The wrapper compared a Triton element type to an NBXDtype member: the flag was False for every
launch. The kernel's only use of it casts the loaded blocks to fp16 — a no-op on an fp16 input
(TTIR/TTGIR/LLIR/PTX identical on sm_70) — so an entry proven under the wrong key proves the right
one and is migrated, not re-proven.

What each test would do if the code were wrong: a migration that touched a non-fp16 input's key, or
left the proof's shape on the old key, or re-keyed twice, fails the first test; a migration that
overwrote an existing right-keyed entry fails the second (it must refuse); the source test fails on
the old comparison (seen red on 3bc85a81's `x.dtype == NBXDtype.float16`).
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import migrate_conv_fp16_key as MG  # noqa: E402

from neurobrix.kernels import autotune_certified as C  # noqa: E402

FP16_KEY = (1, 64, 1, 3206, 32, 1, 3200, 1, 7, 1, 1, 0, 3, 1, 1, 1, False, "fp16", "fp16", "fp16")
FP32_KEY = (1, 64, 1, 3206, 32, 1, 3200, 1, 7, 1, 1, 0, 3, 1, 1, 1, False, "fp32", "fp32", "fp32")


def _entry(key):
    proof = {"shape": list(key), "machine": {"device": {"memory_mb": 16384}}}
    return {"config": {"kwargs": {"BLOCK_SIZE_BHW": 64}}, "proof": proof, "excluded": [],
            "variants": {"32g": {"config": {"kwargs": {}}, "proof": {"shape": list(key), "machine": {"device": {"memory_mb": 32768}}}}}}


def _dir(tmp_path, entries):
    d = tmp_path / "volta"; d.mkdir()
    (d / "conv2d_forward_kernel.fp16.json").write_text(json.dumps(
        {"format": 1, "kernel": MG.KERNEL, "entries": {C.key_repr(k): v for k, v in entries.items()}}))
    return d


def test_an_fp16_input_is_re_keyed_with_its_proof_and_nothing_else(tmp_path):
    d = _dir(tmp_path, {FP16_KEY: _entry(FP16_KEY), FP32_KEY: _entry(FP32_KEY)})
    assert MG.migrate_directory(d, dry=False) == 1
    got = json.loads((d / "conv2d_forward_kernel.fp16.json").read_text())["entries"]
    right = FP16_KEY[:16] + (True,) + FP16_KEY[17:]
    assert set(got) == {C.key_repr(right), C.key_repr(FP32_KEY)}
    e = got[C.key_repr(right)]
    assert e["proof"]["shape"][16] is True and e["variants"]["32g"]["proof"]["shape"][16] is True
    assert got[C.key_repr(FP32_KEY)]["proof"]["shape"][16] is False
    assert MG.migrate_directory(d, dry=False) == 0            # idempotent


def test_an_existing_right_key_is_never_overwritten(tmp_path):
    right = FP16_KEY[:16] + (True,) + FP16_KEY[17:]
    d = _dir(tmp_path, {FP16_KEY: _entry(FP16_KEY), right: _entry(right)})
    with pytest.raises(SystemExit, match="already exists"):
        MG.migrate_directory(d, dry=False)


def test_the_wrapper_flags_an_fp16_input_by_its_nbx_dtype():
    from neurobrix.kernels import wrappers as W
    src = inspect.getsource(W.conv2d_wrapper)
    assert "fp16 = x.dtype == NBXDtype.float16" not in src
    assert "== NBXDtype.float16" in src and "nbx_dtype" in src
