"""`nbxtensor.inspection`: a tensor's values read on the host for a diagnostic — the non-finite scan, the value record
a per-op dump keeps, the last position's row and a digest of the bytes. They lived in the engine
(triton/sequence.py: `_maybe_trace_nan`, `nbx_tid_stats`, `_fingerprint`, `_maybe_fingerprint`) until 2026-11-12.

What this test would do if the code were wrong: a complex tensor scanned or measured on its real parts alone, a NaN
behind an infinity reported as "Inf", a bf16 value decoded inexactly, a last-position window past its row, a digest
over a strided view's storage instead of its elements — RED."""
import hashlib

import numpy as np
import pytest

from nbxtensor import inspection
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_the_non_finite_scan_names_nan_first_and_reads_both_complex_parts():
    _accelerator()
    assert inspection.non_finite(NBXTensor.from_numpy(np.ones((2, 3), np.float32))) is None
    assert inspection.non_finite(NBXTensor.from_numpy(np.array([np.inf, 1.0, np.nan], np.float32))) == "NaN"
    half = NBXTensor.from_numpy(np.array([1.0, -np.inf], np.float32)).to(NBXDtype.bfloat16)
    assert inspection.non_finite(half) == "Inf"
    z = np.array([1 + 1j, complex(2.0, np.inf)], dtype=np.complex64)    # infinite in the imaginary part only
    assert inspection.non_finite(NBXTensor.from_numpy(z)) == "Inf"


def test_the_host_values_are_exact_for_a_half_tensor_and_interleaved_for_a_complex_one():
    _accelerator()
    x = np.array([[1.5, -2.25], [3.0, 0.0078125]], dtype=np.float32)        # every value exact in bf16
    assert np.array_equal(inspection.host_values(NBXTensor.from_numpy(x).to(NBXDtype.bfloat16)), x.reshape(-1))
    z = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    assert inspection.host_values(NBXTensor.from_numpy(z)).tolist() == [1.0, 2.0, 3.0, -4.0]


def test_the_value_record_reads_the_first_row_the_last_position_and_the_norms():
    _accelerator()
    a = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    rec = inspection.value_stats(NBXTensor.from_numpy(a))
    assert rec["shape"] == [2, 4, 3] and rec["is_complex"] is False
    assert rec["head10"] == [0.0, 1.0, 2.0]                          # the first row, never past it
    assert rec["last_pos10"] == [9.0, 10.0, 11.0]                    # axis 1's last index, batch 0
    assert rec["l2_norm"] == pytest.approx(float(np.sqrt((a.astype(np.float64) ** 2).sum())))
    assert rec["batch_norms"] == pytest.approx([float(np.linalg.norm(a[0])), float(np.linalg.norm(a[1]))])
    assert inspection.last_position_row(NBXTensor.from_numpy(a)).tolist() == [9.0, 10.0, 11.0]
    m = np.arange(40, dtype=np.float32).reshape(2, 20)
    assert inspection.value_stats(NBXTensor.from_numpy(m))["last_pos10"] == list(range(20, 30))
    assert inspection.value_stats(NBXTensor.from_numpy(np.arange(5, dtype=np.float32)))["last_pos10"] is None
    assert inspection.last_position_row(NBXTensor.from_numpy(np.arange(5, dtype=np.float32))) is None
    empty = inspection.value_stats(NBXTensor.from_numpy(np.zeros((1, 0, 4), np.float32)))
    assert empty["head10"] == [] and empty["l2_norm"] == 0.0 and empty["last_pos10"] is None


def test_the_digest_covers_the_elements_in_row_major_order():
    _accelerator()
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = NBXTensor.from_numpy(x)
    assert inspection.digest(t, "sha256") == hashlib.sha256(x.tobytes()).hexdigest()
    assert inspection.digest(t, "sha256", 8) == hashlib.sha256(x.tobytes()[:8]).hexdigest()
    assert inspection.digest(t.t(), "blake2b", digest_size=8) == \
        hashlib.blake2b(np.ascontiguousarray(x.T).tobytes(), digest_size=8).hexdigest()
