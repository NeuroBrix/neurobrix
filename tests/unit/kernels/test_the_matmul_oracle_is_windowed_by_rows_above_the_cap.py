"""The certifier's fp64 oracle for the matmul family is computed on ROW WINDOWS above the MAC
cap, never whole on the host. Measured 2026-09-21 23:32 on metatron: a 44 544 x 3 072 x 8 192
product's whole fp64 oracle in numpy (2.2 TFLOP through OpenBLAS's pool) held one certifier at
155 GB of host RSS with its card at 0 % while four of them drove the load to 74 on 80 cores.

Shapes: a product 8 x 8 x 8 (512 MACs) stays whole under a cap of 1 000; 40 x 8 x 8 (2 560 MACs)
is windowed at rows 0-13, 13-26, 27-40 — the first rows, the middle, the last: the masked edges
a wrong tiling shows at, and the interior. The batched form windows the bias by the same rows.
"""
from __future__ import annotations

import numpy as np

from neurobrix.kernels import autotune_certify as CF


def test_a_small_product_keeps_its_whole_oracle():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((8, 8)).astype(np.float32), rng.standard_normal((8, 8)).astype(np.float32)
    assert CF._row_windows(8, 8, 8, cap=1000) is None
    oracle = CF._matmul_oracle_fn(a, b)()
    assert isinstance(oracle, np.ndarray) and oracle.shape == (8, 8)


def test_a_large_product_is_windowed_by_rows_and_measured_on_them(monkeypatch):
    monkeypatch.setattr(CF, "ORACLE_MAX_MACS", 1000)
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((40, 8)).astype(np.float32), rng.standard_normal((8, 8)).astype(np.float32)
    wins = CF._row_windows(40, 8, 8)
    assert wins == [(0, 5), (17, 22), (35, 40)], wins
    oracle = CF._matmul_oracle_fn(a, b)()
    assert isinstance(oracle, CF.RowWindowedOracle) and "row window" in oracle.describe
    full = a.astype(np.float64) @ b.astype(np.float64)
    assert CF.oracle_deviation(full, oracle) == 0.0
    wrong = full.copy(); wrong[-1, -1] += 1.0                       # a masked-edge error at the last row
    assert CF.oracle_deviation(wrong, oracle) > 0.0
    bias = rng.standard_normal((2, 40, 8)).astype(np.float32)
    a3, b3 = np.stack([a, a]), np.stack([b, b])
    o3 = CF._matmul_oracle_fn(a3, b3, bias)()
    full3 = a3.astype(np.float64) @ b3.astype(np.float64) + bias.astype(np.float64)
    assert CF.oracle_deviation(full3, o3) == 0.0
