"""The certifier's oracle is the oracle of the launch the autotuner sees, not of the shape the
census key names.

Measured 2026-09-26 on this rack: the census key `matmul_kernel (4194304, 512, 256, IEEE,
PROMOTE_B, fp32, fp16, fp32)` — mochi's, an output of exactly 2^31 elements — was refused by the
certifier ("every config diverges from the fp64 oracle, best 1.0") while the same wrapper on the
production path was correct at every sampled row around the boundary (relative deviation under
1e-6). The `mm` wrapper band-streams a product above `NBX_MM_MAX_OUTPUT_ELEMS` output elements
and keys every band on the whole shape's bucket; the certifier computed ONE oracle over the
whole synthesized product and cut its comparison windows on whichever band tensor the launch
carried — past the band it read memory the kernel never wrote. An instrument that compares a
band against the whole says "kernel defect" about a correct kernel.

Now the oracle is computed from the launch's own operands (`launch_oracle`, the runtime screen's
`screen_oracle._mm` brick, row-windowed on the launch's M), and a second launch of an already
certified key inside one wrapper call — the next band — runs the chosen configuration. The proof
records how many launches served the key.

Seen RED on d2285fc2 (the band door lowered so a small key streams in two bands: the certifier
refused the key), GREEN here.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src pytest tests/unit/kernels/test_the_certifier_measures_the_launch_it_sees.py -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as AC


def test_the_launch_oracle_is_computed_from_the_launch_operands_not_the_whole(monkeypatch):
    """A band of 8 rows out of a 64-row product: the oracle covers the 8 rows the launch carries."""
    import types
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.matmul import matmul_kernel
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 16)).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)
    band = a[8:16]
    out = NBXTensor.from_numpy(np.zeros((8, 32), np.float32))
    tuner = types.SimpleNamespace(base_fn=matmul_kernel.fn if hasattr(matmul_kernel, "fn") else matmul_kernel,
                                  nargs={"a_ptr": NBXTensor.from_numpy(np.ascontiguousarray(band)),
                                         "b_ptr": NBXTensor.from_numpy(b), "c_ptr": out, "M": 8, "N": 32, "K": 16})
    oracle = AC.launch_oracle("neurobrix.kernels.ops.matmul.matmul_kernel", tuner, {}, out)
    assert hasattr(oracle, "blocks") and [w for w, _ in oracle.blocks] == [(0, 8)], oracle.describe
    np.testing.assert_allclose(oracle.blocks[0][1], band.astype(np.float64) @ b.astype(np.float64), rtol=0, atol=1e-12)
    # a launch over the cap is windowed on ITS rows, never past them
    monkeypatch.setattr(AC, "ORACLE_MAX_MACS", 8 * 32 * 16 // 2)
    windowed = AC.launch_oracle("neurobrix.kernels.ops.matmul.matmul_kernel", tuner, {}, out)
    assert all(r1 <= 8 for (r0, r1), _ in windowed.blocks), windowed.describe
    # families the screen does not row-window, or whose output is not a matrix, keep the
    # synthesized oracle — read from the screen's table and the output's rank, not a name list
    assert AC.launch_oracle("neurobrix.kernels.ops.baddbmm_op.baddbmm_kernel", tuner, {}, out) is None
    out4 = NBXTensor.from_numpy(np.zeros((1, 2, 4, 4), np.float32))
    assert AC.launch_oracle("neurobrix.kernels.ops.matmul.matmul_kernel", tuner, {}, out4) is None


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


def test_a_key_the_wrapper_streams_in_bands_is_certified_on_its_launches(monkeypatch, tmp_path):
    """The band door lowered to 4096 output elements: a (256 x 32) product streams in two bands
    of 128 rows, each its own launch under the one census key. RED on main: the certifier
    refused it ('every config diverges'); GREEN: certified, two launches recorded."""
    pytest.importorskip("triton")
    if _cuda_free_bytes() < (1 << 30):
        pytest.skip("needs a card with 1 GB free")
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay_cache"))
    from neurobrix.kernels import wrappers as W
    from neurobrix.triton import autotune_cache as atc
    monkeypatch.setattr(W, "_NBX_MM_MAX_OUTPUT_ELEMS", 4096)
    AC._bind_hardware_profile()
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    tuner = dict(atc._autotuners())[qual]
    # the key the wrapper forms for this shape, captured at the autotuner's seam
    captured = {}
    saved = tuner.run
    tuner.run = lambda *a, **k: captured.setdefault("key", tuple(atc.key_of(tuner, a, k)))
    try:
        from neurobrix.kernels.nbx_tensor import NBXTensor
        W.mm(NBXTensor.from_numpy(np.ones((256, 16), np.float32)).to("cuda"),
             NBXTensor.from_numpy(np.ones((16, 32), np.float32)).to("cuda").to("float16"))
    finally:
        tuner.run = saved
    key = captured["key"]
    assert int(key[0]) >= 256, key
    entry = AC.certify_key(qual, tuner, key, 1e-4, np.random.default_rng(3))
    proof = entry.get("proof", entry)
    assert int(proof.get("launches", 1)) == 2, f"two bands, two launches: {proof.get('launches')}"
    assert float(proof.get("deviation", entry.get("deviation", 1.0))) <= 1e-4


def test_a_later_band_that_writes_nothing_is_REFUSED_not_counted(monkeypatch, tmp_path):
    """The chosen configuration must hold on every band: the second launch is measured against
    its own oracle. Injection: the second launch's kernel run is skipped (its band is left
    poisoned/unwritten) — the certifier must refuse the key, not count two launches."""
    pytest.importorskip("triton")
    if _cuda_free_bytes() < (1 << 30):
        pytest.skip("needs a card with 1 GB free")
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay_cache"))
    from neurobrix.kernels import wrappers as W
    from neurobrix.triton import autotune_cache as atc
    monkeypatch.setattr(W, "_NBX_MM_MAX_OUTPUT_ELEMS", 4096)
    AC._bind_hardware_profile()
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    tuner = dict(atc._autotuners())[qual]
    captured = {}
    saved = tuner.run
    tuner.run = lambda *a, **k: captured.setdefault("key", tuple(atc.key_of(tuner, a, k)))
    try:
        from neurobrix.kernels.nbx_tensor import NBXTensor
        W.mm(NBXTensor.from_numpy(np.ones((256, 16), np.float32)).to("cuda"),
             NBXTensor.from_numpy(np.ones((16, 32), np.float32)).to("cuda").to("float16"))
    finally:
        tuner.run = saved
    key = captured["key"]
    # The injection: the SECOND band's comparison reads as an unwritten band (infinite
    # deviation) — the first band's candidates are measured as usual, so on a certifier that
    # counts later launches instead of measuring them this key certifies with `launches == 2`.
    orig_dev = AC.deviation_against
    calls = {"n": 0}

    def dev_of_unwritten_second_band(out_, oracle_):
        calls["n"] += 1
        if calls["n"] > 1 and hasattr(oracle_, "blocks") and oracle_.blocks[0][0][1] <= 128:
            return float("inf")
        return orig_dev(out_, oracle_)
    monkeypatch.setattr(AC, "deviation_against", dev_of_unwritten_second_band)
    with pytest.raises(RuntimeError, match="does not hold on every band"):
        AC.certify_key(qual, tuner, key, 1e-4, np.random.default_rng(3))


def test_a_batched_key_with_a_bias_is_certified_by_its_synthesized_oracle(monkeypatch, tmp_path):
    """`baddbmm` is a single-launch family: its batch-aware synthesized oracle stands. The
    guardian's finding (2026-09-26): a launch oracle for it dropped the bias (a constexpr kwarg
    the positional arguments never carry) and every biased key would have been refused."""
    pytest.importorskip("triton")
    if _cuda_free_bytes() < (1 << 30):
        pytest.skip("needs a card with 1 GB free")
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay_cache"))
    from neurobrix.triton import autotune_cache as atc
    AC._bind_hardware_profile()
    qual = "neurobrix.kernels.ops.baddbmm_op.baddbmm_kernel"
    tuner = dict(atc._autotuners())[qual]
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor
    captured = {}
    saved = tuner.run
    tuner.run = lambda *a, **k: captured.setdefault("key", tuple(atc.key_of(tuner, a, k)))
    try:
        B, M, N, K = 2, 64, 32, 16
        W.baddbmm_wrapper(NBXTensor.from_numpy(np.ones((B, M, N), np.float32)).to("cuda"),
                          NBXTensor.from_numpy(np.ones((B, M, K), np.float32)).to("cuda"),
                          NBXTensor.from_numpy(np.ones((B, K, N), np.float32)).to("cuda"))
    finally:
        tuner.run = saved
    key = captured["key"]
    entry = AC.certify_key(qual, tuner, key, 1e-4, np.random.default_rng(5))
    proof = entry.get("proof", entry)
    assert float(proof.get("deviation", 1.0)) <= 1e-4, proof
