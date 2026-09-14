"""After an illegal memory access the CUDA context is dead: every later
launch and malloc fails with the same error, so a certifier that goes on is
reporting a fault once per key. On 2026-09-14 07:38 one such key cost 227
further keys ("GPU malloc failed (error 700) for 256 bytes", each). The
run now stops at the first sticky error, names the key, and exits 1.

Injection: `sticky_cuda_error` forced to False made the first test RED;
restored, green.
"""
from neurobrix.kernels.autotune_certify import after_key_failure, sticky_cuda_error


def test_a_sticky_error_stops_the_run_and_names_the_key():
    said = []
    summary = {"failed": 0, "certified": 12}
    stop = after_key_failure(RuntimeError("GPU malloc failed (error 700) for 256 bytes [device cuda:0 …]"),
                             summary, "(1068480, 2048, 512, …)", said.append)
    assert stop is True
    assert summary["aborted"]["key"] == "(1068480, 2048, 512, …)" and summary["failed"] == 1
    assert any("ABORTED" in s and "poisoned" in s for s in said)


def test_an_ordinary_failure_is_counted_and_the_run_goes_on():
    summary = {"failed": 0}
    assert after_key_failure(RuntimeError("DeviceOOMError: 8 GiB requested, 5 GiB free"), summary, "k", lambda *a: None) is False
    assert summary["failed"] == 1 and "aborted" not in summary


def test_the_marks_are_the_ones_the_runtime_writes():
    for text in ("cudaMemcpy failed rc=700 (kind=2, nbytes=4096) — CUDA errors are STICKY",
                 "NeuroBrix launcher: cuLaunchKernel failed (700: an illegal memory access was encountered)",
                 "GPU malloc failed (error 700) for 1024 bytes"):
        assert sticky_cuda_error(RuntimeError(text)), text
    assert not sticky_cuda_error(RuntimeError("deviation 3e-4 exceeds tolerance 1e-4"))
