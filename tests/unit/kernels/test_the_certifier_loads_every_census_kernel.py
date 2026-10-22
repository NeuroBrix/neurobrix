"""The certifier finds the autotuner of every kernel the census names, its module loaded first
(`autotune_certify.census_tuners`): `gemv_vec_kernel`'s module is imported by `mv_wrapper` only when it launches, and
the certifier skipped its every key as "not an autotuner" (2026-10-22). In a fresh interpreter, where nothing launched.
Injection (seen red): the module import removed -> the GEMV qual absent."""
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[3] / "src")
QUAL = "neurobrix.kernels.ops.gemv_vec.gemv_vec_kernel"


def test_a_lazily_imported_kernel_is_found():
    code = ("import sys; from neurobrix.kernels import autotune_certify as A; "
            f"t = A.census_tuners({{{QUAL!r}: []}}); print({QUAL!r} in t)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env={"PYTHONPATH": SRC, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
                            "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip().splitlines()[-1] == "True", r.stdout[-500:]
