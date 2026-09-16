"""One process, one CUDA runtime: the engine's ctypes loader opens the
`libcudart.so.12` the environment's torch wheel ships (`nvidia/cuda_runtime/lib`)
before any the system offers. Under torch 2.14.0+cu126 beside a 12.2 system
toolkit the other order broke `libc10_cuda.so` (undefined symbol
`cudaGetDriverEntryPointByVersion`, 2026-09-16). Injection: the environment
candidates dropped from the list → the first test found the system's path
first — RED."""
import os

from neurobrix.kernels import nbx_tensor as T


def test_the_environment_s_libcudart_leads_the_candidate_list():
    libs = T._GPU_BACKENDS["cuda"]["rt_libs"]
    env = T._environment_runtime_libs("nvidia.cuda_runtime", ["libcudart.so.12"])
    if not env:
        import pytest
        pytest.skip("this environment ships no nvidia.cuda_runtime package")
    assert libs[0] == env[0] and os.path.isabs(libs[0]) and "nvidia/cuda_runtime/lib" in libs[0]
    assert "libcudart.so.12" in libs      # the system's name stays as the fallback


def test_an_absent_package_adds_nothing():
    assert T._environment_runtime_libs("nvidia.no_such_runtime", ["libcudart.so.12"]) == []
