"""The vendor libraries the backends open through ctypes, named as each system names them: a shared object on Linux
(`libcuda.so.1`), a DLL on Windows (`nvcuda.dll`). Read by the backend tables (`backend.runtime`), the launch driver
(`backend.launcher`), the graph replay (`backend.replay_graph`), the expandable pool (`memory.vmm`) and Level Zero
(`backend.level_zero`). Windows' names are the vendors' (NVIDIA: nvcuda.dll in System32 with the driver, cudart64_12.dll
in the CUDA runtime wheel's `bin`, nvml.dll; AMD HIP SDK: amdhip64_6.dll; Intel: ze_loader.dll, OpenCL.dll).
"""
from __future__ import annotations

import sys

WINDOWS = sys.platform == "win32"

#: the CUDA runtime, after the one the environment's wheel carries (`runtime._environment_runtime_libs`)
CUDA_RUNTIME = ["cudart64_12.dll", "cudart64_110.dll"] if WINDOWS else ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
#: the CUDA runtime file inside the nvidia-cuda-runtime wheel, and the wheel's directory that holds it
CUDA_RUNTIME_IN_WHEEL = ("bin", "cudart64_12.dll") if WINDOWS else ("lib", "libcudart.so.12")
#: the CUDA driver API
CUDA_DRIVER = ["nvcuda.dll"] if WINDOWS else ["libcuda.so.1", "libcuda.so"]
#: NVIDIA's management library
NVML = ["nvml.dll"] if WINDOWS else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
#: the HIP runtime
HIP_RUNTIME = ["amdhip64_6.dll", "amdhip64.dll"] if WINDOWS else ["libamdhip64.so", "libamdhip64.so.5"]
#: the Level Zero loader
LEVEL_ZERO = ["ze_loader.dll"] if WINDOWS else ["libze_loader.so.1", "libze_loader.so"]
#: the OpenCL ICD loader
OPENCL = ["OpenCL.dll"] if WINDOWS else ["libOpenCL.so.1", "libOpenCL.so"]
