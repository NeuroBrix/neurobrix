"""The launch driver of Intel GPUs over Level Zero (`backend.level_zero`'s runtime): the kernels Intel's Triton backend
compiles, launched with the ABI its own driver uses — the sources are in `backend.level_zero`'s docstring. Apart from
that module because it is the launcher's (it imports Triton); detecting the backend does not."""
from __future__ import annotations

import ctypes
import threading
from typing import Dict, Optional, Tuple

from nbxtensor.backend.launcher import _ModuleDriver, _ctypes_slot
from nbxtensor.backend.level_zero import (_GRF_RETRY_FLAG, _KERNEL_FLAG_FORCE_RESIDENCY, _MODULE_FORMAT_IL_SPIRV,
                                          _SPILL_RETRY_BYTES, _ST_KERNEL_DESC, _ST_KERNEL_PROPS, _ST_MODULE_DESC,
                                          LevelZeroRuntime, _as_int, _GroupCount, _KernelDesc, _KernelProps,
                                          _ModuleDesc, EXTENSION_FACTS, device_extensions,
                                          device_file_for, result_name, runtime)


class LevelZeroDriver(_ModuleDriver):
    """Intel GPUs: SPIR-V modules from Intel's Triton backend, kernels by name, arguments by index, an in-order
    immediate command list as the stream (module docstring of `backend.level_zero` for the sources)."""

    _inst: Optional["LevelZeroDriver"] = None

    @classmethod
    def instance(cls) -> "LevelZeroDriver":
        if cls._inst is None:
            cls._inst = cls(runtime())
        return cls._inst

    artifact_kind = "spv"
    backend = "xpu"
    wants_scratch_params = True

    def __init__(self, rt: LevelZeroRuntime) -> None:  # noqa: D107
        self.rt = rt
        self._modules: Dict[Tuple[int, bytes, str], ctypes.c_void_p] = {}
        self._kernels: Dict[int, dict] = {}              # kernel handle -> its argument count, SLM, name
        self._launch_lock = threading.Lock()             # a kernel's arguments are state of its handle (the spec)
        self._target = None

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"NeuroBrix launcher: {what} failed ({result_name(rc)})")

    def target(self):
        """`GPUTarget("xpu", <properties>, 32)` as Intel's driver builds it (driver.py `_construct_target`), torch- and
        SYCL-free: the counts Level Zero answers, the extensions the OpenCL ICD answers (`EXTENSION_FACTS`), and the
        architecture name from the device file its PCI id selects — handed to Intel's compiler through its own
        `knobs.intel.device_arch`, so it does not compile its SYCL arch parser. A fact none of them states is left to
        the compiler's own default (`has_16bit_atomics`: False), as Intel's driver leaves it."""
        if self._target is None:
            from triton import knobs
            from triton.backends.compiler import GPUTarget
            p, c = self.rt.properties(), self.rt.compute_properties()
            name, desc = device_file_for(p.deviceId)
            arch = str(desc.get("compute_capability") or "")
            if not arch:
                raise RuntimeError(f"nbxtensor: the Intel device file {name!r} states no compute_capability (the "
                                   f"architecture name Intel's compiler keys on, e.g. \"pvc\") — never guessed")
            exts = set(device_extensions(p.deviceId))
            sizes = [int(c.subGroupSizes[i]) for i in range(c.numSubGroupSizes)]
            props = {fact: ext in exts for fact, ext in EXTENSION_FACTS.items()}
            props.update({
                "name": p.name.decode(errors="replace"), "vendor": "Intel(R) Corporation",
                "device_id": int(p.deviceId), "driver_version": str(self.rt.driver_version()), "arch": arch,
                "gpu_eu_count": int(p.numSlices * p.numSubslicesPerSlice * p.numEUsPerSubslice),
                "gpu_subslice_count": int(p.numSlices * p.numSubslicesPerSlice),
                "max_work_group_size": int(c.maxTotalGroupSize),
                "max_num_sub_groups": int(c.maxTotalGroupSize) // min(sizes),
                "sub_group_sizes": sizes, "local_mem_size": int(c.maxSharedLocalMemory),
                "core_clock_rate": int(p.coreClockRate), "__intel_already_queried_extensions__": True,
            })
            if not hasattr(knobs, "intel"):
                from nbxtensor.backend.install_help import report
                raise RuntimeError("nbxtensor: this Triton has no Intel backend — Intel GPUs compile with Intel's "
                                   "Triton build (triton-xpu).\n" + report())
            knobs.intel.device_arch = arch
            self._target = GPUTarget("xpu", props, 32)
        return self._target

    def block_for(self, metadata):
        return (int(metadata.threads_per_warp) * int(metadata.num_warps), 1, 1)

    def load_options(self, metadata):
        return {"build_flags": str(getattr(metadata, "build_flags", "") or "")}

    def max_shared_memory_per_block(self) -> int:
        return int(self.rt.compute_properties().maxSharedLocalMemory)

    def _module(self, binary: bytes, flags: str):
        mod, log = ctypes.c_void_p(), ctypes.c_void_p()
        buf = ctypes.create_string_buffer(binary, len(binary))
        desc = _ModuleDesc(stype=_ST_MODULE_DESC, format=_MODULE_FORMAT_IL_SPIRV, inputSize=len(binary),
                           pInputModule=ctypes.cast(buf, ctypes.c_void_p), pBuildFlags=flags.encode())
        rc = self.rt.lib.zeModuleCreate(self.rt.context(), self.rt.device(), ctypes.byref(desc), ctypes.byref(mod),
                                        ctypes.byref(log))
        text = self._build_log(log)
        return rc, mod, text

    def _build_log(self, log) -> str:
        if not log.value:
            return ""
        n = ctypes.c_size_t(0)
        self.rt.lib.zeModuleBuildLogGetString(log, ctypes.byref(n), None)
        buf = ctypes.create_string_buffer(n.value or 1)
        self.rt.lib.zeModuleBuildLogGetString(log, ctypes.byref(n), buf)
        self.rt.lib.zeModuleBuildLogDestroy(log)
        return buf.value.decode(errors="replace")

    def _kernel(self, mod, name: str):
        k = ctypes.c_void_p()
        desc = _KernelDesc(stype=_ST_KERNEL_DESC, flags=_KERNEL_FLAG_FORCE_RESIDENCY, pKernelName=name.encode())
        self._check(self.rt.lib.zeKernelCreate(mod, ctypes.byref(desc), ctypes.byref(k)), f"zeKernelCreate({name})")
        props = _KernelProps(stype=_ST_KERNEL_PROPS)
        self._check(self.rt.lib.zeKernelGetProperties(k, ctypes.byref(props)), "zeKernelGetProperties")
        return k, props

    def load(self, binary: bytes, name: str, shared: int, build_flags: str = ""):
        """The kernel `name` of a SPIR-V module, built once per (device, binary, flags). Rebuilt with 256 registers
        per thread when the build fails or the kernel spills past 1000 bytes and no register mode was chosen —
        Intel's own driver does the same (driver.c)."""
        key = (self.rt._current, binary, build_flags)
        mod = self._modules.get(key)
        if mod is None:
            rc, mod, text = self._module(binary, build_flags)
            grf_chosen = "GRF" in build_flags
            if rc != 0 and not grf_chosen:
                rc, mod, text = self._module(binary, f"{build_flags} {_GRF_RETRY_FLAG}".strip())
                grf_chosen = True
            if rc != 0:
                raise RuntimeError(f"NeuroBrix launcher: zeModuleCreate failed ({result_name(rc)}) for {name}: "
                                   f"{text.strip()[:600]}")
            k, props = self._kernel(mod, name)
            if props.spillMemSize > _SPILL_RETRY_BYTES and not grf_chosen:
                self.rt.lib.zeKernelDestroy(k)
                rc, mod2, text = self._module(binary, f"{build_flags} {_GRF_RETRY_FLAG}".strip())
                if rc == 0:
                    self.rt.lib.zeModuleDestroy(mod)
                    mod = mod2
            self._modules[key] = mod
        k, props = self._kernel(mod, name)
        limit = int(self.rt.compute_properties().maxSharedLocalMemory)
        if shared > limit:
            from triton.runtime.errors import OutOfResources
            raise OutOfResources(shared, limit, "shared memory")
        self._kernels[int(k.value)] = {"args": int(props.numKernelArgs), "shared": int(shared), "name": name}
        return k

    def launch(self, function, grid, block, shared: int, stream: int, params,
               names=None, types=None, trailing=None) -> None:
        if trailing is not None:
            raise RuntimeError(f"NeuroBrix launcher: trailing buffers ({trailing!r}) were computed for a kernel the "
                               f"Level Zero driver cannot bind them for — a driver/metadata mismatch, not launched")
        from nbxtensor.memory.allocator import DeviceAllocator
        h = _as_int(function)
        info = self._kernels.get(h)
        if info is None:
            raise RuntimeError(f"NeuroBrix launcher: kernel handle {h:#x} was not loaded by this driver — not launched")
        slm = int(shared) > 0
        expected = info["args"] - (1 if slm else 0)        # the shared-local-memory argument is the driver's, last
        if len(params) != expected:
            raise TypeError(f"NeuroBrix launcher: the kernel declares {expected} parameters, "
                            f"{len(params)} given — refused, not launched")
        for i, (kind, value) in enumerate(params):
            if kind == "ptr" and value and not DeviceAllocator.holds(int(value)):
                what = f"parameter '{names[i]}'" if names and i < len(names) else f"parameter #{i}"
                raise ValueError(f"NeuroBrix launcher: device address {int(value):#x} ({what}) was not "
                                 f"handed out by the allocator — refused, not launched")
        gx, gy, gz = (tuple(int(g) for g in grid) + (1, 1, 1))[:3]
        bx, by, bz = (tuple(int(b) for b in block) + (1, 1, 1))[:3]
        if gx * gy * gz <= 0:
            return
        lib, k = self.rt.lib, ctypes.c_void_p(h)
        with self._launch_lock:
            for i, (kind, value) in enumerate(params):
                cell = _ctypes_slot(kind)
                cell.value = value
                self._check(lib.zeKernelSetArgumentValue(k, ctypes.c_uint32(i), ctypes.c_size_t(ctypes.sizeof(cell)),
                                                         ctypes.byref(cell)), f"zeKernelSetArgumentValue({i})")
            if slm:
                self._check(lib.zeKernelSetArgumentValue(k, ctypes.c_uint32(len(params)),
                                                         ctypes.c_size_t(int(shared)), None),
                            "zeKernelSetArgumentValue(shared local memory)")
            self._check(lib.zeKernelSetGroupSize(k, ctypes.c_uint32(bx), ctypes.c_uint32(by), ctypes.c_uint32(bz)),
                        "zeKernelSetGroupSize")
            count = _GroupCount(gx, gy, gz)
            self._check(lib.zeCommandListAppendLaunchKernel(self.rt.command_list(stream), k, ctypes.byref(count),
                                                            None, ctypes.c_uint32(0), None),
                        "zeCommandListAppendLaunchKernel")
