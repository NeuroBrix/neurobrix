"""The launch driver of Intel GPUs over Level Zero (`backend.level_zero`'s runtime): the kernels Intel's Triton backend
compiles, launched with the ABI its own driver uses — the sources are in `backend.level_zero`'s docstring. Apart from
that module because it is the launcher's (it imports Triton); detecting the backend does not."""
from __future__ import annotations

import ctypes
import threading
from typing import Dict, Optional, Tuple

from nbxtensor.backend.launcher import ModuleDriver, ctypes_slot
from nbxtensor.backend.level_zero import (EXTENSION_FACTS, IntelDeviceError, LevelZeroRuntime, device_extensions,
                                          device_file_for, result_name, runtime)

#: Intel's own driver (driver.c, `compileLevelZeroObjects`): a module is rebuilt once with 256 registers a thread when
#: its build fails, or when its kernel spills past this many bytes, and no register mode was chosen
GRF_RETRY_FLAG, SPILL_RETRY_BYTES = "-cl-intel-256-GRF-per-thread", 1000


class LevelZeroDriver(ModuleDriver):
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
        self._loaded: Dict[tuple, ctypes.c_void_p] = {}  # ((device, binary, flags), name) -> its kernel, created once
        self._arg_buffers: Dict[Tuple[int, tuple], list] = {}   # (kernel, argument kinds) -> the C cells reused
        self._kernels: Dict[int, dict] = {}              # kernel handle -> its argument count, SLM, name
        self._launch_lock = threading.Lock()             # a kernel's arguments are state of its handle (the spec)
        self._target = None

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise IntelDeviceError(f"NeuroBrix launcher: {what} failed ({result_name(rc)})", what, rc,
                                   device=self.rt.current_device)

    def target(self):
        """`GPUTarget("xpu", <properties>, 32)` as Intel's driver builds it (driver.py `_construct_target`), torch- and
        SYCL-free: the counts Level Zero answers, the extensions the OpenCL ICD answers (`EXTENSION_FACTS`), and the
        architecture name from the device file its PCI id selects — handed to Intel's compiler through its own
        `knobs.intel.device_arch`, so it does not compile its SYCL arch parser. A fact none of them states is left to
        the compiler's own default (`has_16bit_atomics`: False), as Intel's driver leaves it."""
        if self._target is None:
            from triton import knobs
            from triton.backends.compiler import GPUTarget
            if not hasattr(knobs, "intel"):
                from nbxtensor.backend.install_help import report
                raise RuntimeError("nbxtensor: this Triton has no Intel backend — Intel GPUs compile with Intel's "
                                   "Triton build (triton-xpu).\n" + report())
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
                # name, vendor, driver_version and max_num_sub_groups as SYCL's Level Zero adapter states them
                # (intel/llvm unified-runtime level_zero common/device.cpp: the vendor literal "Intel(R) Corporation",
                # Level Zero giving only the id; MAX_NUM_SUB_GROUPS = maxTotalGroupSize / min(subGroupSizes))
                "name": p.name.decode(errors="replace"), "vendor": "Intel(R) Corporation",
                "device_id": int(p.deviceId), "driver_version": self.rt.driver_version(), "arch": arch,
                "gpu_eu_count": int(p.numSlices * p.numSubslicesPerSlice * p.numEUsPerSubslice),
                "gpu_subslice_count": int(p.numSlices * p.numSubslicesPerSlice),
                "max_work_group_size": int(c.maxTotalGroupSize),
                "max_num_sub_groups": int(c.maxTotalGroupSize) // min(sizes),
                "sub_group_sizes": sizes, "local_mem_size": int(c.maxSharedLocalMemory),
                "core_clock_rate": int(p.coreClockRate), "__intel_already_queried_extensions__": True,
            })
            knobs.intel.device_arch = arch
            self._target = GPUTarget("xpu", props, 32)
        return self._target

    def block_for(self, metadata):
        return (int(metadata.threads_per_warp) * int(metadata.num_warps), 1, 1)

    def load_options(self, metadata):
        return {"build_flags": str(metadata.build_flags)}     # Intel's compiler always records it ("" for none)

    def max_shared_memory_per_block(self) -> int:
        return int(self.rt.compute_properties().maxSharedLocalMemory)

    def _kernel(self, mod, name: str):
        """The kernel `name` of `mod` and its facts (declared arguments, spill)."""
        built = self.rt.create_kernel(mod, name)
        self._check(built.result, f"zeKernelCreate({name})")
        facts = self.rt.kernel_properties(built.kernel)
        self._check(facts.result, "zeKernelGetProperties")
        return built.kernel, facts

    def load(self, binary: bytes, name: str, shared: int, build_flags: str = ""):
        """The kernel `name` of a SPIR-V module, built once per (device, binary, flags). Rebuilt with 256 registers
        per thread when the build fails or the kernel spills past 1000 bytes and no register mode was chosen —
        Intel's own driver does the same (driver.c)."""
        key = (self.rt.current_device, binary, build_flags)
        loaded = self._loaded.get((key, name))
        if loaded is not None:
            return loaded
        mod, k = self._modules.get(key), None
        if mod is None:
            built = self.rt.build_module(binary, build_flags)
            grf_chosen = "GRF" in build_flags
            if built.result != 0 and not grf_chosen:
                built = self.rt.build_module(binary, f"{build_flags} {GRF_RETRY_FLAG}".strip())
                grf_chosen = True
            if built.result != 0:
                raise IntelDeviceError(f"NeuroBrix launcher: zeModuleCreate failed ({result_name(built.result)}) for "
                                       f"{name}: {built.log.strip()[:600]}", "zeModuleCreate", built.result,
                                       built.log.strip(), device=self.rt.current_device)
            mod = built.module
            k, props = self._kernel(mod, name)
            if props.spill_bytes > SPILL_RETRY_BYTES and not grf_chosen:
                rebuilt = self.rt.build_module(binary, f"{build_flags} {GRF_RETRY_FLAG}".strip())
                if rebuilt.result == 0:           # the rebuilt module replaces the spilling one, its kernel with it
                    self._check(self.rt.destroy_kernel(k), "zeKernelDestroy")
                    self._check(self.rt.destroy_module(mod), "zeModuleDestroy")
                    mod, k = rebuilt.module, None
            self._modules[key] = mod
        if k is None:
            k, props = self._kernel(mod, name)
        limit = int(self.rt.compute_properties().maxSharedLocalMemory)
        if shared > limit:
            from triton.runtime.errors import OutOfResources
            raise OutOfResources(shared, limit, "shared memory")
        self._kernels[int(k.value)] = {"args": props.declared_args, "shared": int(shared), "name": name}
        self._loaded[(key, name)] = k
        return k

    def launch(self, function, grid, block, shared: int, stream: int, params,
               names=None, types=None, trailing=None) -> None:
        if trailing is not None:
            raise RuntimeError(f"NeuroBrix launcher: trailing buffers ({trailing!r}) were computed for a kernel the "
                               f"Level Zero driver cannot bind them for — a driver/metadata mismatch, not launched")
        from nbxtensor.memory.allocator import DeviceAllocator
        h = int(getattr(function, "value", function) or 0)
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
        k = ctypes.c_void_p(h)
        with self._launch_lock:
            kinds = tuple(kind for kind, _v in params)
            cells = self._arg_buffers.get((h, kinds))
            if cells is None:             # one C cell per argument, built once per (kernel, kinds) as CudaDriver's
                cells = self._arg_buffers[(h, kinds)] = [ctypes_slot(kind) for kind in kinds]
            for i, (cell, (_kind, value)) in enumerate(zip(cells, params)):
                cell.value = value
                self._check(self.rt.set_argument(k, i, ctypes.sizeof(cell), cell), f"zeKernelSetArgumentValue({i})")
            if slm:
                self._check(self.rt.set_argument(k, len(params), int(shared), None),
                            "zeKernelSetArgumentValue(shared local memory)")
            self._check(self.rt.set_group_size(k, bx, by, bz), "zeKernelSetGroupSize")
            self._check(self.rt.append_launch(stream, k, gx, gy, gz), "zeCommandListAppendLaunchKernel")
