#!/usr/bin/env python3
"""R33 execution proof — is torch in `sys.modules` at the end of the process?

R33 (engraved 2026-09-05) says: zero torch in the NeuroBrix Triton branch, at
import and at runtime, on every backend, without exception. A green AST gate
proves nobody *wrote* `import torch`; it cannot prove that nothing pulls it in
transitively, from a C++ extension, or on a cold cache. Only running the thing
and looking at `sys.modules` proves that, and that is what this does.

Each case runs in a FRESH process with a COLD compile cache — a warm cache
once hid a torch import behind an already-built kernel, and that is exactly
the kind of miss this file exists to prevent.

    python tools/r33_execution_proof.py [--out FILE]

The last case is deliberately the one that has always been True: Triton's own
`kernel[grid]`, whose C++ argument binder imports torch on every backend. It
is kept in the table as the negative control — a table where every line reads
False and nothing can read True is not a measurement.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

#: (label, body). Each body runs in a fresh interpreter; the harness appends
#: the verdict print. Order is the order of the engine's own startup.
CASES = [
    ("import neurobrix.kernels",
     "import neurobrix.kernels"),

    ("import the Metal allocator",
     "from neurobrix.kernels import metal_device"),

    ("reach the device (DeviceAllocator.get_device)",
     "from neurobrix.kernels.nbx_tensor import DeviceAllocator\n"
     "assert DeviceAllocator.get_device() is not None"),

    ("allocate + H2D + D2H through NBXTensor",
     "import numpy as np\n"
     "from neurobrix.kernels.nbx_tensor import NBXTensor\n"
     "a = np.arange(1024, dtype=np.float32)\n"
     "t = NBXTensor.from_numpy(a)\n"
     "assert np.array_equal(t.numpy(), a)"),

    ("import kernels.ops.matmul FIRST (asks the driver at import)",
     "import neurobrix.kernels.ops.matmul"),

    ("import the whole wrappers module",
     "import neurobrix.kernels.wrappers"),

    ("import the launcher contract + Metal driver",
     "from neurobrix.triton import launcher_contract, metal_driver"),

    ("activate the launcher (driver registry -> Metal)",
     "from neurobrix.kernels import driver_registry, launcher\n"
     "assert driver_registry.activate() == 'metal'\n"
     "assert launcher.is_installed()\n"
     "assert launcher.active_driver() is not None"),

    ("COLD compile a kernel to MSL (our driver)",
     "from neurobrix.triton.metal_driver import compile_to_msl\n"
     "from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel as k\n"
     "# rms_norm is wrapped in @triton.heuristics; peeling that to the\n"
     "# JITFunction is the launcher's job, so use the launcher's own peel.\n"
     "from neurobrix.kernels.launcher import _unwrap_jit\n"
     "k = _unwrap_jit(k)\n"
     "signature = {'input_ptr':'*fp32','weight_ptr':'*fp32',\n"
     "  'output_ptr':'*fp32','batch_dim':'i32','feat_dim':'i32',\n"
     "  'input_batch_stride':'i32','input_feat_stride':'i32',\n"
     "  'output_batch_stride':'i32','output_feat_stride':'i32',\n"
     "  'eps':'fp32','scale_by_weight':'constexpr',\n"
     "  'BLOCK_SIZE_BATCH':'constexpr','BLOCK_SIZE_FEAT':'constexpr'}\n"
     "constexprs = {'scale_by_weight': True, 'BLOCK_SIZE_BATCH': 4,\n"
     "              'BLOCK_SIZE_FEAT': 128}\n"
     "msl, meta = compile_to_msl(k, signature, constexprs)\n"
     "assert 'kernel void' in msl, msl[:200]"),

    ("COLD compile + LAUNCH a real wrapper through OUR launcher",
     "# The whole path the engine actually takes: a wrapper from\n"
     "# wrappers.py, its kernel[grid] intercepted by the launcher, our\n"
     "# specialization, our driver, our dispatch — and the result checked.\n"
     "import numpy as np\n"
     "from neurobrix.kernels import launcher, wrappers\n"
     "from neurobrix.kernels.nbx_tensor import NBXTensor\n"
     "assert launcher.active_driver() is not None, 'no driver registered'\n"
     "x = np.arange(4096, dtype=np.float32)\n"
     "y = np.ones(4096, dtype=np.float32) * 3.0\n"
     "out = wrappers.add(NBXTensor.from_numpy(x), NBXTensor.from_numpy(y))\n"
     "got = out.numpy()\n"
     "assert np.array_equal(got, x + y), (got[:8], (x + y)[:8])"),

    ("the FULL launcher contract checker, cold",
     "import importlib.util, pathlib\n"
     "path = pathlib.Path('tests/unit/triton/test_launcher_contract.py')\n"
     "spec = importlib.util.spec_from_file_location('contract_check', path)\n"
     "mod = importlib.util.module_from_spec(spec)\n"
     "spec.loader.exec_module(mod)\n"
     "drv = mod._DRIVERS['metal']()\n"
     "assert drv is not None, 'no Metal driver on this machine'\n"
     "mod.test_driver_satisfies_the_launcher_contract(drv)"),

    ("launch through TRITON's own kernel[grid] (NEGATIVE CONTROL)",
     "# The component the launcher replaces. Triton's C++ argument binder\n"
     "# imports torch on every backend; this line must read True or the\n"
     "# table above is not measuring anything.\n"
     "from triton._C.libtriton import native_specialize_impl\n"
     "from triton.backends.compiler import BaseBackend\n"
     "native_specialize_impl(BaseBackend, 16, False, True, True)"),
]

_VERDICT = (
    "\nimport sys\n"
    "print('__TORCH__=%s' % ('torch' in sys.modules))\n"
)


def run_case(label: str, body: str, cache_root: Path) -> tuple[bool, str]:
    """Fresh process, cold cache. Returns (torch_present, error_or_empty)."""
    cache = cache_root / label.replace(" ", "_").replace("/", "_")[:60]
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    env["NBX_METAL_CACHE_DIR"] = str(cache / "metal")
    env["XDG_CACHE_HOME"] = str(cache / "xdg")

    proc = subprocess.run(
        [sys.executable, "-c", body + _VERDICT],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=900)

    for line in proc.stdout.splitlines():
        if line.startswith("__TORCH__="):
            return line.split("=", 1)[1] == "True", ""
    tail = (proc.stderr.strip().splitlines() or ["(no output)"])[-1]
    return False, f"CASE DID NOT RUN: {tail[:160]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows, failures = [], []
    with tempfile.TemporaryDirectory(prefix="r33_proof_") as tmp:
        for label, body in CASES:
            torch_present, error = run_case(label, body, Path(tmp))
            if error:
                failures.append(f"{label}: {error}")
            rows.append((label, torch_present, error))
            print(f"  {label:<58} {'ERROR' if error else torch_present}",
                  flush=True)

    width = max(len(label) for label, _, _ in rows)
    lines = [
        "R33 EXECUTION PROOF — is torch in sys.modules at the end of the "
        "process?",
        "Each case: a fresh process AND a cold compile cache.",
        f"generated by tools/r33_execution_proof.py",
        "",
        f"{'step':<{width}} torch",
        "-" * (width + 7),
    ]
    for label, torch_present, error in rows:
        lines.append(f"{label:<{width}} {'ERROR' if error else torch_present}")
    lines.append("")

    owned = [r for r in rows[:-1]]
    clean = all(not t and not e for _, t, e in owned)
    control_fired = rows[-1][1]

    lines.append("Every step NeuroBrix owns: "
                 + ("TORCH-FREE" if clean else "*** R33 VIOLATION ***"))
    lines.append("Negative control (Triton's own binder) reads True: "
                 + ("yes — the table can detect torch"
                    if control_fired
                    else "NO — this table proves nothing, fix the harness"))
    lines.append("")
    lines.append("The ATen branch is torch BY NATURE and is not covered by "
                 "this table, by design.")
    report = "\n".join(lines) + "\n"

    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"written to {args.out}")

    if failures:
        print("CASES THAT DID NOT RUN:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    return 0 if (clean and control_fired and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
