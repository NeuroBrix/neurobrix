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

The last case deliberately imports torch: it is the DETECTOR CONTROL, and a
table where every line reads False and nothing can read True is not a
measurement. It is a bare `import torch` precisely so that it cannot go
inert — the control before it was Triton's own `kernel[grid]`, whose C++
argument binder imported torch on Triton 3.6, and upstream made the CUDA
driver probe native in 3.7 (triton#9578, #10935). That control then read
False on 3.8 and the table went on printing a verdict, having lost the
ability to detect anything. It is kept, one row above, as an OBSERVATION of
upstream.

Each case declares the machine it needs. A case that does not run on a
machine that should run it is a BROKEN HARNESS, not a violation and not a
pass: silence and success must never be the same reading.
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

#: (label, body) or (label, body, requires). Each body runs in a fresh
#: interpreter; the harness appends the verdict print. Order is the order of
#: the engine's own startup. `requires` names the platform the case needs
#: (`sys.platform`); a case that cannot run HERE is reported apart, and a case
#: that fails to run where it SHOULD turns the table red.
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

    ("install the launcher (its registry resolves the driver)",
     "from neurobrix.kernels import launcher\n"
     "assert launcher.install() is True\n"
     "drv = launcher.active_driver()\n"
     "assert drv is not None and drv.artifact_kind\n"
     "assert launcher.target().backend"),

    ("COLD compile a kernel to MSL (our driver)",
     "from neurobrix.triton.metal_driver import compile_to_msl\n"
     "from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel as k\n"
     "# rms_norm is wrapped in @triton.heuristics. The launcher never\n"
     "# unwraps — it patches JITFunction.run so Triton's own decorators do\n"
     "# their work — but compile_to_msl takes the JITFunction directly, so\n"
     "# this diagnostic peels it here.\n"
     "from triton.runtime.jit import JITFunction\n"
     "while not isinstance(k, JITFunction) and hasattr(k, 'fn'):\n"
     "    k = k.fn\n"
     "signature = {'input_ptr':'*fp32','weight_ptr':'*fp32',\n"
     "  'output_ptr':'*fp32','batch_dim':'i32','feat_dim':'i32',\n"
     "  'input_batch_stride':'i32','input_feat_stride':'i32',\n"
     "  'output_batch_stride':'i32','output_feat_stride':'i32',\n"
     "  'eps':'fp32','scale_by_weight':'constexpr',\n"
     "  'BLOCK_SIZE_BATCH':'constexpr','BLOCK_SIZE_FEAT':'constexpr'}\n"
     "constexprs = {'scale_by_weight': True, 'BLOCK_SIZE_BATCH': 4,\n"
     "              'BLOCK_SIZE_FEAT': 128}\n"
     "msl, meta = compile_to_msl(k, signature, constexprs)\n"
     "assert 'kernel void' in msl, msl[:200]",
     "darwin"),

    ("COLD compile + LAUNCH a real wrapper through the launcher",
     "# The whole path the engine actually takes: a wrapper from\n"
     "# wrappers.py, its kernel[grid] intercepted by the launcher, the\n"
     "# launcher's binder and Triton's compiler, our driver, our dispatch —\n"
     "# and the result checked.\n"
     "import numpy as np\n"
     "from neurobrix.kernels import launcher, wrappers\n"
     "from neurobrix.kernels.nbx_tensor import NBXTensor\n"
     "assert launcher.install() is True\n"
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
     "mod.test_driver_satisfies_the_launcher_contract(drv)",
     "darwin"),

    ("launch through TRITON's own kernel[grid] (OBSERVATION, not the control)",
     "# The component the launcher replaces. On Triton 3.6 this imported torch on\n"
     "# every backend, which is why the replacement exists; on 3.7+ upstream made\n"
     "# the CUDA driver probe native (#9578/#10935) and it no longer does. Kept as\n"
     "# an OBSERVATION of upstream, never again as the control: a control that goes\n"
     "# quiet when upstream changes stops measuring without anyone noticing, which\n"
     "# is what happened between 3.6 and 3.8 (2026-09-17).\n"
     "from triton._C.libtriton import native_specialize_impl\n"
     "from triton.backends.compiler import BaseBackend\n"
     "native_specialize_impl(BaseBackend, 16, False, True, True)"),

    ("import torch on purpose (DETECTOR CONTROL — must read True)",
     "# The control, and it cannot go inert: if THIS reads False the probe is\n"
     "# broken and every False above means nothing.\n"
     "import torch  # noqa: F401"),
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


def build_report(rows: list[tuple[str, bool, str, str | None]],
                 platform: str) -> tuple[str, int]:
    """Turn the measured rows into the report and the exit code.

    Pure, so the verdict can be tested without spending fifteen minutes of card
    time on eight subprocesses. Both defects this function was rewritten for
    (2026-09-17) survived precisely because nothing could reach the verdict
    without running the whole table.

    Three outcomes, kept apart, because folding any two of them together is how
    this table spent weeks printing a violation nobody could act on:
      * torch seen in an owned step        -> R33 VIOLATION
      * a step that should run here didn't -> BROKEN HARNESS (never a pass)
      * the detector control stayed silent -> UNPROVEN (no False means anything)
    """
    width = max(len(label) for label, _, _, _ in rows)
    lines = [
        "R33 EXECUTION PROOF — is torch in sys.modules at the end of the "
        "process?",
        "Each case: a fresh process AND a cold compile cache.",
        "generated by tools/r33_execution_proof.py",
        "",
        f"{'step':<{width}} torch",
        "-" * (width + 7),
    ]
    for label, torch_present, error, requires in rows:
        if error:
            mark = "n/a" if requires and requires != platform else "ERROR"
        else:
            mark = str(torch_present)
        lines.append(f"{label:<{width}} {mark}")
    lines.append("")

    # The observation and the detector control are instruments, not steps the
    # engine owns; an owned step is every other row.
    owned = rows[:-2]
    saw_torch = [lbl for lbl, t, e, _ in owned if t and not e]
    # An ERROR is a step that could not RUN. On a CUDA box the two Metal rows
    # error every single time, and counting that as a torch sighting is what
    # made this table print "*** R33 VIOLATION ***" on BOTH stacks while torch
    # appeared in no owned step at all (measured 2026-09-17, one variable apart).
    # But "could not run" may not be silently benign either, or a step that
    # stops being measured reads exactly like a step that passed — so a case
    # says which platform it needs, and only that case is excused here.
    not_applicable = [lbl for lbl, _, e, req in owned if e and req and req != platform]
    broken = [lbl for lbl, _, e, req in owned if e and not (req and req != platform)]
    control_fired = rows[-1][1]

    if saw_torch:
        verdict, code = ("*** R33 VIOLATION *** — torch in: "
                         + ", ".join(saw_torch)), 1
    elif broken:
        verdict, code = ("*** BROKEN HARNESS *** — a step that should run on "
                         f"{platform} did not: " + ", ".join(broken)), 1
    elif not control_fired:
        verdict, code = ("UNPROVEN — the detector control did not fire, so no "
                         "False above means anything"), 1
    else:
        verdict, code = "TORCH-FREE", 0
    lines.append("Every step NeuroBrix owns: " + verdict)
    if not_applicable:
        lines.append(f"  not applicable on {platform} (declared, not a "
                     "violation): " + ", ".join(not_applicable))
    lines.append("Detector control (a deliberate `import torch`) reads True: "
                 + ("yes — the table can detect torch"
                    if control_fired
                    else "NO — this table proves nothing, fix the harness"))
    lines.append("")
    lines.append("The ATen branch is torch BY NATURE and is not covered by "
                 "this table, by design.")
    return "\n".join(lines) + "\n", code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory(prefix="r33_proof_") as tmp:
        for case in CASES:
            label, body = case[0], case[1]
            requires = case[2] if len(case) > 2 else None
            torch_present, error = run_case(label, body, Path(tmp))
            rows.append((label, torch_present, error, requires))
            mark = ("n/a" if requires and requires != sys.platform else "ERROR") \
                if error else torch_present
            print(f"  {label:<58} {mark}", flush=True)

    report, code = build_report(rows, sys.platform)
    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"written to {args.out}")

    for label, _, error, requires in rows:
        if error and not (requires and requires != sys.platform):
            print(f"CASE THAT SHOULD HAVE RUN DID NOT: {label}: {error}",
                  file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
