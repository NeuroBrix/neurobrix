#!/usr/bin/env python
"""Capability probes that DECIDE without EXECUTING — the census, repeatable.

THE RULE (register 62, owner 2026-09-16)
----------------------------------------
A capability probe — architecture, shared memory, a dtype's native support, a
backend, anything that decides a code path or decides whether a test runs —
EXECUTES the path it decides and VERIFIES the result. Compiling, linking,
importing or reading an attribute are not evidence.

The instance that engraved it: the Metal fork chose `-std=metal4.1` because a
probe COMPILED under it, and the GPU runtime then rejected the metallib.

WHY THIS FILE EXISTS RATHER THAN A READING
------------------------------------------
The census of 2026-09-16 16:xx read the engine (`src/`) and answered "no
compile-only probe found here" — correctly, for what it read. It did not read
the probes that gate TESTS, and that is where the next one was:
`test_autotune_correctness_screen.py` decided with
`_detect_gpu_backend() is not None`, which answers WHICH BACKEND THIS BUILD CAN
ADDRESS — a fact about the install, not about the machine. It returned "cuda"
on a host with no visible device and the five tests behind it failed at their
first allocation with `cudaErrorNoDevice` instead of skipping. A reading that
answers "none" is exactly the reading to turn into an instrument.

WHAT IT CLASSIFIES, AND WHAT IT CANNOT
---------------------------------------
For every function whose name says it decides a capability, the body is read:

* EXECUTES — it allocates, launches, runs or compares (`empty`, `zeros`,
  `set_device`, `run`, `launch`, a wrapper call, an `assert`/comparison on a
  result). Good.
* DETECTS ONLY — every call it makes is an import, an attribute read, a version
  or a name (`_detect_*`, `.__version__`, `importlib`, `hasattr`, `is_available`
  used alone). This is the shape under hunt.
* READS AN AUTHORITY — it reads the driver's own attributes or a configuration
  the project treats as the authority (`config/vendors`, the profile). NOT a
  defect: the driver answering its own attribute IS the execution for that
  question, and the doctrine says read the authority.

It cannot tell a device query that really executes from one that is answered
from a cache, and it says so rather than pretending: the verdict is a
CANDIDATE list to read, in the register's tradition, not a gate.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A name that announces a capability decision.
DECIDES = ("has_", "_has", "is_available", "available", "supported", "supports",
           "can_", "_cuda", "cuda_", "_gpu", "gpu_", "detect", "capab", "_metal", "metal_")

# Calls that mean the probe RAN something on the thing it is deciding about.
EXECUTES = ("empty", "zeros", "ones", "set_device", "device_count", "malloc", "allocate",
            "run", "launch", "compile_and_run", "synchronize", "matmul", "mm", "conv",
            "kernel", "wrapper", "forward", "decode", "encode", "allclose", "assert_")

# Calls that are a NAME, a version or an import — never evidence about a device.
DETECTS = ("import_module", "find_spec", "hasattr", "getattr", "version", "__version__",
           "_detect_gpu_backend", "which", "environ", "platform", "uname", "exists")

# Reading the driver's or the project's authority is not a compile-only probe.
AUTHORITY = ("get_device_properties", "get_device_capability", "device_attribute",
             "max_shared_memory", "arch", "active_vendor_profile", "profile", "vendor",
             "driver", "nvidia_smi", "nvmlDevice")


def _calls(fn: ast.AST) -> list:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                out.append(f.attr)
            elif isinstance(f, ast.Name):
                out.append(f.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
    return out


# A probe whose work is done by ANOTHER of the project's probes is judged by that
# one, and this census cannot follow it — so it says so rather than counting it.
# The four Metal test probes call `_detect_gpu_backend()`, which OPENS the device on
# that backend (`metal_device_available`: "it opens the real device rather than
# checking for the import") and only loads a library on CUDA. The same call is an
# executing probe on one backend and a naming one on the other, which is exactly
# what a call site cannot see — and why the CUDA-side one cost five tests.
DELEGATES_TO = ("_detect_gpu_backend", "metal_device_available", "_triton_cpu_available",
                "triton_metal_available", "metal_shader_compiler_available",
                "device_count", "_gpu_runtime", "_active_backend")


def classify(fn: ast.FunctionDef) -> tuple:
    names = _calls(fn)
    blob = " ".join(names).lower()
    if any(d in names for d in DELEGATES_TO):
        return "DELEGATES", names
    if any(k in blob for k in (c.lower() for c in EXECUTES)):
        return "EXECUTES", names
    if any(k in blob for k in (c.lower() for c in AUTHORITY)):
        return "READS AN AUTHORITY", names
    if any(k in blob for k in (c.lower() for c in DETECTS)):
        return "DETECTS ONLY", names
    return "UNCLEAR", names


# Vendored upstream kept for reference (CLAUDE.md calls it OBSOLETE) and scratch
# investigations are not this project's probes and are not counted as its findings.
EXCLUDED = ("triton_kernels_ref", "tests/scratch", "__pycache__")


def _is_a_decision(fn: ast.FunctionDef) -> bool:
    """The function ANSWERS a yes/no about a capability — it returns a boolean.

    Without this, the name heuristic collects the graph analysers
    (`_detect_residual_chains`, `detect_and_fuse_moe`, `list_available_profiles`)
    whose names say "detect" and whose subject is a DAG, not a machine. 227 of
    them on the first run, against 32 candidates; a census nobody can read is a
    census nobody reads."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, bool):
                return True
            if isinstance(v, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
                return True
    return False


def scan(roots: list, verbose: bool) -> int:
    findings = {"EXECUTES": [], "DETECTS ONLY": [], "READS AN AUTHORITY": [], "DELEGATES": [], "UNCLEAR": []}
    files = 0
    for root in roots:
        for path in sorted((ROOT / root).rglob("*.py")):
            rel_s = str(path.relative_to(ROOT))
            if any(x in rel_s for x in EXCLUDED):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            files += 1
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                low = node.name.lower()
                if not any(k in low for k in DECIDES):
                    continue
                if not _is_a_decision(node):
                    continue    # a name that says "detect" over a DAG is not a capability probe
                kind, names = classify(node)
                findings[kind].append((path.relative_to(ROOT), node.lineno, node.name, names[:6]))
    print(f"python files read                  : {files}")
    for kind in ("DETECTS ONLY", "UNCLEAR", "DELEGATES", "EXECUTES", "READS AN AUTHORITY"):
        print(f"{kind:<20}: {len(findings[kind])}")
    print()
    for kind in ("DETECTS ONLY", "UNCLEAR", "DELEGATES"):
        if not findings[kind]:
            continue
        print(f"### {kind} — read each one; the name says it decides a capability")
        for rel, line, name, names in findings[kind]:
            print(f"  {rel}:{line}  {name}()")
            print(f"      calls: {', '.join(names) or '(none)'}")
        print()
    if verbose:
        for rel, line, name, names in findings["EXECUTES"]:
            print(f"  ok  {rel}:{line} {name}() — {', '.join(names[:4])}")
    return len(findings["DETECTS ONLY"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", default="src,tests,tools",
                    help="comma-separated directories to read (default: the engine, the suite and the tools)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    # A CENSUS, not a gate: it exits 0 and the reading is its output. Every
    # candidate it lists has to be read against its source before it is a finding —
    # on the first run of this file, eight of eight were benign (three ask about a
    # tokenizer, a cache and a DAG rather than a device; two are honest pre-filters
    # that refuse loudly naming the install; three ask whether a FILE is there).
    # An exit code would turn those eight into an alarm, which is how a census
    # becomes something people stop running.
    scan([r for r in a.roots.split(",") if r], a.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
