"""A loop bound is a count and stays 32-bit; addressing is 64-bit through the
program ids. The Metal induction lowering refuses a 64-bit `scf.for` bound
(metal-first-light 22b427c); every program id is 64-bit since aa60c5c
(register 58); so a `range`/`while` bound derived from a program id must be
narrowed where it is defined. Twelve sites were, at the merge of 2026-09-16.
This reads the source: a bound named after a program-id-derived scalar that
is not cast to int32 at its definition is refused.

Injection: one `.to(tl.int32)` removed from cumsum's `block_offset` made this
RED; restored, green.

Only a kernel has a loop bound the Metal lowering reads: the scan reads each
`@triton.jit` / `@gluon.jit` body alone, its program-id names its own. A file
also holds host Python (library f6bd82e put `_col2im1d_recipe` beside the
col2im kernels): read file-wide, a name a kernel derived from its program id
tainted the same name in that host code (`rel`, conv_transpose2d.py), which
runs no loop on a device.
"""
import ast
import importlib.util
import re
from pathlib import Path

#: the kernel files: the engine's still to move and the tensor library's (nbxtensor/ops, M1)
OPS_DIRS = (Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "kernels" / "ops",
            Path(importlib.util.find_spec("nbxtensor").origin).parent / "ops")


def _ops_files(pattern="*.py"):
    return sorted((p for d in OPS_DIRS for p in d.glob(pattern)), key=lambda p: p.name)


def _op_file(name):
    found = [d / name for d in OPS_DIRS if (d / name).exists()]
    assert len(found) == 1, (name, found)
    return found[0]


def loop_bounds_from_program_ids(src: str):
    pidvars = set(re.findall(r"^\s*(\w+)\s*=\s*tl\.program_id", src, re.M))
    narrowed = set()
    for _ in range(3):
        for m in re.finditer(r"^\s*(\w+)\s*=\s*(.+)$", src, re.M):
            rhs = m.group(2)
            # a value loaded from memory takes the pointer's element type, not the program id's
            if any(re.search(rf"\b{v}\b", rhs) for v in pidvars) and "tl.arange" not in rhs and "program_id" not in rhs and "tl.load(" not in rhs:
                if ".to(tl.int32)" in rhs:
                    narrowed.add(m.group(1))
                else:
                    pidvars.add(m.group(1))
    hits = []
    for m in re.finditer(r"(?:tl\.range|range)\(([^)]*)\)|while\s+(.+):", src):
        args = m.group(1) or m.group(2)
        bad = [v for v in pidvars - narrowed if re.search(rf"\b{v}\b", args)]
        if bad:
            hits.append((src[:m.start()].count("\n") + 1, args.strip(), bad))
    return hits


def kernel_bodies(src: str):
    """(first line, source) of every function the file compiles for a device: decorated `@triton.jit` or
    `@gluon.jit` (alone or under the autotune and heuristics decorators), nested ones included."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                ast.unparse(d.func if isinstance(d, ast.Call) else d) in ("triton.jit", "gluon.jit", "jit")
                for d in node.decorator_list):
            out.append((node.lineno, ast.get_source_segment(src, node)))
    return out


def test_no_loop_bound_is_a_64_bit_program_id_value():
    hits = []
    for f in _ops_files():
        for first, body in kernel_bodies(f.read_text(encoding="utf-8")):
            for line, args, bad in loop_bounds_from_program_ids(body):
                hits.append(f"{f.name}:{first + line - 1}: {args} — {bad}")
    assert not hits, ("loop bounds derived from a 64-bit program id (the Metal lowering refuses them): narrow the "
                      "bound scalar with .to(tl.int32) where it is defined:\n  " + "\n  ".join(hits))


def test_the_reader_sees_a_bound_and_its_narrowing():
    src = "    pid = tl.program_id(0).to(tl.int64)\n    start = pid * 4\n    for i in range(start, 8):\n        pass\n"
    assert loop_bounds_from_program_ids(src)
    src2 = "    pid = tl.program_id(0).to(tl.int64)\n    start = (pid * 4).to(tl.int32)\n    for i in range(start, 8):\n        pass\n"
    assert not loop_bounds_from_program_ids(src2)


def test_the_scan_reads_every_kernel_and_no_host_code():
    """Every function of the kernel files that names `tl.program_id` in its code is a kernel the scan reads (a kernel
    the decorator test missed would go unread), and a host function's names are not a kernel's."""
    unread = []
    for f in _ops_files():
        src = f.read_text(encoding="utf-8")
        kernels = {first for first, _ in kernel_bodies(src)}
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno not in kernels
                    and any(isinstance(n, ast.Attribute) and n.attr == "program_id" for n in ast.walk(node))):
                unread.append(f"{f.name}:{node.lineno} {node.name}")
    assert not unread, unread
    src = ("@triton.jit\ndef k(n_ptr):\n    rel = tl.program_id(0)\n    for i in range(0, 4):\n        pass\n\n"
           "def host(n):\n    rel = n + 1\n    return [v for v in range(rel, 8)]\n")
    assert not [h for _first, body in kernel_bodies(src) for h in loop_bounds_from_program_ids(body)]
    bad = "@triton.jit\ndef k(n_ptr):\n    pid = tl.program_id(0)\n    end = pid * 4\n    for i in range(0, end):\n        pass\n"
    assert [h for _first, body in kernel_bodies(bad) for h in loop_bounds_from_program_ids(body)]
