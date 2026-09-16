"""A loop bound is a count and stays 32-bit; addressing is 64-bit through the
program ids. The Metal induction lowering refuses a 64-bit `scf.for` bound
(metal-first-light 22b427c); every program id is 64-bit since aa60c5c
(register 58); so a `range`/`while` bound derived from a program id must be
narrowed where it is defined. Twelve sites were, at the merge of 2026-09-16.
This reads the source: a bound named after a program-id-derived scalar that
is not cast to int32 at its definition is refused.

Injection: one `.to(tl.int32)` removed from cumsum's `block_offset` made this
RED; restored, green.
"""
import re
from pathlib import Path

OPS = Path(__file__).resolve().parents[3] / "src/neurobrix/kernels/ops"


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


def test_no_loop_bound_is_a_64_bit_program_id_value():
    hits = []
    for f in sorted(OPS.glob("*.py")):
        for line, args, bad in loop_bounds_from_program_ids(f.read_text(encoding="utf-8")):
            hits.append(f"{f.name}:{line}: {args} — {bad}")
    assert not hits, ("loop bounds derived from a 64-bit program id (the Metal lowering refuses them): narrow the "
                      "bound scalar with .to(tl.int32) where it is defined:\n  " + "\n  ".join(hits))


def test_the_reader_sees_a_bound_and_its_narrowing():
    src = "    pid = tl.program_id(0).to(tl.int64)\n    start = pid * 4\n    for i in range(start, 8):\n        pass\n"
    assert loop_bounds_from_program_ids(src)
    src2 = "    pid = tl.program_id(0).to(tl.int64)\n    start = (pid * 4).to(tl.int32)\n    for i in range(start, 8):\n        pass\n"
    assert not loop_bounds_from_program_ids(src2)
