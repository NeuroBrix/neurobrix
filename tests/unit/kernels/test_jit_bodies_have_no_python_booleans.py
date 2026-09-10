"""No Python `and` / `or` produces a value inside a @triton.jit body.

A user's A40 report carried four copies of the same deprecation warning per run,
from `tril.py:32`, on a non-scalar `and`. The warning names a scheduled
breakage: Triton will raise on this construct in a future version.

What the construct does TODAY was checked in the compiler rather than assumed,
and the answer is narrower than it first looks. On Triton 3.6.0,
`code_generator.visit_BoolOp` lowers `and` / `or` on tensors to
`semantic.logical_and` / `logical_or`, which bitcast each operand to `int1` and
then apply the bitwise op — elementwise, both operands, no short-circuit. Every
operand at the nineteen sites was already `int1` (a comparison, or an
accumulator declared `tl.int1`), so the rewrite to `&` / `|` was semantically
NEUTRAL there. It is a forward-compatibility fix, not a behaviour fix, and
saying otherwise would credit it with a correctness it did not deliver.

The one case where `and` really does drop an operand is a `constexpr`: the
generator short-circuits on a value that is not a Triton tensor, and the other
mask disappears with no diagnostic. None of the nineteen had one — which is a
fact about those nineteen, not a property of the construct.

An audit of the kernel library found the same construct in **nineteen** places
across twelve files, not one: the reduce combiners of `all`/`any`, every 2-D
`mask = m < M and n < N`, both triangular kernels, `var`, `prod`, `weight_norm`,
the arg-reductions. One report, one line quoted, a family underneath.

Why this AST gate and not the warning: the warning fires only
`if value.type.is_block()`, so the two scalar `combine_fn` sites never emitted
one, and it fires only for a kernel that actually compiles, so the three sites
no cached container reaches (`aten::argmin`, `aten::min`, `aten::var`) would
stay silent under `-W error` too. Promoting the warning covers 14 of the 19.
Reading the source covers 19 of 19.

The distinction this test draws is the one that matters: a boolean operator in
the TEST of an `if` on `tl.constexpr` values is scalar and legitimate — five of
those live in `cumsum` and `groupnorm` and are left alone. A boolean operator
producing a VALUE has tensor operands and is the defect.

One trap worth recording, because it is the real risk here and a careless fix
plants it: `&` binds TIGHTER than `<`, so rewriting `a < N and mask` as
`a < N & mask` parses as `a < (N & mask)` and computes something else entirely
without a word. On `weight_norm` that reads `col_offset < (N & 1)`, i.e. one
column of N, and the norm comes out wrong by roughly sqrt(N) in silence. Every
comparison operand is parenthesised — and this gate does NOT catch that fault,
because there is no BoolOp left to see. The instrument for it is TTIR equality
against the pre-edit form; the plan is in
`docs/internal/kernel_boolean_validation_plan_2026_09_10.md`.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_jit_bodies_have_no_python_booleans.py
"""
from __future__ import annotations

import ast
from pathlib import Path

OPS = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "kernels" / "ops"


def _jit_functions(tree):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and any(
                "jit" in ast.dump(d) for d in fn.decorator_list):
            yield fn


def test_no_boolean_operator_produces_a_value_in_a_kernel():
    offenders = []
    for path in sorted(OPS.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for fn in _jit_functions(tree):
            in_test = set()
            for node in ast.walk(fn):
                if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    for sub in ast.walk(node.test):
                        if isinstance(sub, ast.BoolOp):
                            in_test.add(id(sub))
            for node in ast.walk(fn):
                if isinstance(node, ast.BoolOp) and id(node) not in in_test:
                    op = "and" if isinstance(node.op, ast.And) else "or"
                    offenders.append(f"{path.name}:{node.lineno} in {fn.name}() — `{op}`")
    assert not offenders, (
        "Python boolean operators producing a value inside a @triton.jit body. "
        "Triton will raise on them in a future version, and on a `constexpr` "
        "operand it silently drops the other one today. Use `&` / `|`, and "
        "parenthesise every comparison operand — `&` binds tighter than `<`:"
        "\n  " + "\n  ".join(offenders))


def test_the_scalar_constexpr_tests_are_left_alone():
    """The control: this must not become a rule that forbids scalar logic. A
    boolean operator in an `if` on constexpr values is legitimate, and some
    exist — if this count reaches zero the test above has started over-reaching."""
    kept = 0
    for path in sorted(OPS.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for fn in _jit_functions(tree):
            for node in ast.walk(fn):
                if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    kept += sum(1 for s in ast.walk(node.test)
                                if isinstance(s, ast.BoolOp))
    assert kept > 0, (
        "no scalar constexpr boolean remains in any kernel — the value-rule has "
        "probably been applied to `if` tests it does not govern")


def test_the_where_condition_is_compared_not_borrowed():
    """`aten::where`'s condition crosses the NBX boundary in a uint8 container,
    so the loaded value is an integer. Triton warns on a non-boolean condition
    today and will raise later."""
    src = (OPS / "where.py").read_text()
    assert "tl.where(cond != 0" in src, (
        "where.py hands tl.where the loaded condition directly; a bool arrives "
        "in a uint8 container, so compare it explicitly")
