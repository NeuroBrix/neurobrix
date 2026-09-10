"""No Python `and` / `or` produces a value inside a @triton.jit body.

A user's A40 report carried four copies of the same deprecation warning per run,
from `tril.py:32`, on a non-scalar `and`. Two things were true at once: the
warning names a scheduled breakage — Triton will raise on it — and the construct
was never right in the first place. Python's `and` on two tensors does not
combine them elementwise; it evaluates the truthiness of the first, which is not
what a mask means.

An audit of the kernel library found the same construct in **nineteen** places
across twelve files, not one: the reduce combiners of `all`/`any`, every 2-D
`mask = m < M and n < N`, both triangular kernels, `var`, `prod`, `weight_norm`,
the arg-reductions. One report, one line quoted, a family underneath.

The distinction this test draws is the one that matters: a boolean operator in
the TEST of an `if` on `tl.constexpr` values is scalar and legitimate — five of
those live in `cumsum` and `groupnorm` and are left alone. A boolean operator
producing a VALUE has tensor operands and is the defect.

One trap worth recording, because a careless fix introduces a silent one: `&`
binds TIGHTER than `<`, so rewriting `a < N and mask` as `a < N & mask` parses as
`a < (N & mask)` and computes something else entirely without a word. Every
comparison operand is parenthesised.

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
        "They do not combine tensors elementwise, and Triton will raise on them "
        "in a future version. Use `&` / `|`, and parenthesise every comparison "
        "operand — `&` binds tighter than `<`:\n  " + "\n  ".join(offenders))


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
