"""Kernel-library discipline: no bare `return` in a @triton.jit body.

A bare return at the head of a kernel body is unstructured control flow with
no lowering on every backend (the Mac's Metal census flagged five in
fft_op.py, 2026-09-05); the portable form is a mask on the loads and stores.
This gate refuses any bare return in any @triton.jit function of the engine —
the fixes are proven bit-identical on CUDA by their own tests. Portability
discipline, not optimisation.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def _is_jit(dec: ast.expr) -> bool:
    node = dec.func if isinstance(dec, ast.Call) else dec
    name = ast.unparse(node)
    return name in ("triton.jit", "jit") or name.endswith(".jit")


def bare_returns(root: Path = SRC) -> dict[str, list[tuple[str, int]]]:
    """{file: [(kernel, line), ...]} for every bare return in a @triton.jit body."""
    found: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(root.rglob("*.py")):
        if "triton_kernels_ref" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:               # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and any(_is_jit(d) for d in node.decorator_list)):
                continue
            hits = [(node.name, sub.lineno) for sub in ast.walk(node)
                    if isinstance(sub, ast.Return) and sub.value is None]
            if hits:
                found.setdefault(str(path.relative_to(root)), []).extend(hits)
    return found


def test_no_bare_return_in_any_triton_jit_body():
    found = bare_returns()
    assert not found, (
        "bare `return` inside a @triton.jit body (unstructured control flow, no lowering "
        "on every backend) — replace the exit by a mask on the loads and stores:\n"
        + "\n".join(f"  {f}: " + ", ".join(f"{k}@{l}" for k, l in hits) for f, hits in found.items())
    )


def test_the_gate_is_seen_failing_on_an_injected_bare_return(tmp_path):
    (tmp_path / "k.py").write_text(
        "import triton\n@triton.jit\ndef k(x):\n    if x:\n        return\n    return x\n"
        "@triton.jit\ndef ok(x):\n    return x\n"
        "def plain():\n    return\n")
    assert bare_returns(tmp_path) == {"k.py": [("k", 5)]}
