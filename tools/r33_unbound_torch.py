#!/usr/bin/env python3
"""List the functions that use `torch.` without binding it — the residue of a
peel: a module whose head import became TYPE_CHECKING-only must bind torch
in every function that still calls it (a local import), or the first call
raises NameError on the compiled branch.

    python tools/r33_unbound_torch.py src/neurobrix/core/runtime/graph_executor.py ...
"""
from __future__ import annotations

import ast
import re
import sys


def check(path):
    src = open(path).read()
    tree = ast.parse(src)
    def _binds(n):
        # `import torch`, `import torch.nn.functional as F` (binds `torch` too)
        return isinstance(n, ast.Import) and any(
            (a.name == "torch" and a.asname in (None, "torch")) or (a.name.startswith("torch.") and a.asname is None)
            for a in n.names)

    head_binds = any(_binds(n) for n in tree.body)
    if head_binds:
        return []
    out = []
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent

    def _enclosing_binds(node):
        p = getattr(node, "_parent", None)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(_binds(n) for n in ast.walk(p)):
                return True
            p = getattr(p, "_parent", None)
        return False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _enclosing_binds(node):
            continue  # a closure over the enclosing function's binding
        body_src = "\n".join(src.split("\n")[node.body[0].lineno - 1:node.end_lineno])
        code = re.sub(r"(\"\"\".*?\"\"\"|'[^'\n]*'|\"[^\"\n]*\")", "", body_src, flags=re.S)  # strings and docstrings out
        uses = [ln for ln in code.split("\n") if not ln.strip().startswith(("import ", "from ")) and re.search(r"(?<![\w.])torch\.(?!Tensor\b|dtype\b|device\b)", ln.split("#")[0])]
        if not uses:
            continue
        binds = any(_binds(n) for n in ast.walk(node)) or any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "torch" for t in n.targets)
            for n in ast.walk(node))
        if not binds:
            out.append((node.name, node.lineno, uses[0].strip()[:80]))
    return out


if __name__ == "__main__":
    bad = 0
    for path in sys.argv[1:]:
        for name, line, use in check(path):
            print(f"{path}:{line} {name}: {use}")
            bad += 1
    print(f"{bad} unbound use(s)")
    sys.exit(1 if bad else 0)
