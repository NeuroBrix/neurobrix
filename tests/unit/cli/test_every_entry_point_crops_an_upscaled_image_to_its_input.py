"""Every entry point that pops an upscaler's pre-pad size hands it to the save's crop.

The input processor reflect-pads an image to the model's window (`load_upscale_image`, HAT's 16) and returns the
size before the pad as `_upscale_orig_hw`; the crop to input x scale lives in ONE place, `output_dispatch.save_image`
(`orig_hw=`). `nbx upscale` and the server passed it; `neurobrix run` popped it and dropped it — HAT-S at 150x100
returned 640x448 with the apple's mirrored bottom under it, not the vendor's 600x400 (2026-10-21).

What this test does if the code is wrong: a module whose `.pop("_upscale_orig_hw", ...)` result is discarded, or
never reaches a `save_output(..., orig_hw=...)` call, fails by name (run.py before the fix: discarded).
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
KEY = "_upscale_orig_hw"


def _pops(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == KEY):
            yield node


def test_the_popped_size_reaches_the_save_in_every_module_that_pops_it():
    modules = [p for p in SRC.rglob("*.py") if KEY in p.read_text(encoding="utf-8")]
    assert modules, f"no module mentions {KEY}: the test reads the wrong tree"
    checked = 0
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for call in _pops(tree):
            checked += 1
            stmt = parents[call]
            if isinstance(stmt, ast.keyword) and stmt.arg == "orig_hw":
                continue                       # popped straight into the save's crop (the server)
            assert isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name), \
                f"{path.relative_to(SRC)}:{call.lineno} pops {KEY} and keeps nothing — the crop never sees the size"
            name = stmt.targets[0].id
            uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
            assert uses, f"{path.relative_to(SRC)}:{call.lineno} keeps {KEY} as {name!r} and never reads it"
    assert checked, f"no `.pop({KEY!r}, ...)` found"


def test_run_saves_with_the_size_it_popped():
    tree = ast.parse((SRC / "cli" / "commands" / "run.py").read_text(encoding="utf-8"))
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "save_output"
             and any(k.arg == "orig_hw" for k in n.keywords)]
    assert saves, "neurobrix run saves an image without the size before the window pad"
