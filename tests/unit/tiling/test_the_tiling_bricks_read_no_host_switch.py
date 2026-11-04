"""The tiling engine and the vendor tiling read no host switch: no environment variable decides what they compute or
print (the supervisor's 3.21 decision, 2026-11-04 13:40 — zero host setting: a fact of the plan or the profile, or
nothing). They read NBX_DISABLE_INPLACE_ADD, NBX_DISABLE_INPLACE_UNARY, NBX_INPLACE_MIN_BYTES, NBX_TRITON_CHAIN_WRAPPER,
NBX_S5_BASELINE_AUDIT, NBX_TILE_CENSUS and NBX_CHAIN_DIAG.

What this test would do if the code were wrong: any `os.environ` / `os.getenv` read in either file — red, naming its
line (seen red on f45a1baf, which read all seven)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src/neurobrix/core/module"


def _env_reads(path: Path):
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv", "putenv"):
            hits.append(f"{path.name}:{node.lineno}")
    return hits


def test_neither_tiling_brick_reads_the_environment():
    hits = [h for f in ("tiling_engine.py", "vendor_tiling.py") for h in _env_reads(ROOT / f)]
    assert not hits, f"a tiling brick reads the environment: {hits}"
