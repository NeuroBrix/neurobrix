"""The flash wrapper asks what the device can hold before it launches.

The seam. `largest_tile_within_smem` is worth nothing if the flash path keeps
its constant, so this pins the call itself: it fails again the day someone
re-freezes a tile.

Checked on the source rather than by running the kernel, because running the
defect needs the sm_86 card it was reported on. The predicate's own behaviour is
pinned next door, in test_tile_fits_the_declared_smem.py.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_flash_wrapper_asks_the_device.py
"""
from __future__ import annotations


def test_the_flash_wrapper_consults_the_budget_before_launching():
    """The seam. The predicate is worth nothing if the flash path keeps its
    constant: this fails again the day someone re-freezes one.

    Checked on the source rather than by running the kernel, because running it
    needs the card this defect is about.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
           / "kernels" / "wrappers.py")
    tree = ast.parse(src.read_text(), filename=str(src))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "scaled_dot_product_attention_wrapper"), None)
    assert fn is not None, "the flash wrapper is no longer where this test looks"
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "largest_tile_within_smem" in called or "_largest_tile_within_smem" in called, (
        "the flash wrapper chooses BLOCK_M/BLOCK_N without asking what the "
        "device can hold — the A40 defect, verbatim")
