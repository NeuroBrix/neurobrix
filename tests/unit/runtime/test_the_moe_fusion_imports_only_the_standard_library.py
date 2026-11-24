"""core/runtime/graph/moe_fusion.py imports nothing of the engine (red line 28): standard library only, so that it can
be kept byte for byte where no engine is installed. Its NeuroTax tokens are given by the caller, from the NeuroTax
engine of the caller's own branch; an entry point called without them is refused by name.

What it would do if the claim were wrong: an import of neurobrix or nbxtensor anywhere in the file, lazy ones included
(red: the walk below names it); a call with no tokens, or with one of the six words missing, fused with tokens of its
own choosing (red: the refusal is asserted)."""
import ast
import sys
from pathlib import Path

import pytest

from neurobrix.core.runtime.graph import moe_fusion as MF
from neurobrix.pytorch import neurotax as pytorch_neurotax

FILE = Path(MF.__file__)


def _imported_roots(tree):
    """Every module the file imports, top level and inside functions alike, as its first dotted segment."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield "." if node.level else node.module.split(".")[0]
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "import_module", "__import__"):
            yield "<dynamic import>"


def test_the_file_imports_only_the_standard_library():
    """Every import of the file, lazy ones included, is a standard-library module."""
    roots = set(_imported_roots(ast.parse(FILE.read_text())))
    assert roots, "the walk found no import at all: the instrument is broken"
    foreign = sorted(r for r in roots if r not in sys.stdlib_module_names)
    assert not foreign, f"{FILE.name} imports outside the standard library: {foreign}"


def test_an_entry_point_called_without_tokens_is_refused_by_name():
    """No tokens, or one vendor word missing: refused, the missing name in the message."""
    with pytest.raises(TypeError, match="tokens"):
        MF.detect_and_fuse_moe({"ops": {}, "execution_order": []}, "llm")
    partial = dict(pytorch_neurotax.moe_tokens())
    partial.pop("down_proj")
    with pytest.raises(ValueError, match="down_proj"):
        MF.detect_and_fuse_moe({"ops": {}, "execution_order": []}, "llm", tokens=partial)


def test_both_branches_give_the_same_six_tokens():
    """The PyTorch and Triton NeuroTax twins hand the fusion the same six tokens."""
    from neurobrix.triton import neurotax as triton_neurotax
    assert pytorch_neurotax.moe_tokens() == triton_neurotax.moe_tokens()
    assert set(pytorch_neurotax.moe_tokens()) == set(MF._VENDOR_WORDS)
