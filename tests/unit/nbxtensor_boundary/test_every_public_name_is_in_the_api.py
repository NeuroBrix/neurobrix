"""Every name nbxtensor exports is in docs/nbxtensor/API.md, and every README example runs (supervisor 16:42; DESIGN
rule 10: the exported names are the API, everything else is `_private`).

A public name is a top-level name without a leading underscore of a module whose path has no private part — or, where
the module states `__all__`, exactly those names. Read with `ast`, never imported, so it runs on any machine.
TODAY'S UNDOCUMENTED names are listed in docs/nbxtensor/api_undocumented.txt and the list SHRINKS ONLY: a new public
name must be in API.md; a listed name that is documented, made private or removed must leave the list. Most of them
are internals to make private (rule 10), not to document. The README half is nbxtensor/tests/test_readme.py.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "nbxtensor" / "src" / "nbxtensor"
API = REPO / "docs" / "nbxtensor" / "API.md"
LISTED = REPO / "docs" / "nbxtensor" / "api_undocumented.txt"


def public_names(lib: Path = LIB) -> list:
    out = []
    for p in sorted(lib.rglob("*.py")):
        rel = p.relative_to(lib.parent)
        if any(part.startswith("_") and part != "__init__.py" for part in rel.parts):
            continue
        mod = ".".join(rel.with_suffix("").parts).replace(".__init__", "")
        tree = ast.parse(p.read_text())
        stated = None
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
                stated = [str(x) for x in ast.literal_eval(n.value)]
        names = stated if stated is not None else [
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not n.name.startswith("_")]
        out.extend(f"{mod}.{name}" for name in names)
    return out


def undocumented(api_text: str, lib: Path = LIB) -> set:
    return {q for q in public_names(lib) if not re.search(r"\b%s\b" % re.escape(q.rsplit(".", 1)[1]), api_text)}


def _listed() -> set:
    return {l.strip() for l in LISTED.read_text().splitlines() if l.strip() and not l.startswith("#")}


def test_no_new_public_name_without_its_api_entry():
    new = sorted(undocumented(API.read_text()) - _listed())
    assert not new, ("public names absent from docs/nbxtensor/API.md — document them there, or make them private "
                     "(a leading underscore, DESIGN rule 10):\n  " + "\n  ".join(new))


def test_the_list_of_undocumented_names_only_shrinks():
    gone = sorted(_listed() - undocumented(API.read_text()))
    assert not gone, ("documented, private or removed now — delete them from docs/nbxtensor/api_undocumented.txt:\n  "
                      + "\n  ".join(gone))


def test_the_guard_sees_an_undocumented_name(tmp_path):
    lib = tmp_path / "nbxtensor"
    (lib / "ops").mkdir(parents=True)
    (lib / "ops" / "a.py").write_text("def documented(): pass\ndef forgotten(): pass\ndef _inner(): pass\n")
    (lib / "_private.py").write_text("def hidden(): pass\n")
    (lib / "ops" / "b.py").write_text("__all__ = ['stated']\ndef stated(): pass\ndef helper(): pass\n")
    assert undocumented("`documented` and `stated`", lib) == {"nbxtensor.ops.a.forgotten"}
