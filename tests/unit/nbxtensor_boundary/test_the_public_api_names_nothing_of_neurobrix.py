"""The published library names nothing of NeuroBrix (Hocine 2026-10-29 16:28; API.md section 11).

nbxtensor is the alternative to torch and only that. A NeuroBrix notion in its public API — a vendor file, a memory
rung, the certification census, a NeuroBrix hardware profile, a memory class — is either generic in the library (a
measured fact, an optional setting) or goes back to the engine, which applies it through the public API.

Read from the source (`ast`), never imported, so it runs on any machine: every public top-level name (no leading
underscore) of every public module (no private package or module on its path), and every `nbxtensor.host` service.
TODAY'S OFFENDERS are listed below and the list SHRINKS ONLY: a new name is refused, and a name that left the API must
leave the list (an exact match both ways), so the list states what is left of the boundary work. M3's exit: empty.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

LIB = Path(__file__).resolve().parents[3] / "nbxtensor" / "src" / "nbxtensor"
NEUROBRIX_NAMES = re.compile(r"vendor|rung|census|neurobrix|memory_class|profile", re.I)

# What is left (API.md section 11 names each row's fate and owner). Shrinks only.
OFFENDERS = {
    "nbxtensor.host:census",
    "nbxtensor.host:census_table",
    "nbxtensor.host:memory_class_from_profile_id",
    "nbxtensor.host:memory_class_gb",
    "nbxtensor.host:profile_triton_has_fp64",
    "nbxtensor.host:vendor_config",
    "nbxtensor.host:vendor_param",
    "nbxtensor.host:vendors_dir",
    "nbxtensor.ops.autotune_certified.active_profile",
    "nbxtensor.ops.autotune_certified.census_row",
    "nbxtensor.ops.autotune_certified.covered_memory_classes",
    "nbxtensor.ops.autotune_certified.entry_for_memory_class",
    "nbxtensor.ops.autotune_certified.executing_memory_class",
    "nbxtensor.ops.autotune_certified.memory_class_from_profile_id",
    "nbxtensor.ops.autotune_certified.memory_class_gb",
    "nbxtensor.ops.autotune_certified.proof_memory_class",
    "nbxtensor.ops.wrappers.get_hardware_profile",
    "nbxtensor.ops.wrappers.set_hardware_profile",
}


def _public_modules():
    for p in sorted(LIB.rglob("*.py")):
        rel = p.relative_to(LIB.parent)
        if any(part.startswith("_") and part != "__init__.py" for part in rel.parts):
            continue
        yield ".".join(rel.with_suffix("").parts).replace(".__init__", ""), ast.parse(p.read_text())


def _top_level_names(tree):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield n.name
        elif isinstance(n, ast.Assign):
            yield from (t.id for t in n.targets if isinstance(t, ast.Name))


def _host_services():
    tree = ast.parse((LIB / "host.py").read_text())
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == "SERVICES" for t in targets) and isinstance(n.value, ast.Dict):
                return [k.value for k in n.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("nbxtensor/host.py: no literal SERVICES table to read")


def neurobrix_names() -> set:
    found = {f"{mod}.{name}" for mod, tree in _public_modules() for name in _top_level_names(tree)
             if not name.startswith("_") and NEUROBRIX_NAMES.search(name)}
    found |= {f"nbxtensor.host:{s}" for s in _host_services() if NEUROBRIX_NAMES.search(s)}
    return found


def test_no_new_neurobrix_name_enters_the_public_api():
    new = sorted(neurobrix_names() - OFFENDERS)
    assert not new, ("a NeuroBrix name entered nbxtensor's public API — make it generic (a measured fact, an optional "
                     "setting) or keep it in the engine (API.md section 11):\n  " + "\n  ".join(new))


def test_the_list_of_offenders_only_shrinks():
    gone = sorted(OFFENDERS - neurobrix_names())
    assert not gone, ("these names left the public API: delete them from OFFENDERS, the list states what is left:\n  "
                      + "\n  ".join(gone))


def test_the_guard_sees_a_neurobrix_name(tmp_path):
    """The scan is seen to catch: a public `set_rung` in a public module is found."""
    pkg = tmp_path / "nbxtensor"
    (pkg / "memory").mkdir(parents=True)
    (pkg / "host.py").write_text("SERVICES = {'vendor_config': ''}\n")
    (pkg / "memory" / "limits.py").write_text("def set_rung(mb):\n    pass\n\ndef _census():\n    pass\n")
    (pkg / "_private.py").write_text("def vendor_param():\n    pass\n")
    globals()["LIB"], saved = pkg, LIB
    try:
        assert neurobrix_names() == {"nbxtensor.memory.limits.set_rung", "nbxtensor.host:vendor_config"}
    finally:
        globals()["LIB"] = saved
