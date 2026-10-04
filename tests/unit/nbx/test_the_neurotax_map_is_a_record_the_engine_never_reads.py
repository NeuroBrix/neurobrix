"""A component's `neurotax_map.json` is a record (the neurotaxe's rule 2), never a runtime input.

From the single write of 2026-10-04 every container carries one per component ({source key: key}; the supervisor's decision
of 2026-10-04). The compiled loader used to read that file and add each source name as an alias of
its tensor — the partial vocabulary's keys would have come back beside the complete ones (the
finders that scan it by substring would have seen both). No engine module reads the file.

What this would do with the alias path restored: fail, naming the module.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
# The parser's docstring names the file; the container validator checks it is PRESENT (a
# container's completeness), and binds nothing.
ALLOWED = {"nbx/neurotax.py", "core/validators/nbx_validator.py"}


def test_no_engine_module_reads_the_map():
    readers = []
    for p in ROOT.rglob("*.py"):
        if "triton_kernels_ref" in p.parts:
            continue
        code = "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("#"))
        if "neurotax_map" in code and p.relative_to(ROOT).as_posix() not in ALLOWED:
            readers.append(p.relative_to(ROOT).as_posix())
    assert readers == [], f"modules that read neurotax_map.json: {readers}"
