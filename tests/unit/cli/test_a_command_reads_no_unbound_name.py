"""A refactor that moves a local out of a long command function leaves every later read of it
unbound — and nothing fails until a run reaches that line. 2026-09-29: `request_input_config`
was extracted from `cmd_run` (the one InputConfig the plan and the derived census share); three
reads of `num_frames`, `height` and `width` stayed behind, and every video request died at its
inputs with NameError while `--explain-plan` (which returns before them) stayed green.

This walks each function of the run command for names it reads and nothing binds — not the
function, not an enclosing function, not the module, not the builtins. Injection: the
`height, width, num_frames = ...` line of `cmd_run` removed -> `cmd_run` named, RED."""
import builtins
import symtable
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "cli" / "commands" / "run.py"


def unbound_reads(path: Path) -> dict:
    """{function: [names]} — names a function (or a function nested in it) reads that resolve,
    by Python's own scoping (`symtable`), to the module level, where nothing defines them."""
    top = symtable.symtable(path.read_text(), str(path), "exec")
    module = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    known = module | set(dir(builtins))
    out = {}

    def walk(table, owner):
        for sym in table.get_symbols():
            if sym.is_referenced() and sym.is_global() and sym.get_name() not in known:
                out.setdefault(owner, set()).add(sym.get_name())
        for child in table.get_children():
            walk(child, owner)

    for child in top.get_children():
        if child.get_type() == "function":
            walk(child, child.get_name())
    return {k: sorted(v) for k, v in out.items()}


def test_the_run_command_reads_no_unbound_name():
    assert unbound_reads(SRC) == {}
