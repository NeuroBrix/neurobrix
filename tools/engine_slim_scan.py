"""The engine's slimming scan: what the engine carries that it should not, now that it imports nbxtensor.

The rule: the engine imports nbxtensor and keeps only its own job — no duplication, no dead
function, no adaptation layer. This scan lists, in `src/neurobrix` (both branches), four kinds of finding:

  dead        a definition nothing in the repository refers to (tests are not proof of use): a module-level function or
              class, or a method, whose name appears in no other code as a name, an attribute, an imported name, a
              keyword or a string (a string reaches `getattr` and the registries), in no config or packaging file, and
              that no decorator registers;
  duplicate   a function whose normalised body (docstring dropped, its parameters and locals renamed in order of
              appearance) is the body of another function of the engine or of nbxtensor;
  wrapper     a function that only hands its parameters on to another callable, a class that only renames its base,
              a module-level alias of an imported name, a name the engine imports from nbxtensor for other modules to
              import from it (a re-export), and a parameter no call site ever passes (it is always its default);
  fact        engine code computing a fact nbxtensor exposes: a definition carrying the name of a public nbxtensor fact
              function (or one of its synonyms, `SYNONYMS`), and a literal table of dtype sizes;
and, beside them, `pending`: a CALLER PENDING marker the stale rule no longer holds (CALLER PENDING, below).

Every finding ends deleted, or stays with its reason in one sentence: the engine's list of them is
`tools/engine_slim_list.tsv` (kind, path, symbol, why it stays), whose header records the nbxtensor release it was scanned
against, the engine's code-line count and the number of findings. The findings count can only go down
(`tests/unit/nbxtensor_boundary/test_the_engine_only_slims.py`); the code-line count is a measure, never a refusal
(ruling 17/11 16:11). The list is the output of `--write-list`, which keeps every reason already written and refuses
to write a findings count above its header. The library read is the INSTALLED package
(the release `pyproject.toml` pins), never a copy inside this repository.

THE TWO BRANCHES (RED_LINES.md lines 16-22): a doubled brick lives at `src/neurobrix/pytorch/<brick>/` and
`src/neurobrix/triton/<brick>/`. The scan reads that layout by PATH and NAME only, never from a maintained list:
  * a TWIN: a definition in `triton/<brick>/<rel>.py` is the twin of the definition of the same qualified name in
    `pytorch/<brick>/<rel>.py`. A twin is not a duplicate of its original, a reference from the other side of its
    brick does not keep a definition alive, and a twin's finding that its original carries too (same kind, same symbol,
    the side's name in an import read as `pytorch`) is the original's finding: it is folded into it, never counted
    twice. A twin finding its original lacks is its own finding;
  * the BRANCH DOOR: `branch_door.names("<brick>", "<sub>", "<m>", ...)` names the members `<m>` of
    `neurobrix.pytorch.<brick>[.<sub>]` AND of `neurobrix.triton.<brick>[.<sub>]` (only string literals written at
    the call; a package's member followed to the submodule its `__init__` imports it from): a member named through
    the door is alive on both sides.
`scan(..., twin_aware=False)` is the raw scan (twins counted separately, the door ignored): `--counts` prints both.

CALLER PENDING (ruling 24/11 11:24). A definition written one hand-in before its caller is declared IN THE CODE, by
one comment on the line(s) directly above its `def`/`class` (above its decorators, if any), never in the list:

    # CALLER PENDING: <the card id whose hand-in brings its caller> <YYYY-MM-DD>

e.g. `# CALLER PENDING: 1365 2026-11-24`. The card id is digits (the harness's card ids), never all zeros (0000 is a
placeholder); any other `# CALLER PENDING` comment is refused by name. A comment is not code: it is no reference
(the definition stays dead to every other rule) and no code line (the measure is unchanged). A marked definition with
no caller is PENDING, not dead: not a finding, not counted. `--write-list` writes every pending entry (path, symbol,
the hand-in, the date; a twin folded into its original's entry as findings are) into the list's one section
`## CALLER PENDING`, below the findings; the test holds that section to the markers of the code. The section is
EMPTY at the release gate: `--release` exits non-zero while one entry stands.
THE STALE RULE: a pending entry stops being pending, and is a finding of kind `pending` (counted, so the suite turns
red until the marker goes), when
  * its definition HAS a caller (some other code refers to it: the hand-in that brings the caller has landed, the
    marker is left behind), or
  * its FIRST date is more than PENDING_DAYS days before the scan's day (the named hand-in did not bring the caller
    in time: land the caller, or delete the definition), or
  * its marker's date differs from the date the list's CALLER PENDING section recorded for it (a marker is never
    re-dated). The list's section is the record of each entry's first date: `--write-list` writes an entry's date
    once, from the code's marker, and from then on keeps the date it recorded; the scan reads the marker against it.
The first is exact (the scan's own reference rule decides it); the second bounds a promise no caller kept, with a
date the code states and the list keeps, never a git history the tool would have to interpret.

CALL SITES, MODULE-AWARE. The parameter rule of `wrapper` reads the call sites of a MODULE-LEVEL function `f` of
module M as module plus name (`_calls_module`): a bare `f(...)` counts; an attribute call counts only when its receiver
resolves to M through the calling file's imports (`m.f(...)`, `pkg.m.f(...)`), or is the calling module itself
(`sys.modules[__name__].f(...)`, or a name assigned it). `self.f(...)` or `obj.f(...)` of another class never keeps a
parameter of M's `f` alive: a coincidence of names hid `mark(sync)` behind tools/container_judge.py's own `self.mark`.
A method's (or a nested function's) call sites are still read by name: its receiver is an object, not a module.

    python tools/engine_slim_scan.py            # the findings, one per line: kind, path, symbol, detail
    python tools/engine_slim_scan.py --lines    # the engine's code-line count
    python tools/engine_slim_scan.py --counts   # findings and code lines both ways: raw, and twin/door-aware
    python tools/engine_slim_scan.py --write-list   # rewrite tools/engine_slim_list.tsv (refused if the findings rose)
    python tools/engine_slim_scan.py --release  # the release gate: non-zero while a CALLER PENDING entry stands

Standard library only (ast). A scan, not a proof: its rules are syntactic, so a finding is a question a person answers
(delete it, or write why it stays) — never deleted on the scan's word alone.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "src" / "neurobrix"
LIBRARY = Path(importlib.util.find_spec("nbxtensor").origin).resolve().parent    # the installed library
#: the engine's own list of its findings, each with why it stays (its header: the findings count, which only goes
#: down, and the code-line count, a measure)
LIST = Path(__file__).with_name("engine_slim_list.tsv")
#: the code whose references prove a definition is used (tests excluded, by the order); the library's own code (the
#: installed package, read beside these roots) counts: a library that calls an engine definition back is a use the scan
#: must not call dead
#: and the benchmark harness outside the library (benchmarks/) is a reader: 81641a40 deleted classification.is_triton_op,
#: which only the harness called, before benchmarks/ was a root
USAGE_ROOTS = ("src", "tools", "benchmarks")
#: the receiver by which a module names itself (`_module_bindings`)
_SELF_MODULE = "sys.modules[__name__]"
#: non-Python files a name is reached from (entry points, registries written as data)
DATA_GLOBS = ("pyproject.toml", "src/**/*.yml", "src/**/*.yaml", "src/**/*.json", "src/**/*.toml")
#: a duplicate body shorter than this many AST nodes is a coincidence of idiom, not a duplicated function
#: (`return self._x`, `raise NotImplementedError`): the smallest body the scan compares
MIN_DUPLICATE_NODES = 25
#: the library's fact modules: every public function there names a fact the engine must read, not recompute
#: memory/guarded.py is NOT here: guarded() and place() are OPERATIONS of the element-wise proof (a buffer ending at an
#: unmapped guard, a placement before it), not facts; an engine helper that shares the name is no finding (qa 1326-1).
#: A memory module is named here when it holds facts; a new one is added by name, never swept in by its package.
_MEMORY_FACTS = (
    "memory/__init__.py",
    "memory/allocator.py",
    "memory/arena.py",
    "memory/block_upload.py",
    "memory/chunk_rule.py",
    "memory/deferred_frees.py",
    "memory/device_memory.py",
    "memory/device_transfer.py",
    "memory/expert_slots.py",
    "memory/fixed_state.py",
    "memory/host_spill.py",
    "memory/kv_host_pages.py",
    "memory/kv_host_tier.py",
    "memory/kv_session.py",
    "memory/layout.py",
    "memory/memory_pool.py",
    "memory/metal.py",
    "memory/oom_advice.py",
    "memory/paged_kv.py",
    "memory/pages.py",
    "memory/pointer_tables.py",
    "memory/pool.py",
    "memory/shadow.py",
    "memory/slab.py",
    "memory/slot_stores.py",
    "memory/step_stabilizer.py",
    "memory/to_device_census.py",
    "memory/trace_replay.py",
    "memory/vmm.py")
FACT_MODULES = ("dtype", "shape") + _MEMORY_FACTS + ("backend/device_facts.py", "ops/_cost.py")
#: other names under which the engine has written a library fact (read in the engine before this scan, 2026-10-29)
SYNONYMS = {
    "element_size": "itemsize", "bytes_per_element": "itemsize", "dtype_bytes": "itemsize", "dtype_nbytes": "itemsize",
    "num_elements": "numel", "prod_shape": "numel", "shape_numel": "numel",
    "strides_of": "contiguous_strides", "row_major_strides": "contiguous_strides", "default_strides": "contiguous_strides",
    "broadcast_shape": "broadcast_shapes", "broadcasted_shape": "broadcast_shapes",
    "wider_dtype": "binary_result", "promote_dtypes": "binary_result", "result_dtype": "binary_result",
    "result_type": "binary_result",
}
#: dtype names a literal size table is recognised by
DTYPE_NAMES = {"float16", "float32", "float64", "bfloat16", "int8", "int16", "int32", "int64", "uint8", "bool",
               "fp16", "fp32", "fp64", "bf16", "half", "float", "double", "complex64", "complex128"}

Finding = Tuple[str, str, str, str]          # (kind, path relative to the repository, symbol, detail)
#: a CALLER PENDING marker: one comment directly above a definition (the module docstring)
PENDING_PREFIX = re.compile(r"^\s*#\s*CALLER PENDING\b")
#: its one valid form: a card id (digits, never all zeros: 0000 is a placeholder) and a day
PENDING_MARKER = re.compile(r"^\s*#\s*CALLER PENDING:\s*(?P<card>[0-9]+)\s+(?P<date>\d{4}-\d{2}-\d{2})\s*$")
#: the days a CALLER PENDING marker holds after its date; past them the entry is a finding (the stale rule)
PENDING_DAYS = 14
#: the list's one section of pending entries, written by --write-list below the findings
PENDING_SECTION = "## CALLER PENDING"


# ─────────────────────────────── reading the code ───────────────────────────────

def _py_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(p: Path, repo: Path = REPO, library: Path = LIBRARY) -> str:
    """A path of the repository, relative to it; a file of the library, as `nbxtensor/...` (its package's path)."""
    return str(p.relative_to(library.parent) if p.is_relative_to(library) else p.relative_to(repo))


class ScanRefused(RuntimeError):
    """A file the scan must read and cannot (named): a file skipped would read as 'nothing found'."""


def installed_library(repo: Path = REPO, library: Path = LIBRARY) -> Path:
    """The library package the scan reads: the installed release, refused when it resolves inside the repository."""
    if library.is_relative_to(repo):
        raise ScanRefused(f"nbxtensor resolves to {library}, inside this repository: the scan reads the installed "
                          f"release only; install the version pyproject.toml pins and take that path off PYTHONPATH")
    return library


def library_version(library: Path = LIBRARY) -> str:
    """The `__version__` the installed package declares, read from its source (the scan imports nothing of it)."""
    for n in ast.parse((library / "__init__.py").read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets):
            return str(ast.literal_eval(n.value))
    raise ScanRefused(f"{library / '__init__.py'} declares no __version__: the list records the release it was "
                      f"scanned against")


def library_stamp(library: Path = LIBRARY) -> str:
    """What the list records it was scanned against, in the form the engine pins it (tools/library_pin.py): the
    library's commit when the imported one states it (a checkout, a git install), else its released version."""
    spec = importlib.util.spec_from_file_location("library_pin", Path(__file__).with_name("library_pin.py"))
    pin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pin)
    return (pin.installed_commit() if library == LIBRARY else None) or library_version(library)


def read_list(path: Path = LIST) -> Tuple[Dict[str, str], Dict[Tuple[str, str, str], str]]:
    """({header name: value}, {(kind, path, symbol): why it stays}) of the engine's list."""
    head, rows = {}, {}
    for line in path.read_text().splitlines():
        if line.startswith("# ") and ": " in line and " " not in line[2:].split(": ", 1)[0]:
            k, v = line[2:].split(": ", 1)
            head[k] = v
        elif line.strip() and not line.startswith("#") and not line.startswith("kind\t") \
                and not line.startswith("pending\t"):
            kind, rel, symbol, why = (line.split("\t") + [""])[:4]
            rows[(kind, rel, symbol)] = why
    return head, rows


def read_pending(path: Path = LIST) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """{(path, symbol): (the card that brings its caller, the FIRST date the list recorded)} of the list's CALLER
    PENDING section ({} when there is no list)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.startswith("pending\t"):
            _, rel, symbol, handin, date = (line.split("\t") + ["", ""])[:5]
            out[(rel, symbol)] = (handin, date)
    return out


def write_list(repo: Path = REPO, library: Path = LIBRARY, path: Path = LIST,
               today: Optional[datetime.date] = None) -> int:
    """Rewrite the list from a fresh scan, every reason already written kept; refused if the findings rose over its
    header (the code-line count is written as a measure, never refused). The CALLER PENDING section is written from
    the code's markers (`pending`), on the day `today` (default: the day of the scan)."""
    library = installed_library(repo, library)
    recorded = read_pending(path)
    scanned = {(k, p, s) for k, p, s, _ in scan(repo, library, today=today, recorded=recorded)}
    head, why = read_list(path) if path.exists() else ({}, {})
    found = sorted(scanned)
    lines = engine_lines(repo)
    if "findings" in head and len(found) > int(head["findings"]):
        raise ScanRefused(f"{path.name}: findings rose from {head['findings']} to {len(found)}; the engine only slims: "
                          f"delete what was added, never raise the list")
    out = ["# The engine's slimming findings (tools/engine_slim_scan.py --write-list): kind, path, symbol, why it stays.",
           "# A new finding is written into the engine: delete it, or move it into nbxtensor. Rule (ruling 17/11 16:11):",
           "# the findings count, and the scatter count of scatter_list.tsv, only go down; engine_lines is a measure:",
           "# the test prints the current count against it and never fails on it.",
           f"# nbxtensor: {library_stamp(library)}", f"# engine_lines: {lines}", f"# findings: {len(found)}",
           "kind\tpath\tsymbol\twhy it stays"]
    out += ["\t".join((k, p, s, why.get((k, p, s), ""))) for k, p, s in found]
    out += [f"{PENDING_SECTION}: a definition whose caller a named hand-in brings (written from the code's markers,",
            "## never by hand; not a finding; EMPTY at the release gate: --release). pending, path, symbol, hand-in, date"]
    out += ["\t".join(("pending", p, s, h, d))
            for (p, s), (h, d) in sorted(pending(repo, library, today, recorded=recorded).items())]
    path.write_text("\n".join(out) + "\n")
    return len(found)


def pending_marker(lines: List[str], node, where: str = "?") -> Optional[Tuple[str, str]]:
    """(card id, date) of the CALLER PENDING comment directly above `node` (above its decorators), else None: the
    comment block that ends on the line before the definition is read, upward, to its first non-comment line. A
    CALLER PENDING comment not of the form `# CALLER PENDING: <card id: digits> <YYYY-MM-DD>`, or whose card id is
    all zeros (a placeholder), is refused by name."""
    i = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 2
    while i >= 0 and lines[i].lstrip().startswith("#"):
        if PENDING_PREFIX.match(lines[i]):
            m = PENDING_MARKER.match(lines[i])
            if m is None or not m["card"].strip("0"):
                raise ScanRefused(f"{where}:{i + 1}: {lines[i].strip()!r} is not a CALLER PENDING marker: write "
                                  f"`# CALLER PENDING: <card id> <YYYY-MM-DD>`, the card id in digits and not a "
                                  f"placeholder (0000)")
            return m["card"], m["date"]
        i -= 1
    return None


def _definitions(t: ast.Module):
    """The definitions the dead rule reads: (node, owning class or None) — module-level ones and methods."""
    defs = [(n, None) for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for c in [n for n in t.body if isinstance(n, ast.ClassDef)]:
        defs += [(m, c) for m in c.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return defs


def pending(repo: Path = REPO, library: Path = LIBRARY, today: Optional[datetime.date] = None,
            twin_aware: bool = True, recorded: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None
            ) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """{(path, symbol): (hand-in, date)} of the definitions that are CALLER PENDING today: marked, no caller, inside
    PENDING_DAYS of their date (a marker past the stale rule is a finding of `scan`, not an entry here). A twin entry
    its original carries too is folded into the original's, as findings are."""
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    scan(repo, library, twin_aware, today, out, recorded)
    if not twin_aware:
        return out
    twins = Twins(repo / "src" / "neurobrix")
    folded = {}
    for (rel, symbol), entry in out.items():
        orig = twins.original_of(repo / rel)
        if orig is None or (_rel(orig, repo, library), symbol) not in out:
            folded[(rel, symbol)] = entry
    return folded


def _parse(p: Path) -> ast.Module:
    try:
        return ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise ScanRefused(f"{p}: not readable as Python ({type(e).__name__}: {e}) — the scan reads every file of its "
                          f"roots, a skipped one would hide its findings") from None


def _module_name(p: Path, repo: Path = REPO, library: Path = LIBRARY) -> str:
    """`src/neurobrix/core/x.py` -> `neurobrix.core.x` (a package's `__init__` is the package)."""
    for root in (repo / "src", library.parent):
        if p.is_relative_to(root):
            parts = list(p.relative_to(root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
    return _rel(p, repo, library)


def engine_lines(repo: Path = REPO) -> int:
    """The engine's CODE lines under src/neurobrix — no docstring, comment-only or blank line: the brick ratchet's own
    count (`tools/brick_ratchet.code_lines`), one definition for both ratchets, so cutting a catch-all into named
    pieces with their one-sentence docstrings, or writing a reason in a comment, never reads as growth."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("brick_ratchet", Path(__file__).with_name("brick_ratchet.py"))
    bricks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bricks)
    return sum(len(bricks.code_lines(p.read_text(encoding="utf-8"))) for p in _py_files(repo / "src" / "neurobrix"))


def engine_lines_twin_aware(repo: Path = REPO) -> Tuple[int, List[Tuple[str, int, int]]]:
    """(the engine's code lines with every twin file counted only for what it holds past its original's lines, the
    twin files that grew past their original: [(twin path, original's lines, twin's lines)]). The raw count is
    `engine_lines`: the two differ by what the twins repeat of their originals."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("brick_ratchet", Path(__file__).with_name("brick_ratchet.py"))
    bricks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bricks)
    engine = repo / "src" / "neurobrix"
    twins = Twins(engine)
    count = lambda q: len(bricks.code_lines(q.read_text(encoding="utf-8")))
    total, grew = 0, []
    for p in _py_files(engine):
        orig = twins.original_of(p)
        if orig is None:
            total += count(p)
            continue
        n, base = count(p), count(orig)
        total += max(0, n - base)
        if n > base:
            grew.append((_rel(p, repo), base, n))
    return total, grew


# ─────────────────────────────── the two branches ───────────────────────────────

#: the two branches of the engine, each a directory of src/neurobrix holding its side of every doubled brick
SIDES = ("pytorch", "triton")


class Twins:
    """The engine's doubled bricks, read from its tree by path: a brick is doubled when a directory of the same name
    stands under `pytorch/` and under `triton/`; a file `triton/<brick>/<rel>` is the twin of `pytorch/<brick>/<rel>`
    when that file exists. No list: the layout is the relation."""

    def __init__(self, engine: Path, enabled: bool = True):
        self.engine = engine
        here = [engine / s for s in SIDES]
        self.bricks: Set[str] = set()      # disabled (the raw scan): no brick is doubled, no file is a twin
        if enabled and all(d.is_dir() for d in here):
            self.bricks = {d.name for d in here[0].iterdir() if d.is_dir() and d.name != "__pycache__"
                           and (here[1] / d.name).is_dir()}

    def side_of(self, p: Path) -> Optional[Tuple[str, str]]:
        """(side, brick) of a file inside a doubled brick, else None."""
        if not p.is_relative_to(self.engine):
            return None
        parts = p.relative_to(self.engine).parts
        if len(parts) >= 3 and parts[0] in SIDES and parts[1] in self.bricks:
            return parts[0], parts[1]
        return None

    def other_side(self, p: Path) -> Optional[Path]:
        """The directory of the other side of `p`'s brick (None outside a doubled brick)."""
        sb = self.side_of(p)
        if sb is None:
            return None
        return self.engine / SIDES[1 - SIDES.index(sb[0])] / sb[1]

    def original_of(self, p: Path) -> Optional[Path]:
        """For a file of a brick's Triton side, the PyTorch file it twins (same brick, same relative path), when it
        exists; else None."""
        sb = self.side_of(p)
        if sb is None or sb[0] != "triton":
            return None
        orig = self.engine / "pytorch" / p.relative_to(self.engine / "triton")
        return orig if orig.is_file() else None

    def twin_pair(self, p: Path, q: Path) -> bool:
        """`p` and `q` are a twin and its original, in either order."""
        return self.original_of(p) == q or self.original_of(q) == p


#: the receiver text a door call is recognised by (`_branch_door.names`, `branch_door.module`,
#: `importlib.import_module("neurobrix.branch_door").module`)
DOOR = "branch_door"


def door_references(tree: ast.AST) -> Set[Tuple[str, str]]:
    """Every (module, member) a branch-door call names, on BOTH sides: `names(brick, sub, *members)` gives
    (`neurobrix.<side>.<brick>[.<sub>]`, member) for each member; `module(brick, sub)` gives (that module, "").
    Only string literals written at the call are read: a name computed at run time is not resolved."""
    out: Set[Tuple[str, str]] = set()
    for c in ast.walk(tree):
        if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("names", "module")
                and DOOR in ast.unparse(c.func.value)):
            continue
        args = list(c.args)
        if any(isinstance(a, ast.Starred) for a in args):
            continue
        kw = {k.arg: k.value for k in c.keywords if k.arg}
        brick = args[0] if args else kw.get("brick")
        sub = args[1] if len(args) > 1 else kw.get("sub")
        lit = lambda n: n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None
        if lit(brick) is None or (sub is not None and lit(sub) is None):
            continue
        members = [lit(a) for a in args[2:]] if c.func.attr == "names" else [""]
        for side in SIDES:
            mod = f"neurobrix.{side}.{lit(brick)}" + (f".{lit(sub)}" if sub is not None and lit(sub) else "")
            out.update((mod, m) for m in members if m is not None)
    return out


def _package_reexports(trees: Dict[Path, ast.Module], repo: Path, library: Path) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """{(package, exported name): (module it is imported from, its name there)} of the engine's `__init__` files: a
    member a door names on a package is followed to the submodule that defines it."""
    out = {}
    for p, t in trees.items():
        if p.name != "__init__.py":
            continue
        pkg = _module_name(p, repo, library)
        for n in t.body:
            if not isinstance(n, ast.ImportFrom):
                continue
            base = pkg.split(".")[: len(pkg.split(".")) - (n.level - 1)] if n.level else []
            src = ".".join(base + ([n.module] if n.module else [])) if n.level else (n.module or "")
            for a in n.names:
                out[(pkg, a.asname or a.name)] = (src, a.name)
    return out


def _twin_symbol(symbol: str) -> str:
    """A finding's symbol read on the PyTorch side: an import of `neurobrix.triton.<x>` names `neurobrix.pytorch.<x>`
    (a twin imports its own side where its original imports the other)."""
    return symbol.replace("neurobrix.triton.", "neurobrix.pytorch.")


class _Refs(ast.NodeVisitor):
    """Every name a module refers to: identifiers, attributes, imported names, keyword names, string constants."""

    def __init__(self):
        self.names: Set[str] = set()
        self.calls: Dict[str, List[ast.Call]] = defaultdict(list)

    def visit_Name(self, n):
        self.names.add(n.id)

    def visit_Attribute(self, n):
        self.names.add(n.attr)
        self.generic_visit(n)

    def visit_alias(self, n):
        self.names.add(n.name.split(".")[-1])
        if n.asname:
            self.names.add(n.asname)

    def visit_keyword(self, n):
        if n.arg:
            self.names.add(n.arg)
        self.generic_visit(n)

    def visit_Constant(self, n):
        if isinstance(n.value, str):
            self.names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", n.value))

    def visit_Call(self, n):
        f = n.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name:
            self.calls[name].append(n)
        self.generic_visit(n)


def _definition_refs(tree: ast.Module) -> Dict[int, Set[str]]:
    """Per definition node (by id), the names its own body refers to — a definition that only names itself (a recursive
    call) is not used by that."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            r = _Refs()
            for child in node.body:
                r.visit(child)
            out[id(node)] = r.names
    return out


# ─────────────────────────────── the four kinds ───────────────────────────────

def _docstring_dropped(body: List[ast.stmt]) -> List[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


class _Canon(ast.NodeTransformer):
    """Parameters and locals renamed in order of appearance; positions dropped (ast.dump without attributes)."""

    def __init__(self, params: Iterable[str]):
        self.map: Dict[str, str] = {}
        for p in params:
            self._name(p)

    def _name(self, s: str) -> str:
        if s not in self.map:
            self.map[s] = f"v{len(self.map)}"
        return self.map[s]

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Store) or n.id in self.map:
            return ast.copy_location(ast.Name(id=self._name(n.id), ctx=n.ctx), n)
        return n

    def visit_arg(self, n):
        n.arg = self._name(n.arg)
        return n


def _params(fn) -> List[str]:
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] + \
           ([a.vararg.arg] if a.vararg else []) + ([a.kwarg.arg] if a.kwarg else [])


def body_key(fn) -> Optional[str]:
    """The normalised body of `fn`, or None when it is shorter than MIN_DUPLICATE_NODES."""
    body = _docstring_dropped(fn.body)
    if not body:
        return None
    mod = ast.Module(body=[ast.parse(ast.unparse(s)).body[0] for s in body], type_ignores=[])
    mod = _Canon(_params(fn)).visit(mod)
    if sum(1 for _ in ast.walk(mod)) < MIN_DUPLICATE_NODES:
        return None
    return hashlib.sha256(ast.dump(mod, include_attributes=False).encode()).hexdigest()[:16]


def pass_through(fn) -> Optional[str]:
    """The callable `fn` only hands its parameters on to (its whole body, docstring aside, is `return g(params...)` or
    `g(params...)`, every parameter passed once and nothing else computed), or None."""
    body = _docstring_dropped(fn.body)
    if len(body) != 1:
        return None
    st = body[0]
    call = st.value if isinstance(st, (ast.Return, ast.Expr)) else None
    if not isinstance(call, ast.Call):
        return None
    params = [p for p in _params(fn) if p not in ("self", "cls")]
    passed = [a.id for a in call.args if isinstance(a, ast.Name)] + \
             [a.value.id for a in call.args if isinstance(a, ast.Starred) and isinstance(a.value, ast.Name)] + \
             [k.value.id for k in call.keywords if isinstance(k.value, ast.Name)]
    others = [a for a in call.args if not isinstance(a, (ast.Name, ast.Starred, ast.Constant))] + \
             [k for k in call.keywords if not isinstance(k.value, (ast.Name, ast.Constant))]
    if others or sorted(passed) != sorted(params):
        return None
    return ast.unparse(call.func)


def _is_registered(node) -> bool:
    """A definition a decorator registers (anything but the three built-in method decorators) is reached through
    that registry, not by its name."""
    for d in getattr(node, "decorator_list", []):
        name = d.id if isinstance(d, ast.Name) else d.attr if isinstance(d, ast.Attribute) else \
            (d.func.id if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) else
             d.func.attr if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) else None)
        if name not in ("staticmethod", "classmethod", "property", "cached_property", "lru_cache", "wraps",
                        "contextmanager", "abstractmethod"):
            return True
    return False


def _dtype_size_table(node: ast.Dict) -> bool:
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    vals = [v for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, int)]
    return len(set(keys) & DTYPE_NAMES) >= 3 and len(vals) == len(node.values) and len(vals) >= 3


def _library_facts(repo: Path = REPO, library: Path = LIBRARY) -> Dict[str, str]:
    """{public fact function name: its library module} from FACT_MODULES and every `*_cost` declaration."""
    out = {}
    for spec in FACT_MODULES:
        target = library / spec
        if not target.exists():
            continue
        files = [target] if target.suffix == ".py" else _py_files(target)
        for p in files:
            tree = _parse(p)
            if tree is None:
                continue
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
                    out[n.name] = _module_name(p, repo, library)
    for p in (_py_files(library / "ops") if (library / "ops").is_dir() else []):
        tree = _parse(p)
        for n in (tree.body if tree else []):
            if isinstance(n, ast.FunctionDef) and n.name.endswith("_cost") and not n.name.startswith("_"):
                out[n.name] = _module_name(p, repo, library)
    return out


def scan(repo: Path = REPO, library: Path = LIBRARY, twin_aware: bool = True, today: Optional[datetime.date] = None,
         pending_out: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
         recorded: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None) -> List[Finding]:
    """Every finding of the repository at `repo` against the library package at `library` (the installed one by
    default), sorted (kind, path, symbol). `twin_aware` (the default) reads the two branches (the module docstring);
    False gives the raw scan: twins counted separately, the door ignored. `today` (default: the day of the scan) is the
    day the CALLER PENDING stale rule counts from; `pending_out`, when given, receives the pending entries (card, first
    date); `recorded` is the list's CALLER PENDING section (default: the one of `repo`'s list), each entry's FIRST date."""
    today = today or datetime.date.today()
    if recorded is None:
        recorded = read_pending(repo / "tools" / LIST.name)
    ENGINE, LIBRARY = repo / "src" / "neurobrix", library
    twins = Twins(ENGINE, enabled=twin_aware)
    usage_files = [p for r in [repo / u for u in USAGE_ROOTS] + [library] if r.is_dir() for p in _py_files(r)]
    trees = {p: _parse(p) for p in usage_files}
    lib_trees = {p: t for p, t in trees.items() if p.is_relative_to(LIBRARY)}
    eng_trees = {p: t for p, t in trees.items() if p.is_relative_to(ENGINE) and t is not None}

    # every reference of the usage corpus, per file, and the data files' words
    refs: Dict[Path, _Refs] = {}
    for p, t in trees.items():
        if t is not None:
            r = _Refs()
            r.visit(t)
            refs[p] = r
    data_words: Set[str] = set()
    for g in DATA_GLOBS:
        for p in repo.glob(g):
            if p.is_file():
                data_words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", p.read_text(encoding="utf-8", errors="replace")))
    naming: Dict[str, Set[Path]] = defaultdict(set)    # name -> the files referring to it
    for p, r in refs.items():
        for n in r.names:
            naming[n].add(p)
    door: Set[Tuple[str, str]] = set()
    if twin_aware:
        for t in trees.values():
            door |= door_references(t)
        reexp = _package_reexports(eng_trees, repo, library)
        todo = list(door)
        while todo:
            key = todo.pop()
            target = reexp.get(key)
            if target and target not in door:
                door.add(target)
                todo.append(target)
    calls: Dict[str, List[Tuple[Path, ast.Call]]] = defaultdict(list)     # name -> (file, call site)
    for q, r in refs.items():
        for k, v in r.calls.items():
            calls[k].extend((q, c) for c in v)

    bindings: Dict[Path, Dict[str, str]] = {}

    def bound_of(q: Path) -> Dict[str, str]:
        """The import bindings of usage file `q` (`_module_bindings`), read once per file."""
        if q not in bindings:
            bindings[q] = _module_bindings(trees[q], _module_name(q, repo, library))
        return bindings[q]

    findings: List[Finding] = []

    # 1. dead
    for p, t in eng_trees.items():
        own = _Refs()
        own.visit(t)
        self_refs = _definition_refs(t)
        lines = p.read_text(encoding="utf-8").splitlines()
        for node, owner in _definitions(t):
            name = node.name
            symbol = f"{owner.name}.{name}" if owner is not None else name
            marker = pending_marker(lines, node, _rel(p, repo, library))
            if marker is not None:
                orig = twins.original_of(p)
                first = recorded.get((_rel(p, repo, library), symbol)) or \
                    (recorded.get((_rel(orig, repo, library), symbol)) if orig is not None else None)
                _date(marker[1], p, symbol)
                if first is not None and first[1] != marker[1]:
                    findings.append(("pending", _rel(p, repo, library), symbol,
                                     f"its CALLER PENDING marker is dated {marker[1]}, the list recorded it first on "
                                     f"{first[1]}: a marker is never re-dated"))
                    continue
                marker = (marker[0], first[1] if first is not None else marker[1])
            if marker is not None and _is_dead(p, t, node, owner, own, self_refs, naming, door, data_words, twins,
                                               repo, library):
                due = _date(marker[1], p, symbol) + datetime.timedelta(days=PENDING_DAYS)
                if today <= due:
                    if pending_out is not None:
                        pending_out[(_rel(p, repo, library), symbol)] = marker
                else:
                    findings.append(("pending", _rel(p, repo, library), symbol,
                                     f"its CALLER PENDING marker ({marker[0]}, {marker[1]}) is past {due}: no caller "
                                     f"came; land the caller or delete the definition"))
                continue
            if marker is not None:
                findings.append(("pending", _rel(p, repo, library), symbol,
                                 f"its CALLER PENDING marker ({marker[0]}, {marker[1]}) stands and it has a caller: "
                                 f"delete the marker"))
                continue
            if _is_dead(p, t, node, owner, own, self_refs, naming, door, data_words, twins, repo, library):
                findings.append(("dead", _rel(p, repo, library), symbol, "no reference outside its own body (tests excluded)"))

    # 2. duplicate
    groups: Dict[str, List[Tuple[Path, str, str]]] = defaultdict(list)
    for p, t in list(eng_trees.items()) + [(p, t) for p, t in lib_trees.items() if t is not None]:
        for node, qual in _functions_qualified(t):
            k = body_key(node)
            if k:
                groups[k].append((p, node.name, qual))
    for k, members in groups.items():
        if len(members) < 2:
            continue
        for p, name, qual in members:
            if p.is_relative_to(ENGINE):
                # a twin and its original (same qualified name, twin files) are one definition written per branch
                others = [(q, m) for q, m, qq in members if (q, qq) != (p, qual)
                          and not (qq == qual and twins.twin_pair(p, q))]
                if others:
                    findings.append(("duplicate", _rel(p, repo, library), name,
                                     "same body as " + ", ".join(f"{_rel(q, repo, library)}::{m}" for q, m in others)))

    # 3. wrapper
    lib_names_by_module: Dict[str, Set[str]] = {}
    for p, t in eng_trees.items():
        mod = _module_name(p, repo, library)
        imported_from_lib: Set[str] = set()
        for n in t.body:
            if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "nbxtensor":
                imported_from_lib.update(a.asname or a.name for a in n.names)
        lib_names_by_module[mod] = imported_from_lib
        module_level = {id(n) for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = pass_through(node)
                if target:
                    findings.append(("wrapper", _rel(p, repo, library), node.name, f"only hands its parameters to {target}"))
                sites = calls.get(node.name, [])
                if id(node) in module_level:
                    sites = [(q, c) for q, c in sites if _calls_module(c, mod, bound_of(q))]
                for param, default in _defaulted(node):
                    if _never_passed(node, param, [c for _, c in sites]):
                        findings.append(("wrapper", _rel(p, repo, library), f"{node.name}({param})",
                                         f"no call site passes it: always its default {default}"))
            if isinstance(node, ast.ClassDef):
                body = _docstring_dropped(node.body)
                if len(node.bases) == 1 and all(isinstance(s, ast.Pass) for s in body):
                    findings.append(("wrapper", _rel(p, repo, library), node.name, f"only renames {ast.unparse(node.bases[0])}"))
        for n in t.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and isinstance(n.value, (ast.Name, ast.Attribute)) and n.targets[0].id != "__all__":
                rhs = ast.unparse(n.value)
                root = rhs.split(".")[0]
                if root in imported_from_lib or root in _imported_names(t):
                    findings.append(("wrapper", _rel(p, repo, library), n.targets[0].id, f"an alias of {rhs}"))
    # re-exports: a library name another engine file imports from this engine module
    for p, t in eng_trees.items():
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module in lib_names_by_module and n.module.startswith("neurobrix"):
                for a in n.names:
                    if a.name in lib_names_by_module[n.module]:
                        findings.append(("wrapper", _rel(p, repo, library), f"from {n.module} import {a.name}",
                                         "a library name imported through an engine module (a re-export)"))

    # 4. fact
    facts = _library_facts(repo, library)
    for p, t in eng_trees.items():
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bare = node.name.lstrip("_")
                lib = bare if bare in facts else SYNONYMS.get(bare)
                if lib and lib in facts and pass_through(node) is None:
                    findings.append(("fact", _rel(p, repo, library), node.name, f"nbxtensor exposes {facts[lib]}.{lib}"))
            if isinstance(node, ast.Dict) and _dtype_size_table(node):
                findings.append(("fact", _rel(p, repo, library), f"dtype size table at line {node.lineno}",
                                 "nbxtensor.dtype exposes itemsize / itemsize_table"))
    if twin_aware:
        findings = _fold_twins(findings, twins, repo, library)
    return sorted(set(findings))


def _date(text: str, p: Path, symbol: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise ScanRefused(f"{p}: {symbol}'s CALLER PENDING marker dates it {text!r}, not a day (YYYY-MM-DD)") from None


def _is_dead(p, t, node, owner, own, self_refs, naming, door, data_words, twins, repo, library) -> bool:
    """The dead rule (the module docstring): `node` of file `p` is referred to by nothing outside its own body."""
    name = node.name
    if name.startswith("__") and name.endswith("__"):
        return False
    if _is_registered(node) or name in data_words:
        return False
    if owner is None and (_module_name(p, repo, library), name) in door:
        return False                                   # named through the branch door: alive on both sides
    # used elsewhere: another file names it (the other side of its brick is not a use); or this file names it
    # outside its own body
    other = twins.other_side(p)
    if [q for q in naming.get(name, ()) if q != p and not (other and q.is_relative_to(other))]:
        return False
    body_names = self_refs.get(id(node), set())
    return not (name in own.names and (name not in body_names or _count_name(t, name) > _count_name(node, name)))


def _fold_twins(findings: List[Finding], twins: Twins, repo: Path, library: Path) -> List[Finding]:
    """Every twin finding its original carries too (same kind, same symbol read on the PyTorch side) folded into the
    original's: the original's entry covers its twin, never listed twice."""
    keys = {(k, p, _twin_symbol(s)) for k, p, s, _ in findings}
    out = []
    for k, p, s, d in findings:
        orig = twins.original_of(repo / p)
        if orig is not None and (k, _rel(orig, repo, library), _twin_symbol(s)) in keys:
            continue
        out.append((k, p, s, d))
    return out


def _functions_qualified(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
    """Every function of `tree` (nested ones included) with its qualified name (`Class.method`, `outer.inner`)."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    out.append((child, qual))
                walk(child, qual + ".")
            else:
                walk(child, prefix)
    walk(tree, "")
    return out


def _count_name(node, name: str) -> int:
    c = 0
    for n in ast.walk(node):
        if (isinstance(n, ast.Name) and n.id == name) or (isinstance(n, ast.Attribute) and n.attr == name):
            c += 1
    return c


def _imported_names(t: ast.Module) -> Set[str]:
    out = set()
    for n in t.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in n.names)
    return out


def _defaulted(fn) -> List[Tuple[str, str]]:
    a = fn.args
    pos = a.posonlyargs + a.args
    out = [(arg.arg, ast.unparse(d)) for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)]
    out += [(arg.arg, ast.unparse(d)) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    return out


def findings(repo: Path = REPO, library: Path = LIBRARY, path: Path = LIST) -> Set[Tuple[str, str, str]]:
    """What the list must hold: the scan's findings (kind, path, symbol), read against the list's CALLER PENDING dates."""
    return {(k, p, s) for k, p, s, _ in scan(repo, library, recorded=read_pending(path))}


def _module_bindings(t: ast.Module, here: str) -> Dict[str, str]:
    """{a name this file binds: the module (or module member) it names} from its imports, relative ones resolved
    against `here` (the file's module): `import a.b as x` -> x: a.b; `import a.b` -> a.b: a.b; `from a import b` ->
    b: a.b. A module naming itself is read too: `sys.modules[__name__]` is `here`, and so is a name assigned it
    (`me = sys.modules[__name__]`, at any depth of the file)."""
    out = {_SELF_MODULE: here}
    pkg = here.split(".")
    for n in ast.walk(t):
        if isinstance(n, ast.Assign) and ast.unparse(n.value) == _SELF_MODULE:
            out.update((x.id, here) for x in n.targets if isinstance(x, ast.Name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom):
            base = ".".join(pkg[: len(pkg) - n.level] + ([n.module] if n.module else [])) if n.level else n.module or ""
            for a in n.names:
                out[a.asname or a.name] = f"{base}.{a.name}" if base else a.name
    return out


def _calls_module(c: ast.Call, mod: str, bound: Dict[str, str]) -> bool:
    """A call site of a MODULE-LEVEL function of module `mod`: a bare name (`f(...)`), or an attribute whose receiver
    names that module through the file's imports (`m.f(...)`, `pkg.m.f(...)`) or is the calling module itself
    (`sys.modules[__name__].f(...)`, or a name assigned it). `self.f(...)` or `obj.f(...)` of another class is not a
    call of `mod.f`: an attribute call is read as module plus name, never by name alone."""
    if isinstance(c.func, ast.Name):
        return True
    if not isinstance(c.func, ast.Attribute):
        return False
    recv = ast.unparse(c.func.value)
    if recv in bound:
        return bound[recv] == mod
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", recv):
        return False
    root, _, rest = recv.partition(".")
    return root in bound and bound[root] + ("." + rest if rest else "") == mod


def _never_passed(fn, param: str, sites: List[ast.Call]) -> bool:
    """No call site of `fn`'s name passes `param` (by keyword, or positionally at its index); False when there is no
    call site (dead, the first kind) or one forwards *args / **kwargs (unknowable)."""
    if not sites:
        return False
    pos = [x.arg for x in fn.args.posonlyargs + fn.args.args]
    idx = pos.index(param) if param in pos else None
    offset = 1 if pos and pos[0] in ("self", "cls") else 0
    for c in sites:
        if any(isinstance(a, ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords):
            return False
        if any(k.arg == param for k in c.keywords):
            return False
        if idx is not None and len(c.args) + offset > idx:
            return False
    return True


def release_gate(repo: Path = REPO, library: Path = LIBRARY, today: Optional[datetime.date] = None) -> int:
    """`--release`: 0 when no CALLER PENDING entry stands, else 1, every entry printed (the section is EMPTY at the
    release gate)."""
    left = pending(repo, library, today)
    for (path, symbol), (handin, date) in sorted(left.items()):
        print(f"CALLER PENDING\t{path}\t{symbol}\t{handin}\t{date}")
    print(f"release gate: {len(left)} CALLER PENDING entr{'y' if len(left) == 1 else 'ies'}"
          + ("" if not left else ": the section is empty at the release gate (land the caller, or delete the "
             "definition)"))
    return 1 if left else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the engine's slimming scan")
    ap.add_argument("--lines", action="store_true", help="print the engine's Python line count")
    ap.add_argument("--write-list", action="store_true", help=f"rewrite {LIST.name} (refused if the findings rose)")
    ap.add_argument("--release", action="store_true",
                    help="the release gate: exit non-zero while a CALLER PENDING entry stands")
    ap.add_argument("--counts", action="store_true",
                    help="print the findings and code lines both ways: raw (twins counted separately, the door "
                         "ignored) and twin/door-aware")
    a = ap.parse_args(argv)
    if a.lines:
        print(engine_lines())
        return 0
    if a.counts:
        lib = installed_library()
        # counted as the list counts them: distinct (kind, path, symbol)
        raw = {f[:3] for f in scan(REPO, lib, twin_aware=False)}
        aware = {f[:3] for f in scan(REPO, lib)}
        lines, (lines_aware, grew) = engine_lines(), engine_lines_twin_aware()
        by_kind = lambda f: ", ".join(f"{k} {sum(1 for x in f if x[0] == k)}"
                                      for k in ("dead", "duplicate", "wrapper", "fact", "pending"))
        print(f"findings: raw {len(raw)} ({by_kind(raw)}); twin/door-aware {len(aware)} ({by_kind(aware)}); "
              f"the list's header: {read_list()[0].get('findings')}")
        print(f"engine_lines: raw {lines}; twin-aware {lines_aware} (a twin counted past its original's lines); "
              f"the list's header: {read_list()[0].get('engine_lines')}")
        for rel, base, n in grew:
            print(f"twin grew past its original: {rel} {n} code lines (its original: {base})")
        return 0
    if a.write_list:
        print(f"{LIST.name}: {write_list()} findings")
        return 0
    if a.release:
        return release_gate(REPO, installed_library())
    for kind, path, symbol, detail in scan(REPO, installed_library()):
        print(f"{kind}\t{path}\t{symbol}\t{detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
