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
              function (or one of its synonyms, `SYNONYMS`), and a literal table of dtype sizes.

Every finding ends deleted, or stays with its reason in one sentence: the engine's list of them is
`tools/engine_slim_list.tsv` (kind, path, symbol, why it stays), whose header records the nbxtensor release it was scanned
against, the engine's code-line count and the number of findings. The findings count can only go down
(`tests/unit/nbxtensor_boundary/test_the_engine_only_slims.py`); the code-line count is a measure, never a refusal
(ruling 17/11 16:11). The list is the output of `--write-list`, which keeps every reason already written and refuses
to write a findings count above its header. The library read is the INSTALLED package
(the release `pyproject.toml` pins), never a copy inside this repository.

    python tools/engine_slim_scan.py            # the findings, one per line: kind, path, symbol, detail
    python tools/engine_slim_scan.py --lines    # the engine's code-line count
    python tools/engine_slim_scan.py --write-list   # rewrite tools/engine_slim_list.tsv (refused if the findings rose)

Standard library only (ast). A scan, not a proof: its rules are syntactic, so a finding is a question a person answers
(delete it, or write why it stays) — never deleted on the scan's word alone.
"""
from __future__ import annotations

import argparse
import ast
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
#: non-Python files a name is reached from (entry points, registries written as data)
DATA_GLOBS = ("pyproject.toml", "src/**/*.yml", "src/**/*.yaml", "src/**/*.json", "src/**/*.toml")
#: a duplicate body shorter than this many AST nodes is a coincidence of idiom, not a duplicated function
#: (`return self._x`, `raise NotImplementedError`): the smallest body the scan compares
MIN_DUPLICATE_NODES = 25
#: the library's fact modules: every public function there names a fact the engine must read, not recompute
FACT_MODULES = ("dtype", "shape", "memory", "backend/device_facts.py", "ops/_cost.py")
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
        elif line.strip() and not line.startswith("#") and not line.startswith("kind\t"):
            kind, rel, symbol, why = (line.split("\t") + [""])[:4]
            rows[(kind, rel, symbol)] = why
    return head, rows


def write_list(repo: Path = REPO, library: Path = LIBRARY, path: Path = LIST) -> int:
    """Rewrite the list from a fresh scan, every reason already written kept; refused if the findings rose over its
    header (the code-line count is written as a measure, never refused)."""
    library = installed_library(repo, library)
    found = sorted({(k, p, s) for k, p, s, _ in scan(repo, library)})
    lines = engine_lines(repo)
    head, why = read_list(path) if path.exists() else ({}, {})
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
    path.write_text("\n".join(out) + "\n")
    return len(found)


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


def scan(repo: Path = REPO, library: Path = LIBRARY) -> List[Finding]:
    """Every finding of the repository at `repo` against the library package at `library` (the installed one by
    default), sorted (kind, path, symbol)."""
    ENGINE, LIBRARY = repo / "src" / "neurobrix", library
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
    total = defaultdict(int)                       # name -> number of files referring to it
    for p, r in refs.items():
        for n in r.names:
            total[n] += 1
    calls: Dict[str, List[ast.Call]] = defaultdict(list)
    for r in refs.values():
        for k, v in r.calls.items():
            calls[k].extend(v)

    findings: List[Finding] = []

    # 1. dead
    for p, t in eng_trees.items():
        own = _Refs()
        own.visit(t)
        self_refs = _definition_refs(t)
        defs = [(n, None) for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for c in [n for n in t.body if isinstance(n, ast.ClassDef)]:
            defs += [(m, c) for m in c.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node, owner in defs:
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            if _is_registered(node) or name in data_words:
                continue
            # used elsewhere: another file names it; or this file names it outside its own body
            elsewhere = total.get(name, 0) - (1 if name in own.names else 0)
            if elsewhere > 0:
                continue
            body_names = self_refs.get(id(node), set())
            in_file_outside = name in own.names and (name not in body_names or _count_name(t, name) > _count_name(node, name))
            if in_file_outside:
                continue
            symbol = f"{owner.name}.{name}" if owner is not None else name
            findings.append(("dead", _rel(p, repo, library), symbol, "no reference outside its own body (tests excluded)"))

    # 2. duplicate
    groups: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for p, t in list(eng_trees.items()) + [(p, t) for p, t in lib_trees.items() if t is not None]:
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                k = body_key(node)
                if k:
                    groups[k].append((p, node.name))
    for k, members in groups.items():
        if len(members) < 2:
            continue
        for p, name in members:
            if p.is_relative_to(ENGINE):
                others = ", ".join(f"{_rel(q, repo, library)}::{m}" for q, m in members if (q, m) != (p, name))
                findings.append(("duplicate", _rel(p, repo, library), name, f"same body as {others}"))

    # 3. wrapper
    lib_names_by_module: Dict[str, Set[str]] = {}
    for p, t in eng_trees.items():
        mod = _module_name(p, repo, library)
        imported_from_lib: Set[str] = set()
        for n in t.body:
            if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "nbxtensor":
                imported_from_lib.update(a.asname or a.name for a in n.names)
        lib_names_by_module[mod] = imported_from_lib
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = pass_through(node)
                if target:
                    findings.append(("wrapper", _rel(p, repo, library), node.name, f"only hands its parameters to {target}"))
                for param, default in _defaulted(node):
                    if _never_passed(node, param, calls.get(node.name, [])):
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
    return sorted(set(findings))


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the engine's slimming scan")
    ap.add_argument("--lines", action="store_true", help="print the engine's Python line count")
    ap.add_argument("--write-list", action="store_true", help=f"rewrite {LIST.name} (refused if the findings rose)")
    a = ap.parse_args(argv)
    if a.lines:
        print(engine_lines())
        return 0
    if a.write_list:
        print(f"{LIST.name}: {write_list()} findings")
        return 0
    for kind, path, symbol, detail in scan(REPO, installed_library()):
        print(f"{kind}\t{path}\t{symbol}\t{detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
