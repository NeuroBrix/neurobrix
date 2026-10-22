#!/usr/bin/env python3
"""What a derived census row is derived FROM, measured where the derivation runs — the scope of a re-derivation.

A model's census rows are a function of three things and nothing else:

    the plans     every (mode, request, rung) plan record the derivation read (`derived_census.plan_record`, the
                  engine's own `--explain-plan --json` in a subprocess): hashed whole, so a planner change is
                  judged by the plans it emits, never by the code it touched;
    the code      every function the derivation EXECUTED in its own processes (a global trace function sees each
                  frame's code once, at its call), as the unit of `gate_scope_code.units_of` holding it, hashed in
                  its normal form; with the residue of every module loaded from the tree (`<module>`, each class's
                  `C.<body>`: its constants and its class attributes, executed at import);
    the files     every file the derivation opened for reading (the `open` audit event): the container's graphs,
                  its configs, the hardware profile, the registries — hashed by content.

`record()` writes them (the table's input record, `census_table.replace_inputs`); `static_move()` says the first
code unit or file that moved since, by re-hashing the same names in the current tree. A plan that moved is found
by explaining it again (derived_census `--only-moved`). A function the derivation never executed cannot change
its rows unless executed code or a read file changed to reach it, and both are hashed.
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = (REPO / "src", REPO / "tools")

import gate_scope_code as GC                   # noqa: E402  (tools/ is on the path of every tool)

_codes: set = set()
_reads: set = set()
#: what the run itself writes — the table, its input record, its probes' directory: an output read back is no input
OUTPUTS: list = []
_state = {"hook": False, "on": False}

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _tracer(frame, event, arg):
    # the GLOBAL trace function: called once per new frame ('call'); returning None leaves the frame untraced
    _codes.add(frame.f_code)
    return None


def _audit(event, args):
    if event != "open" or not _state["on"]:
        return
    path, mode, flags = (tuple(args) + (None, None, None))[:3]
    if isinstance(path, int) or path is None:
        return
    if mode is not None:
        if any(c in str(mode) for c in "wax+"):
            return
    elif isinstance(flags, int) and flags & _WRITE_FLAGS:
        return
    try:
        _reads.add(os.fsdecode(path))
    except (TypeError, ValueError):
        pass


def install() -> None:
    """Start measuring in THIS process (and the threads it starts); a forked worker inherits it."""
    if not _state["hook"]:
        sys.addaudithook(_audit)
        _state["hook"] = True
    _state["on"] = True
    sys.settrace(_tracer)
    threading.settrace(_tracer)


def reset() -> None:
    _codes.clear()
    _reads.clear()


def _in_tree(p: str) -> bool:
    try:
        rp = Path(p).resolve()
    except (OSError, RuntimeError):
        return False
    return any(rp.is_relative_to(r) for r in ROOTS)


def take() -> dict:
    """What this process executed and read since `reset()`, in a picklable form: the executed code objects of the
    tree (file, first line, name), the files opened for reading, the tree's modules loaded."""
    codes = sorted({(c.co_filename, c.co_firstlineno, c.co_name) for c in list(_codes) if _in_tree(c.co_filename)})
    mods = sorted({m.__file__ for m in list(sys.modules.values())
                   if getattr(m, "__file__", None) and str(m.__file__).endswith(".py") and _in_tree(m.__file__)})
    return {"codes": codes, "reads": sorted(_reads), "mods": mods}


def merge(parts) -> dict:
    out = {"codes": set(), "reads": set(), "mods": set()}
    for p in parts:
        for k in out:
            out[k].update(tuple(x) if isinstance(x, list) else x for x in p[k])
    return out


# ------------------------------------------------------------------------------------------------------------
# naming and hashing
# ------------------------------------------------------------------------------------------------------------

_parsed: dict = {}


def _units(path: Path) -> tuple:
    """(units {qualname: node}, the function units' line ranges [(first, last, qualname)]) of one source file."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _parsed:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        units = GC.units_of(tree)
        ranges = []
        for q, n in units.items():
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = min([n.lineno] + [d.lineno for d in n.decorator_list])
                ranges.append((first, n.end_lineno, q))
        _parsed[key] = (units, ranges)
    return _parsed[key]


def _unit_sha(node) -> str:
    return hashlib.sha256(GC.dump(GC.normalized(node)).encode()).hexdigest()[:16]


def _rel(path: Path) -> str:
    """A path as the record writes it: relative to the repository (`repo:`) or to the catalogue (`cache:`),
    absolute otherwise."""
    from neurobrix.core.paths import cache_dir
    rp = path.resolve()
    for tag, root in (("repo", REPO), ("cache", Path(cache_dir()).resolve())):
        if rp.is_relative_to(root):
            return f"{tag}:{rp.relative_to(root)}"
    return str(rp)


def _abs(name: str, repo: Path = REPO) -> Path:
    from neurobrix.core.paths import cache_dir
    tag, _, rest = name.partition(":")
    if tag == "repo":
        return repo / rest
    if tag == "cache":
        return Path(cache_dir()) / rest
    return Path(name)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _skip_read(p: Path) -> bool:
    if p.suffix in (".py", ".pyc", ".so", ".pth") or not p.is_file():
        return True
    rp = p.resolve()
    if any(rp == o or rp.is_relative_to(o) for o in (Path(x).resolve() for x in OUTPUTS)):
        return True
    s = str(p)
    if s.startswith(("/proc/", "/sys/", "/dev/")):
        return True
    return any(s.startswith(str(Path(x).resolve()) + os.sep) for x in {sys.prefix, sys.base_prefix})


def units_and_files(measured: dict) -> tuple:
    """({unit name: hash}, {file name: hash}) of a merged measurement."""
    units = {}
    by_file: dict = {}
    for f, line, _name in measured["codes"]:
        by_file.setdefault(f, set()).add(line)
    for f in set(by_file) | set(measured["mods"]):
        path = Path(f)
        if not path.is_file():
            continue
        us, ranges = _units(path)
        rel = _rel(path)
        hit = {q for q in us if q == "<module>" or q.endswith(".<body>")}
        for line in by_file.get(f, ()):
            owners = [(last - first, q) for first, last, q in ranges if first <= line <= last]
            if owners:
                hit.add(min(owners)[1])
        for q in hit:
            units[f"{rel}::{q}"] = _unit_sha(us[q])
    files = {}
    for r in measured["reads"]:
        p = Path(r)
        if not p.is_absolute():
            p = (Path.cwd() / p)
        if _skip_read(p):
            continue
        files[_rel(p)] = _file_sha(p)
    return units, files


def static_move(record: dict, repo: Path = REPO) -> str:
    """The first code unit or file of `record` whose hash moved in the tree at `repo` (this one by default), named
    with why; '' if none."""
    for name, sha in sorted(record.get("units", {}).items()):
        rel, _, q = name.partition("::")
        path = _abs(rel, repo)
        if not path.is_file():
            return f"code {name}: its file is gone"
        us, _ = _units(path)
        if q not in us:
            return f"code {name}: the unit is gone"
        if _unit_sha(us[q]) != sha:
            return f"code {name} changed"
    for name, sha in sorted(record.get("files", {}).items()):
        path = _abs(name, repo)
        if not path.is_file():
            return f"file {name}: gone"
        if _file_sha(path) != sha:
            return f"file {name} changed"
    return ""


def digest(obj) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
