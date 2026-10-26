#!/usr/bin/env python3
"""The code side of the computed gate scope (tools/gate_scope.py): what a tree's functions are, which of
them a gate cell's process can reach, and the folds that show two versions of a function equal where the
cell runs them.

    SourceIndex     every module of `src/neurobrix` of one tree, parsed once: its imports, its functions
                    (`f`, `C.m`; a function nested in a function belongs to its parent), its classes, and
                    the residues an import executes (`<module>`, `C.<body>`).
    references()    what a function names: the functions and classes it resolves through the module's
                    imports and attribute chains, the attribute names it reads on objects it cannot type,
                    the modules it imports, the classes and modules it names by string.
    reach()         forward reachability from the cell's entry (rapid type analysis: an attribute read on an
                    untyped object reaches the methods of that name of the classes the reachable code
                    instantiates — or names, decorates, or names by string), with the argv every cell of a
                    mode is launched with folded in first (the tree's own parser parses each cell's argv; an
                    attribute equal across the cells is a constant of that mode).
    Folder          constant substitution, boolean simplification, dead branches, constant propagation and
                    single-use inlining — what makes `if not certified_only and _try_warm_path(args)` a
                    dead call, and `_stop = not _done and decode_stop.requested(n)` (requested() constant
                    False where nothing reachable binds an event) the code it replaced.

Nothing here knows a model, a flow or a file of the engine: the entry, the argv seed and every registry are
data (tools/gate_scope.yml).
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import pickle
from pathlib import Path

# -------------------------------------------------------------------------------------------------------------
# normalization: docstrings, imports and `__main__` blocks out
# -------------------------------------------------------------------------------------------------------------


def _strip_doc(body: list) -> list:
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _is_main_guard(s) -> bool:
    return isinstance(s, ast.If) and isinstance(s.test, ast.Compare) and isinstance(s.test.left, ast.Name) \
        and s.test.left.id == "__name__"


class Norm(ast.NodeTransformer):
    """Docstrings and import statements out (an import binds a name; what the name is used for is in the
    code that uses it — the import's module EXECUTION is tracked by reach()), `if __name__ == "__main__":`
    out (never run by an import), expression statements that are bare constants out (comments-as-strings)."""

    def _body(self, body):
        out = []
        for s in _strip_doc(body):
            if isinstance(s, (ast.Import, ast.ImportFrom)) or _is_main_guard(s):
                continue
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
                continue
            out.append(self.visit(s))
        return out or [ast.Pass()]

    def generic_visit(self, node):
        for field, v in ast.iter_fields(node):
            if field in ("body", "orelse", "finalbody") and isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                setattr(node, field, self._body(v))
            elif isinstance(v, list):
                setattr(node, field, [self.visit(x) if isinstance(x, ast.AST) else x for x in v])
            elif isinstance(v, ast.AST):
                setattr(node, field, self.visit(v))
        return node


def normalized(node):
    return Norm().visit(copy.deepcopy(node))


def dump(node) -> str:
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def module_name(rel: str) -> str:
    """src/neurobrix/a/b.py -> neurobrix.a.b ; src/neurobrix/a/__init__.py -> neurobrix.a"""
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# -------------------------------------------------------------------------------------------------------------
# units: the functions of a module
# -------------------------------------------------------------------------------------------------------------

def units_of(tree: ast.Module) -> dict:
    """{qualname: node}: every function (`f`, `C.m`, `C.D.m`), each class's residue (`C.<body>`: its
    statements that are not functions or classes, its bases and decorators) and the module's residue
    (`<module>`: its statements that are not functions or classes, and the decorators of every function —
    they execute at import; a class is its own `C.<body>` unit)."""
    out = {}

    def walk(body, prefix):
        rest = []
        for s in body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + s.name] = s
                if s.decorator_list:
                    rest.append(ast.Expr(ast.Tuple(elts=list(s.decorator_list), ctx=ast.Load())))
            elif isinstance(s, ast.ClassDef):
                residue = [x for x in s.body if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                   ast.ClassDef))]
                walk(s.body, prefix + s.name + ".")
                out[prefix + s.name + ".<body>"] = ast.Module(
                    body=[ast.Expr(ast.Tuple(elts=list(s.bases) + [k.value for k in s.keywords]
                                             + list(s.decorator_list), ctx=ast.Load()))] + residue,
                    type_ignores=[])
            else:
                rest.append(s)
        return rest

    out["<module>"] = ast.Module(body=walk(tree.body, ""), type_ignores=[])
    return out


# -------------------------------------------------------------------------------------------------------------
# the index of one tree
# -------------------------------------------------------------------------------------------------------------

class ModInfo:
    __slots__ = ("name", "rel", "tree", "imports", "defs", "classes", "bases", "is_pkg", "globals_")

    def __init__(self, name, rel, tree, is_pkg):
        self.name, self.rel, self.tree, self.is_pkg = name, rel, tree, is_pkg
        self.imports = {}      # local name -> ("mod", dotted) | ("sym", dotted, attr) | ("ext", dotted)
        self.defs = {}         # qual -> FunctionDef | ClassDef
        self.classes = {}      # qual -> ClassDef
        self.bases = {}        # class qual -> [base expr]
        self.globals_ = set()  # names bound at module level by assignment


def _import_targets(node, pkg_parts: list, is_pkg: bool) -> list:
    """[(local name, kind, dotted, attr|None)] of one import statement; the modules it executes are the
    dotted prefixes of every target."""
    out = []
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.asname:
                out.append((a.asname, "mod", a.name, None))
            else:
                out.append((a.name.split(".")[0], "mod", a.name.split(".")[0], None))
                out.append((None, "mod", a.name, None))          # executes a.b.c too
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            base = pkg_parts if is_pkg else pkg_parts[:-1]
            base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
            mod = ".".join(base + ([node.module] if node.module else []))
        else:
            mod = node.module or ""
        for a in node.names:
            if a.name == "*":
                out.append(("*", "star", mod, None))
            else:
                out.append((a.asname or a.name, "sym", mod, a.name))
    return out


class SourceIndex:
    """Every module of a tree's `src/` package (the package named by `package`)."""

    def __init__(self, src: Path, package: str = "neurobrix"):
        self.src = Path(src)
        self.mods: dict = {}
        for f in sorted((self.src / package).rglob("*.py")):
            rel = str(f.relative_to(self.src.parent))
            try:
                tree = ast.parse(f.read_text(errors="replace"))
            except SyntaxError:
                continue
            name = module_name(rel)
            mi = ModInfo(name, rel, tree, f.name == "__init__.py")
            self.mods[name] = mi
        for mi in self.mods.values():
            self._scan(mi)
        # method name -> [def id] ; class simple name -> [class id]
        self.methods_by_name: dict = {}
        self.classes_by_name: dict = {}
        self.funcs_by_name: dict = {}
        for mi in self.mods.values():
            for q, n in mi.defs.items():
                did = f"{mi.name}:{q}"
                if isinstance(n, ast.ClassDef):
                    self.classes_by_name.setdefault(n.name, []).append(did)
                elif "." in q and q.rsplit(".", 1)[0] in mi.classes:
                    self.methods_by_name.setdefault(n.name, []).append(did)
                else:
                    self.funcs_by_name.setdefault(n.name, []).append(did)
        self._mro_cache = {}

    # -- scanning ------------------------------------------------------------------------------------------
    def _scan(self, mi: ModInfo) -> None:
        pkg_parts = mi.name.split(".")
        for node in ast.walk(mi.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and self._at_module_level(mi.tree, node):
                for local, kind, dotted, attr in _import_targets(node, pkg_parts, mi.is_pkg):
                    if local is None or local == "*":
                        continue
                    mi.imports[local] = (kind, dotted, attr)

        def walk(body, prefix):
            for s in body:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mi.defs[prefix + s.name] = s
                elif isinstance(s, ast.ClassDef):
                    mi.defs[prefix + s.name] = s
                    mi.classes[prefix + s.name] = s
                    mi.bases[prefix + s.name] = list(s.bases)
                    walk(s.body, prefix + s.name + ".")
                elif not prefix:
                    for t in ast.walk(s):
                        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                            mi.globals_.add(t.id)
                    if isinstance(s, (ast.If, ast.Try)):
                        walk(getattr(s, "body", []), prefix)
                        walk(getattr(s, "orelse", []), prefix)
                        for h in getattr(s, "handlers", []):
                            walk(h.body, prefix)
        walk(mi.tree.body, "")

    @staticmethod
    def _at_module_level(tree, node) -> bool:
        for s in tree.body:
            if s is node:
                return True
            if isinstance(s, (ast.If, ast.Try)):
                for sub in ast.walk(s):
                    if sub is node:
                        return not any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                       for x in ast.walk(s) if x is not s and _contains(x, node))
        return False

    # -- resolution ----------------------------------------------------------------------------------------
    def module_exists(self, dotted: str) -> bool:
        return dotted in self.mods

    def resolve_in_module(self, mod: str, name: str, depth: int = 0):
        """What `mod.name` is: ("def", id) | ("mod", dotted) | None — following re-exports."""
        if depth > 12:
            return None
        mi = self.mods.get(mod)
        if mi is None:
            return None
        if name in mi.defs:
            return ("def", f"{mod}:{name}")
        sub = f"{mod}.{name}"
        if sub in self.mods and name not in mi.imports:
            return ("mod", sub)
        imp = mi.imports.get(name)
        if imp:
            return self._resolve_import(imp, depth + 1)
        if sub in self.mods:
            return ("mod", sub)
        return None

    def _resolve_import(self, imp, depth=0):
        kind, dotted, attr = imp
        if kind == "mod":
            return ("mod", dotted) if dotted in self.mods else None
        if kind == "sym":
            sub = f"{dotted}.{attr}"
            if sub in self.mods:
                return ("mod", sub)
            return self.resolve_in_module(dotted, attr, depth)
        return None

    def class_mro(self, cid: str) -> list:
        """[class id] of a class and its bases resolved in the index (depth first, each once); a base the
        index cannot resolve is marked by the id '<external>'."""
        if cid in self._mro_cache:
            return self._mro_cache[cid]
        out, seen = [], set()

        def visit(c, depth=0):
            if c in seen or depth > 30:
                return
            seen.add(c)
            out.append(c)
            mod, q = c.split(":", 1)
            mi = self.mods.get(mod)
            for b in (mi.bases.get(q, []) if mi else []):
                r = self.resolve_expr(mod, b, {})
                if r and r[0] == "def" and isinstance(self.node(r[1]), ast.ClassDef):
                    visit(r[1], depth + 1)
                elif not (isinstance(b, ast.Name) and b.id == "object"):
                    if "<external>" not in out:
                        out.append("<external>")
        visit(cid)
        self._mro_cache[cid] = out
        return out

    def node(self, did: str):
        mod, q = did.split(":", 1)
        mi = self.mods.get(mod)
        return mi.defs.get(q) if mi else None

    def resolve_expr(self, mod: str, expr, local_imports: dict):
        """A Name or an attribute chain on a Name: ("def", id) | ("mod", dotted) | None."""
        chain = []
        e = expr
        while isinstance(e, ast.Attribute):
            chain.append(e.attr)
            e = e.value
        if not isinstance(e, ast.Name):
            return None
        chain.reverse()
        r = None
        if e.id in local_imports:
            r = self._resolve_import(local_imports[e.id])
        if r is None:
            r = self.resolve_in_module(mod, e.id)
        for a in chain:
            if r is None:
                return None
            if r[0] == "mod":
                r = self.resolve_in_module(r[1], a)
            else:
                n = self.node(r[1])
                if isinstance(n, ast.ClassDef):
                    hit = None
                    for c in self.class_mro(r[1]):
                        if c == "<external>":
                            continue
                        m, q = c.split(":", 1)
                        if f"{q}.{a}" in self.mods[m].defs:
                            hit = ("def", f"{m}:{q}.{a}")
                            break
                    r = hit
                else:
                    return None
        return r


def _contains(tree, node) -> bool:
    return any(x is node for x in ast.walk(tree))


def load_index(src: Path, store: Path, package: str = "neurobrix") -> SourceIndex:
    """The index of a tree, pickled under the scope's scratch keyed by the tree's file stamps."""
    files = sorted((Path(src) / package).rglob("*.py"))
    h = hashlib.sha256()
    for f in files:
        st = f.stat()
        h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    key = h.hexdigest()[:20]
    p = Path(store) / f"index_{key}.pkl"
    # The pickle is this tool's own cache, written by it under the scope's scratch from a tree it parsed
    # (never a downloaded or shared file).
    if p.is_file():
        try:
            return pickle.loads(p.read_bytes())
        except Exception:                           # noqa: BLE001 — a stale pickle is rebuilt
            pass
    idx = SourceIndex(src, package)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(pickle.dumps(idx, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(p)
    return idx


# -------------------------------------------------------------------------------------------------------------
# folding
# -------------------------------------------------------------------------------------------------------------

_BOOL_FUNCS = {"isinstance", "issubclass", "bool", "callable", "hasattr"}
#: builtins whose call has no effect beyond its value (the purity of an argument a fold drops)
PURE_BUILTINS = {"len", "isinstance", "issubclass", "bool", "callable", "hasattr", "id", "type"}


def is_pure(e) -> bool:
    """An expression whose evaluation has no effect: names, constants, attribute reads, comparisons and
    boolean/arithmetic of those, calls to the pure builtins."""
    for n in ast.walk(e):
        if isinstance(n, (ast.Call,)):
            if not (isinstance(n.func, ast.Name) and n.func.id in PURE_BUILTINS):
                return False
        elif isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Lambda)):
            return False
    return True


def bool_valued(e) -> bool:
    if isinstance(e, ast.Constant):
        return isinstance(e.value, bool)
    if isinstance(e, ast.Compare):
        return True
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return True
    if isinstance(e, ast.BoolOp):
        return all(bool_valued(v) for v in e.values)
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in _BOOL_FUNCS:
        return True
    return False


def _const(e):
    return isinstance(e, ast.Constant)


_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Is: lambda a, b: a is b,
        ast.IsNot: lambda a, b: a is not b, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


class Folder(ast.NodeTransformer):
    """Substitutes the constants a cell's context gives (`subst(node) -> ast.Constant | None`), then folds:
    constant comparisons, `not c`, boolean operators with constant operands (an operand is dropped only when
    what is left keeps its value and no effect is lost), `if c:` to its live branch, `x if c else y`."""

    def __init__(self, subst=None):
        self.subst = subst
        self.applied = []

    def visit(self, node):
        if self.subst is not None and isinstance(node, ast.expr):
            r = self.subst(node)
            if r is not None:
                self.applied.append(dump(node)[:160])
                return r
        return super().visit(node)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and _const(node.operand):
            return ast.Constant(not node.operand.value)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and _const(node.left) and _const(node.comparators[0]) and type(node.ops[0]) in _CMP:
            try:
                return ast.Constant(bool(_CMP[type(node.ops[0])](node.left.value, node.comparators[0].value)))
            except Exception:                    # noqa: BLE001
                return node
        # `c in (c1, c2, ...)` / `not in`, every element a literal: the membership is a literal
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and _const(node.left) and \
                isinstance(node.comparators[0], (ast.Tuple, ast.List, ast.Set)) and \
                all(_const(x) for x in node.comparators[0].elts):
            try:
                v = node.left.value in [x.value for x in node.comparators[0].elts]
            except Exception:                    # noqa: BLE001
                return node
            return ast.Constant(v if isinstance(node.ops[0], ast.In) else not v)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot)) and _const(node.comparators[0]) \
                and node.comparators[0].value is None and _const(node.left):
            v = node.left.value is None
            return ast.Constant(v if isinstance(node.ops[0], ast.Is) else not v)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        absorbing, neutral = (False, True) if is_and else (True, False)
        vals = []
        for v in node.values:
            if _const(v) and isinstance(v.value, bool) and v.value is absorbing:
                if all(is_pure(x) for x in vals):
                    return ast.Constant(absorbing) if not vals or all(bool_valued(x) for x in vals) else \
                        (ast.BoolOp(op=node.op, values=vals + [v]) if vals else ast.Constant(absorbing))
                vals.append(v)
                break
            vals.append(v)
        # a neutral constant is dropped when the operands left are booleans (the value is the same)
        keep = [v for v in vals if not (_const(v) and v.value is neutral and isinstance(v.value, bool))]
        if len(keep) != len(vals) and keep and all(bool_valued(x) for x in keep):
            vals = keep
        elif not keep:
            return ast.Constant(neutral)
        if len(vals) == 1:
            return vals[0]
        node.values = vals
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if _const(node.test):
            return node.body if node.test.value else node.orelse
        return node

    def _stmts(self, body):
        out = []
        for s in body:
            r = self.visit(s)
            if r is None:
                continue
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)
        return out

    def generic_visit(self, node):
        for field, v in ast.iter_fields(node):
            if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                nv = self._stmts(v)
                if not nv and field in ("body",):
                    nv = [ast.Pass()]
                setattr(node, field, nv)
            elif isinstance(v, list):
                setattr(node, field, [self.visit(x) if isinstance(x, ast.AST) else x for x in v])
            elif isinstance(v, ast.AST):
                setattr(node, field, self.visit(v))
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if _const(node.test):
            live = node.body if node.test.value else node.orelse
            live = [s for s in live if not isinstance(s, ast.Pass)]
            return live or None
        if all(isinstance(s, ast.Pass) for s in node.body) and not node.orelse and is_pure(node.test):
            return None
        return node


def _stores(fn) -> dict:
    """{name: [store nodes]} of a function body (assignments, aug-assignments, loop/with/except targets)."""
    out = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.setdefault(n.id, []).append(n)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for x in n.names:
                out.setdefault(x, []).extend([n, n])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.setdefault(n.name, []).append(n)
    return out


def _loads(fn) -> dict:
    out = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.setdefault(n.id, []).append(n)
    return out


def _params(fn) -> set:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return set()
    a = fn.args
    return {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | \
        ({a.vararg.arg} if a.vararg else set()) | ({a.kwarg.arg} if a.kwarg else set())


class _Replace(ast.NodeTransformer):
    def __init__(self, name, value):
        self.name, self.value, self.count = name, value, 0

    def visit_Name(self, node):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            self.count += 1
            return copy.deepcopy(self.value)
        return node


def _blocks(fn):
    for n in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            v = getattr(n, field, None)
            if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                yield v


def propagate(fn) -> list:
    """Constant propagation and single-use inlining inside one function, to a fixpoint. Returns what it did.

    - a local assigned exactly once (one syntactic store, a plain `name = <constant>`) is replaced by the
      constant at every read after it, the assignment removed;
    - a local assigned once to a pure expression without calls, read exactly once, in the statement right
      after its assignment in the same block, by a statement that stores none of the expression's names, is
      inlined there, the assignment removed."""
    done = []
    for _ in range(50):
        changed = False
        stores, loads = _stores(fn), _loads(fn)
        params = _params(fn)
        for block in list(_blocks(fn)):
            for i, s in enumerate(block):
                if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)):
                    continue
                name = s.targets[0].id
                if name in params or len(stores.get(name, [])) != 1:
                    continue
                if _const(s.value):
                    r = _Replace(name, s.value)
                    for t in block[i + 1:]:
                        r.visit(t)
                    if r.count != len(loads.get(name, [])):
                        # a read outside the statements after it (a loop's earlier statement): undo
                        return done if not changed else done + ["(stopped: non-dominated read of %s)" % name]
                    block.pop(i)
                    if not block:
                        block.append(ast.Pass())
                    done.append(f"const {name}")
                    changed = True
                    break
                if len(loads.get(name, [])) == 1 and i + 1 < len(block) and is_pure(s.value) \
                        and not any(isinstance(x, ast.Call) for x in ast.walk(s.value)):
                    nxt = block[i + 1]
                    if any(isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Load)
                           for x in ast.walk(nxt)):
                        used = {x.id for x in ast.walk(s.value) if isinstance(x, ast.Name)}
                        if not (used & set(_stores(nxt))):
                            r = _Replace(name, s.value)
                            r.visit(nxt)
                            block.pop(i)
                            done.append(f"inline {name}")
                            changed = True
                            break
            if changed:
                break
        if not changed:
            break
        Folder().visit(fn)
    return done


def fold(node, subst=None) -> tuple:
    """(folded copy, [what was applied]) of a unit."""
    n = copy.deepcopy(node)
    f = Folder(subst)
    n = f.visit(n)
    applied = list(f.applied)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        applied += propagate(n)
    return n, applied


# -------------------------------------------------------------------------------------------------------------
# references and reachability
# -------------------------------------------------------------------------------------------------------------

def _local_imports(mi: ModInfo, node, index: SourceIndex) -> tuple:
    """(local name -> import target, [modules the function's imports execute]) of the imports inside a
    function."""
    out, execd = {}, []
    pkg_parts = mi.name.split(".")
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for local, kind, dotted, attr in _import_targets(n, pkg_parts, mi.is_pkg):
                if kind == "sym" and f"{dotted}.{attr}" in index.mods:
                    execd.append(f"{dotted}.{attr}")
                execd.append(dotted)
                if local and local != "*":
                    out[local] = (kind, dotted, attr)
    return out, execd


class Refs:
    __slots__ = ("defs", "mods", "attrs", "strings", "calls")

    def __init__(self):
        self.defs, self.mods, self.attrs, self.strings = set(), set(), set(), set()
        self.calls = []   # (call node, resolved def id | None)


def references(index: SourceIndex, mod: str, node) -> Refs:
    """What one unit names (see the module docstring)."""
    mi = index.mods[mod]
    local, execd = _local_imports(mi, node, index)
    r = Refs()
    for d in execd:
        parts = d.split(".")
        for i in range(1, len(parts) + 1):
            if ".".join(parts[:i]) in index.mods:
                r.mods.add(".".join(parts[:i]))
    seen_attr_nodes = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and id(n) not in seen_attr_nodes:
            res = index.resolve_expr(mod, n, local)
            # mark the inner chain as handled
            e = n.value
            while isinstance(e, ast.Attribute):
                seen_attr_nodes.add(id(e))
                e = e.value
            if res is not None:
                (r.defs if res[0] == "def" else r.mods).add(res[1])
                # a module attribute read executes the module; the chain's modules too
                if res[0] == "def":
                    r.mods.add(res[1].split(":")[0])
            else:
                # the longest prefix that resolves is named; the rest is attribute reads on objects
                e = n
                attrs = []
                while isinstance(e, ast.Attribute):
                    attrs.append(e.attr)
                    pre = index.resolve_expr(mod, e.value, local)
                    if pre is not None:
                        (r.defs if pre[0] == "def" else r.mods).add(pre[1])
                        break
                    e = e.value
                r.attrs.update(attrs)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            res = index.resolve_expr(mod, n, local)
            if res is not None:
                (r.defs if res[0] == "def" else r.mods).add(res[1])
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 200:
            r.strings.add(n.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "hasattr",
                                                                                          "setattr") \
                and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
            r.attrs.add(n.args[1].value)
    return r


def unit_nodes(index: SourceIndex) -> dict:
    """{unit id: node} over the whole index: `mod:qual` for functions and methods (nested functions in
    their parent), `mod:C.<body>`, `mod:<module>`."""
    out = {}
    for mod, mi in index.mods.items():
        for q, n in units_of(mi.tree).items():
            out[f"{mod}:{q}"] = n
    return out


def owner_class(index: SourceIndex, uid: str):
    """The class id a method unit belongs to, else None."""
    mod, q = uid.split(":", 1)
    if "." not in q:
        return None
    c = q.rsplit(".", 1)[0]
    if c.endswith(">"):
        return None
    return f"{mod}:{c}" if c in index.mods[mod].classes else None


class Reach:
    """Forward reachability from an entry (see the module docstring). `fold_for(uid, node)` folds a unit
    before its references are read (the argv constants of a mode); `strings_name_units` makes a string equal
    to a class name or a module name a reference."""

    def __init__(self, index: SourceIndex, units: dict, fold_for=None):
        self.index, self.units, self.fold_for = index, units, fold_for
        self.reached = {}           # uid -> the uid that reached it first (a witness path)
        self.instantiated = set()
        self.attr_names = set()
        self.folds = {}             # uid -> [folds applied]
        self._method_units = {}
        for uid in units:
            c = owner_class(index, uid)
            if c:
                self._method_units.setdefault(c, []).append(uid)

    def _class_methods(self, cid, names=None, dunders=False, all_=False):
        for c in self.index.class_mro(cid):
            if c == "<external>":
                continue
            for m in self._method_units.get(c, []):
                nm = m.rsplit(".", 1)[1]
                if all_ or (dunders and nm.startswith("__") and nm.endswith("__")) or (names and nm in names):
                    yield m

    def run(self, entries: list) -> dict:
        todo = []

        def add(uid, why):
            if uid in self.units and uid not in self.reached:
                self.reached[uid] = why
                todo.append(uid)

        def add_module(mod, why):
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):
                m = ".".join(parts[:i])
                if m in self.index.mods:
                    add(f"{m}:<module>", why)

        def instantiate(cid, why):
            if cid in self.instantiated:
                return
            self.instantiated.add(cid)
            mod, q = cid.split(":", 1)
            add_module(mod, why)
            add(f"{cid}.<body>", why)
            external = "<external>" in self.index.class_mro(cid)
            for m in self._class_methods(cid, names=self.attr_names, dunders=True, all_=external):
                add(m, why)
            for c in self.index.class_mro(cid):
                if c != "<external>":
                    add(f"{c}.<body>", why)

        for e in entries:
            if e.endswith(":<module>"):
                add_module(e.split(":")[0], "<entry>")
            add(e, "<entry>")
        while todo:
            uid = todo.pop()
            node = self.units[uid]
            mod = uid.split(":")[0]
            if self.fold_for is not None:
                node, applied = self.fold_for(uid, node)
                if applied:
                    self.folds[uid] = applied
            refs = references(self.index, mod, node)
            for m in refs.mods:
                add_module(m, uid)
            for d in refs.defs:
                n = self.index.node(d)
                if isinstance(n, ast.ClassDef):
                    instantiate(d, uid)
                else:
                    add_module(d.split(":")[0], uid)
                    add(d, uid)
            for s in refs.strings:
                if s in self.index.mods:
                    add_module(s, uid)
                for cid in self.index.classes_by_name.get(s, ()):
                    instantiate(cid, uid)
            new_attrs = refs.attrs - self.attr_names
            if new_attrs:
                self.attr_names |= new_attrs
                for cid in list(self.instantiated):
                    for m in self._class_methods(cid, names=new_attrs):
                        add(m, uid)
            # a decorated function or class of a module whose residue runs is registered by its decorator
            if uid.endswith(":<module>"):
                mi = self.index.mods[mod]
                for q, n in mi.defs.items():
                    if "." in q:
                        continue
                    if getattr(n, "decorator_list", None):
                        if isinstance(n, ast.ClassDef):
                            instantiate(f"{mod}:{q}", uid)
                        else:
                            add(f"{mod}:{q}", uid)
            # a nested class in a reached method or class body is instantiated with it
        return self.reached

    def path(self, uid: str, limit: int = 12) -> list:
        out = [uid]
        while len(out) < limit and self.reached.get(out[-1]) not in (None, "<entry>"):
            out.append(self.reached[out[-1]])
        return out


# -------------------------------------------------------------------------------------------------------------
# the argv namespace: the parameters that hold the parsed argv
# -------------------------------------------------------------------------------------------------------------

def namespace_params(index: SourceIndex, units: dict, seed_uid: str, scope_pkg: str) -> dict:
    """{uid: {local names holding the parsed argv}} — the seed function's names assigned from
    `<parser>.parse_args(...)`, then every parameter of a function of `scope_pkg` that ALL its call sites in
    the index pass a namespace name at that position (a function some call site hands something else is not
    folded: the fold must hold for every call)."""
    ns = {}
    seed = units[seed_uid]
    names = set()
    for n in ast.walk(seed):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) \
                and n.value.func.attr == "parse_args":
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    ns[seed_uid] = names
    # every call site in the index, by callee
    sites = {}
    for uid, node in units.items():
        mod = uid.split(":")[0]
        mi = index.mods[mod]
        local, _ = _local_imports(mi, node, index)
        for c in ast.walk(node):
            if isinstance(c, ast.Call):
                r = index.resolve_expr(mod, c.func, local)
                if r and r[0] == "def":
                    sites.setdefault(r[1], []).append((uid, c))
    for _ in range(20):
        grew = False
        for callee, calls in sites.items():
            if not callee.startswith(scope_pkg) or callee not in units:
                continue
            fn = units[callee]
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            pnames = [a.arg for a in fn.args.posonlyargs + fn.args.args]
            have = set()
            for i, p in enumerate(pnames):
                ok = True
                for (caller, c) in calls:
                    cn = ns.get(caller, set())
                    arg = c.args[i] if i < len(c.args) else next((k.value for k in c.keywords if k.arg == p), None)
                    if not (isinstance(arg, ast.Name) and arg.id in cn):
                        ok = False
                        break
                if ok:
                    have.add(p)
            if have - ns.get(callee, set()):
                ns[callee] = ns.get(callee, set()) | have
                grew = True
        if not grew:
            break
    return {k: v for k, v in ns.items() if v}


def argv_subst(ns_names: set, values: dict, stored: set):
    """The substitution of a namespace function: `args.x`, `getattr(args, "x"[, d])` -> the constant every
    cell of the mode gives `x` (never an attribute some code assigns)."""
    def subst(node):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) \
                and node.value.id in ns_names and node.attr in values and node.attr not in stored:
            return ast.Constant(values[node.attr])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in ns_names \
                and isinstance(node.args[1], ast.Constant) and node.args[1].value in values \
                and node.args[1].value not in stored:
            return ast.Constant(values[node.args[1].value])
        return None
    return subst


def stored_ns_attrs(units: dict, ns: dict) -> set:
    out = set()
    for uid, names in ns.items():
        for n in ast.walk(units[uid]):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) \
                    and n.value.id in names:
                out.add(n.attr)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" and n.args \
                    and isinstance(n.args[0], ast.Name) and n.args[0].id in names and len(n.args) > 1 \
                    and isinstance(n.args[1], ast.Constant):
                out.add(n.args[1].value)
    return out


BUILTIN_NAMES = set(dir(builtins))


# =============================================================================================================
# the program of one tree: canonical references, call sites, slots, the normal form of a unit
# =============================================================================================================

SIG = "§"            # a canonical reference: Name(id="§pkg.mod:qual") — never a valid Python identifier
TOP = ("top",)


def _meet(a, b):
    """The meet of two slot values: None (no information yet) | ("c", literal) | TOP."""
    if a is None:
        return b
    if b is None:
        return a
    if a == TOP or b == TOP:
        return TOP
    return a if (a[1] == b[1] and type(a[1]) is type(b[1])) else TOP


def is_ref(n) -> bool:
    return isinstance(n, ast.Name) and n.id.startswith(SIG)


def ref_id(n) -> str:
    return n.id[len(SIG):]


def fn_params(fn) -> list:
    """[(name, kind, default node | None)] of a function's parameters, kind in pos|kwonly|vararg|kwarg."""
    a = fn.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    out = [(p.arg, "pos", d) for p, d in zip(pos, defaults)]
    if a.vararg:
        out.append((a.vararg.arg, "vararg", None))
    out += [(p.arg, "kwonly", d) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg:
        out.append((a.kwarg.arg, "kwarg", None))
    return out


def fn_locals(fn) -> set:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    out = set(_params(fn)) | set(_stores(fn))
    for n in ast.walk(fn):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not fn:
            out |= _params(n)
            if not isinstance(n, ast.Lambda):
                out.add(n.name)
        elif isinstance(n, ast.ClassDef):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
    return out


class Canon(ast.NodeTransformer):
    """Every name and attribute chain that resolves outside the function — a function, a class, a module, a
    module-level global of the index — becomes one canonical reference `§module:qual`, whatever import or
    alias spelled it."""

    def __init__(self, index: SourceIndex, mod: str, local_imports: dict, locals_: set):
        self.index, self.mod, self.local, self.locals_ = index, mod, local_imports, locals_

    def ref(self, node):
        idx, mod = self.index, self.mod
        if isinstance(node, ast.Name):
            if node.id in self.locals_ or node.id.startswith(SIG):
                return None
            r = idx.resolve_expr(mod, node, self.local)
            if r:
                return r[1]
            mi = idx.mods[mod]
            if node.id in mi.globals_:
                return f"{mod}:{node.id}"
            imp = self.local.get(node.id) or mi.imports.get(node.id)
            if imp and imp[0] == "sym" and imp[1] in idx.mods and imp[2] in idx.mods[imp[1]].globals_:
                return f"{imp[1]}:{imp[2]}"
            return None
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name) or root.id in self.locals_:
                return None
            r = idx.resolve_expr(mod, node, self.local)
            if r:
                return r[1]
            pre = idx.resolve_expr(mod, node.value, self.local)
            if pre and pre[0] == "mod" and node.attr in idx.mods[pre[1]].globals_:
                return f"{pre[1]}:{node.attr}"
        return None

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            r = self.ref(node)
            if r:
                return ast.copy_location(ast.Name(id=SIG + r, ctx=ast.Load()), node)
        return node

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            r = self.ref(node)
            if r:
                return ast.copy_location(ast.Name(id=SIG + r, ctx=ast.Load()), node)
        self.generic_visit(node)
        return node


def alpha(fn):
    """Locals that are not parameters renamed `_v<i>` in order of first appearance (a rename is no change)."""
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return fn
    params = _params(fn)
    names = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and not n.id.startswith(SIG) and n.id not in params \
                and n.id not in BUILTIN_NAMES and isinstance(n.ctx, ast.Store) and n.id not in names:
            names[n.id] = f"_v{len(names)}"
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in names:
            n.id = names[n.id]
    return fn


def path_sites(fn):
    """[(node, [path condition terms])] of every expression statement's calls and value references in a unit:
    the tests of the enclosing `if`s (negated in `else`), and the negated tests of the earlier `if T:` siblings
    whose body always leaves (return / raise)."""
    out = []

    def leaves(body):
        return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))

    def exprs_of(s):
        for field, v in ast.iter_fields(s):
            if field in ("body", "orelse", "finalbody", "handlers"):
                continue
            if isinstance(v, ast.AST):
                yield v
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, ast.AST) and not isinstance(x, ast.stmt):
                        yield x

    def block(body, cond):
        cond = list(cond)
        for s in body:
            for e in exprs_of(s):
                for n in ast.walk(e):
                    if isinstance(n, (ast.Call, ast.Name)):
                        out.append((n, list(cond)))
            if isinstance(s, ast.If):
                block(s.body, cond + [s.test])
                block(s.orelse, cond + [ast.UnaryOp(op=ast.Not(), operand=s.test)])
                if leaves(s.body) and not s.orelse:
                    cond = cond + [ast.UnaryOp(op=ast.Not(), operand=s.test)]
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                block(s.body, cond)
            else:
                for field in ("body", "orelse", "finalbody"):
                    v = getattr(s, field, None)
                    if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                        block(v, cond)
                for h in getattr(s, "handlers", []) or []:
                    block(h.body, cond)
    body = fn.body if hasattr(fn, "body") else []
    block(body, [])
    return out


_BOT = object()          # the optimistic bottom of the type lattice (no value seen yet)


class Program:
    """One tree, seen from the gate: its units, their canonical form, every call site with its path condition,
    the constants the gate gives (`K`: calls constant at the gate, read by the probe in that tree), the
    parameter and attribute slots those constants fill (a parameter every reached call site passes the same
    constant), and the normal form of a unit under all of it."""

    def __init__(self, index: SourceIndex, units: dict, reached: set, *, K=None, slot_exclude=(),
                 max_inline_depth: int = 4):
        self.index, self.units, self.reached = index, units, set(reached)
        self.K = dict(K or {})
        self.slot_exclude = tuple(slot_exclude)        # packages whose call sites fill no slot (the planner)
        self.max_inline_depth = max_inline_depth
        self._canon, self._nf, self._nf_busy = {}, {}, set()
        self.absorbed = {}                              # uid -> {callee ids inlined into its normal form}
        self.children = {}
        for mi in index.mods.values():
            for q in mi.classes:
                cid = f"{mi.name}:{q}"
                for b in index.class_mro(cid)[1:]:
                    if b != "<external>":
                        self.children.setdefault(b, set()).add(cid)
        self._sites = None
        self._pslots, self._aslots, self._atypes = {}, {}, {}
        self._atypes_open = False
        self._slots_done = False

    # -- canonical form ----------------------------------------------------------------------------------
    def canon(self, uid):
        if uid in self._canon:
            return self._canon[uid]
        node = self.units[uid]
        mod = uid.split(":")[0]
        mi = self.index.mods[mod]
        local, _ = _local_imports(mi, node, self.index)
        n = normalized(node)
        n = Canon(self.index, mod, local, fn_locals(n)).visit(n)
        self._canon[uid] = n
        return n

    def owner(self, uid):
        return owner_class(self.index, uid)

    def subclasses(self, cid) -> set:
        out, todo = set(), [cid]
        while todo:
            c = todo.pop()
            for ch in self.children.get(c, ()):
                if ch not in out:
                    out.add(ch)
                    todo.append(ch)
        return out

    def lookup(self, cid, name):
        for c in self.index.class_mro(cid):
            if c == "<external>":
                return None
            m, q = c.split(":", 1)
            if f"{q}.{name}" in self.index.mods[m].defs:
                return f"{m}:{q}.{name}"
        return None

    def is_class(self, did) -> bool:
        return isinstance(self.index.node(did), ast.ClassDef)

    def self_name(self, uid):
        fn = self.units.get(uid)
        if self.owner(uid) and isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.args.args:
            if not any(isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod")
                       for d in fn.decorator_list):
                return fn.args.args[0].arg
        return None

    def recv_type(self, uid, expr):
        """The class of a receiver expression in `uid` (see `vtype`)."""
        sn = self.self_name(uid)
        if isinstance(expr, ast.Name) and sn and expr.id == sn:
            return self.owner(uid)
        if getattr(self, "_sites_busy", False) and not self._slots_done:
            # The call sites are being built and the slot types are built FROM them: the receiver is untyped here,
            # and `targets` answers with every method of that name — wider, never narrower (gate tha4's scope died
            # on the re-entry when a commit touching a module global made `gate_constants` ask for the sites first,
            # 2026-10-26). The answers given meanwhile are forgotten once the sites are built (`sites`).
            return None
        self._ensure_slots()
        return self.vtype(uid, expr)

    def vtype(self, uid, e):
        """The one class a value expression of `uid`'s canonical form is an instance of, or None: a constructor
        call; `self`; a parameter every call site (none a reference) passes such a value for; a local bound
        exactly once, by `x = E`; `X.a` with X typed and attribute a typed."""
        t = self._vt(uid, e, frozenset())
        return None if t is _BOT else t

    def _vt(self, uid, e, seen):
        """`vtype` on the optimistic lattice: _BOT (no value seen yet: a cycle back to itself, an attribute not
        yet typed), a class, None (no one class). Memoised while the attribute types hold still, for an answer
        no cycle cut short (a cut answer is only true inside its cycle)."""
        key = (uid, id(e))
        memo = self.__dict__.setdefault("_vt_memo", {})
        if key in memo:
            return memo[key]
        hits = self.__dict__.setdefault("_vt_cuts", [0])
        before = hits[0]
        r = self._vt1(uid, e, seen)
        if hits[0] == before:
            memo[key] = r
        return r

    def _vt1(self, uid, e, seen):
        if isinstance(e, ast.Call) and is_ref(e.func) and self.is_class(ref_id(e.func)):
            return ref_id(e.func)
        sn = self.self_name(uid)
        fn = self.units.get(uid)
        if isinstance(e, ast.Name):
            if sn and e.id == sn:
                return self.owner(uid)
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return None
            st = _stores(fn).get(e.id, [])
            if e.id in _params(fn):
                key = (uid, e.id)
                if key in seen:
                    self.__dict__.setdefault("_vt_cuts", [0])[0] += 1
                    return _BOT                           # a cycle adds nothing of its own
                if getattr(self, "_sites_busy", False):
                    return None                           # while the call sites are being found: not from them
                if st or e.id in ((fn.args.vararg.arg,) if fn.args.vararg else ()) or \
                        e.id in ((fn.args.kwarg.arg,) if fn.args.kwarg else ()):
                    return None
                t = _BOT
                for (caller, call, recv, ctor, cond, kind) in self.sites().get(uid, ()):
                    if kind != "call":
                        return None
                    arg = self._arg_for(fn, call, recv, ctor, e.id)
                    if arg[0] != "arg":
                        return None
                    x = self._vt(caller, arg[1], seen | {key})
                    if x is _BOT:
                        continue
                    if x is None or (t is not _BOT and x != t):
                        return None
                    t = x
                return t
            if len(st) != 1:
                return None
            for n in ast.walk(self.canon(uid)):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                        and n.targets[0].id == e.id:
                    key = (uid, "=" + e.id)
                    if key in seen:
                        self.__dict__.setdefault("_vt_cuts", [0])[0] += 1
                        return _BOT
                    return self._vt(uid, n.value, seen | {key})
            return None
        if isinstance(e, ast.Attribute):
            t = self._vt(uid, e.value, seen)
            if t is _BOT or t is None:
                return t
            return self._atypes.get((t, e.attr), _BOT if self._atypes_open else None)
        # an element read from a mapping attribute: `R.a[k]`, `R.a.get(k)`, `R.a.get(k, None)`
        cont = None
        if isinstance(e, ast.Subscript) and isinstance(e.ctx, ast.Load):
            cont = e.value
        elif isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "get" and \
                not e.keywords and (len(e.args) == 1 or (len(e.args) == 2 and isinstance(e.args[1], ast.Constant)
                                                         and e.args[1].value is None)):
            cont = e.func.value
        if isinstance(cont, ast.Attribute):
            t = self._vt(uid, cont.value, seen)
            if t is _BOT or t is None:
                return t
            return self.mapping_value_class(t, cont.attr)
        return None

    def mapping_value_class(self, cid, a):
        """The class of the values of the mapping attribute `a` of class `cid`, read from its declared
        annotation (`Dict[K, C]`, `Mapping[K, C]`, ... on the class body or on a `self.a: ...` store), or None.
        A declared annotation is the code's own statement of what the mapping holds."""
        memo = self.__dict__.setdefault("_mvc", {})
        if (cid, a) in memo:
            return memo[(cid, a)]
        found = set()
        for c in self.index.class_mro(cid):
            if c == "<external>":
                continue
            cls = self.index.node(c)
            if not isinstance(cls, ast.ClassDef):
                continue
            mod = c.split(":", 1)[0]
            anns = [s.annotation for s in cls.body if isinstance(s, ast.AnnAssign)
                    and isinstance(s.target, ast.Name) and s.target.id == a]
            for n in ast.walk(cls):
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Attribute) and n.target.attr == a \
                        and isinstance(n.target.value, ast.Name) and n.target.value.id == "self":
                    anns.append(n.annotation)
            for an in anns:
                found.add(self._ann_value_class(mod, an))
        r = next(iter(found)) if len(found) == 1 else None
        memo[(cid, a)] = r
        return r

    def _ann_value_class(self, mod, an):
        if isinstance(an, ast.Constant) and isinstance(an.value, str):
            try:
                an = ast.parse(an.value, mode="eval").body
            except SyntaxError:
                return None
        if not isinstance(an, ast.Subscript):
            return None
        head = an.value.attr if isinstance(an.value, ast.Attribute) else getattr(an.value, "id", "")
        if head not in ("Dict", "dict", "Mapping", "MutableMapping", "OrderedDict", "DefaultDict", "defaultdict"):
            return None
        sl = an.slice
        if not (isinstance(sl, ast.Tuple) and len(sl.elts) == 2):
            return None
        v = sl.elts[1]
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            try:
                v = ast.parse(v.value, mode="eval").body
            except SyntaxError:
                return None
        if not isinstance(v, ast.Name):
            return None
        r = self.index.resolve_in_module(mod, v.id)
        if r and r[0] == "def" and self.is_class(r[1]):
            return r[1]
        return None

    def dataclass_fields(self, cid):
        """The `__init__` fields of a dataclass (bases first), or None when `cid` is not one with a generated
        `__init__`."""
        out, any_dc = [], False
        for c in reversed([x for x in self.index.class_mro(cid) if x != "<external>"]):
            cls = self.index.node(c)
            if not isinstance(cls, ast.ClassDef):
                return None
            dc = None
            for d in cls.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                if nm.rsplit(":", 1)[-1].rsplit(".", 1)[-1] == "dataclass":
                    dc = d
            if dc is None:
                continue
            if isinstance(dc, ast.Call) and any(k.arg == "init" and not (isinstance(k.value, ast.Constant)
                                                                     and k.value.value) for k in dc.keywords):
                return None
            any_dc = True
            for st in cls.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and \
                        "ClassVar" not in ast.unparse(st.annotation) and "InitVar" not in ast.unparse(st.annotation):
                    if st.target.id not in out:
                        out.append(st.target.id)
        if not any_dc:
            return None
        if self.lookup(cid, "__init__"):
            return None                                   # an __init__ written by hand: its stores are its own
        return out

    def targets(self, uid, call, exact=False):
        """[(callee id, receiver expr | None, is_constructor)] of a call node of `uid`'s canonical form. With
        `exact`, only a statically unique target (no override in a subclass, no name-based guess)."""
        f = call.func
        if isinstance(f, ast.Subscript):                     # kernel[grid](...)
            f = f.value
        if is_ref(f):
            did = ref_id(f)
            if did in self.units and not self.is_class(did):
                return [(did, None, False)]
            if self.is_class(did):
                init = self.lookup(did, "__init__")
                return [(init, None, True)] if init else []
            return []
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name) and f.value.func.id == "super":
                own = self.owner(uid)
                if own is None:
                    return []
                for c in self.index.class_mro(own)[1:]:
                    if c == "<external>":
                        return []
                    m = self.lookup(c, f.attr)
                    if m:
                        return [(m, f.value, False)]
                return []
            if is_ref(f.value) and self.is_class(ref_id(f.value)):
                m = self.lookup(ref_id(f.value), f.attr)
                return [(m, None, False)] if m else []
            t = self.recv_type(uid, f.value)
            if t:
                m = self.lookup(t, f.attr)
                out = [(m, f.value, False)] if m else []
                over = [self.lookup(s, f.attr) for s in self.subclasses(t)]
                over = [o for o in over if o and o != m]
                if over:
                    if exact:
                        return []
                    out += [(o, f.value, False) for o in over]
                return out
            if exact or (f.attr.startswith("__") and f.attr.endswith("__")):
                return []
            return [(m, f.value, False) for m in self.index.methods_by_name.get(f.attr, ())]
        return []

    # -- call sites ----------------------------------------------------------------------------------------
    def sites(self):
        """{callee id: [(caller uid, node, receiver | None, ctor, [path terms], kind)]} over the reached units;
        kind "call", or "ref" for a function named as a value (a table entry, a callback)."""
        if self._sites is not None:
            return self._sites
        if getattr(self, "_sites_busy", False):
            raise RuntimeError("sites() re-entered while building")
        self._sites_busy = True
        try:
            return self._build_sites()
        finally:
            self._sites_busy = False
            self._vt_memo = {}                            # answers given without the call sites

    def _build_sites(self):
        out = {}
        for uid in self.reached:
            if uid not in self.units:
                continue
            fn = self.canon(uid)
            called = set()
            for node, cond in path_sites(fn):
                if isinstance(node, ast.Call):
                    called.add(id(node.func))
                    if isinstance(node.func, ast.Subscript):
                        called.add(id(node.func.value))
                    for did, recv, ctor in self.targets(uid, node):
                        if did:
                            out.setdefault(did, []).append((uid, node, recv, ctor, cond, "call"))
            for node, cond in path_sites(fn):
                if is_ref(node) and id(node) not in called:
                    did = ref_id(node)
                    if did in self.units and not self.is_class(did):
                        out.setdefault(did, []).append((uid, node, None, False, cond, "ref"))
        self._sites = out
        return out

    # -- slots ---------------------------------------------------------------------------------------------
    def _arg_for(self, callee_fn, call, recv, ctor, pname):
        """The argument node a call passes for parameter `pname` ("<default>", node) | ("<none>",) |
        ("<star>",)."""
        params = fn_params(callee_fn)
        pos = [p for p in params if p[1] == "pos"]
        skip = 1 if (recv is not None or ctor) else 0
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            return ("<star>",)
        for k in call.keywords:
            if k.arg == pname:
                return ("arg", k.value)
        for i, p in enumerate(pos):
            if p[0] == pname:
                j = i - skip
                if 0 <= j < len(call.args):
                    return ("arg", call.args[j])
                return ("<default>", p[2]) if p[2] is not None else ("<none>",)
        for p in params:
            if p[0] == pname and p[1] == "kwonly":
                return ("<default>", p[2]) if p[2] is not None else ("<none>",)
        return ("<none>",)

    def _absval(self, uid, e):
        """A slot value or a reference to another slot: ("c", v) | TOP | ("p", (uid, name)) | ("a", (cid, a))."""
        if isinstance(e, ast.Constant):
            return ("c", e.value)
        if isinstance(e, ast.Call) and not e.keywords and all(is_pure(a) for a in e.args):
            if isinstance(e.func, ast.Name) and e.func.id == "bool" and len(e.args) == 1:
                v = self._absval(uid, e.args[0])
                return ("bool", v)
            for did, recv, ctor in self.targets(uid, e, exact=True):
                if did in self.K:
                    return ("c", self.K[did])
            return TOP
        if isinstance(e, ast.Call):
            for did, recv, ctor in self.targets(uid, e, exact=True):
                if did in self.K and all(is_pure(a) for a in e.args) and all(is_pure(k.value) for k in e.keywords):
                    return ("c", self.K[did])
            return TOP
        fn = self.units.get(uid)
        if isinstance(e, ast.Name) and isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if e.id in _params(fn) and len(_stores(fn).get(e.id, [])) == 0:
                return ("p", (uid, e.id))
            return TOP
        if isinstance(e, ast.Attribute):
            sn = self.self_name(uid)
            if isinstance(e.value, ast.Name) and sn and e.value.id == sn:
                return ("a", (self.owner(uid), e.attr))
            t = self.recv_type(uid, e.value)
            if t:
                return ("a", (t, e.attr))
        return TOP

    def _ensure_slots(self):
        if self._slots_done:
            return
        self._slots_done = True                       # (re-entrancy: types are read while building)
        sites = self.sites()
        # attribute stores, by attribute name: (uid, receiver kind, value node)
        stores, untyped, dyn_self = {}, set(), set()
        self.dynamic_setattr = []
        for uid in self.reached:
            if uid not in self.units:
                continue
            fn = self.canon(uid)
            sn = self.self_name(uid)
            for n in ast.walk(fn):
                tgts = []
                if isinstance(n, ast.Assign):
                    tgts = [(t, n.value) for t in n.targets]
                elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and getattr(n, "value", None) is not None:
                    tgts = [(n.target, n.value if isinstance(n, ast.AnnAssign) else None)]
                for t, v in tgts:
                    for tt in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if isinstance(tt, ast.Attribute):
                            if isinstance(tt.value, ast.Name) and sn and tt.value.id == sn and v is not None \
                                    and not isinstance(t, ast.Tuple) and not isinstance(n, ast.AugAssign):
                                stores.setdefault(tt.attr, []).append((uid, v))
                            else:
                                untyped.add(tt.attr)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("setattr", "delattr") \
                        and len(n.args) >= 2:
                    if isinstance(n.args[1], ast.Constant):
                        untyped.add(n.args[1].value)
                    elif isinstance(n.args[0], ast.Name) and sn and n.args[0].id == sn and self.owner(uid):
                        dyn_self.add(self.owner(uid))     # setattr(self, <name>, v): that class family's slots
                    else:
                        self.dynamic_setattr.append(uid)  # on another object: named, not assumed
        self._attr_stores, self._attr_untyped = stores, untyped
        poisoned = set()
        for c in dyn_self:
            poisoned |= set(x for x in self.index.class_mro(c) if x != "<external>") | self.subclasses(c)
        self._poisoned = poisoned
        # a dataclass's generated __init__ stores each field from its constructor's call sites: those stores are
        # the call's arguments (a field no site passes keeps its class-level default, met below)
        ctor = {}
        for uid in self.reached:
            if uid not in self.units:
                continue
            for n in ast.walk(self.canon(uid)):
                f = n.func if isinstance(n, ast.Call) else None
                if f is not None and n.keywords and (getattr(f, "attr", None) or getattr(f, "id", "")).rsplit(
                        ":", 1)[-1].rsplit(".", 1)[-1] == "replace":
                    untyped.update(k.arg for k in n.keywords if k.arg)   # dataclasses.replace(obj, a=v): unseen stores
                if isinstance(n, ast.Call) and is_ref(n.func) and self.is_class(ref_id(n.func)):
                    cid = ref_id(n.func)
                    flds = self.dataclass_fields(cid)
                    if flds is None:
                        continue
                    if any(isinstance(x, ast.Starred) for x in n.args) or any(k.arg is None for k in n.keywords):
                        for f in flds:
                            ctor.setdefault((cid, f), []).append((uid, None))
                        continue
                    for f, x in zip(flds, n.args):
                        ctor.setdefault((cid, f), []).append((uid, x))
                    for k in n.keywords:
                        ctor.setdefault((cid, k.arg), []).append((uid, k.value))
        self._ctor_stores = ctor

        def family(cid):
            fam = set(c for c in self.index.class_mro(cid) if c != "<external>") | self.subclasses(cid)
            return fam

        self._family = family
        # attribute types: every store of `self.a` (or constructor field a) is a value of one class — a
        # constructor call, a parameter every call site passes such a value for, a local bound once to one, an
        # attribute already typed; to a fixpoint, since a type found makes the next one findable
        self._atypes, self._atypes_open = {}, True
        keys = {}
        for a, lst in stores.items():
            for uid, v in lst:
                c = self.owner(uid)
                if c:
                    keys.setdefault((c, a), None)
        for (cid, a) in ctor:
            keys.setdefault((cid, a), None)
        # optimistic: an attribute is untyped (_BOT) until a store gives it a class; it only ever moves up
        # (_BOT -> a class -> None), so the loop ends; what it ends on satisfies every store
        for _ in range(2 * len(keys) + 2):
            changed = False
            for (c, a) in keys:
                fam = family(c)
                if a in untyped or (fam & poisoned):
                    t = None
                else:
                    vals = [(uid, v) for uid, v in stores.get(a, ()) if self.owner(uid) in fam]
                    vals += [(uid, v) for fc in fam for uid, v in ctor.get((fc, a), ())]
                    ts = {self._vt(uid, v, frozenset()) if v is not None else None for uid, v in vals} - {_BOT}
                    t = None if (len(ts) != 1 or None in ts) else next(iter(ts))
                    if not ts:
                        continue
                for fc in fam:
                    cur = self._atypes.get((fc, a), _BOT)
                    if cur is None or cur == t:
                        continue
                    self._atypes[(fc, a)] = t if cur is _BOT else None
                    self._vt_memo = {}
                    changed = True
            if not changed:
                break
        self._atypes = {k: v for k, v in self._atypes.items() if v is not None}
        self._atypes_open = False
        self._vt_memo = {}
        # parameter and attribute slots: optimistic fixpoint
        pkeys = set()
        for did in sites:
            fn = self.units.get(did)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for p in fn_params(fn):
                    if p[1] in ("pos", "kwonly"):
                        pkeys.add((did, p[0]))
        val = {}

        def resolve(av, seen):
            if av is None or av == TOP:
                return av
            if av[0] == "c":
                return av
            if av[0] == "bool":
                r = resolve(av[1], seen)
                return ("c", bool(r[1])) if r and r != TOP and r[0] == "c" else r
            if av[0] == "p":
                return val.get(("p",) + av[1])
            if av[0] == "a":
                return val.get(("a",) + av[1])
            return TOP

        def pslot(did, pname):
            fn = self.units[did]
            out = None
            any_site = False
            for (caller, call, recv, ctor, cond, kind) in sites.get(did, ()):
                if caller.split(":")[0].startswith(self.slot_exclude):
                    continue
                any_site = True
                if kind == "ref":
                    return TOP
                arg = self._arg_for(fn, call, recv, ctor, pname)
                if arg[0] in ("<star>", "<none>"):
                    return TOP
                if arg[0] == "<default>":
                    v = ("c", arg[1].value) if isinstance(arg[1], ast.Constant) else TOP
                else:
                    v = resolve(self._absval(caller, arg[1]), set())
                out = _meet(out, v)
                if out == TOP:
                    return TOP
            return out if any_site else TOP

        def aslot(cid, a):
            fam = family(cid)
            if a in self._attr_untyped or (fam & self._poisoned):
                return TOP
            out = None
            for uid, v in self._attr_stores.get(a, ()):
                if self.owner(uid) in fam:
                    out = _meet(out, resolve(self._absval(uid, v), set()))
                    if out == TOP:
                        return TOP
            for fc in fam:
                for uid, v in self._ctor_stores.get((fc, a), ()):
                    out = _meet(out, TOP if v is None else resolve(self._absval(uid, v), set()))
                    if out == TOP:
                        return TOP
            # a class-level default
            for c in fam:
                m, q = c.split(":", 1)
                cls = self.index.mods[m].classes.get(q)
                for s in (cls.body if cls else []):
                    if isinstance(s, (ast.Assign, ast.AnnAssign)):
                        tg = s.targets if isinstance(s, ast.Assign) else [s.target]
                        if any(isinstance(t, ast.Name) and t.id == a for t in tg):
                            v = getattr(s, "value", None)
                            out = _meet(out, ("c", v.value) if isinstance(v, ast.Constant) else TOP)
            return out if out is not None else TOP

        akeys = set()
        for a, lst in self._attr_stores.items():
            for uid, v in lst:
                c = self.owner(uid)
                if c:
                    for fc in family(c):
                        akeys.add((fc, a))
        for (c, a) in self._ctor_stores:
            for fc in family(c):
                akeys.add((fc, a))
        for _ in range(40):
            changed = False
            for (did, p) in pkeys:
                v = pslot(did, p)
                if val.get(("p", did, p)) != v:
                    val[("p", did, p)] = v
                    changed = True
            for (c, a) in akeys:
                v = aslot(c, a)
                if val.get(("a", c, a)) != v:
                    val[("a", c, a)] = v
                    changed = True
            if not changed:
                break
        self._pslots = {(k[1], k[2]): v[1] for k, v in val.items() if k[0] == "p" and v and v != TOP and v[0] == "c"}
        self._aslots = {(k[1], k[2]): v[1] for k, v in val.items() if k[0] == "a" and v and v != TOP and v[0] == "c"}

    def pslots(self):
        self._ensure_slots()
        return self._pslots

    def aslots(self):
        self._ensure_slots()
        return self._aslots

    # -- the normal form -----------------------------------------------------------------------------------
    def nf(self, uid, depth=0):
        """The unit's normal form (a fresh copy): canonical references; the gate's constant calls and the
        constant parameter/attribute slots substituted, constant parameters stripped from signatures and call
        sites; single-expression callees inlined; folded; locals alpha-renamed."""
        if uid in self._nf:
            return copy.deepcopy(self._nf[uid])
        if uid in self._nf_busy or depth > self.max_inline_depth:
            return None
        self._nf_busy.add(uid)
        try:
            self._ensure_slots()
            n = copy.deepcopy(self.canon(uid))
            s = _Simplify(self, uid, depth)
            n = s.visit(n)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s.strip_signature(n)
            n = Folder().visit(n)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                propagate(n)
                s.drop_dead_self_stores(n)
                n = Folder().visit(n)
                propagate(n)
                drop_dead_stores(n)
            n = alpha(n)
            ast.fix_missing_locations(n)
            self._nf[uid] = n
            self.absorbed[uid] = s.absorbed
            return copy.deepcopy(n)
        finally:
            self._nf_busy.discard(uid)

    def return_expr(self, did, depth):
        """The single expression a callee's normal form returns (an `if T: return A` chain ended by `return B`
        folded to `A if T else B`), with its parameter list; None when it is more than that."""
        fn = self.nf(did, depth + 1)
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        if any(isinstance(d, ast.Name) and d.id in ("property",) for d in fn.decorator_list):
            return None
        body = [s for s in fn.body if not isinstance(s, ast.Pass)]
        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            return None
        e = body[-1].value
        for s in reversed(body[:-1]):
            if isinstance(s, ast.If) and not s.orelse and len(s.body) == 1 and isinstance(s.body[0], ast.Return) \
                    and s.body[0].value is not None:
                e = ast.IfExp(test=s.test, body=s.body[0].value, orelse=e)
            else:
                return None
        if any(isinstance(x, (ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr, ast.Lambda, ast.ListComp,
                              ast.SetComp, ast.DictComp, ast.GeneratorExp)) for x in ast.walk(e)):
            return None
        return fn, e


def _simple(e) -> bool:
    return isinstance(e, (ast.Name, ast.Constant)) or (isinstance(e, ast.Attribute) and _simple(e.value))


class _Simplify(ast.NodeTransformer):
    def __init__(self, prog: Program, uid: str, depth: int):
        self.p, self.uid, self.depth = prog, uid, depth
        self.absorbed = set()
        fn = prog.units[uid]
        self.fn = fn if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        self.sn = prog.self_name(uid)
        self.const_params = {}
        if self.fn is not None:
            stores = _stores(self.fn)
            for (did, pname), v in prog.pslots().items():
                if did == uid and not stores.get(pname):
                    self.const_params[pname] = v

    def strip_signature(self, fn):
        a = fn.args
        if not self.const_params:
            return
        pos = a.posonlyargs + a.args
        defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        keep = [(p, d) for p, d in zip(pos, defaults) if p.arg not in self.const_params]
        npo = len([p for p in a.posonlyargs if p.arg not in self.const_params])
        a.posonlyargs = [p for p, _ in keep[:npo]]
        a.args = [p for p, _ in keep[npo:]]
        ds = [d for _, d in keep]
        while ds and ds[0] is None:
            ds.pop(0)
        if any(d is None for d in ds):
            ds = [d for d in ds if d is not None]
        a.defaults = ds
        kw = [(p, d) for p, d in zip(a.kwonlyargs, a.kw_defaults) if p.arg not in self.const_params]
        a.kwonlyargs = [p for p, _ in kw]
        a.kw_defaults = [d for _, d in kw]

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.const_params:
            return ast.Constant(self.const_params[node.id])
        return node

    def _attr_const(self, node):
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            return None
        if self.sn and isinstance(node.value, ast.Name) and node.value.id == self.sn:
            k = (self.p.owner(self.uid), node.attr)
        else:
            t = self.p.recv_type(self.uid, node.value)
            if not t:
                return None
            k = (t, node.attr)
        if k in self.p.aslots():
            return ast.Constant(self.p.aslots()[k])
        return None

    def visit_Attribute(self, node):
        c = self._attr_const(node)
        if c is not None:
            return c
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        tg = self.p.targets(self.uid, node, exact=True)
        if len(tg) != 1:
            return node
        did, recv, ctor = tg[0]
        if did is None:
            return node
        args_pure = all(is_pure(a) for a in node.args) and all(is_pure(k.value) for k in node.keywords)
        if did in self.p.K and args_pure and not ctor:
            return ast.Constant(self.p.K[did])
        fn = self.p.units.get(did)
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
        # constant parameters of the callee leave its call sites (the callee's normal form substitutes them; a
        # slot whose constant differs between the trees is a changed unit, `Program.pslots` compared by the
        # caller); the call carries them aside (`_consts`) so that its executable form passes them again
        consts = {pn: v for (d, pn), v in self.p.pslots().items() if d == did}
        if consts and not any(isinstance(a, ast.Starred) for a in node.args):
            pos = [p[0] for p in fn_params(fn) if p[1] == "pos"]
            skip = 1 if (recv is not None or ctor) else 0
            kept = [(i, consts[pos[i + skip]]) for i, a in enumerate(node.args)
                    if i + skip < len(pos) and pos[i + skip] in consts]
            kept += [(k.arg, consts[k.arg]) for k in node.keywords if k.arg in consts]
            node.args = [a for i, a in enumerate(node.args)
                         if not (i + skip < len(pos) and pos[i + skip] in consts)]
            node.keywords = [k for k in node.keywords if k.arg not in consts]
            if kept:
                node._consts = kept
        r = self._inline(node, did, recv, ctor)
        return node if r is None else r

    def _inline(self, node, did, recv, ctor):
        """The callee's single return expression with the call's arguments bound, or None."""
        if ctor:
            return None
        r = self.p.return_expr(did, self.depth)
        if r is None:
            return None
        cfn, e = r
        params = fn_params(cfn)
        if any(p[1] in ("vararg", "kwarg") for p in params) or any(isinstance(a, ast.Starred) for a in node.args) \
                or any(k.arg is None for k in node.keywords):
            return None
        bind = {}
        pos = [p for p in params if p[1] == "pos"]
        args = list(node.args)
        if recv is not None:
            if not pos:
                return None
            bind[pos[0][0]] = recv
            pos = pos[1:]
        for p, a in zip(pos, args):
            bind[p[0]] = a
        if len(args) > len(pos):
            return None
        kws = {k.arg: k.value for k in node.keywords}
        for p in params:
            if p[0] in bind:
                continue
            if p[0] in kws:
                bind[p[0]] = kws.pop(p[0])
            elif p[2] is not None and isinstance(p[2], ast.Constant):
                bind[p[0]] = p[2]
            else:
                return None
        if kws:
            return None
        uses = _loads(e)
        for pn, a in bind.items():
            if len(uses.get(pn, [])) > 1 and not _simple(a):
                return None
        free = {n.id for n in ast.walk(e) if isinstance(n, ast.Name) and not n.id.startswith(SIG)}
        if free - set(bind) - BUILTIN_NAMES:
            return None
        out = copy.deepcopy(e)

        class Sub(ast.NodeTransformer):
            def visit_Name(s, n):
                if isinstance(n.ctx, ast.Load) and n.id in bind:
                    return copy.deepcopy(bind[n.id])
                return n
        out = Sub().visit(out)
        self.absorbed.add(did)
        self.absorbed |= self.p.absorbed.get(did, set())
        return self.visit(out)

    def drop_dead_self_stores(self, fn):
        """`self.a = <pure>` where the slot (class, a) is a constant every read of which is substituted."""
        if not self.sn:
            return
        own = self.p.owner(self.uid)
        for block in list(_blocks(fn)):
            for i in range(len(block) - 1, -1, -1):
                s = block[i]
                if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Attribute) \
                        and isinstance(s.targets[0].value, ast.Name) and s.targets[0].value.id == self.sn \
                        and (own, s.targets[0].attr) in self.p.aslots() and is_pure(s.value):
                    block.pop(i)
            if not block:
                block.append(ast.Pass())


def drop_dead_stores(fn) -> None:
    """`name = <pure, call-free>` whose name is never read in the function (and is no global): removed."""
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    glob = {x for n in ast.walk(fn) if isinstance(n, (ast.Global, ast.Nonlocal)) for x in n.names}
    for _ in range(20):
        loads = _loads(fn)
        hit = False
        for block in list(_blocks(fn)):
            for i in range(len(block) - 1, -1, -1):
                st = block[i]
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                        and st.targets[0].id not in loads and st.targets[0].id not in glob and is_pure(st.value) \
                        and not any(isinstance(x, ast.Call) for x in ast.walk(st.value)):
                    block.pop(i)
                    hit = True
            if not block:
                block.append(ast.Pass())
        if not hit:
            return


def nf_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    return dump(a) == dump(b)


class _R(ast.NodeTransformer):
    """`§id` -> `__R("id")`: a canonical form made executable in a probe; a call's stripped constant arguments
    (`_consts`, the slot's constant) passed again."""

    def visit_Call(self, node):
        self.generic_visit(node)
        kept = getattr(node, "_consts", None)
        if kept:
            args = list(node.args)
            for i, v in sorted((x for x in kept if isinstance(x[0], int)), key=lambda x: x[0]):
                args.insert(i, ast.Constant(v))
            node.args = args
            node.keywords = list(node.keywords) + [ast.keyword(k, ast.Constant(v)) for k, v in kept
                                                   if isinstance(k, str)]
        return node

    def visit_Name(self, node):
        if node.id.startswith(SIG):
            return ast.copy_location(ast.Call(func=ast.Name(id="__R", ctx=ast.Load()),
                                              args=[ast.Constant(node.id[len(SIG):])], keywords=[]), node)
        return node


def executable(node) -> str:
    n = _R().visit(copy.deepcopy(node))
    ast.fix_missing_locations(n)
    return ast.unparse(n)
