#!/usr/bin/env python3
"""The computed gate scope: which cells of an anti-regression gate a change can reach.

A gate replays only the cells the diff reaches; every other cell is CARRIED with its provenance (the gate and
row where it was last green). The scope is computed, never chosen:

    1. the diff            the units (functions, class bodies, module residues) whose normalized code changed,
                           the data files and the tools that changed;
    2. the functions       a unit counts only where a gate cell's process reaches it (`python -m neurobrix
                           <cell argv>`, each mode's argv parsed by the tree's own parser and folded in), and
                           only if its NORMAL FORM differs: canonical references, the calls the gate makes
                           constant (probed in each tree on this rack's profile), the parameters and
                           attributes every reached call site fills with one constant, single-expression
                           callees inlined, folded (gate_scope_code.Program);
    3. what they serve     a changed unit is followed to what dispatches it: the launcher's op maps (the op
                           and the argument signatures its guard admits, evaluated in the tree), an op or flow
                           type it is compared against, a registration decorator, the planner (judged by the
                           plan it emits for each cell, old tree against new), or a record domain its points
                           are probed over (the graph argument records the cell's containers hold);
    4. the cells           those features intersected with each container's runtime graph
                           (gate_scope_index) and census table, in the modes whose process reaches the unit.

What cannot be mapped falls back to every cell that loads its module, and is NAMED in scope.json.

Outputs (in --out): queue_<cls>.txt in the sched format ("model modes timeout", one line per cell),
carried.tsv (the cells not replayed, each with the gate and row it is carried from), scope.json (every
changed unit, its verdict and why).

    gate_scope.py --base 2b95ba03 --head HEAD --gate-dir <mq24> --out <dir>
    gate_scope.py --range 192a41cf..6ce86b39 ...        gate_scope.py --commits a29261aa,68333e77 ...

CPU only: every probe runs with CUDA_VISIBLE_DEVICES='' and reads the shared cache, never writes it.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import gate_scope_code as C                    # noqa: E402
import gate_scope_index as GI                  # noqa: E402

SCRATCH = Path(os.environ.get("NBX_GATE_SCOPE_SCRATCH", "/home/mlops/nbx/tmp/gate_scope"))
PYTHON = os.environ.get("NEUROBRIX_PYTHON", "/home/mlops/venvs/nbx_t214/bin/python")
CFG_PATH = TOOLS / "gate_scope.yml"
CACHE = Path.home() / ".neurobrix" / ("ca" + "che")
MARK = "@@GATE_SCOPE_PROBE@@"
DIMENSIONS = ("op", "dtype", "flow", "record", "mode", "planner")


def load_cfg(path: Path = CFG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


# =============================================================================================================
# git and trees
# =============================================================================================================

def git(repo: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: git {' '.join(args)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def full_sha(repo: Path, rev: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()


def main_checkout(repo: Path) -> Path:
    """The main working tree of the repository (a worktree's machine files live there)."""
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    return common.parent


def tree_of(repo: Path, sha: str, scratch: Path = SCRATCH) -> Path:
    """The tree of one commit, materialized once under scratch/trees/<sha12> (git archive), with the hardware
    profiles the engine reads but git does not carry (config/hardware/*.yml are machine files of the main
    checkout)."""
    d = scratch / "trees" / sha[:12]
    if (d / ".complete").is_file():
        return d
    tmp = d.with_name(d.name + f".{os.getpid()}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arch = subprocess.run(["git", "-C", str(repo), "archive", sha, "src", "tools"], capture_output=True)
    if arch.returncode != 0:
        raise SystemExit(f"REFUSED: git archive {sha} failed: {arch.stderr.decode()[:200]}")
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch.stdout, check=True)
    hw_src = main_checkout(repo) / "src" / "neurobrix" / "config" / "hardware"
    hw_dst = tmp / "src" / "neurobrix" / "config" / "hardware"
    hw_dst.mkdir(parents=True, exist_ok=True)
    for y in hw_src.glob("*.yml"):
        if not (hw_dst / y.name).exists():
            shutil.copy2(y, hw_dst / y.name)
    (tmp / ".complete").write_text(sha + "\n")
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)
    return d


def change_pairs(repo: Path, base=None, head=None, rng=None, commits=None) -> list:
    """[(old sha, new sha)] — one pair per constituent: `--base B --head H`, `--range A..B`, or `--commits`
    (each `sha` against its first parent, or `A..B`)."""
    out = []
    if base or head:
        if not (base and head):
            raise SystemExit("REFUSED: --base and --head go together")
        out.append((full_sha(repo, base), full_sha(repo, head)))
    if rng:
        a, b = rng.split("..")
        out.append((full_sha(repo, a), full_sha(repo, b)))
    for c in (commits or "").split(","):
        c = c.strip()
        if not c:
            continue
        if ".." in c:
            a, b = c.split("..")
            out.append((full_sha(repo, a), full_sha(repo, b)))
        else:
            s = full_sha(repo, c)
            out.append((full_sha(repo, s + "^"), s))
    if not out:
        raise SystemExit("REFUSED: no change named (--base/--head, --range or --commits)")
    return out


# =============================================================================================================
# the probe (tools/gate_scope_probe.py) — the tree's own code, run on this rack's profile, no card
# =============================================================================================================

def probe(tree: Path, hw, jobs: list, cfg: dict, rack: bool = True) -> dict:
    """Run probe jobs inside one tree (cached by tree, class and jobs)."""
    req = {"src": str(tree / "src"), "hw": hw, "jobs": jobs}
    if rack:
        req["rack"] = dict(cfg["rack"], package=cfg["package"])
    key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:24]
    cf = SCRATCH / "probes" / f"{tree.name}_{key}.json"
    if cf.is_file():
        return json.loads(cf.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NBX_", "NEUROBRIX_"))}
    for k in cfg.get("env_unset", []):
        env.pop(k, None)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(tree / "src"), PYTHONNOUSERSITE="1")
    p = subprocess.run([PYTHON, str(TOOLS / "gate_scope_probe.py")], input=json.dumps(req), capture_output=True,
                       text=True, env=env, cwd=str(SCRATCH))
    if MARK not in p.stdout:
        err = (p.stdout + p.stderr).strip()[-600:]
        raise SystemExit(f"REFUSED: the probe failed in {tree.name}: {err}")
    out = json.loads(p.stdout.rsplit(MARK, 1)[1])
    if "error" in out and len(out) == 1:
        raise SystemExit(f"REFUSED: the probe refused in {tree.name}: {out['error']}")
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


# =============================================================================================================
# one tree: index, units, the reach of each mode
# =============================================================================================================

class Tree:
    def __init__(self, repo: Path, sha: str, cfg: dict):
        self.sha, self.cfg = sha, cfg
        self.path = tree_of(repo, sha)
        self.index = C.load_index(self.path / "src", SCRATCH / "code")
        self.units = C.unit_nodes(self.index)
        self.reach = {}          # mode -> {uid: witness}
        self._prog = None
        self.K = {}

    def file_units(self, rel: str) -> dict:
        p = self.path / rel
        if not p.is_file():
            return {}
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            return {}
        return C.units_of(tree)

    def mode_reach(self, cells_by_mode: dict, cell_argvs: dict) -> None:
        """Each mode's reach: the argv attributes every cell of the mode is launched with, parsed by the tree's
        own parser, folded into the functions that receive the namespace."""
        a = self.cfg["argv"]
        argvs = {f"{m}|{mode}": cell_argvs[(m, mode)] for (m, mode) in cell_argvs}
        res = probe(self.path, None, [{"id": "argv", "kind": "argv", "parser": a["parser"], "argvs": argvs}],
                    self.cfg, rack=False)["argv"]
        if "error" in res:
            raise SystemExit(f"REFUSED: the parser of {self.sha[:8]} failed: {res['error']}")
        ns = C.namespace_params(self.index, self.units, a["seed"], a["scope"])
        env_subst = env_unset_subst(self.index, self.units, self.cfg.get("env_unset", []))
        self.argv_consts = {}
        for mode, models in cells_by_mode.items():
            vals, seen = None, {}
            for m in models:
                v = res[f"{m}|{mode}"]
                if "__error__" in v:
                    raise SystemExit(f"REFUSED: {self.sha[:8]}'s parser refuses the cell {m}/{mode}: {v['__error__']}")
                v = {k: x for k, x in v.items() if k != "__func__"}
                for k, x in v.items():
                    seen.setdefault(k, []).append(x)
                if vals is None:
                    vals = dict(v)
                else:
                    vals = {k: x for k, x in vals.items() if k in v and v[k] == x and type(v[k]) is type(x)}
            vals = {k: x for k, x in (vals or {}).items() if x is None or isinstance(x, (bool, int, float, str))}
            # an attribute that varies across the cells: its values, for the comparisons they all agree on
            nonnull = {k: xs for k, xs in seen.items() if k not in vals and len(xs) == len(models)
                       and all(x is None or isinstance(x, (bool, int, float, str)) for x in xs)}
            # optimistic: every attribute the cells agree on is a constant, until a store of it survives in the
            # code folded under that assumption (a store in a branch the constants kill does not count)
            for _ in range(20):
                folded = {uid: C.fold(self.units[uid], ns_subst(names, vals, nonnull))[0]
                          for uid, names in ns.items()}
                live = C.stored_ns_attrs(folded, ns) & (set(vals) | set(nonnull))
                if not live:
                    break
                vals = {k: x for k, x in vals.items() if k not in live}
                nonnull = {k: x for k, x in nonnull.items() if k not in live}
            self.argv_consts[mode] = dict(vals, **{k: f"<{len(set(map(repr, x)))} values>" for k, x in nonnull.items()})

            folds = {}                                   # (unit, the slots of its class) -> its fold

            def make_fold(asl, P, psl=None, vals=vals, nonnull=nonnull):
                by_cls, by_fn = {}, {}
                for (c, a), v in asl.items():
                    by_cls.setdefault(c, []).append((a, repr(v)))
                for (d, pn), v in (psl or {}).items():
                    by_fn.setdefault(d, []).append((pn, repr(v)))

                def fold_for(uid, node):
                    own = P.owner(uid) if P is not None else None
                    key = (uid, tuple(sorted(by_cls.get(own, ()))), tuple(sorted(by_fn.get(uid, ()))))
                    if key not in folds or folds[key][0] is not node:
                        folds[key] = (node, fold_one(uid, node))
                    return folds[key][1]

                def fold_one(uid, node):
                    names = ns.get(uid)
                    argv = ns_subst(names, vals, nonnull) if names else None
                    sn, own = (P.self_name(uid), P.owner(uid)) if P is not None else (None, None)
                    # a parameter every reached call passes the same literal (never rebound in the body)
                    pc = {pn: (psl or {})[(uid, pn)] for pn, _ in by_fn.get(uid, ())
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and not C._stores(node).get(pn)}

                    def subst(n):
                        r = (argv(n) if argv else None) or env_subst(n)
                        if r is None and isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in pc:
                            r = ast.Constant(pc[n.id])
                        if r is None and sn and isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) \
                                and isinstance(n.value, ast.Name) and n.value.id == sn and (own, n.attr) in asl:
                            r = ast.Constant(asl[(own, n.attr)])
                        # getattr(self, "a"[, d]): the slot's literal — with a default, only when it is that literal
                        # too (an instance the attribute was never stored on answers d)
                        if r is None and sn and isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
                                n.func.id == "getattr" and not n.keywords and len(n.args) in (2, 3) and \
                                isinstance(n.args[0], ast.Name) and n.args[0].id == sn and \
                                isinstance(n.args[1], ast.Constant) and (own, n.args[1].value) in asl:
                            v = asl[(own, n.args[1].value)]
                            if len(n.args) == 2 or (isinstance(n.args[2], ast.Constant) and n.args[2].value == v
                                                    and type(n.args[2].value) is type(v)):
                                r = ast.Constant(v)
                        return r
                    return C.fold(node, subst)
                return fold_for
            fold_for = make_fold({}, None)
            R = C.Reach(self.index, self.units, fold_for)
            R.run([self.cfg["entry"]])
            # the attribute slots this mode's own reach makes constant (every store of `self.a` it reaches writes
            # the same literal — an argv value handed down to a constructor, say): folded in, reach re-run, while
            # it shrinks. Sound at every round: the slots come from the stores of a reach that over-approximates
            # the mode's execution, so the reach they fold over-approximates it too; the smallest one is kept.
            for _ in range(len(self.units)):
                fu = dict(self.units)
                fu.update({u: fold_for(u, self.units[u])[0] for u in R.reached if u in self.units})
                # every reached call site fills a slot here, the planner's too (it runs in the mode's process)
                P = C.Program(self.index, fu, set(R.reached))
                asl = {k: v for k, v in P.aslots().items() if v is None or isinstance(v, (bool, int, float, str))}
                psl = {k: v for k, v in P.pslots().items() if v is None or isinstance(v, (bool, int, float, str))}
                f2 = make_fold(asl, P, psl)
                R2 = C.Reach(self.index, self.units, f2)
                R2.run([self.cfg["entry"]])
                if len(R2.reached) >= len(R.reached):
                    break
                R, fold_for = R2, f2
            self.reach[mode] = R

    def reached(self) -> set:
        out = set()
        for R in self.reach.values():
            out |= set(R.reached)
        return out

    def program(self) -> C.Program:
        if self._prog is None:
            self._prog = C.Program(self.index, self.units, self.reached(), K=self.K,
                                   slot_exclude=tuple(self.cfg["planner"]["packages"]))
        return self._prog

    def reset_program(self):
        self._prog = None


def ns_subst(names, vals, domain):
    """The argv substitution of a namespace function, plus a comparison of an attribute the cells set to
    different values (`args.x is None`, `getattr(args, "x", d) == "lit"`, `args.x in (...)`) when every cell's
    value gives it the same result."""
    base = C.argv_subst(names, vals, set())

    def lit(e):
        if isinstance(e, ast.Constant):
            return True, e.value
        if isinstance(e, (ast.Tuple, ast.List, ast.Set)) and all(isinstance(x, ast.Constant) for x in e.elts):
            return True, tuple(x.value for x in e.elts)
        return False, None

    def subst(n):
        r = base(n)
        if r is not None:
            return r
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in C._CMP:
            x = n.left
            attr = None
            if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in names:
                attr = x.attr
            elif isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "getattr" \
                    and len(x.args) >= 2 and isinstance(x.args[0], ast.Name) and x.args[0].id in names \
                    and isinstance(x.args[1], ast.Constant):
                attr = x.args[1].value
            ok, v = lit(n.comparators[0])
            if attr in domain and ok:
                try:
                    res = {bool(C._CMP[type(n.ops[0])](d, v)) for d in domain[attr]}
                except Exception:              # noqa: BLE001
                    return None
                if len(res) == 1:
                    return ast.Constant(res.pop())
        return None
    return subst


def env_unset_subst(index, units, names):
    """`<os>.environ.get("N"[, d])`, `os.getenv("N"[, d])` -> d (None) for a variable the gate never sets and
    no code of the tree sets — a cell's process reads it unset."""
    names = set(names)
    for uid, node in units.items():
        for n in ast.walk(node):
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.slice, ast.Constant):
                names.discard(n.slice.value)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("setdefault", "update") \
                    and n.args and isinstance(n.args[0], ast.Constant):
                names.discard(n.args[0].value)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "putenv" \
                    and n.args and isinstance(n.args[0], ast.Constant):
                names.discard(n.args[0].value)

    def subst(n):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value in names and not n.keywords:
            f = n.func
            env_get = f.attr == "get" and isinstance(f.value, ast.Attribute) and f.value.attr == "environ"
            getenv = f.attr == "getenv"
            if env_get or getenv:
                if len(n.args) == 1:
                    return ast.Constant(None)
                if len(n.args) == 2 and isinstance(n.args[1], ast.Constant):
                    return ast.Constant(n.args[1].value)
        return None
    return subst


# =============================================================================================================
# the gate's constants: calls whose value is fixed where a cell runs (probed, then checked for writers)
# =============================================================================================================

MUTATORS = {"append", "extend", "insert", "remove", "pop", "popitem", "clear", "update", "setdefault", "add",
            "discard", "__setitem__", "__delitem__", "set", "put"}


def global_writers(tree: Tree) -> dict:
    """{(module, global name): {writer uid: {store lines}}} — a `global` store, an attribute/subscript store or a
    mutating method call on it, setattr on it; a module's own residue (its initializer) is not a writer."""
    P = tree.program()
    out = {}
    for uid in tree.units:
        mod = uid.split(":")[0]
        try:
            fn = P.canon(uid)
        except Exception:                     # noqa: BLE001 — a unit the index cannot canonicalize writes nothing known
            continue
        decl = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Global):
                decl |= set(n.names)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id in decl:
                out.setdefault((mod, n.id), {}).setdefault(uid, set()).add(getattr(n, "lineno", 0))
            root = None
            if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
                root = n.value
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS:
                root = n.func.value
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("setattr", "delattr") \
                    and n.args:
                root = n.args[0]
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if root is not None and C.is_ref(root):
                rid = C.ref_id(root)
                if uid != f"{rid.split(':')[0]}:<module>":
                    out.setdefault(tuple(rid.split(":", 1)), {}).setdefault(uid, set()).add(getattr(n, "lineno", 0))
    return out


def _line_globals(tree: Tree, lines) -> set:
    """The module globals the executed lines read: {(module, name)}."""
    by_file = {}
    for rel, ln in lines:
        by_file.setdefault(rel, set()).add(ln)
    out = set()
    rel2mod = {str(Path(mi.rel).relative_to("src")) if mi.rel.startswith("src/") else mi.rel: mi
               for mi in tree.index.mods.values()}
    for rel, lns in by_file.items():
        mi = rel2mod.get(rel)
        if mi is None:
            continue
        for n in ast.walk(mi.tree):
            if getattr(n, "lineno", None) in lns:
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in mi.globals_:
                    out.add((mi.name, n.id))
                elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                    r = tree.index.resolve_expr(mi.name, n.value, {})
                    if r and r[0] == "mod" and n.attr in tree.index.mods[r[1]].globals_:
                        out.add((r[1], n.attr))
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    imp = mi.imports.get(n.id)
                    if imp and imp[0] == "sym" and imp[1] in tree.index.mods and imp[2] in tree.index.mods[imp[1]].globals_:
                        out.add((imp[1], imp[2]))
    return out


def gate_constants(tree: Tree, candidates: set, hw: str, writers: dict) -> tuple:
    """({callee id: literal}, {callee id: why refused}, {callee id: [env names read]})."""
    jobs = []
    for did in sorted(candidates):
        fn = tree.units.get(did)
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = fn.args
        req = len(a.posonlyargs + a.args) - len(a.defaults)
        if any(d is None for d in a.kw_defaults):
            continue
        mod, q = did.split(":", 1)
        jobs.append({"id": did, "kind": "const", "target": f"{mod}:{q}", "nparams": req})
    if not jobs:
        return {}, {}, {}
    res = probe(tree.path, hw, jobs, tree.cfg)
    setup = tree.cfg["rack"]["setup"]
    P = tree.program()
    sites = P.sites()
    reached = tree.reached()
    # the rack setup and the functions only it calls
    allowed = {setup}
    for _ in range(20):
        grew = False
        for did, ss in sites.items():
            if did not in allowed and ss and all(s[0] in allowed for s in ss):
                allowed.add(did)
                grew = True
        if not grew:
            break
    # a writer is harmless when it is unreachable, the rack setup (or only called by it), or itself a gate
    # constant whose probe never executed its store line: an optimistic fixpoint over the candidates
    rel_of = {uid: tree.index.mods[uid.split(":")[0]].rel for uid in tree.units}

    def src_rel(uid):
        r = rel_of[uid]
        return r[len("src/"):] if r.startswith("src/") else r
    ok = {j["id"] for j in jobs if res.get(j["id"], {}).get("constant")}
    why = {}
    for _ in range(20):
        drop = set()
        for jid in ok:
            r = res[jid]
            ran = {(f, ln) for f, ln in r.get("lines", [])}
            bad = []
            for g in _line_globals(tree, r.get("lines", [])):
                for w, lns in writers.get(g, {}).items():
                    if w in allowed or w not in reached:
                        continue
                    if (w == jid or w in ok) and not any((src_rel(w), ln) in ran or
                                                         (src_rel(w), ln) in {(f, l) for f, l in res.get(w, {}).get("lines", [])}
                                                         for ln in lns):
                        continue
                    bad.append(f"{g[0]}:{g[1]} written by {w}")
            if bad:
                drop.add(jid)
                why[jid] = "; ".join(sorted(bad)[:3])
        if not drop:
            break
        ok -= drop
    K, refused, envs = {}, {}, {}
    for j in jobs:
        r = res.get(j["id"], {})
        if j["id"] in ok:
            K[j["id"]] = r["value"]
            if r.get("env"):
                envs[j["id"]] = r["env"]
        elif j["id"] in why:
            refused[j["id"]] = why[j["id"]]
    return K, refused, envs


# =============================================================================================================
# the diff
# =============================================================================================================

def changed(old: Tree, new: Tree) -> dict:
    """{"units": [(kind, file, uid)], "data": [file], "tools": [file], "other": [file]}."""
    files = git(REPO, "diff", "--name-only", old.sha, new.sha).split()
    out = {"units": [], "data": [], "tools": [], "other": []}
    pkg_src = new.cfg["src"]
    for f in files:
        if f.startswith(pkg_src + "/") and f.endswith(".py"):
            mod = C.module_name(f)
            ua, ub = old.file_units(f), new.file_units(f)
            for q in sorted(set(ua) | set(ub)):
                uid = f"{mod}:{q}"
                if q not in ua:
                    out["units"].append(("add", f, uid))
                elif q not in ub:
                    out["units"].append(("del", f, uid))
                elif C.dump(C.normalized(ua[q])) != C.dump(C.normalized(ub[q])):
                    out["units"].append(("chg", f, uid))
        elif f.startswith(pkg_src + "/"):
            out["data"].append(f)
        elif f.startswith("tools/"):
            out["tools"].append(f)
        else:
            out["other"].append(f)
    return out


def call_candidates(tree: Tree, uids) -> set:
    """The statically unique callees of units (and of the callees a normal form may inline, two levels)."""
    P = tree.program()
    out, frontier = set(), set(u for u in uids if u in tree.units)
    for _ in range(3):
        nxt = set()
        for uid in frontier:
            try:
                fn = P.canon(uid)
            except Exception:                 # noqa: BLE001
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    for did, recv, ctor in P.targets(uid, n, exact=True):
                        if did and not ctor and did not in out:
                            out.add(did)
                            nxt.add(did)
        frontier = nxt
    return out


# =============================================================================================================
# the cell side: the gate's cells, their containers' features
# =============================================================================================================

class Cells:
    """The cells of the gate the scope is computed against: {cls: {(model, mode)}} from its lists, each cell's
    request from the gate's rows (the request the cell ran), each container's record (gate_scope_index)."""

    @staticmethod
    def name(recorded: str) -> str:
        """The container a gate record names, in the cache today: its current name once the container was
        renamed there (`container_renames.current_name`, the harness's table), the recorded one until then."""
        import regression_matrix as RM
        now = RM.current_name(recorded)
        return now if now != recorded and (CACHE / now).is_dir() else recorded

    def __init__(self, gate_dir: Path, cfg: dict, src: Path):
        import regression_matrix as RM
        self.RM, self.dir, self.cfg = RM, Path(gate_dir), cfg
        g = cfg["gate"]
        self.cells, self.rows, self.request = {}, {}, {}
        for cls in cfg["classes"]:
            lst = self.dir / g["lists"].format(cls=cls)
            if not lst.is_file():
                raise SystemExit(f"REFUSED: the gate {self.dir} has no list for {cls} ({lst})")
            # a list names a container as it was called when the gate ran; it is one cell with today's name (the
            # harness's own rename record, `container_renames.current_name`, as `load_rows` reads rows)
            self.cells[cls] = {(self.name(m), mo) for m, mo in RM.cells_of_lists([lst]) if mo in RM.GATE_MODES}
            rd = self.dir / g["rows"].format(cls=cls)
            self.rows[cls] = {(self.name(r.get("recorded_as", r["model"])), r["mode"]): r
                              for r in RM.load_rows(rd)} if rd.is_dir() else {}
            for (m, _), r in self.rows[cls].items():
                if r.get("request") is not None:
                    self.request.setdefault(m, r["request"])
        self.timeouts = {}
        tf = self.dir / g["timeouts"]
        if tf.is_file():
            for line in tf.read_text().splitlines():
                p = line.split()
                if len(p) >= 2 and not p[0].startswith("#"):
                    self.timeouts[self.name(p[0])] = int(p[1])
        self.models = sorted({m for cs in self.cells.values() for m, _ in cs})
        self.recs = GI.records(CACHE, self.models, SCRATCH / "index", PYTHON, src)
        self.gone = sorted(m for m, r in self.recs.items() if r.get("gone"))
        # a container whose graphs are newer than the gate's row of it was retraced after the gate ran: the gate
        # judged another container, so every cell of it is replayed, whatever the diff
        self.retraced = {}
        for cls in self.cells:
            for (m, mo) in self.cells[cls]:
                r = self.rows[cls].get((m, mo))
                t = RM._row_time(r) if r else 0.0
                newest = max((st[2] for st in self.recs[m]["stamp"]), default=0)
                if r is None or newest > t:
                    self.retraced.setdefault(m, "no gate row" if r is None else
                                             f"graphs of {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(newest))}"
                                             f" newer than its gate row ({r.get('recorded_as', m)}, {r['date']})")
        self.argv = {}
        for cs in self.cells.values():
            for m, mode in cs:
                self.argv[(m, mode)] = RM.cell_argv(m, mode, "/dev/null", req=self.request.get(m))
        self.by_mode = {}
        for cs in self.cells.values():
            for m, mode in cs:
                self.by_mode.setdefault(mode, set()).add(m)
        # what a value of each kind names: the models whose graphs hold it
        self.holders = {}
        for m, r in self.recs.items():
            for x in r["ops"]:
                self.holders.setdefault(("op", x), set()).add(m)
            for x in r["arg_records"]:
                self.holders.setdefault(("rec", x), set()).add(m)
            # a string an op's argument record holds, as written (the values a parameter fed from a record takes)
            for x in r["strings"]:
                self.holders.setdefault(("str", x), set()).add(m)
            # a dtype the graph holds anywhere — an argument's or a tensor's (the values of a dtype a value probe
            # builds a tensor in)
            for x in set(r["explicit_dtypes"]) | set(r["tensor_dtypes"]):
                self.holders.setdefault(("dtype", x), set()).add(m)
            if r.get("flow"):
                self.holders.setdefault(("flow", r["flow"]), set()).add(m)
            for t, sigs in r["sigs"].items():
                for s in sigs:
                    self.holders.setdefault(("sig", t, s), set()).add(m)
        self.all_ops = {k[1] for k in self.holders if k[0] == "op"}
        self.all_flows = {k[1] for k in self.holders if k[0] == "flow"}

    def of(self, cls, models, modes) -> set:
        return {(m, mode) for (m, mode) in self.cells[cls] if m in models and mode in modes}

    def every(self, cls, modes) -> set:
        return {(m, mode) for (m, mode) in self.cells[cls] if mode in modes}


# =============================================================================================================
# statement-level diff of two normal forms
# =============================================================================================================

def _header(s) -> str:
    h = copy.copy(s)
    for f in ("body", "orelse", "finalbody", "handlers"):
        if isinstance(getattr(h, f, None), list):
            setattr(h, f, [])
    return C.dump(h)


def stmt_diff(a: list, b: list, path=()) -> list:
    """[(kind, [old stmts], [new stmts], path)] — kind ins|del|rep; `path` the (statement, branch) chain of the
    compound statements above (both sides share it). Compound statements with equal headers are descended."""
    da, db = [C.dump(s) for s in a], [C.dump(s) for s in b]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=da, b=db, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        sa, sb = a[i1:i2], b[j1:j2]
        if op == "replace" and len(sa) == len(sb):
            for x, y in zip(sa, sb):
                if type(x) is type(y) and _header(x) == _header(y) and \
                        any(isinstance(getattr(x, f, None), list) for f in ("body", "orelse")):
                    for f in ("body", "orelse", "finalbody"):
                        bx, by = getattr(x, f, None), getattr(y, f, None)
                        if isinstance(bx, list) and isinstance(by, list) and C.dump(ast.Module(bx, [])) != \
                                C.dump(ast.Module(by, [])):
                            out += stmt_diff(bx, by, path + ((y, f),))
                    hx, hy = getattr(x, "handlers", None), getattr(y, "handlers", None)
                    if hx and hy and len(hx) == len(hy):
                        for k, (u, v) in enumerate(zip(hx, hy)):
                            if C.dump(u) != C.dump(v):
                                out += stmt_diff(u.body, v.body, path + ((y, f"handler{k}"),))
                elif C.dump(x) != C.dump(y):
                    out.append(("rep", [x], [y], path))
            continue
        out.append(({"insert": "ins", "delete": "del", "replace": "rep"}[op], sa, sb, path))
    return out


def _leaves(body) -> bool:
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))


def region_guard(fn, target_ids: set, extra_test=None):
    """The condition under which execution reaches the first statement of a region of `fn` (identified by
    node ids), as a list of terms over the function's parameters — the enclosing tests (negated in an else),
    the negated tests of earlier `if T:` siblings whose body always leaves — plus the simple assignments that
    precede it on the path (they bind names the terms read). None when the region is inside a loop, a `with`
    or a `try` (the path is not a condition of the parameters)."""
    res = []

    def block(body, cond, pre):
        cond, pre = list(cond), list(pre)
        for s in body:
            if id(s) in target_ids:
                res.append((cond + ([extra_test] if extra_test is not None else []), pre))
                return True
            if isinstance(s, ast.If):
                if block(s.body, cond + [s.test], pre) or \
                        block(s.orelse, cond + [ast.UnaryOp(op=ast.Not(), operand=s.test)], pre):
                    return True
                if _leaves(s.body) and not s.orelse:
                    cond.append(ast.UnaryOp(op=ast.Not(), operand=s.test))
            elif isinstance(s, ast.Try) and any(id(x) in target_ids for x in ast.walk(ast.Module(s.body, []))):
                # the try body is entered unconditionally; an earlier raise in it leaves the region unreached in
                # both trees alike (the probe runs the same prefix), and the region's own raise is its result
                if block(s.body, cond, pre):
                    return True
                res.append(None)
                return True
            elif isinstance(s, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                for f in ("body", "orelse", "finalbody"):
                    v = getattr(s, f, None)
                    if isinstance(v, list) and any(id(x) in target_ids for x in ast.walk(ast.Module(v, []))):
                        res.append(None)
                        return True
                for h in getattr(s, "handlers", []) or []:
                    if any(id(x) in target_ids for x in ast.walk(h)):
                        res.append(None)
                        return True
                pre.append(None)                  # a compound statement on the path: what it binds is unknown
            elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
                pre.append(s)
        return False
    block(fn.body, [], [])
    if not res or res[0] is None:
        return None
    return res[0]


def guard_source(fn, cond: list, pre: list, name: str) -> str:
    """A probe function with `fn`'s parameters returning whether the path `cond` holds (short-circuit, as the
    code evaluates it); a binding the path needs but the prefix cannot give raises — a hit, conservatively."""
    g = ast.FunctionDef(name=name, args=copy.deepcopy(fn.args), body=[], decorator_list=[], returns=None,
                        type_comment=None, type_params=[])
    for s in pre:
        if s is not None:
            g.body.append(copy.deepcopy(s))
    test = ast.BoolOp(op=ast.And(), values=[ast.Constant(True)] + [copy.deepcopy(t) for t in cond]) \
        if cond else ast.Constant(True)
    g.body.append(ast.Return(ast.Call(func=ast.Name("bool", ast.Load()), args=[test], keywords=[])))
    m = ast.Module([g], [])
    ast.fix_missing_locations(m)
    return C.executable(m)


def free_names(fn) -> set:
    """Names a function reads that it neither binds nor gets from builtins (a canonical reference is bound)."""
    import builtins
    bound = C.fn_locals(fn) | set(dir(builtins))
    for n in ast.walk(fn):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in n.names}
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                bound |= {x.id for x in ast.walk(g.target) if isinstance(x, ast.Name)}
        if isinstance(n, ast.Lambda):
            bound |= C._params(n)
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and not n.id.startswith(C.SIG)} - bound


# =============================================================================================================
# judging one changed unit (one change pair, one memory class)
# =============================================================================================================

def V(verdict, why="", cells=(), **kw) -> dict:
    return dict(verdict=verdict, why=why, cells=set(cells), **kw)


class Judge:
    """What one changed unit serves, for one class: the cells whose process can see the change."""

    def __init__(self, old: Tree, new: Tree, cells: Cells, cls: str, changed_ids: set, cfg: dict):
        self.old, self.new, self.cells, self.cls, self.cfg = old, new, cells, cls, cfg
        self.hw = cfg["classes"][cls]
        self.changed_ids = changed_ids
        self.Po, self.Pn = old.program(), new.program()
        self.plan_cells = set()
        self.probed = {}            # (tree, unit) -> [(first line, last line)] of the code a probe executed
        self._ops_of = None
        self._writers = {}
        self._readers = {}
        self.limits = cfg["limits"]

    # -- helpers -------------------------------------------------------------------------------------------
    def modes(self, uid) -> set:
        return {m for t in (self.old, self.new) for m, R in t.reach.items() if uid in R.reached}

    def k_equal(self, uid) -> bool:
        return uid in self.old.K and uid in self.new.K and self.old.K[uid] == self.new.K[uid] \
            and type(self.old.K[uid]) is type(self.new.K[uid])

    def run_points(self, tree, mod, source, entry, points, jid, guard_builtins=False):
        job = {"id": jid, "kind": "points", "module": mod, "source": source, "entry": entry, "points": points}
        if guard_builtins:
            job["guard_builtins"] = True
        return probe(tree.path, self.hw, [job], self.cfg)[jid]

    def writers(self, tree):
        if tree.sha not in self._writers:
            self._writers[tree.sha] = global_writers(tree)
        return self._writers[tree.sha]

    def readers(self, tree, g) -> set:
        """The reached units of a tree that read the module global g = (module, name)."""
        key = (tree.sha, g)
        if key not in self._readers:
            ref = C.SIG + f"{g[0]}:{g[1]}"
            P = tree.program()
            out = set()
            for uid in tree.reached():
                if uid not in tree.units:
                    continue
                try:
                    fn = P.canon(uid)
                except Exception:                 # noqa: BLE001
                    continue
                if any(isinstance(n, ast.Name) and n.id == ref for n in ast.walk(fn)):
                    out.add(uid)
            self._readers[key] = out
        return self._readers[key]

    def neutral_global(self, g) -> tuple:
        """(True, "") when every reached reader of global g, in both trees, other than its writers, is a gate
        constant equal in both trees — its value cannot reach a cell — else (False, the readers that can)."""
        bad = []
        for t in (self.old, self.new):
            ws = set(self.writers(t).get(g, {}))
            for r in sorted(self.readers(t, g) - ws):
                if not self.k_equal(r):
                    bad.append(r)
        return (not bad), sorted(set(bad))

    # -- the verdict ---------------------------------------------------------------------------------------
    def judge(self, kind, uid) -> dict:
        modes = self.modes(uid)
        if not modes:
            return V("unreached", "no gate cell's process reaches it")
        if kind in ("add", "del"):
            t = self.new if kind == "add" else self.old
            node = t.units.get(uid)
            sites = (self.Pn if kind == "add" else self.Po).sites().get(uid)
            if getattr(node, "decorator_list", None) and not sites and not uid.endswith((".<body>", ":<module>")):
                return self.walk(uid, None, [], modes, why=f"{kind}ed with a decorator (it registers itself)")
            return V("callers", f"{kind}ed; judged at the units that call it (each changed one is judged)")
        a, b = self.Po.nf(uid), self.Pn.nf(uid)
        if C.nf_equal(a, b):
            return V("equal", "normal forms equal at the gate")
        if self.k_equal(uid):
            return V("k-equal", f"a gate constant equal in both trees ({self.new.K[uid]!r})")
        if a is None or b is None:
            return self.fallback(uid, modes, "no normal form (inlining depth or recursion)")
        if uid.split(":")[0].startswith(tuple(self.cfg["planner"]["packages"])):
            cs = self.cells.every(self.cls, modes)
            self.plan_cells |= cs
            return V("plan", f"planner unit: judged by the plan each of {len(cs)} cells receives", plan=True)
        if uid.endswith(":<module>") or uid.endswith(".<body>"):
            return self.module_rule(uid, a, b, modes)
        diffs = stmt_diff(a.body, b.body)
        if not diffs:
            return self.fallback(uid, modes, "signature changed")
        if self.writer_drop(uid, diffs):
            return V("writer-drop", "inserted writer calls whose globals only equal gate constants read")
        r = self.point_probe(uid, a, b, modes)
        if r is not None:
            return r
        r = self.region_probe(uid, a, b, diffs, modes)
        if r is not None:
            return r
        r = self.expr_probe(uid, a, b, diffs, modes)
        if r is not None:
            return r
        r = self.suffix_probe(uid, a, b, modes)
        if r is not None:
            return r
        return self.walk(uid, (a, b), diffs, modes)

    def fallback(self, uid, modes, why) -> dict:
        return V("fallback", why, self.cells.every(self.cls, modes), fallback=True)

    # -- module residues: each changed global assignment ---------------------------------------------------
    def module_rule(self, uid, a, b, modes) -> dict:
        mod = uid.split(":")[0]
        names, other = set(), []
        # the decorators of a definition added or deleted (the residue carries them as a tuple): judged with
        # that definition, which is its own unit
        deco = {}
        for t, o, k in ((self.new, self.old, "ins"), (self.old, self.new, "del")):
            for u, n in t.units.items():
                if u.split(":")[0] == mod and u not in o.units and getattr(n, "decorator_list", None):
                    deco.setdefault(k, []).append(len(n.decorator_list))
        for kind, sa, sb, path in stmt_diff(a.body, b.body):
            for s in sa + sb:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Pass)):
                    continue                       # a definition is its own unit, judged as one
                if kind in deco and not path and isinstance(s, ast.Expr) and isinstance(s.value, ast.Tuple) \
                        and len(s.value.elts) in deco[kind]:
                    deco[kind].remove(len(s.value.elts))
                    continue
                tg = s.targets if isinstance(s, ast.Assign) else [s.target] if isinstance(s, ast.AnnAssign) else None
                if path or tg is None or not all(isinstance(x, ast.Name) for x in tg):
                    other.append(ast.unparse(s)[:80])
                    continue
                names |= {x.id for x in tg}
        why = []
        for name in sorted(names):
            if uid.endswith(".<body>"):
                other.append(f"class attribute {name}")
                continue
            src = f"def __v():\n    return {name}\n"
            ra = self.run_points(self.old, mod, src, "__v", [{}], f"val:{mod}:{name}")
            rb = self.run_points(self.new, mod, src, "__v", [{}], f"val:{mod}:{name}")
            if "error" not in ra and "error" not in rb and ra["results"] == rb["results"]:
                why.append(f"{name}: value equal")
                continue
            ok, bad = self.neutral_global((mod, name))
            if ok:
                why.append(f"{name}: read only by equal gate constants")
                continue
            other.append(f"{name} read by {', '.join(bad[:4])}")
        if other:
            return self.fallback(uid, modes, "module statements: " + "; ".join(other[:6]))
        return V("module-equal", "; ".join(why))

    # -- an inserted writer call whose effect no cell can read -------------------------------------------
    def writer_drop(self, uid, diffs) -> bool:
        w = self.writers(self.new)
        for kind, sa, sb, path in diffs:
            if kind != "ins":
                return False
            for s in sb:
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and C.is_ref(s.value.func)):
                    return False
                if uid != self.cfg["rack"]["setup"] and any(isinstance(x, ast.Call) for a in s.value.args
                                                             for x in ast.walk(a)):
                    return False                    # an argument computed off the rack setup could raise
                did = C.ref_id(s.value.func)
                gs = {g for g, ws in w.items() if did in ws}
                if not gs or not all(self.neutral_global(g)[0] for g in gs):
                    return False
        return True

    # -- point probe: the unit over the finite domain its parameters take at the gate -------------------------
    def domain(self, nparams):
        recs = sorted({k[1] for k in self.cells.holders if k[0] == "rec"})
        strs = sorted({k[1] for k in self.cells.holders if k[0] == "str"})
        if nparams == 1:
            return [{"lit": json.loads(r)} for r in recs] + [{"lit": s} for s in strs], "records+strings"
        return [{"lit": s} for s in strs], "strings"

    def _sources(self, a, b):
        sa, sb = C.executable(ast.Module([a], [])), C.executable(ast.Module([b], []))
        return sa, sb, ast.parse(sa).body[0], ast.parse(sb).body[0]

    def point_probe(self, uid, a, b, modes):
        if not isinstance(a, ast.FunctionDef) or not isinstance(b, ast.FunctionDef):
            return None
        if C.dump(a.args) != C.dump(b.args) or a.args.vararg or a.args.kwarg or a.args.kwonlyargs:
            return None
        params = [x.arg for x in a.args.posonlyargs + a.args.args]
        used = [p for p in params if any(isinstance(n, ast.Name) and n.id == p for f in (a, b) for n in ast.walk(f))]
        sn = self.Pn.self_name(uid)
        if not used or (sn and sn in used) or free_names(a) or free_names(b):
            return None
        dom, dname = self.domain(len(used))
        n = len(dom) ** len(used)
        if n > self.limits["points"]:
            return None
        import itertools
        points = [dict({p: {"lit": None} for p in params}, **dict(zip(used, combo)))
                  for combo in itertools.product(dom, repeat=len(used))]
        sa, sb, ta, tb = self._sources(a, b)
        mod = uid.split(":")[0]
        ra = self.run_points(self.old, mod, sa, a.name, points, f"pts:{uid}")
        rb = self.run_points(self.new, mod, sb, b.name, points, f"pts:{uid}")
        if "error" in ra or "error" in rb:
            return None
        # the domain must execute every changed region, in the tree that holds it
        # (an inserted or deleted early exit — `if T: ... return` — its body too: a domain that never makes T
        # true proves nothing about the inputs that do)
        for kind, xa, xb, path in stmt_diff(ta.body, tb.body):
            for xs, r in ((xa, ra), (xb, rb)):
                if xs and not any(ln in r["covered"] for s in xs for ln in range(s.lineno, s.end_lineno + 1)):
                    return None
                for s in xs:
                    if kind in ("ins", "del") and isinstance(s, ast.If) and _leaves(s.body) and \
                            not any(x.lineno in r["covered"] for x in s.body):
                        return None
        diff = [i for i, (x, y) in enumerate(zip(ra["results"], rb["results"])) if x != y]
        models, shown = set(), []
        for i in diff:
            vals = [points[i][p]["lit"] for p in used]
            for v in vals:
                key = ("rec", json.dumps(v, sort_keys=True)) if isinstance(v, dict) else ("str", v)
                models |= self.cells.holders.get(key, set())
            if len(shown) < 4:
                shown.append(f"{vals!r}: {ra['results'][i]} -> {rb['results'][i]}")
        self._probed(uid, [a], [b])
        return V("points", f"{len(dom)}-value {dname} domain; {len(diff)} points differ: " + "; ".join(shown),
                 self.cells.of(self.cls, models, modes), dimension="record")

    # -- expression probe: a replaced return / assignment over the gate's values, the instance unknown -----------
    def expr_probe(self, uid, a, b, diffs, modes):
        """Each replaced `return E` (or `x = E`) evaluated in its own tree over the finite domain the parameters
        take at the gate (as the point probe), after the simple assignments its path runs, the instance opaque.
        A point is dropped only where BOTH trees' path conditions evaluate False; an expression that touches the
        instance, or a path binding the prefix cannot give, counts as a difference (a hit, conservatively)."""
        if not isinstance(a, ast.FunctionDef) or not isinstance(b, ast.FunctionDef):
            return None
        if C.dump(a.args) != C.dump(b.args) or a.args.vararg or a.args.kwarg or a.args.kwonlyargs:
            return None
        for k, sa, sb, _ in diffs:
            if k != "rep" or not sa or not sb:
                return None
            x, y = sa[-1], sb[-1]
            # a block that always ends in a return, replaced by another such block (or one `x = E` by `x = E'`)
            if not ((isinstance(x, ast.Return) and isinstance(y, ast.Return) and x.value is not None
                     and y.value is not None) or
                    (len(sa) == len(sb) == 1 and isinstance(x, ast.Assign) and isinstance(y, ast.Assign) and
                     [C.dump(t) for t in x.targets] == [C.dump(t) for t in y.targets])):
                return None
        params = [x.arg for x in a.args.posonlyargs + a.args.args]
        sn = self.Pn.self_name(uid)
        mod = uid.split(":")[0]
        import itertools
        models, shown, ndiff, npts = set(), [], 0, 0
        for n, (_, sa, sb, _p) in enumerate(diffs):
            parts = {}
            for side, fn, blk in (("old", a, sa), ("new", b, sb)):
                g = region_guard(fn, {id(blk[0])})
                if g is None:
                    return None
                cond, pre = g
                tail = [copy.deepcopy(x) for x in blk] if isinstance(blk[-1], ast.Return) else \
                    [ast.Return(copy.deepcopy(blk[0].value))]
                f = ast.FunctionDef(name="__x", args=copy.deepcopy(fn.args), body=[copy.deepcopy(x) for x in pre
                                    if x is not None] + tail, decorator_list=[],
                                    returns=None, type_comment=None, type_params=[])
                m = ast.Module([f], [])
                ast.fix_missing_locations(m)
                # what the probe reads must be bound: by itself, or at the top of the module it runs in (the
                # tree's own) — any other name decides nothing
                mi = (self.old if side == "old" else self.new).program().index.mods.get(mod)
                top = set(mi.defs) | set(mi.imports) | set(mi.globals_) if mi else set()
                g = ast.FunctionDef(name="__g", args=copy.deepcopy(fn.args), body=[copy.deepcopy(x) for x in pre
                                    if x is not None] + [ast.Expr(ast.Tuple([copy.deepcopy(t) for t in cond],
                                                                            ast.Load()))], decorator_list=[],
                                    returns=None, type_comment=None, type_params=[])
                if (free_names(f) | free_names(g)) - top:
                    return None
                parts[side] = (C.executable(m), guard_source(fn, cond, pre, "__g"), f)
            used = [p for p in params if p != sn and any(isinstance(z, ast.Name) and z.id == p
                                                         for side in parts.values() for z in ast.walk(side[2]))]
            if not used:
                return None
            dom, dname = self.domain(len(used))
            if len(dom) ** len(used) > self.limits["points"]:
                return None
            points = [dict({p: ({"opaque": p} if p == sn else {"lit": None}) for p in params}, **dict(zip(used, c)))
                      for c in itertools.product(dom, repeat=len(used))]
            npts += len(points)
            r = {}
            for side, t in (("old", self.old), ("new", self.new)):
                ex = self.run_points(t, mod, parts[side][0], "__x", points, f"xpr:{uid}:{n}:{side}", True)
                gd = self.run_points(t, mod, parts[side][1], "__g", points, f"xpg:{uid}:{n}:{side}", True)
                if "error" in ex or "error" in gd:
                    return None
                r[side] = (ex["results"], gd["results"])
            for i, pt in enumerate(points):
                (x, gx), (y, gy) = (r["old"][0][i], r["old"][1][i]), (r["new"][0][i], r["new"][1][i])
                if gx == "False" and gy == "False":
                    continue
                if any(z.startswith(("raise Touched", "unbuildable")) or z == "<opaque>" for z in (x, y)):
                    return None                 # the expression needs the instance: the probe decides nothing
                if x == y:
                    continue
                ndiff += 1
                vals = [pt[p]["lit"] for p in used]
                for v in vals:
                    key = ("rec", json.dumps(v, sort_keys=True)) if isinstance(v, dict) else ("str", v)
                    models |= self.cells.holders.get(key, set())
                if len(shown) < 4:
                    shown.append(f"{vals!r}: {x} -> {y}")
        self._probed(uid, [x for _, sa, _, _ in diffs for x in sa], [y for _, _, sb, _ in diffs for y in sb])
        return V("expr", f"replaced expressions over the {dname} domain ({npts} points); {ndiff} differ: "
                 + "; ".join(shown), self.cells.of(self.cls, models, modes), dimension="record")

    # -- suffix probe: the code from the first difference on, every parameter unknown, on this rack ----------------
    def suffix_probe(self, uid, a, b, modes):
        """Both trees' function from its first differing top-level statement to its end, run with EVERY parameter
        opaque (any use but an attribute read or a hand-over to the declared rack reader raises, isinstance and
        hasattr included), after the common prefix: a path that reaches the suffix carries the same state into it
        in both trees; the prefix runs as such, its early exits (`if T: ... return/raise`, no else) left out — the parameters
        being unknown, T is too. Equal non-raising results mean the change answers the same on this rack for every argument;
        anything else (a raise, an opaque result, a prefix that binds a name) decides nothing."""
        if not isinstance(a, ast.FunctionDef) or not isinstance(b, ast.FunctionDef):
            return None
        if C.dump(a.args) != C.dump(b.args) or a.args.vararg or a.args.kwarg:
            return None
        i = 0
        while i < min(len(a.body), len(b.body)) and C.dump(a.body[i]) == C.dump(b.body[i]):
            i += 1
        keep = [st for st in a.body[:i] if not (isinstance(st, ast.If) and not st.orelse and _leaves(st.body))
                and not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
        params = [x.arg for x in a.args.posonlyargs + a.args.args + a.args.kwonlyargs]
        src = {}
        for side, f in (("old", a), ("new", b)):
            g = copy.deepcopy(f)
            g.name, g.decorator_list = "__s", []
            g.body = [copy.deepcopy(x) for x in keep] + [copy.deepcopy(x) for x in f.body[i:]]
            if not g.body:
                return None
            m = ast.Module([g], [])
            ast.fix_missing_locations(m)
            src[side] = C.executable(m)
        point = [{p: {"opaque": p} for p in params}]
        mod = uid.split(":")[0]
        ra = self.run_points(self.old, mod, src["old"], "__s", point, f"sfx:{uid}", guard_builtins=True)
        rb = self.run_points(self.new, mod, src["new"], "__s", point, f"sfx:{uid}", guard_builtins=True)
        if "error" in ra or "error" in rb:
            return None
        x, y = ra["results"][0], rb["results"][0]
        if x != y or x.startswith(("raise ", "unbuildable ")) or x == "<opaque>":
            return None
        return V("suffix-equal", f"from statement {i} on, every parameter unknown, both trees answer {x} on this rack")

    def _probed(self, uid, xs, ys):
        for tag, ns in (("old", xs), ("new", ys)):
            self.probed.setdefault((tag, uid), []).extend(
                (n.lineno, n.end_lineno) for n in ns if getattr(n, "lineno", None) and getattr(n, "end_lineno", None))

    def callers_probed(self, uid):
        """A changed unit every call of which, in both trees, lies inside code a probe of its (changed) caller
        EXECUTED — in the caller's own tree, this unit included — or inside a planner unit (judged by the plan
        each of its cells receives): its change reaches a cell only through those verdicts, which already hold it.
        None when one use is anything else (a reference, an entry, a call outside a probed region)."""
        planner = tuple(self.cfg["planner"]["packages"])
        plan_callers = set()
        for tag, P, t in (("old", self.Po, self.old), ("new", self.Pn, self.new)):
            R = t.reached()
            if any(M.reached.get(uid) == "<entry>" for M in t.reach.values()):
                return None
            for (caller, call, recv, ctor, cond, kind) in P.sites().get(uid, ()):
                if caller not in R:
                    continue
                if kind != "call":
                    return None
                if caller.split(":")[0].startswith(planner):
                    plan_callers.add(caller)
                    continue
                ln = getattr(call, "lineno", None)
                if ln is None or not any(lo <= ln <= hi for lo, hi in self.probed.get((tag, caller), ())):
                    return None
        cs = set()
        for c in plan_callers:
            cs |= self.cells.every(self.cls, self.modes(c))
        self.plan_cells |= cs
        return V("callers-probed", "every call lies in a probed region of its changed caller"
                 + (f" or in a planner unit ({len(cs)} cells judged by their plan: {', '.join(sorted(plan_callers))})"
                    if plan_callers else ""))

    # -- region probe: a replaced assignment or block over an external domain ------------------------------------
    def region_probe(self, uid, a, b, diffs, modes):
        """Each replaced region — one `x = E` by `x = E'`, or a block of statements by another — run in its own
        tree over the external domain (gate_scope.yml `external_domains`) of every function local it reads, the
        instance opaque, under the region's path condition. A single assignment answers its value; a block answers
        the values, after it, of the names it stores that the function holds outside it. A point where either tree
        touches the instance, or cannot build a value, counts as a difference (a hit, conservatively). A local
        whose stores are not all built by one external library decides nothing (None)."""
        if not all(k == "rep" and sa and sb for k, sa, sb, _ in diffs):
            return None
        mod = uid.split(":")[0]
        sn = self.Pn.self_name(uid)
        imports = {}
        ext_of = {}
        for side, fn in (("old", a), ("new", b)):
            # the function's own imports, read off its source (the normal form drops them)
            src_fn = (self.old if side == "old" else self.new).units.get(uid, fn)
            imports[side] = [s for s in ast.walk(src_fn) if isinstance(s, (ast.Import, ast.ImportFrom))]
            mi = (self.old if side == "old" else self.new).index.mods[mod]
            e = {}
            for s in imports[side]:
                if isinstance(s, ast.Import):
                    for al in s.names:
                        e[al.asname or al.name.split(".")[0]] = al.name
            for k, v in mi.imports.items():
                if v[0] == "ext":
                    e.setdefault(k, v[1])
            ext_of[side] = e
        top = set()
        for t in (self.old, self.new):
            mi = t.program().index.mods.get(mod)
            if mi:
                top |= set(mi.defs) | set(mi.imports) | set(mi.globals_)
        import builtins
        top |= set(dir(builtins))

        def names(nodes, ctx):
            return {n.id for x in nodes for n in ast.walk(x) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)
                    and not n.id.startswith(C.SIG)}

        def imported(nodes):
            return {al.asname or al.name.split(".")[0] for x in nodes for s in ast.walk(x)
                    if isinstance(s, (ast.Import, ast.ImportFrom)) for al in s.names}

        models, shown, ndiff = set(), [], 0
        for _, sa, sb, path in diffs:
            single = len(sa) == len(sb) == 1 and isinstance(sa[0], ast.Assign) and isinstance(sb[0], ast.Assign) \
                and [C.dump(x) for x in sa[0].targets] == [C.dump(x) for x in sb[0].targets]
            guards = {}
            for side, fn, blk in (("old", a, sa), ("new", b, sb)):
                g = region_guard(fn, {id(blk[0])})
                if g is None:
                    return None
                guards[side] = g[0]
            if single:
                bodies = {"old": [ast.Return(copy.deepcopy(sa[0].value))], "new": [ast.Return(copy.deepcopy(sb[0].value))]}
                live = []
                reads = names([sa[0].value, sb[0].value], ast.Load)
            else:
                ids = {id(x) for blk in (sa, sb) for st in blk for x in ast.walk(st)}
                outside = {n.id for fn in (a, b) for n in ast.walk(fn) if isinstance(n, ast.Name) and id(n) not in ids}
                stored = names(sa + sb, ast.Store)
                live = sorted(stored & outside)
                # a name a block binds before reading it is its own; one it reads unbound comes from outside
                reads = set()
                for blk in (sa, sb):
                    bound = set(imported(blk))
                    for st in blk:
                        reads |= names([st], ast.Load) - bound
                        bound |= names([st], ast.Store) | imported([st])
                reads |= set(live)
                if any(isinstance(x, (ast.Return, ast.Yield, ast.YieldFrom, ast.Break, ast.Continue, ast.Global,
                                      ast.Nonlocal, ast.FunctionDef, ast.ClassDef, ast.Lambda))
                       for blk in (sa, sb) for st in blk for x in ast.walk(st)):
                    return None
                tup = ast.Tuple([ast.Name(x, ast.Load()) for x in live], ast.Load())
                bodies = {"old": [copy.deepcopy(x) for x in sa] + [ast.Return(tup)],
                          "new": [copy.deepcopy(x) for x in sb] + [ast.Return(copy.deepcopy(tup))]}
            fn_imp = {side: imported(imports[side]) for side in imports}
            free = sorted(x for x in reads if x not in top and x not in (fn_imp["old"] & fn_imp["new"]))
            binds, values = {}, {}
            for name in free:
                if name == sn:
                    binds[name] = [{"opaque": name}]          # the instance: any use but a read raises (a hit)
                    continue
                dom = None
                for fn, side in ((a, "old"), (b, "new")):
                    for st in C._stores(fn).get(name, []):
                        asg = next((x for x in ast.walk(fn) if isinstance(x, ast.Assign) and st in x.targets), None)
                        if asg is None:
                            dom = False
                            break
                        root = asg.value.func if isinstance(asg.value, ast.Call) else None
                        while isinstance(root, (ast.Attribute, ast.Call)):     # np.frombuffer(...).reshape(...)
                            root = root.value if isinstance(root, ast.Attribute) else root.func
                        if isinstance(root, ast.Name) and root.id == name:
                            continue                          # a method of the value itself: its own library
                        ext = ext_of[side].get(root.id) if isinstance(root, ast.Name) else None
                        cand = next((k for k in self.cfg.get("external_domains", {}) if ext and
                                     k.split(".")[0] == ext.split(".")[0]), None)
                        if cand is None or (dom is not None and dom != cand):
                            dom = False
                            break
                        dom = cand
                    if dom is False:
                        break
                if not dom:
                    return None
                d = self.cfg["external_domains"][dom]
                binds[name] = [json.loads(json.dumps(d["spec_each"]).replace('"$v"', json.dumps(v)))
                               for v in d["values"]]
                values[name] = d["values"]
            # a path term over names the probe binds filters the domain; any other term is dropped (the path
            # taken wider: more points compared, never fewer)
            known = set(binds) | top | fn_imp["old"] | fn_imp["new"]
            guards = {side: [t for t in gs if names([t], ast.Load) <= known] for side, gs in guards.items()}
            fa = ast.FunctionDef(name="__r", args=ast.arguments(posonlyargs=[], args=[ast.arg(x) for x in free],
                                 kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[],
                                 returns=None, type_comment=None, type_params=[])
            src = {}
            for side in ("old", "new"):
                test = ast.BoolOp(ast.And(), [ast.Constant(True)] + [copy.deepcopy(c) for c in guards[side]])
                f = copy.deepcopy(fa)
                f.body = [copy.deepcopy(s) for s in imports[side]] + [
                    ast.If(test=test, body=bodies[side], orelse=[ast.Return(ast.Constant("@@off-path"))])]
                m = ast.Module([f], [])
                ast.fix_missing_locations(m)
                src[side] = C.executable(m)
            import itertools
            nm = list(binds)
            combos = list(itertools.product(*[range(len(binds[x])) for x in nm]))
            if len(combos) > self.limits["points"]:
                return None
            points = [{x: binds[x][i] for x, i in zip(nm, c)} for c in combos]
            ra = self.run_points(self.old, mod, src["old"], "__r", points, f"reg:{uid}:{len(shown)}:old")
            rb = self.run_points(self.new, mod, src["new"], "__r", points, f"reg:{uid}:{len(shown)}:new")
            if "error" in ra or "error" in rb:
                return None
            for c, x, y in zip(combos, ra["results"], rb["results"]):
                blind = any(z.startswith(("raise Touched", "unbuildable", "raise NameError",
                                          "raise UnboundLocalError", "raise ImportError",
                                          "raise ModuleNotFoundError")) or z == "<opaque>" for z in (x, y))
                if x != y or blind:
                    ndiff += 1
                    for n_, i in zip(nm, c):
                        if n_ in values:
                            models |= self.cells.holders.get(("dtype", GI.canonical_dtype(values[n_][i])), set())
                    if not values:
                        models |= set(self.cells.models)
                    if len(shown) < 4:
                        shown.append(f"{dict((n_, values[n_][i]) for n_, i in zip(nm, c) if n_ in values)}: "
                                     f"{x} -> {y}")
        self._probed(uid, [x for _, sa, _, _ in diffs for x in sa], [y for _, _, sb, _ in diffs for y in sb])
        return V("region", f"replaced regions over their external domain; {ndiff} points differ "
                 + "; ".join(shown), self.cells.of(self.cls, models, modes), dimension="dtype")

    # -- the reverse walk: from the unit to what dispatches it -------------------------------------------------
    def ops_of(self) -> dict:
        """{function uid: {op types it serves}} from each tree's op maps (the launcher's own tables)."""
        if self._ops_of is None:
            out = {}
            for t in (self.old, self.new):
                jobs = []
                for i, om in enumerate(self.cfg["op_maps"]):
                    j = {"id": f"opmap{i}", "kind": "opmap", "op_types": sorted(self.cells.all_ops)}
                    j.update({k: om[k] for k in ("builder", "table", "key") if k in om})
                    jobs.append(j)
                res = probe(t.path, self.hw, jobs, self.cfg)
                for i, om in enumerate(self.cfg["op_maps"]):
                    r = res[f"opmap{i}"]
                    if "error" in r:
                        raise SystemExit(f"REFUSED: the op map {om} of {t.sha[:8]}: {r['error']}")
                    by_key = {}
                    for typ, key in r["keys"].items():
                        by_key.setdefault(key if om.get("key") else typ, set()).add(typ)
                    for key, target in r["table"].items():
                        out.setdefault(target, set()).update(by_key.get(key, set()))
            self._ops_of = out
        return self._ops_of

    def literal_anchor(self, cond) -> set:
        lits = set()
        for term in cond:
            if isinstance(term, ast.UnaryOp) and isinstance(term.op, ast.Not):
                continue
            for n in ast.walk(term):
                if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.In)):
                    for e in [n.left] + n.comparators:
                        xs = [e] if isinstance(e, ast.Constant) else list(e.elts) \
                            if isinstance(e, (ast.Tuple, ast.List, ast.Set)) else []
                        lits |= {x.value for x in xs if isinstance(x, ast.Constant) and isinstance(x.value, str)}
        return lits & (self.cells.all_ops | self.cells.all_flows)

    def unit_guard(self, uid, pair, diffs):
        """The source of `__g(<the unit's parameters>)`: whether a call reaches a changed region (an inserted
        early exit: only where its test holds). None when a region's path is not a condition of the
        parameters."""
        if pair is None:
            return None
        a, b = pair
        parts = []
        for kind, sa, sb, path in diffs:
            fn, xs = (b, sb) if sb else (a, sa)
            s0 = xs[0]
            extra = s0.test if (len(xs) == 1 and isinstance(s0, ast.If) and not s0.orelse and _leaves(s0.body)
                                and kind in ("ins", "del")) else None
            g = region_guard(fn, {id(s0)}, extra)
            if g is None:
                return None
            parts.append(g)
        node = (self.new.units.get(uid) or self.old.units.get(uid))
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        srcs, names = [], []
        for i, (cond, pre) in enumerate(parts):
            fn_tmp = ast.FunctionDef(name="x", args=node.args, body=[], decorator_list=[])
            srcs.append(guard_source(fn_tmp, cond, pre, f"__g{i}"))
            names.append(f"__g{i}")
        args = node.args
        plist = [x.arg for x in args.posonlyargs + args.args]
        call = ", ".join(plist)
        hm = ast.Module([ast.FunctionDef(name="__g", args=copy.deepcopy(args), body=[ast.Pass()], decorator_list=[],
                                         returns=None, type_comment=None, type_params=[])], [])
        ast.fix_missing_locations(hm)
        hdr = ast.unparse(hm).split("\n")[0]
        body = f"{hdr}\n    return " + " or ".join(f"{n}({call})" for n in names) + "\n"
        return "\n".join(srcs) + "\n" + body

    def sig_spec(self, a):
        st = self.cfg["stubs"]
        if "t" in a:
            return {"obj": st["tensor"]["class"], "attrs": {st["tensor"]["dtype_slot"]: {
                "member": st["dtype"]["enum"], "name": a["t"], "suffixes": st["dtype"]["suffixes"]}}}
        if "s" in a:
            return {"lit": a["s"]}
        if "l" in a:
            return {"list": [self.sig_spec(x) for x in a["l"]]}
        r = a.get("r") or {}
        if r.get(self.cfg["records"]["tag"]) == self.cfg["records"]["dtype_tag"] and isinstance(r.get("value"), str):
            return {"member": st["dtype"]["enum"], "name": GI.canonical_dtype(r["value"]),
                    "suffixes": st["dtype"]["suffixes"]}
        return {"lit": r.get("value")}

    def guarded_models(self, uid, ops, guard) -> tuple:
        """The models whose signatures of `ops` reach the changed region: the guard (source, entry) — a function
        of `uid`'s parameters — evaluated per signature, in both trees; a raise is a hit."""
        guard_src, entry = guard
        node = self.new.units.get(uid) or self.old.units.get(uid)
        plist = [x.arg for x in node.args.posonlyargs + node.args.args]
        sn = self.Pn.self_name(uid)
        pts, keys = [], []
        for t in sorted(ops):
            for k in self.cells.holders:
                if k[0] == "sig" and k[1] == t:
                    s = json.loads(k[2])
                    pos = [p for p in plist if p != sn]
                    pt = {p: {"lit": None} for p in plist}
                    for p, a in zip(pos, s["args"]):
                        pt[p] = self.sig_spec(a)
                    for kw, a in s["kwargs"].items():
                        if kw in pt:
                            pt[kw] = self.sig_spec(a)
                    pts.append(pt)
                    keys.append(k)
        if not pts:
            return set(), 0, 0
        mod = uid.split(":")[0]
        hits = set()
        for t in (self.old, self.new):
            r = self.run_points(t, mod, guard_src, entry, pts, f"guard:{uid}:{entry}")
            if "error" in r:
                return None, 0, len(pts)
            for k, res in zip(keys, r["results"]):
                if res == "True" or res.startswith("raise") or res.startswith("unbuildable"):
                    hits.add(k)
        models = set()
        for k in hits:
            models |= self.cells.holders[k]
        return models, len(hits), len(pts)

    def caller_guard(self, P, tree, callee, guard, caller, node, recv, ctor, kind):
        """The guard of `callee` carried to one of its call sites: the path to the call in `caller` AND the
        callee's guard over the call's arguments. None when it cannot be written (a reference, a receiver,
        a starred argument, an argument the call does not give)."""
        if guard is None or kind != "call" or recv is not None or ctor:
            return None
        cfn = tree.units.get(callee)
        if not isinstance(cfn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        args = []
        for p in [x.arg for x in cfn.args.posonlyargs + cfn.args.args]:
            got = P._arg_for(cfn, node, recv, ctor, p)
            if got[0] not in ("arg", "<default>"):
                return None
            args.append(copy.deepcopy(got[1]))
        try:
            fn = P.canon(caller)
        except Exception:                          # noqa: BLE001
            return None
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None                            # a module-level call site: no parameters to guard over
        stmt = None
        for x in ast.walk(fn):
            if isinstance(x, ast.stmt) and x is not fn and any(y is node for y in ast.walk(x)):
                if stmt is None or any(y is x for y in ast.walk(stmt)):
                    stmt = x
        if stmt is None:
            return None
        src, entry = guard
        test = ast.Call(func=ast.Name(entry, ast.Load()), args=args, keywords=[])
        g = region_guard(fn, {id(stmt)}, test)
        if g is None:
            return None
        self._gn = getattr(self, "_gn", 0) + 1
        name = f"__h{self._gn}"
        return src + "\n" + guard_source(fn, g[0], g[1], name), name

    def closed_sites(self, u, guard, callers) -> set:
        """The indices of the call sites whose constant arguments make `u`'s guard False (the changed region is
        unreachable through them: `div(x, t)` never reaches a `rounding_mode == "trunc"` branch). Every
        argument the call does not give as a constant is opaque: any use of it raises, a hit."""
        if guard is None:
            return set()
        cfn = self.new.units.get(u) or self.old.units.get(u)
        if not isinstance(cfn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return set()
        params = [x.arg for x in cfn.args.posonlyargs + cfn.args.args]
        pts, idx = [], []
        for i, (P, t, (caller, node, recv, ctor, cond, kind)) in enumerate(callers):
            if kind != "call":
                continue
            pt = {}
            for p in params:
                got = P._arg_for(cfn, node, recv, ctor, p)
                e = got[1] if got[0] in ("arg", "<default>") else None
                pt[p] = {"lit": e.value} if isinstance(e, ast.Constant) else {"opaque": p}
            pts.append(pt)
            idx.append(i)
        if not pts:
            return set()
        r = self.run_points(self.new, u.split(":")[0], guard[0], guard[1], pts, f"close:{u}:{guard[1]}")
        if "error" in r:
            return set()
        return {i for i, res in zip(idx, r["results"]) if res == "False"}

    def walk(self, uid, pair, diffs, modes, why="") -> dict:
        """From the unit up its callers (both trees) to what dispatches it: an op map's entry (whose
        signatures the region's guard, carried along each call, is evaluated over — and whose own callers are
        walked on, an op wrapper being callable by another) or a call under a literal op/flow test."""
        g0 = self.unit_guard(uid, pair, diffs)
        guard0 = (g0, "__g") if g0 is not None else None
        ops_of = self.ops_of()
        anchors, named = [], []
        seen, todo = set(), [(uid, guard0)]
        progs = ((self.Pn, self.new), (self.Po, self.old))
        # the op maps' own references to their entries are the anchor itself, not callers
        tables = {om["builder"] for om in self.cfg["op_maps"] if "builder" in om} | \
            {om["table"].split(":")[0] + ":<module>" for om in self.cfg["op_maps"] if "table" in om}
        visits, pruned = {}, []
        while todo:
            u, guard = todo.pop()
            visits[u] = visits.get(u, 0) + 1
            if visits[u] > 8:
                guard = None                       # many paths (or a cycle): the guard is dropped, every holder
            if (u, guard) in seen:
                continue
            seen.add((u, guard))
            if len(seen) > self.limits["walk"]:
                named.append(f"walk limit at {u}")
                break
            callers = [(P, t, c) for P, t in progs for c in P.sites().get(u, [])]
            if ops_of.get(u):
                anchors.append(("op", u, ops_of[u], guard))
                callers = [x for x in callers if not (x[2][5] == "ref" and x[2][0] in tables)]
            elif not callers:
                named.append(u)
                continue
            closed = self.closed_sites(u, guard, callers)
            for i, (P, t, (caller, node, recv, ctor, cond, kind)) in enumerate(callers):
                if i in closed:
                    pruned.append(f"{caller}->{u.split(':')[1]}")
                    continue
                lits = self.literal_anchor(cond)
                if lits:
                    anchors.append(("lit", caller, lits, None))
                else:
                    todo.append((caller, self.caller_guard(P, t, u, guard, caller, node, recv, ctor, kind)))
        if named:
            return V("fallback", (why + "; " if why else "") + "reached from " + ", ".join(named[:3])
                     + " with no op, flow or op-map anchor on the path", self.cells.every(self.cls, modes),
                     fallback=True, walked=len(seen))
        # each anchor selects in the modes whose process reaches the dispatch: the op map's builder (or table)
        # for an op-map entry, the caller for a literal test — never wider than the unit's own reach
        disp = set().union(*[self.modes(om["builder"] if "builder" in om else om["table"].split(":")[0] + ":<module>")
                             for om in self.cfg["op_maps"]]) or modes
        cells, desc = set(), []
        for kind, u, xs, guard in anchors:
            if kind == "op":
                every = set().union(*[self.cells.holders.get(("op", t), set()) for t in xs])
                if guard is not None:
                    got, nh, npts = self.guarded_models(u, xs, guard)
                    if got is None:
                        got = every
                        desc.append(f"op map {u} ({', '.join(sorted(xs))}): guard not evaluable, every holder")
                    else:
                        desc.append(f"op map {u} ({', '.join(sorted(xs))}): guard holds on {nh}/{npts} signatures")
                else:
                    got = every
                    desc.append(f"op map {u} ({', '.join(sorted(xs))})")
                cells |= self.cells.of(self.cls, got, modes & disp)
            else:
                got = set().union(*[self.cells.holders.get(("op", x), set()) | self.cells.holders.get(("flow", x), set())
                                    for x in xs])
                cells |= self.cells.of(self.cls, got, modes & (self.modes(u) or modes))
                desc.append(f"{u} under {' / '.join(sorted(xs))}")
        if pruned:
            desc.append(f"{len(pruned)} call sites closed by the guard (e.g. {', '.join(sorted(set(pruned))[:3])})")
        return V("walk", (why + "; " if why else "") + "; ".join(sorted(set(desc))[:8]),
                 cells, dimension="op", walked=len(seen))


# =============================================================================================================
# the plan differential: a planner change judged by the plan each cell receives, old tree against new
# =============================================================================================================

def explain_plan(tree: Tree, hw: str, argv: list, cfg: dict) -> dict:
    """The tree's `--explain-plan --json` for one cell's argv on one class's profile (no card; cached)."""
    argv = [a for i, a in enumerate(argv) if a != "--output" and (i == 0 or argv[i - 1] != "--output")]
    cmd = ["-m", cfg["package"], *argv, "--hardware", hw, "--explain-plan", "--json"]
    key = hashlib.sha256(json.dumps([tree.sha, cmd]).encode()).hexdigest()[:24]
    cf = SCRATCH / "plans" / f"{key}.json"
    if cf.is_file():
        return json.loads(cf.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NBX_", "NEUROBRIX_"))}
    env.update(cfg["planner"]["env"])
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(tree.path / "src"), PYTHONNOUSERSITE="1")
    p = subprocess.run([PYTHON, *cmd], capture_output=True, text=True, env=env, cwd=str(SCRATCH))
    try:
        plan = json.loads(p.stdout[p.stdout.index("{"):])
        out = {"plan": {k: plan.get(k) for k in cfg["planner"]["plan_fields"]}}
    except (ValueError, json.JSONDecodeError):
        lines = [x for x in (p.stderr or p.stdout).strip().splitlines() if x.strip()]
        out = {"error": f"rc={p.returncode}: " + (lines[-1][:300] if lines else "")}
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out))
    return out


def plan_view(plan: dict, cfg: dict) -> dict:
    """The plan's compared fields, without what the planning process MEASURES of itself (planner.measured:
    {measured field: [fields that add it]}): a measurement of the planning process is not the plan."""
    view = {k: copy.deepcopy(plan.get(k)) for k in cfg["planner"]["plan_fields"]}
    for m, adds in (cfg["planner"].get("measured") or {}).items():
        path = m.split(".")
        d = view
        for x in path[:-1]:
            d = d.get(x) if isinstance(d, dict) else None
        if not isinstance(d, dict) or path[-1] not in d:
            continue
        v = d.pop(path[-1])
        for f in adds:
            fp = f.split(".")
            e = view
            for x in fp[:-1]:
                e = e.get(x) if isinstance(e, dict) else None
            if isinstance(e, dict) and isinstance(e.get(fp[-1]), (int, float)) and isinstance(v, (int, float)):
                e[fp[-1]] -= v
    return view


def plan_differential(old: Tree, new: Tree, cells: Cells, cls: str, todo: set, cfg: dict) -> dict:
    """{(model, mode): why} — the cells whose plan differs between the trees (a planner that fails in one tree
    and not the other differs)."""
    hw = cfg["classes"][cls]
    jobs = [(t, c) for c in sorted(todo) for t in (old, new)]
    with ThreadPoolExecutor(max_workers=cfg["limits"]["plan_jobs"]) as ex:
        res = list(ex.map(lambda tc: explain_plan(tc[0], hw, cells.argv[tc[1]], cfg), jobs))
    by = {}
    for (t, c), r in zip(jobs, res):
        by.setdefault(c, {})[t.sha] = r
    out = {}
    for c, r in by.items():
        a, b = r[old.sha], r[new.sha]
        if "plan" in a and "plan" in b:
            pa, pb = plan_view(a["plan"], cfg), plan_view(b["plan"], cfg)
            ks = [k for k in cfg["planner"]["plan_fields"] if pa.get(k) != pb.get(k)]
            if ks:
                out[c] = "plan differs in " + ", ".join(ks)
        elif "error" in a and "error" in b:
            # no plan in either tree decides nothing about the cell (its gate run had one)
            out[c] = f"no plan in either tree: {b['error'][:160]}"
        elif a != b:
            out[c] = f"plan {'refused' if 'error' in b else 'restored'}: {(b.get('error') or a.get('error'))[:160]}"
    return out


# =============================================================================================================
# data files and tools
# =============================================================================================================

def _flatten(d, pre=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flatten(v, pre + (str(k),))
    else:
        yield pre, d


def data_units(old: Tree, new: Tree, f: str) -> tuple:
    """(changed leaf keys, reached units of either tree holding one of them as a string literal) of a data file
    of the package; a file that does not parse as YAML/JSON names every reached unit holding its basename."""
    def load(t):
        p = t.path / f
        if not p.is_file():
            return {}
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:                          # noqa: BLE001
            return None
    a, b = load(old), load(new)
    if a is None or b is None:
        keys = {Path(f).name}
    else:
        fa, fb = dict(_flatten(a)), dict(_flatten(b))
        # a reader reaches a changed leaf only by naming the TOPMOST key of the path that is absent from one
        # side (an added or removed subtree), or the leaf itself when the path exists on both sides
        pa = {k[:i] for k in fa for i in range(1, len(k) + 1)}
        pb = {k[:i] for k in fb for i in range(1, len(k) + 1)}
        keys = set()
        for k in set(fa) | set(fb):
            if not k or fa.get(k) == fb.get(k):
                continue
            top = next((k[:i] for i in range(1, len(k) + 1) if (k[:i] in pa) != (k[:i] in pb)), k)
            keys.add(top[-1])
    units = set()
    for t in (old, new):
        for uid in t.reached():
            node = t.units.get(uid)
            if node is None:
                continue
            if any(isinstance(n, ast.Constant) and n.value in keys for n in ast.walk(node)):
                units.add(uid)
    return keys, units


def harness_closure(repo_tree: Path, cfg: dict) -> set:
    """The tools a cell's harness imports (its own module and the tools/ modules it imports, transitively)."""
    out, todo = set(), list(cfg["harness"])
    while todo:
        rel = todo.pop()
        if rel in out:
            continue
        out.add(rel)
        p = repo_tree / rel
        if not p.is_file():
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] \
                if isinstance(n, ast.ImportFrom) and n.module else []
            for nm in names:
                cand = f"tools/{nm.split('.')[0]}.py"
                if (repo_tree / cand).is_file():
                    todo.append(cand)
    return out


# =============================================================================================================
# the scope
# =============================================================================================================

def class_constants(tree: Tree, uids: set, hw: str) -> dict:
    tree.K = {}
    tree.reset_program()
    cand = call_candidates(tree, uids) | {u for u in uids if u in tree.units}
    K, refused, envs = gate_constants(tree, cand, hw, global_writers(tree))
    tree.K = K
    tree.reset_program()
    return {"K": K, "refused": refused, "env": envs}


def compute(pairs: list, gate_dir: Path, cfg: dict, prunes=(), log=print) -> dict:
    newest = tree_of(REPO, pairs[-1][1])
    cells = Cells(gate_dir, cfg, newest / "src")
    log(f"gate {gate_dir}: " + ", ".join(f"{c} {len(v)} cells" for c, v in cells.cells.items())
        + f"; {len(cells.recs)} containers indexed")
    sel = {cls: {} for cls in cfg["classes"]}
    report = {"gate": str(gate_dir), "pairs": [], "prunes": [], "gone": cells.gone}
    if cells.gone:
        log("retraced away since the gate ran (scoped from the record indexed while present): " + ", ".join(cells.gone))

    def pick(cls, cs, why):
        for c in cs:
            sel[cls].setdefault(c, []).append(why)

    report["retraced"] = cells.retraced
    for m, why in sorted(cells.retraced.items()):
        log(f"container changed since the gate ran: {m} ({why}): every cell of it")
        for cls in cfg["classes"]:
            pick(cls, {c for c in cells.cells[cls] if c[0] == m}, f"container changed since the gate ran: {why}")

    for a, b in pairs:
        t0 = time.time()
        old, new = Tree(REPO, a, cfg), Tree(REPO, b, cfg)
        for t in (old, new):
            t.mode_reach(cells.by_mode, cells.argv)
        ch = changed(old, new)
        pr = {"old": a, "new": b, "files": {k: v for k, v in ch.items() if k != "units"}, "units": [],
              "tools": [], "data": []}
        hc = harness_closure(new.path, cfg) | harness_closure(old.path, cfg)
        for f in ch["tools"]:
            if f in hc:
                for cls in cfg["classes"]:
                    pick(cls, cells.cells[cls], f"harness tool {f} ({a[:8]}..{b[:8]})")
                pr["tools"].append({"file": f, "verdict": "harness: every cell"})
            else:
                pr["tools"].append({"file": f, "verdict": "not in a cell's harness: no cell"})
        changed_code = {u for _, _, u in ch["units"]}       # judged on their own diff below
        data_u = {}
        rack_files = cfg["rack"]["reader_files"].format(*cfg["rack"]["reader_values"])
        for f in ch["data"]:
            if fnmatch.fnmatch(f, cfg["rack"]["reader_files"].format(*["*"] * len(cfg["rack"]["reader_values"]))) \
                    and f != rack_files:
                pr["data"].append({"file": f, "verdict": f"another rack's profile (this rack reads {rack_files})"})
                continue
            keys, units = data_units(old, new, f)
            pr["data"].append({"file": f, "keys": sorted(keys)[:20], "units": sorted(units)})
            for u in units - changed_code:
                data_u.setdefault(u, f)
        reached = {u for _, _, u in ch["units"] if u in old.reached() or u in new.reached()}
        changed_ids = {u for _, _, u in ch["units"]}
        for cls, hw in cfg["classes"].items():
            ks = {t.sha: class_constants(t, reached | set(data_u), hw) for t in (old, new)}
            J = Judge(old, new, cells, cls, changed_ids, cfg)
            # a constant parameter slot whose constant differs between the trees changes its function, though its
            # file did not (the normal form substitutes the slot's constant)
            pa, pb = old.program().pslots(), new.program().pslots()
            slot_units = sorted({d for (d, pn) in set(pa) & set(pb) if pa[(d, pn)] != pb[(d, pn)]} - changed_ids)
            units = list(ch["units"]) + [("chg", "(constant slot)", u) for u in slot_units
                                         if u in old.reached() or u in new.reached()]
            judged = []
            for kind, f, uid in units:
                if uid not in reached and f != "(constant slot)":
                    if cls == next(iter(cfg["classes"])):
                        pr["units"].append({"unit": uid, "kind": kind, "verdict": "unreached"})
                    continue
                judged.append((kind, uid, J.judge(kind, uid)))
            for kind, uid, v in judged:
                if v["cells"] and kind == "chg":
                    w = J.callers_probed(uid)
                    if w is not None:
                        v = dict(w, why=f"{w['why']} (its own verdict, {v['verdict']} on {len(v['cells'])} "
                                         f"cells, subsumed)")
                log(f"  {cls} {v['verdict']:<12} {len(v['cells']):>3} {uid}  {v['why'][:160]}")
                pick(cls, v["cells"], f"{uid}: {v['verdict']}")
                pr["units"].append({"unit": uid, "kind": kind, "class": cls, "verdict": v["verdict"], "why": v["why"],
                                    "cells": len(v["cells"]), "fallback": v.get("fallback", False)})
            for uid, f in sorted(data_u.items()):
                v = J.walk(uid, None, [], J.modes(uid), why=f"holds a changed key of {f}")
                pick(cls, v["cells"], f"{uid} ({f}): {v['verdict']}")
                pr["units"].append({"unit": uid, "kind": "data", "class": cls, "verdict": v["verdict"],
                                    "why": v["why"], "cells": len(v["cells"]), "fallback": v.get("fallback", False)})
            if J.plan_cells:
                diff = plan_differential(old, new, cells, cls, J.plan_cells, cfg)
                for c, why in diff.items():
                    pick(cls, {c}, f"plan ({a[:8]}..{b[:8]}): {why}")
                pr.setdefault("plans", {})[cls] = {"cells": len(J.plan_cells), "differ": len(diff),
                                                   "which": {f"{m}/{mo}": w for (m, mo), w in sorted(diff.items())}}
            pr.setdefault("constants", {})[cls] = {t.sha[:8]: {"K": len(k["K"]), "refused": k["refused"]}
                                                   for t, k in zip((old, new), ks.values())}
        pr["seconds"] = round(time.time() - t0, 1)
        report["pairs"].append(pr)
        log(f"{a[:8]}..{b[:8]}: {len(ch['units'])} changed units, {len(reached)} reached; "
            + ", ".join(f"{cls} {len(s)} cells" for cls, s in sel.items()) + f" ({pr['seconds']} s)")
    for p in prunes:
        cls, model, mode, reason = p
        if not reason.strip():
            raise SystemExit("REFUSED: a manual prune carries its reason")
        if sel[cls].pop((model, mode), None) is not None:
            report["prunes"].append({"class": cls, "cell": f"{model}/{mode}", "reason": reason})
    report["selected"] = {cls: {f"{m}/{mo}": w for (m, mo), w in sorted(s.items())} for cls, s in sel.items()}
    return {"cells": cells, "sel": sel, "report": report}


def write_outputs(res: dict, out: Path, cfg: dict) -> None:
    cells, sel, report = res["cells"], res["sel"], res["report"]
    out.mkdir(parents=True, exist_ok=True)
    pruned = {(p["class"], *p["cell"].split("/")): p["reason"] for p in report["prunes"]}
    order = cells.RM.GATE_MODES
    lines_c = ["class\tmodel\tmode\tcarried_from\trow_date\trc\tverdict\tengine\tlast_row\treason"]
    for cls in cfg["classes"]:
        q = []
        by_model = {}
        for (m, mode) in sel[cls]:
            by_model.setdefault(m, set()).add(mode)
        for m in sorted(by_model):
            if m not in cells.timeouts:
                raise SystemExit(f"REFUSED: the gate {cells.dir} names no timeout for {m} ({cfg['gate']['timeouts']})")
            q.append(f"{m} {','.join(x for x in order if x in by_model[m])} {cells.timeouts[m]}")
        f = out / f"queue_{cls}.txt"
        f.write_text("".join(x + "\n" for x in q))
        rows_dir = cells.dir / cfg["gate"]["rows"].format(cls=cls)
        for c in sorted(cells.cells[cls] - set(sel[cls])):
            r = cells.rows[cls].get(c)
            reason = pruned.get((cls, *c)) or "out of the computed scope"
            if r is None:
                lines_c.append(f"{cls}\t{c[0]}\t{c[1]}\t{rows_dir}\t-\t-\t-\t-\tNO ROW\t{reason}")
                continue
            src = r.get("carried") or {}
            lines_c.append("\t".join(str(x) for x in (
                cls, c[0], c[1], src.get("source_gate") or rows_dir, r.get("date", "-"), r.get("rc"),
                r.get("verdict", "-"), (r.get("engine") or "-")[:12], "RED" if cells.RM.is_red(r) else "green",
                reason + (f" (carried in that gate: {src.get('reason')})" if src else ""))))
    (out / "carried.tsv").write_text("\n".join(lines_c) + "\n")
    (out / "scope.json").write_text(json.dumps(report, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--range")
    ap.add_argument("--commits", help="comma-separated constituents: a sha (against its first parent) or A..B")
    ap.add_argument("--gate-dir", required=True, help="the gate the unreached cells are carried from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prune", action="append", default=[],
                    help="cls:model:mode:reason — a manual prune, its reason written in scope.json and carried.tsv")
    ap.add_argument("--config", default=str(CFG_PATH))
    a = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print("note: the scope never touches a card (every probe runs with CUDA_VISIBLE_DEVICES='')", file=sys.stderr)
    cfg = load_cfg(Path(a.config))
    pairs = change_pairs(REPO, a.base, a.head, a.range, a.commits)
    prunes = []
    for p in a.prune:
        parts = p.split(":", 3)
        if len(parts) != 4 or not parts[3].strip():
            raise SystemExit(f"REFUSED: --prune {p!r}: cls:model:mode:reason, the reason required")
        prunes.append(tuple(parts))
    res = compute(pairs, Path(a.gate_dir), cfg, prunes, log=lambda s: print(s, flush=True))
    write_outputs(res, Path(a.out), cfg)
    sel = res["sel"]
    fb = [f"{u['unit']} ({u['class']}): {u['why'][:120]}" for p in res["report"]["pairs"] for u in p["units"]
          if u.get("fallback")]
    print(f"scope: " + ", ".join(f"{cls} {len(s)} cells / {len(res['cells'].cells[cls])}" for cls, s in sel.items())
          + f" -> {a.out}")
    for x in sorted(set(fb)):
        print("  fallback:", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
