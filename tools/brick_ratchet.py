"""The brick ratchet: measures what makes code spaghetti and refuses a lot that makes any figure worse.

The owner's rule (Hocine Benkelaya, 2026-11-10): the code is written in bricks — everything that decides or loops is a
named component with one responsibility and a written contract. This tool is its guard 6. It decides nothing about a
design (the design page and the other agent's reread do); it counts, in the packages `tools/brick_ratchet.json` names:

  length     a function's CODE lines — no docstring, no comment-only line, no blank line;
  depth      the if / for / while / try nested inside one function (an `elif` and a `with` add none);
  direction  an import against the declared direction between components (the json's `direction` table);
  private    a private name taken from another module (`from x import _name`);
  reach      `x._name` where x is not self / cls — counted per package, too heuristic to name each one;
  module     a module over the module limit, and a module that states no responsibility (no docstring).

Two tiers. NEW code is held to the limits of its package (the json: today's 95th percentile, dated; lowered when the
percentile falls, never raised). EXISTING offenders are listed in `tools/brick_ratchet.baseline.json`: none may grow,
none may be added, and one that no longer offends must leave the list (`--update` rewrites it, downward only).
A `@...jit` kernel body is exempt from the length limit — one loop nest is one kernel — recognised by its decorator.

    python tools/brick_ratchet.py             # check: every refusal by name, exit 1 when there is one
    python tools/brick_ratchet.py --update    # rewrite the baseline; refused while anything got worse
    python tools/brick_ratchet.py --stats     # the percentiles the limits are drawn from
    python tools/brick_ratchet.py --header    # one line for a lot's report

Standard library only (ast, tokenize): the code is read, never imported.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".json")
BASELINE = Path(__file__).with_name("brick_ratchet.baseline.json")

_BRANCH = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)
_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
_SILENT = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
           tokenize.ENDMARKER)


def _docstring_lines(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef) + _FUNC) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.update(range(first.lineno, first.end_lineno + 1))
    return out


def code_lines(source: str, tree: ast.AST = None) -> set:
    """The line numbers of `source` that hold code: a token other than a comment, outside every docstring."""
    tree = ast.parse(source) if tree is None else tree
    held = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in _SILENT:
            held.update(range(tok.start[0], tok.end[0] + 1))
    return held - _docstring_lines(tree)


def depth(node: ast.AST, at: int = 0) -> int:
    """The deepest branch or loop under `node`, nested scopes left out; an `elif` stays at its `if`'s depth."""
    deepest = at
    for name, value in ast.iter_fields(node):
        for child in value if isinstance(value, list) else [value]:
            if not isinstance(child, ast.AST) or isinstance(child, _SCOPE):
                continue
            elif_ = name == "orelse" and isinstance(node, ast.If) and isinstance(child, ast.If) and len(value) == 1
            step = 1 if isinstance(child, _BRANCH) and not elif_ else 0
            deepest = max(deepest, depth(child, at + step))
    return deepest


def _is_kernel(fn: ast.AST) -> bool:
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (isinstance(target, ast.Attribute) and target.attr == "jit") or (isinstance(target, ast.Name) and target.id == "jit"):
            return True
    return False


def _functions(tree: ast.AST, held: set):
    """(qualified name, code lines, depth, is a kernel) of every function, a repeated name numbered in order."""
    seen = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNC):
                name = prefix + child.name
                seen[name] = seen.get(name, 0) + 1
                key = name if seen[name] == 1 else f"{name}#{seen[name]}"
                span = range(child.lineno, child.end_lineno + 1)
                yield key, sum(1 for i in span if i in held), depth(child), _is_kernel(child)
                yield from walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, prefix + child.name + ".")
            else:
                yield from walk(child, prefix)
    yield from walk(tree, "")


def _module_name(path: Path, root: Path, top: str) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([top] + parts)


def _imports(tree: ast.AST, module: str, is_package: bool):
    """(imported module, imported name or None) of every import, a relative one resolved against `module`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            base = module.split(".")
            if node.level:
                base = base[:len(base) - node.level + (1 if is_package else 0)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            for alias in node.names:
                yield target, alias.name


def _reaches(tree: ast.AST) -> int:
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and n.attr.startswith("_") and not n.attr.startswith("__")
               and not (isinstance(n.value, ast.Name) and n.value.id in ("self", "cls")))


def _against(rel: str, target: str, rules) -> bool:
    return any(rel.startswith(r["in"]) and any(target == n or target.startswith(n + ".") for n in r["never"])
               for r in rules)


def measure(repo: Path = REPO, config: dict = None) -> dict:
    """Everything the ratchet counts, per package: {package: {functions, private, direction, reach, modules, mute}}."""
    config = config or json.loads(CONFIG.read_text())
    state = {}
    for name, pkg in config["packages"].items():
        root, st = repo / pkg["root"], {"functions": {}, "private": [], "direction": [], "reach": 0, "modules": {},
                                        "mute": []}
        for path in sorted(root.rglob("*.py")):
            source, rel = path.read_text(encoding="utf-8"), path.relative_to(repo).as_posix()
            tree = ast.parse(source)
            held = code_lines(source, tree)
            for key, length, deep, kernel in _functions(tree, held):
                st["functions"][f"{rel}::{key}"] = [length, deep, kernel]
            module = _module_name(path, root, pkg["top"])
            for target, taken in _imports(tree, module, path.name == "__init__.py"):
                if taken and taken.startswith("_") and not taken.startswith("__") and target != "__future__":
                    st["private"].append(f"{rel} <- {target}.{taken}")
                if _against(rel, target, config["direction"]):
                    st["direction"].append(f"{rel} -> {target}")
            st["reach"] += _reaches(tree)
            st["modules"][rel] = len(held)
            if held and ast.get_docstring(tree) is None:
                st["mute"].append(rel)
        st["private"], st["direction"] = sorted(set(st["private"])), sorted(set(st["direction"]))
        state[name] = st
    return state


def offenders(state: dict, config: dict) -> dict:
    """The part of a measure a baseline records: what is over a limit, or should not exist."""
    out = {}
    for name, st in state.items():
        lim = config["packages"][name]
        over = {k: [n, d] for k, (n, d, kernel) in st["functions"].items()
                if (n > lim["max_code_lines"] and not kernel) or d > lim["max_depth"]}
        out[name] = {"functions": over, "private": st["private"], "direction": st["direction"], "reach": st["reach"],
                     "modules": {m: n for m, n in st["modules"].items() if n > config["max_module_code_lines"]},
                     "mute": st["mute"]}
    return out


def _grown(kind: str, now: dict, base: dict, unit: str):
    for key, value in now.items():
        if key not in base:
            yield f"{kind}: {key} is new and over the limit ({value} {unit})"
        elif any(a > b for a, b in zip(_seq(value), _seq(base[key]))):
            yield f"{kind}: {key} grew ({base[key]} -> {value} {unit}): a listed offender only shrinks"


def _seq(value):
    return value if isinstance(value, list) else [value]


def check(found: dict, baseline: dict) -> list:
    """Every refusal, by name: what is worse than the baseline, then what the baseline lists that no longer offends."""
    worse, stale = [], []
    for name, now in found.items():
        base = baseline.get(name) or {"functions": {}, "private": [], "direction": [], "reach": 0, "modules": {}, "mute": []}
        worse += _grown(f"{name} function", now["functions"], base["functions"], "code lines, depth")
        worse += _grown(f"{name} module", now["modules"], base["modules"], "code lines")
        for kind, why in (("private", "a private name taken from another module"),
                          ("direction", "an import against the declared direction"),
                          ("mute", "a module that states no responsibility (no docstring)")):
            worse += [f"{name} {kind}: {x} — {why}" for x in now[kind] if x not in base[kind]]
            stale += [f"{name} {kind}: {x}" for x in base[kind] if x not in now[kind]]
        if now["reach"] > base["reach"]:
            worse.append(f"{name} reach: {now['reach']} uses of x._name outside self / cls (baseline {base['reach']})")
        if now["reach"] < base["reach"]:
            stale.append(f"{name} reach: {base['reach']} -> {now['reach']}")
        for kind in ("functions", "modules"):
            stale += [f"{name} {kind}: {k} {base[kind][k]} -> {now[kind].get(k, 'under the limit')}"
                      for k in base[kind] if now[kind].get(k) != base[kind][k]]
    return worse + ([] if worse else [f"resolved, run --update: {s}" for s in stale])


def _percentile(values, p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else 0


def stats(state: dict) -> list:
    out = []
    for name, st in state.items():
        lengths = [n for n, _, kernel in st["functions"].values() if not kernel]
        depths = [d for _, d, _ in st["functions"].values()]
        out.append(f"{name}: {len(st['functions'])} functions | code lines p95 {_percentile(lengths, .95)} "
                   f"p99 {_percentile(lengths, .99)} max {max(lengths)} | depth p95 {_percentile(depths, .95)} "
                   f"p99 {_percentile(depths, .99)} max {max(depths)} | code lines {sum(st['modules'].values())}")
    return out


def header(found: dict) -> str:
    return "brick ratchet: " + "; ".join(
        f"{name} {len(o['functions'])} functions over, {len(o['modules'])} long modules, {len(o['private'])} private "
        f"imports, {len(o['direction'])} against direction, {o['reach']} reaches, {len(o['mute'])} mute"
        for name, o in found.items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the brick ratchet")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline (refused while anything got worse)")
    ap.add_argument("--stats", action="store_true", help="the percentiles the limits are drawn from")
    ap.add_argument("--header", action="store_true", help="one line for a lot's report")
    a = ap.parse_args(argv)
    config = json.loads(CONFIG.read_text())
    state = measure(REPO, config)
    if a.stats:
        print("\n".join(stats(state)))
        return 0
    found = offenders(state, config)
    if a.header:
        print(header(found))
        return 0
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    refusals = [] if baseline is None else check(found, baseline)
    if a.update:
        worse = [r for r in refusals if not r.startswith("resolved")]
        if worse:
            print("\n".join(worse), file=sys.stderr)
            return 1
        BASELINE.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
        return 0
    if baseline is None:
        print(f"{BASELINE} does not exist: run --update once to record today's state", file=sys.stderr)
        return 1
    print("\n".join(refusals) if refusals else header(found))
    return 1 if refusals else 0


if __name__ == "__main__":
    sys.exit(main())
