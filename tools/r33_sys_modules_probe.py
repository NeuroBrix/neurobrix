#!/usr/bin/env python3
"""R33 by measurement: run one request in-process and report, at exit, whether
torch is in sys.modules and the FIRST import path that brought it in.

A directory audit and a static import scan are proxies; this asks the process.
An import hook on `sys.meta_path` records the Python stack at the first request
for `torch` (before anything is loaded), so the answer names the frame — ours,
or upstream Triton's — that pulled it, with its file and line.

    python tools/r33_sys_modules_probe.py --triton --model TinyLlama-1.1B-Chat \\
        --prompt "The sky is" --max-tokens 8 --output /tmp/x.txt
Exit code 1 when torch is in sys.modules at the end of a --triton run.
"""
from __future__ import annotations

import atexit
import importlib.abc
import os
import sys
import traceback

_first = {"stack": None, "already": "torch" in sys.modules}


_importer = {}   # neurobrix module -> (file, line, importing module) of its FIRST import


def _nearest_frame():
    """The frame that issued the import: the innermost one outside importlib."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if "importlib" in fr.filename or "<frozen" in fr.filename or fr.filename.endswith("r33_sys_modules_probe.py"):
            continue
        return fr.filename, fr.lineno
    return None, None


class _TorchWatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "torch" or fullname.startswith("torch.")) and _first["stack"] is None:
            _first["stack"] = traceback.format_stack()[:-1]
            _first["name"] = fullname
        if fullname.startswith("neurobrix") and fullname not in _importer:
            _importer[fullname] = _nearest_frame()
        return None


def _module_of(path):
    if not path:
        return None
    marker = "/neurobrix/"
    if marker not in path:
        return os.path.basename(path)
    return "neurobrix." + path.split(marker, 1)[1].removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _report():
    present = "torch" in sys.modules
    print("\n" + "=" * 70, flush=True)
    print(f"[R33 probe] torch in sys.modules at exit: {present}"
          f"{' (already imported before the probe started)' if _first['already'] else ''}", flush=True)
    torch_mods = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
    print(f"[R33 probe] {len(torch_mods)} torch modules loaded", flush=True)
    if _first["stack"]:
        print(f"[R33 probe] first import of {_first['name']!r} — the stack that requested it:", flush=True)
        frames = [l for l in _first["stack"] if "importlib" not in l and "frozen" not in l]
        for line in frames[-14:]:
            print("    " + line.rstrip().replace("\n", "\n    "), flush=True)
    # The peeling list: every NeuroBrix module LOADED by this run whose source
    # carries a torch import (AST) — the order in which torch must leave the
    # triton branch, by directory.
    import ast, collections, importlib.util
    ours = collections.defaultdict(list)
    for name, mod in list(sys.modules.items()):
        if not name.startswith("neurobrix") or getattr(mod, "__file__", None) is None:
            continue
        try:
            tree = ast.parse(open(mod.__file__).read())
        except Exception:
            continue
        # Three kinds of torch import, three meanings:
        #   H  at the module head — executed at import: the peel target;
        #   L  inside a function — executed only if that function runs on
        #      this branch (the sys.modules verdict above is what counts);
        #   T  under `if TYPE_CHECKING:` — never executed.
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node
        kinds = {}
        for n in ast.walk(tree):
            is_imp = (isinstance(n, ast.Import) and any(a.name == "torch" or a.name.startswith("torch.") for a in n.names)) \
                or (isinstance(n, ast.ImportFrom) and n.module and (n.module == "torch" or n.module.startswith("torch.")))
            if not is_imp:
                continue
            kind, p = "H", n
            while hasattr(p, "_parent"):
                p = p._parent
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "L"
                    break
                if isinstance(p, ast.If) and "TYPE_CHECKING" in ast.dump(p.test):
                    kind = "T"
                    break
            kinds.setdefault(kind, []).append(n.lineno)
        if kinds:
            ours[name.split(".")[1] if "." in name else name].append((name, kinds))
    heads = sum(1 for items in ours.values() for _, k in items if "H" in k)
    total = sum(len(v) for v in ours.values())
    print(f"[R33 probe] {heads} NeuroBrix module(s) loaded by this run import torch at their HEAD "
          f"({total} carry a torch import of any kind), by package — H head / L function-local / T TYPE_CHECKING:", flush=True)
    for pkg, items in sorted(ours.items(), key=lambda kv: -len(kv[1])):
        print(f"    {pkg}: {sum(1 for _, k in items if 'H' in k)} head, {len(items)} any", flush=True)
        for name, kinds in sorted(items, key=lambda it: ("H" not in it[1], it[0])):
            desc = " ".join(f"{k}{v[:3]}" for k, v in sorted(kinds.items()))
            print(f"        {'H' if 'H' in kinds else ' '} {name} {desc}", flush=True)
    # The import tree: for every torch-importing NeuroBrix module, the chain of
    # importers back to the entry — the ROOTS are where a peel removes a
    # whole subtree (an eager registry import, a shared module's head).
    print("[R33 probe] importer chains (module <- importer file:line), torch-importing modules only:", flush=True)
    roots = collections.Counter()
    for pkg, items in sorted(ours.items(), key=lambda kv: -len(kv[1])):
        for name, kinds in sorted(items):
            if "H" not in kinds:
                continue
            chain, cur, seen = [], name, set()
            while cur and cur not in seen:
                seen.add(cur)
                f, ln = _importer.get(cur, (None, None))
                if f is None:
                    break
                chain.append(f"{os.path.relpath(f) if f.startswith('/') else f}:{ln}")
                cur = _module_of(f)
            if chain:
                roots[chain[0]] += 1
            print(f"    {name} <- " + " <- ".join(chain[:6]) + (" …" if len(chain) > 6 else ""), flush=True)
    print("[R33 probe] direct importers of torch-importing modules (peel roots), by count:", flush=True)
    for site, n in roots.most_common():
        print(f"    {n:3d}  {site}", flush=True)
    print("=" * 70, flush=True)
    if present and "--triton" in sys.argv:
        os._exit(1)


def main():
    sys.meta_path.insert(0, _TorchWatcher())
    atexit.register(_report)
    from neurobrix.cli import main as cli_main
    sys.argv = ["neurobrix", "run"] + sys.argv[1:]
    try:
        cli_main()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"[R33 probe] the run exited {e.code}", flush=True)


if __name__ == "__main__":
    main()
