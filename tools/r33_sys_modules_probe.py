#!/usr/bin/env python3
"""R33 by measurement: run one request in-process and report, at exit, whether
torch is in sys.modules and the FIRST import path that brought it in.

A directory audit and a static import scan are proxies; this asks the process.
An import hook on `sys.meta_path` records the Python stack at the first request
for `torch` (before anything is loaded), so the answer names the frame — ours,
or upstream Triton's — that pulled it, with its file and line.

    python tools/r33_sys_modules_probe.py --engine triton --model TinyLlama-1.1B-Chat \\
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


class _TorchWatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "torch" or fullname.startswith("torch.")) and _first["stack"] is None:
            _first["stack"] = traceback.format_stack()[:-1]
            _first["name"] = fullname
        return None


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
        lines = [n.lineno for n in ast.walk(tree)
                 if (isinstance(n, ast.Import) and any(a.name == "torch" or a.name.startswith("torch.") for a in n.names))
                 or (isinstance(n, ast.ImportFrom) and n.module and (n.module == "torch" or n.module.startswith("torch.")))]
        if lines:
            ours[name.split(".")[1] if "." in name else name].append((name, lines))
    total = sum(len(v) for v in ours.values())
    print(f"[R33 probe] {total} NeuroBrix module(s) loaded by this run import torch (AST), by package:", flush=True)
    for pkg, items in sorted(ours.items(), key=lambda kv: -len(kv[1])):
        print(f"    {pkg}: {len(items)}", flush=True)
        for name, lines in sorted(items):
            print(f"        {name} (lines {lines[:4]}{'…' if len(lines) > 4 else ''})", flush=True)
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
