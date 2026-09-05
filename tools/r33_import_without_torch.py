#!/usr/bin/env python3
"""Import modules with torch BLOCKED — the import-time half of the R33 proof
for the shared orchestrator (the modules both engines load).

A finder at the head of sys.meta_path refuses `torch` and every submodule,
so a module that pulls torch at import — itself or through anything it
imports — fails here with the importer's traceback. Exit 1 on any failure.

    python tools/r33_import_without_torch.py neurobrix.core.flow neurobrix.core.dtype ...
"""
from __future__ import annotations

import importlib
import importlib.abc
import sys
import traceback


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"R33: {fullname!r} is blocked in this process — the ATen branch is not loaded")
        return None


def main(names):
    sys.meta_path.insert(0, _Blocker())
    failed = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — the point is to name the importer
            frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/neurobrix/" in f.filename]
            where = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "?"
            failed.append((name, where, f"{type(exc).__name__}: {exc}"))
    present = "torch" in sys.modules
    for name, where, err in failed:
        print(f"FAIL {name} <- {where}: {err}")
    print(f"{len(names) - len(failed)}/{len(names)} imported without torch; torch in sys.modules: {present}")
    return 1 if failed or present else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
