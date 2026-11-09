"""The grammar engine (rust/nbx_grammar) for the tests that need it: imported when installed, else BUILT from the crate
(cargo, the pinned toolchain) and loaded from the repository — a test never silently skips because the library was not
built (the Dell's review 2026-11-09 23:06). Skipped only on a machine without a Rust toolchain at all, said."""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CRATE = Path(__file__).resolve().parents[2] / "rust" / "nbx_grammar"


def load():
    try:
        return importlib.import_module("nbx_grammar")
    except ImportError:
        pass
    if shutil.which("cargo") is None:
        pytest.skip("no Rust toolchain on this machine: the grammar engine (rust/nbx_grammar) cannot be built here")
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=CRATE, check=True)
    name = {"darwin": "libnbx_grammar.dylib", "win32": "nbx_grammar.dll"}.get(sys.platform, "libnbx_grammar.so")
    shutil.copy2(CRATE / "target" / "release" / name, CRATE / "python" / "nbx_grammar" / name)
    sys.path.insert(0, str(CRATE / "python"))
    sys.modules.pop("nbx_grammar", None)
    return importlib.import_module("nbx_grammar")      # fails, never skips, once the crate is built
