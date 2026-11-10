"""The grammar engine (nbxgrammar, its own package; its crate in rust/nbxgrammar until its repository exists) for the
tests that need it: the INSTALLED package when there is one (the wheel the dependency installs), else BUILT from the
crate (cargo, the pinned toolchain) and loaded from the repository — a test never silently skips because the library was
not built (the Dell's review 2026-11-09 23:06). Skipped only where neither a wheel nor a Rust toolchain exists, said."""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CRATE = Path(__file__).resolve().parents[2] / "rust" / "nbxgrammar"


def load():
    try:
        return importlib.import_module("nbxgrammar")
    except ImportError:
        pass
    if shutil.which("cargo") is None:
        pytest.skip("neither the nbxgrammar wheel nor a Rust toolchain on this machine: the grammar engine cannot be "
                    "installed or built here")
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=CRATE, check=True)
    name = {"darwin": "libnbxgrammar.dylib", "win32": "nbxgrammar.dll"}.get(sys.platform, "libnbxgrammar.so")
    shutil.copy2(CRATE / "target" / "release" / name, CRATE / "python" / "nbxgrammar" / name)
    sys.path.insert(0, str(CRATE / "python"))
    sys.modules.pop("nbxgrammar", None)
    return importlib.import_module("nbxgrammar")       # fails, never skips, once the crate is built
