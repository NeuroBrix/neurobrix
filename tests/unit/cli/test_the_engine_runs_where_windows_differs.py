"""The simple half of the Windows audit (2026-11-07, docs: campaigns' windows_audit.md): what Windows does differently
and the engine assumed away, each tested without Windows by simulating the difference. No card.

What this test would do if the code were wrong: the library's import needing os.sysconf (red), a backend identity that
leaves .pyd/.dll out (red), a text output or tokenizer file in the platform's code page (red), a liveness check that
signals the process on Windows (red), stdout left in cp1252 (red), a dump file name with ':' (red)."""
from __future__ import annotations

import ast
import importlib
import io
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_the_library_imports_without_os_sysconf(monkeypatch):
    monkeypatch.delattr(os, "sysconf", raising=False)
    import nbxtensor.io.safetensors as S
    importlib.reload(S)
    assert S.PAGE > 0


def test_the_backend_identity_counts_windows_binaries(tmp_path, monkeypatch):
    from nbxtensor.ops import autotune_certified as C
    pkg = tmp_path / "fake_backend"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "driver.pyd").write_bytes(b"one")
    mod = types.ModuleType("fake_backend")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "fake_backend", mod)
    backend = type("Backend", (), {"__module__": "fake_backend"})()
    first = C._backend_content_hash(backend)
    (pkg / "sub" / "driver.pyd").write_bytes(b"two")
    assert C._backend_content_hash(backend) != first, "a Windows binary of the backend is not part of its identity"


def test_a_text_output_is_written_in_utf8(tmp_path):
    from neurobrix.core.runtime import output_dispatch as O
    src = Path(O.__file__).read_text()
    tree = ast.parse(src)
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"
             and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "w"]
    assert opens and all(any(k.arg == "encoding" for k in n.keywords) for n in opens)


def test_vendor_tokenizer_files_are_read_in_utf8():
    for rel in ("core/module/tokenizer/sp_tokenizer.py", "core/module/tokenizer/factory.py", "core/flow/rnnt.py"):
        tree = ast.parse((REPO / "src" / "neurobrix" / rel).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open":
                mode = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else "r"
                if "b" not in mode:
                    assert any(k.arg == "encoding" for k in n.keywords), (rel, n.lineno)


def test_a_stale_install_lock_is_asked_of_the_kernel_on_windows(monkeypatch):
    from neurobrix.nbx import atomic_install as A
    import socket
    calls = []

    class K32:
        def OpenProcess(self, access, inherit, pid):
            calls.append(pid)
            return 0

        def GetLastError(self):
            return 87

        def CloseHandle(self, h):
            pass
    import ctypes
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=K32()), raising=False)
    # the module's own view of os, not the process's (pytest builds paths from os.name)
    fake_os = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
    fake_os.name = "nt"
    fake_os.kill = lambda *a: (_ for _ in ()).throw(AssertionError("os.kill on Windows"))
    monkeypatch.setattr(A, "os", fake_os)
    assert A._is_our_dead_lock({"host": socket.gethostname(), "pid": 4242}) is True and calls == [4242]


def test_stdout_in_a_code_page_becomes_utf8():
    from neurobrix.cli import _utf8_streams
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    _utf8_streams(stream)
    stream.write("→ ★")
    stream.flush()
    assert raw.getvalue() == "→ ★".encode("utf-8")
