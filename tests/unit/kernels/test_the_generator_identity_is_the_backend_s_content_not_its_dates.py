"""A certificate proves a code: the out-of-tree backend's identity comes from its files' CONTENT, never their dates.
triton's own backend hash folds in mtimes; the Mac (2026-10-20) restored its served AppleGPU backend byte for byte with
`cp -R` and no certificate served any more.

What this test does if the code is wrong: red when touching a file's date moves the identity (the date-based hash),
red when changing a file's bytes does NOT move it (an identity blind to the code it names), and red when the backend's
CPython binding (an extension module: its bytes differ per Python, the code it generates does not) moves it. The Mac,
2026-11-08: the macOS 14 Metal wheels for Python 3.10-3.13 carried byte-identical plugins and four identities, through
`metal_native.cpython-31x-darwin.so`; 235-236 kernels compiled under each produced byte-identical .ttir, .ttgir, .msl
and .metallib to the published build's (campagnes/2026_11_08_metal_wheels_macos14/proof).
"""
import os
import sys
import types

from nbxtensor.ops import autotune_certified as AC


def _fake_backend(tmp_path, monkeypatch):
    pkg = tmp_path / "fake_backend"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "compiler.py").write_text("X = 1\n")
    (pkg / "libplugin.dylib").write_bytes(b"\x00\x01plugin")
    (pkg / "native.cpython-312-darwin.so").write_bytes(b"binding for 3.12")
    mod = types.ModuleType("fake_backend.compiler")
    mod.__file__ = str(pkg / "compiler.py")
    mod.PLUGIN_LIBRARY = pkg / "libplugin.dylib"
    monkeypatch.setitem(sys.modules, "fake_backend.compiler", mod)
    cls = type("Backend", (), {"__module__": "fake_backend.compiler", "CODEGEN_ENV": ()})
    return cls(), pkg


def test_the_same_bytes_with_new_dates_keep_the_identity(tmp_path, monkeypatch):
    b, pkg = _fake_backend(tmp_path, monkeypatch)
    first = AC._backend_content_hash(b)
    for f in pkg.iterdir():
        os.utime(f, (1_000_000_000, 1_000_000_000))
    assert AC._backend_content_hash(b) == first and first.startswith("content-")


def test_new_bytes_move_the_identity(tmp_path, monkeypatch):
    b, pkg = _fake_backend(tmp_path, monkeypatch)
    first = AC._backend_content_hash(b)
    (pkg / "libplugin.dylib").write_bytes(b"\x00\x02plugin")
    assert AC._backend_content_hash(b) != first


def test_the_python_binding_is_not_the_generator(tmp_path, monkeypatch):
    b, pkg = _fake_backend(tmp_path, monkeypatch)
    first = AC._backend_content_hash(b)
    (pkg / "native.cpython-312-darwin.so").unlink()
    (pkg / "native.cpython-313-darwin.so").write_bytes(b"binding for 3.13, other bytes")
    assert AC._backend_content_hash(b) == first
