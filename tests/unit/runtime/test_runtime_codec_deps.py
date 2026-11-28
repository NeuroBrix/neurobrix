"""Unit tests for the generic Engine Runtime Codec / External Component Resolver."""
from __future__ import annotations

import builtins
import importlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from neurobrix.core.components import runtime_deps
from neurobrix.core.module.vision import image_dsp
from neurobrix.core.runtime import output_dispatch


def _fake_tty(monkeypatch, *, stdin_tty: bool = True, stderr_tty: bool = True) -> None:
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: stdin_tty))
    monkeypatch.setattr(sys, "stderr", types.SimpleNamespace(
        isatty=lambda: stderr_tty,
        write=lambda _s: 0,
        flush=lambda: None,
    ))


def _clear_non_interactive_env(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NBX_NONINTERACTIVE", raising=False)


def _block_import(monkeypatch, target: str) -> None:
    orig_import = importlib.import_module
    monkeypatch.delitem(sys.modules, target, raising=False)

    def _import(name, *args, **kwargs):
        if name == target:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _import)


# ---------------------------------------------------------------------------
# ensure_python_dep
# ---------------------------------------------------------------------------


def test_ensure_python_dep_returns_already_installed_module(monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **kw: called.append(a))
    mod = runtime_deps.ensure_python_dep("json", "json>=0", "parsing json")
    assert mod.__name__ == "json"
    assert called == []


@pytest.mark.parametrize(
    "stdin_tty, stderr_tty, env_key, env_val, allow_prompt",
    [
        (False, False, None, None, True),
        (True, False, None, None, True),
        (False, True, None, None, True),
        (True, True, "CI", "true", True),
        (True, True, "NBX_NONINTERACTIVE", "1", True),
        (True, True, None, None, False),
    ],
)
def test_ensure_python_dep_missing_non_interactive_or_ci_raises(
    monkeypatch, stdin_tty, stderr_tty, env_key, env_val, allow_prompt
):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=stdin_tty, stderr_tty=stderr_tty)
    if env_key is not None:
        monkeypatch.setenv(env_key, env_val)

    _block_import(monkeypatch, "missing_codec_pkg")
    monkeypatch.setattr(
        builtins,
        "input",
        lambda _p: (_ for _ in ()).throw(AssertionError("prompt should not be shown")),
    )

    with pytest.raises(RuntimeError) as exc_info:
        runtime_deps.ensure_python_dep(
            "missing_codec_pkg",
            "missing-codec-pkg>=1.2.3",
            "decoding test media",
            allow_prompt=allow_prompt,
        )

    msg = str(exc_info.value)
    assert "missing-codec-pkg>=1.2.3" in msg
    assert "decoding test media" in msg
    assert f"{sys.executable} -m pip install missing-codec-pkg>=1.2.3" in msg


@pytest.mark.parametrize("answer", ["n", "N", "", "no"])
def test_ensure_python_dep_tty_declined_raises(monkeypatch, answer):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=True, stderr_tty=True)
    _block_import(monkeypatch, "some_pkg")
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda p: (prompts.append(p), answer)[1])
    pip_calls = []
    monkeypatch.setattr(subprocess, "check_call", lambda cmd: pip_calls.append(cmd))

    with pytest.raises(RuntimeError, match=r"pip install some-pkg>=0\.1"):
        runtime_deps.ensure_python_dep("some_pkg", "some-pkg>=0.1", "testing decline")

    assert len(prompts) == 1
    assert sys.executable in prompts[0]
    assert "[y/N]" in prompts[0]
    assert pip_calls == []


def test_ensure_python_dep_tty_eof_raises(monkeypatch):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=True, stderr_tty=True)
    _block_import(monkeypatch, "some_pkg")
    monkeypatch.setattr(builtins, "input", lambda _p: (_ for _ in ()).throw(EOFError()))

    with pytest.raises(RuntimeError, match=r"pip install some-pkg>=0\.1"):
        runtime_deps.ensure_python_dep("some_pkg", "some-pkg>=0.1", "testing EOF")


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
def test_ensure_python_dep_tty_accepted_installs_and_imports(monkeypatch, answer):
    _clear_non_interactive_env(monkeypatch)
    stderr_buf = []
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stderr", types.SimpleNamespace(
        isatty=lambda: True,
        write=lambda s: stderr_buf.append(s) or len(s),
        flush=lambda: None,
    ))

    installed = {"done": False}
    fake_mod = types.SimpleNamespace(__name__="installed_pkg", sentinel=42)
    orig_import = importlib.import_module

    def _import_module(name, *args, **kwargs):
        if name == "installed_pkg":
            if not installed["done"]:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return fake_mod
        return orig_import(name, *args, **kwargs)

    pip_calls = []
    cache_invalidations = []

    def _check_call(cmd):
        pip_calls.append(cmd)
        installed["done"] = True

    monkeypatch.setattr(importlib, "import_module", _import_module)
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: cache_invalidations.append(True))
    monkeypatch.setattr(subprocess, "check_call", _check_call)
    monkeypatch.setattr(builtins, "input", lambda _p: answer)

    mod = runtime_deps.ensure_python_dep(
        "installed_pkg", "installed-pkg>=2.0", "reading video streams"
    )
    assert mod is fake_mod
    assert pip_calls == [[sys.executable, "-m", "pip", "install", "installed-pkg>=2.0"]]
    assert cache_invalidations == [True]
    assert "reading video streams" in "".join(stderr_buf)


def test_ensure_python_dep_pip_failure_raises_actionable_error(monkeypatch):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=True, stderr_tty=True)
    _block_import(monkeypatch, "bad_pkg")
    monkeypatch.setattr(builtins, "input", lambda _p: "y")
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda cmd: (_ for _ in ()).throw(subprocess.CalledProcessError(2, cmd)),
    )

    with pytest.raises(RuntimeError, match=r"Failed to install 'bad-pkg>=1\.0'.*exit code 2"):
        runtime_deps.ensure_python_dep("bad_pkg", "bad-pkg>=1.0", "testing pip fail")


# ---------------------------------------------------------------------------
# resolve_ffmpeg_exe
# ---------------------------------------------------------------------------


def test_resolve_ffmpeg_exe_prefers_imageio_ffmpeg(monkeypatch):
    fake_iio_ffmpeg = types.SimpleNamespace(get_ffmpeg_exe=lambda: "/opt/bundled/ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_iio_ffmpeg)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda _name: (_ for _ in ()).throw(AssertionError("shutil.which should not run")),
    )
    assert runtime_deps.resolve_ffmpeg_exe() == "/opt/bundled/ffmpeg"


def test_resolve_ffmpeg_exe_accepts_mocked_string_from_imageio_ffmpeg(monkeypatch):
    fake_iio_ffmpeg = types.SimpleNamespace(get_ffmpeg_exe=lambda: "fake-ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_iio_ffmpeg)
    assert runtime_deps.resolve_ffmpeg_exe() == "fake-ffmpeg"


def test_resolve_ffmpeg_exe_falls_back_to_system_path_when_imageio_ffmpeg_missing(monkeypatch):
    _block_import(monkeypatch, "imageio_ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    assert runtime_deps.resolve_ffmpeg_exe(allow_prompt=False) == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_exe_falls_back_to_system_path_when_get_ffmpeg_exe_raises(monkeypatch):
    def _broken():
        raise RuntimeError("No ffmpeg exe could be found")

    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=_broken),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)

    assert runtime_deps.resolve_ffmpeg_exe(allow_prompt=False) == "/usr/local/bin/ffmpeg"


def test_resolve_ffmpeg_exe_prompts_and_installs_when_all_missing_tty_yes(monkeypatch):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=True, stderr_tty=True)
    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    installed = {"done": False}
    orig_import = importlib.import_module
    fake_mod = types.SimpleNamespace(get_ffmpeg_exe=lambda: "/venv/bin/ffmpeg")

    def _import(name, *a, **kw):
        if name == "imageio_ffmpeg":
            if not installed["done"]:
                raise ModuleNotFoundError("No module named 'imageio_ffmpeg'")
            return fake_mod
        return orig_import(name, *a, **kw)

    pip_calls = []

    def _check_call(cmd):
        pip_calls.append(cmd)
        installed["done"] = True

    monkeypatch.setattr(importlib, "import_module", _import)
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(subprocess, "check_call", _check_call)
    monkeypatch.setattr(builtins, "input", lambda _p: "y")

    exe = runtime_deps.resolve_ffmpeg_exe()
    assert exe == "/venv/bin/ffmpeg"
    assert pip_calls == [[sys.executable, "-m", "pip", "install", "imageio-ffmpeg>=0.4.0"]]


def test_resolve_ffmpeg_exe_raises_when_all_missing_declined_or_non_tty(monkeypatch):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=False, stderr_tty=False)
    _block_import(monkeypatch, "imageio_ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match=r"pip install imageio-ffmpeg>=0\.4\.0"):
        runtime_deps.resolve_ffmpeg_exe()


# ---------------------------------------------------------------------------
# Call-site wiring: output_dispatch + image_dsp + R33 torch-free
# ---------------------------------------------------------------------------


def test_write_video_h264_uses_resolved_system_ffmpeg_without_imageio_ffmpeg(tmp_path, monkeypatch):
    out = tmp_path / "clip.mp4"
    _block_import(monkeypatch, "imageio_ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    seen_cmd = []

    class _FakeProc:
        returncode = 0

        def communicate(self, input=None):
            out.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return b"", b""

    def _popen(cmd, *a, **kw):
        seen_cmd.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _popen)

    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    output_dispatch._write_video_h264(str(out), frames, 16)
    assert seen_cmd and seen_cmd[0][0] == "/usr/bin/ffmpeg"


def test_image_dsp_video_reader_uses_ensure_python_dep(monkeypatch):
    _clear_non_interactive_env(monkeypatch)
    _fake_tty(monkeypatch, stdin_tty=False, stderr_tty=False)
    _block_import(monkeypatch, "imageio")

    with pytest.raises(RuntimeError, match=r"pip install imageio>=2\.30\.0"):
        image_dsp.native_patch_grid_video_np("dummy.mp4", {})


def test_runtime_deps_imports_without_torch():
    repo = Path(__file__).resolve().parents[3]
    tool = repo / "tools" / "r33_import_without_torch.py"
    proc = subprocess.run(
        [sys.executable, str(tool), "neurobrix.core.components.runtime_deps"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
