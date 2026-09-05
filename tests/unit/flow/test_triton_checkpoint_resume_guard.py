"""The triton flow's checkpoint-resume guard must resolve `os` — every warm
triton diffusion request crashed with NameError at that line (found by the
serve-warm battery cells image-triton / video-triton, 2026-09-05)."""
import os
from types import SimpleNamespace

from neurobrix.triton.flow import iterative_process as tflow


def _handler():
    cls = tflow.TritonIterativeProcessHandler
    return cls.__new__(cls)


def test_resume_guard_returns_zero_without_the_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("NBX_RENDER_RESUME", raising=False)
    ck = SimpleNamespace(path=tmp_path / "absent.ckpt")   # nothing on disk
    assert _handler()._checkpoint_resume(ck, "global.hidden_states", None) == 0
    assert _handler()._checkpoint_resume(None, "global.hidden_states", None) == 0


def test_module_binds_os():
    assert tflow.os is os
