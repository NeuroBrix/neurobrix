"""The triton flow's checkpoint-resume guard must resolve `os` — every warm
triton diffusion request crashed with NameError at that line (found by the
serve-warm battery cells image-triton / video-triton, 2026-09-05)."""
import os

import pytest

from neurobrix.triton.flow import iterative_process as tflow


def test_resume_guard_returns_zero_without_the_opt_in(monkeypatch):
    monkeypatch.delenv("NBX_RENDER_RESUME", raising=False)
    handler = tflow.IterativeProcessTritonHandler.__new__(tflow.IterativeProcessTritonHandler) \
        if hasattr(tflow, "IterativeProcessTritonHandler") else None
    if handler is None:  # locate the handler class by its method
        cls = next(c for c in vars(tflow).values()
                   if isinstance(c, type) and hasattr(c, "_checkpoint_resume"))
        handler = cls.__new__(cls)
    assert handler._checkpoint_resume(object(), "global.hidden_states", None) == 0
    assert handler._checkpoint_resume(None, "global.hidden_states", None) == 0


def test_module_binds_os():
    assert tflow.os is os
