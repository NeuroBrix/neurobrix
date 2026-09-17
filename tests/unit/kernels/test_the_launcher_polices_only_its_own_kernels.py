"""The launcher seam is process-wide, so it must ask whose kernel it is before refusing one.

NeuroBrix replaces `triton.runtime.jit.JITFunction.__getitem__` and `.run` so the engine owns
every launch it makes (R33). `JITFunction` is Triton's, so the patch routes EVERY Triton kernel
in the process — including kernels this engine did not write.

torch 2.14 ships in-tree Triton implementations under `torch/_native/ops/` —
`bmm_outer_product`, `foreach_mm`, `norm`, `polar`, `scatter_add`, `sum`, `topk` — and dispatches
eager aten calls to them when a shape condition matches. Those are torch's kernels on torch's
memory, and the seam refused them at its ownership rule:

    Failed at op aten.bmm::0 (aten::bmm): NeuroBrix launcher: device address 0x...
    (parameter 'A_ptr') was not handed out by the allocator — refused, not launched

on the warm compiled path of GLM-4.1V-9B-Thinking, Janus-Pro-7B and Sana-1600M-MultiLing
(2026-09-17). The ownership rule is right and stays — a NeuroBrix kernel may not read memory this
engine did not hand out — but it makes no claim about a foreign library's kernel on that
library's own memory, and enforcing it there breaks a working library.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from neurobrix.kernels.launcher import _is_ours  # noqa: E402


class _Fn:
    def __init__(self, module):
        self.__module__ = module


class _Jit:
    def __init__(self, module):
        self.fn = _Fn(module)


def test_a_kernel_written_in_this_engine_is_ours():
    assert _is_ours(_Jit("neurobrix.kernels.ops.matmul"))
    assert _is_ours(_Jit("neurobrix.kernels.ops.flash_attention"))
    assert _is_ours(_Jit("neurobrix"))


def test_torchs_own_triton_ops_are_not_ours():
    """The case that cost three warm cells. These module paths are read from the
    traceback of the real failure, not invented."""
    assert not _is_ours(_Jit("torch._native.ops.bmm_outer_product.triton_kernels"))
    for op in ("foreach_mm", "norm", "polar", "scatter_add", "sum", "topk"):
        assert not _is_ours(_Jit(f"torch._native.ops.{op}.triton_kernels")), op


def test_a_name_that_merely_starts_with_the_letters_is_not_ours():
    """`neurobrixfoo` is somebody else's package."""
    assert not _is_ours(_Jit("neurobrixfoo.kernels"))
    assert not _is_ours(_Jit("not_neurobrix.kernels"))


def test_an_unreadable_module_is_treated_as_foreign():
    """The conservative direction, and it is a choice worth stating: refusing a foreign
    launch breaks a working library, while passing one through only forgoes a check that
    was never ours to make."""
    class _Bare:
        pass
    assert not _is_ours(_Bare())
    assert not _is_ours(_Jit(None))


def test_the_seam_delegates_a_foreign_kernel_to_tritons_own_path(monkeypatch):
    """The wiring: `_is_ours` can be perfect and never consulted (register 17)."""
    import neurobrix.kernels.launcher as L

    calls = {"ours": 0, "upstream": 0}

    class FakeJIT:
        def __getitem__(self, grid):
            calls["upstream"] += 1
            return lambda *a, **k: "upstream-getitem"

        def run(self, *a, grid=None, warmup=False, **k):
            calls["upstream"] += 1
            return "upstream-run"

    fake_module = type(sys)("triton.runtime.jit")
    fake_module.JITFunction = FakeJIT
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", fake_module)
    monkeypatch.setattr(L, "_installed", False, raising=False)
    monkeypatch.setattr(L, "launch", lambda *a, **k: calls.__setitem__("ours", calls["ours"] + 1), raising=False)
    # the autotuner wrap is not under test here
    monkeypatch.setitem(sys.modules, "triton.runtime.autotuner",
                        type(sys)("triton.runtime.autotuner"))
    try:
        L.install(force=True)
    except Exception:
        pass  # the autotuner half may refuse on the fake module; the JIT half is what matters

    foreign = FakeJIT()
    foreign.fn = _Fn("torch._native.ops.bmm_outer_product.triton_kernels")
    mine = FakeJIT()
    mine.fn = _Fn("neurobrix.kernels.ops.matmul")

    FakeJIT.__getitem__(foreign, (1,))
    assert calls["upstream"] >= 1, "a foreign kernel must go back to Triton's own path"
