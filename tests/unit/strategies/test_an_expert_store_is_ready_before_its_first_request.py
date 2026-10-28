"""An expert store's setup is part of its component's load, never of the first request (S3 1c, 2026-10-28): when zero3
installs on an `expert_store` component, the Triton sequence is compiled, the store built (its other cards' blocks
copied there) and the dense part primed — inside the weight load, a load span. Before: built at the first op of the
first run, 7.7 s of copies and 3.4 s of compile counted in Qwen3-30B-A3B's 19 s time to first token. No card."""
from types import SimpleNamespace

from neurobrix.core.strategies.zero3 import Zero3Strategy


def _strategy(monkeypatch, alloc):
    z = Zero3Strategy.__new__(Zero3Strategy)
    z.context = SimpleNamespace(allocations={"model": alloc})
    z.exec_device = "cuda:0"
    z._loaded_components, z._pinned_components, z._installed, z._ratchet = set(), set(), {}, {}
    calls = []
    monkeypatch.setattr(Zero3Strategy, "_pin_cpu_weights", lambda self, c, e: None)
    monkeypatch.setattr(Zero3Strategy, "_install", lambda self, c, e: self._installed.__setitem__(c, e))

    def build(self, c, e):
        calls.append(("build", e._triton_seq is not None))
        st = {"primed": False}
        self._ratchet[c] = st
        return st

    def prime(self, st, c):
        calls.append(("prime",))
        st["primed"] = True
    monkeypatch.setattr(Zero3Strategy, "_build_ratchet_state", build)
    monkeypatch.setattr(Zero3Strategy, "_ratchet_prime", prime)
    return z, calls


def _executor(calls):
    ex = SimpleNamespace(mode="triton", _triton_seq=None, _triton_device_index=lambda: 2)

    def compile_(dev):
        calls.append(("compile", dev))
        ex._triton_seq = object()
    ex._ensure_triton_compiled = compile_
    return ex


def test_an_expert_store_is_built_and_primed_at_install(monkeypatch):
    z, calls = _strategy(monkeypatch, {"strategy": "expert_store"})
    z.install_for_executor("model", _executor(calls))
    assert calls == [("compile", 2), ("build", True), ("prime",)], calls


def test_a_streamed_zero3_component_keeps_its_first_op_build(monkeypatch):
    z, calls = _strategy(monkeypatch, {"strategy": "zero3"})
    z.install_for_executor("model", _executor(calls))
    assert calls == [], calls
