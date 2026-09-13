"""ExecutionPlan.explain() surfaces the strategy, its reason, and the cost.

A user asked for `--explain-plan`. Everything it prints already travels with
the plan; explain() only surfaces it, and it must include the reason -- the
field exists because an engine can decide badly and the reason is what lets a
user see the badness.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/core/test_prism_explain_plan.py -v
"""
from __future__ import annotations

import pytest


def _plan(**over):
    from neurobrix.core.prism.solver import ExecutionPlan, ComponentMemory
    import inspect

    cm = None
    for ctor in (lambda: ComponentMemory(weights_mb=1000.0),
                 lambda: ComponentMemory(1000.0),
                 lambda: ComponentMemory()):
        try:
            cm = ctor(); break
        except Exception:
            continue
    kw = dict(total_memory_mb=1000.0, strategy="single_gpu",
              components=[], target_dtype="bf16",
              component_memory={"model": cm} if cm is not None else {},
              selection_reason="fits on one device with 27% headroom",
              transient_components=["vae"])
    kw.update(over)
    sig = inspect.signature(ExecutionPlan.__init__)
    kw = {k: v for k, v in kw.items() if k in sig.parameters}
    return ExecutionPlan(**kw)


def test_explain_names_the_strategy_and_the_reason():
    text = _plan().explain()
    assert "single_gpu" in text, "the strategy must be named"
    assert "27% headroom" in text or "fits on one device" in text, (
        "the selection reason must be surfaced -- it is why the field exists")


def test_explain_lists_the_transient_components():
    text = _plan().explain()
    assert "vae" in text and "transient" in text.lower(), (
        "a transient component (released after use) changes the memory budget "
        "and must be visible in the plan explanation")


def test_explain_is_never_empty_even_with_a_bare_plan():
    text = _plan(selection_reason="", transient_components=[]).explain()
    assert "strategy" in text and text.strip(), (
        "even a plan with no reason must print its strategy -- a blank "
        "explanation is the vacuous form")
