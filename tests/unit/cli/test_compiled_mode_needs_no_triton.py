"""A machine with no Triton wheel must still start the engine.

macOS has no Triton wheel. The PyTorch (compiled) path needs nothing from
Triton, yet `cli/commands/run.py` imported `kernels.wrappers` unconditionally
to configure a capability surface that belongs to the Triton wrappers — and
that import pulls the kernel op modules, which carry real @triton.jit
decorators and cannot exist without the wheel.

Measured 2026-09-10 by blocking the module on a machine that has it: a
compiled run died with `No module named 'triton'` AFTER Prism had chosen
`single_gpu` and the engine had printed `Engine: COMPILED`. A fresh Mac could
not start the engine in any mode.

These tests block the module the way a Mac's absent wheel does, rather than
asserting on the source, because the defect was never visible in the source:
every import site looked ordinary.
"""
import importlib
import importlib.abc
import sys

import pytest


class _NoTriton(importlib.abc.MetaPathFinder):
    """The wheel that isn't there."""

    def find_spec(self, name, path=None, target=None):
        if name == "triton" or name.startswith("triton."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return None


@pytest.fixture
def without_triton():
    """Block the module, and put the module table back exactly as it was.

    The first version restored only `triton.*`. The `neurobrix.*` modules that
    these tests delete and re-import were left in sys.modules in the state
    they took WHILE Triton was blocked — and a later test in the same session
    then read that state: `test_ampere_profile_does_not_clamp` saw its budget
    clamped to 2 stages instead of 5. Measured 2026-09-10; the test passed
    alone and failed in the battery, which is the signature.

    A test that reaches into sys.modules owes the session the table it found.
    """
    saved = dict(sys.modules)
    for k in [k for k in sys.modules
              if k == "triton" or k.startswith("triton.")]:
        del sys.modules[k]
    finder = _NoTriton()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        # Drop anything imported during the block, then restore what was here.
        for k in [k for k in sys.modules if k not in saved]:
            del sys.modules[k]
        sys.modules.update(saved)


def _fresh(module: str):
    """Import `module` as a process that has never imported it would."""
    for k in [k for k in sys.modules if k == module or k.startswith(module + ".")]:
        del sys.modules[k]
    return importlib.import_module(module)


def test_the_cli_imports(without_triton):
    _fresh("neurobrix.cli")


def test_the_runtime_imports(without_triton):
    _fresh("neurobrix.core.runtime.executor")


def test_the_serving_engine_imports(without_triton):
    _fresh("neurobrix.serving.engine")


def test_the_compiled_run_path_does_not_reach_the_triton_wrappers(without_triton):
    """The one that would have caught it.

    `cmd_run` configures the Triton wrapper surface only when the mode is not
    compiled. Reading the source is not enough — the guard is exercised by
    importing the module that carries it and checking the gate is there in
    the compiled branch, then by the absence tests above which prove the
    chain it protects.
    """
    run = _fresh("neurobrix.cli.commands.run")
    import inspect
    src = inspect.getsource(run.cmd_run)
    i = src.index("set_hardware_profile")
    before = src[:i]
    assert 'execution_mode != "compiled"' in before, (
        "the Triton wrapper surface is configured without asking the mode; "
        "on a machine with no Triton wheel that import is fatal and the "
        "compiled path never needed it")


def test_the_wrappers_module_cannot_import_without_the_wheel(without_triton):
    """Why the gate is the fix, and a lazy import is not.

    A first attempt deferred `import triton` in wrappers.py, dispatch.py and
    ops/_configs.py. It moved the failure from line 13 to line 118 and
    changed no measurement: the module imports the kernel op modules, which
    carry real @triton.jit decorators and cannot exist without the wheel. A
    fix that does not move the measurement that motivated it was reverted.

    So this module legitimately requires Triton, and the only correct answer
    is that the compiled path must never import it. This test pins that
    requirement, so that a future lazy-import attempt is measured against it
    rather than assumed.
    """
    with pytest.raises(ModuleNotFoundError, match="triton"):
        _fresh("neurobrix.kernels.wrappers")
