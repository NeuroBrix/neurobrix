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

    A test that reaches into sys.modules owes the session the table it found — and the package
    attributes: a re-import rebinds `neurobrix.cli.commands.run` on its package, and restoring the
    table alone left `commands.run` (the fresh module) and `sys.modules[...run]` (the old one) two
    objects. A later test patched one and the code read the other: the calibrate test passed alone
    and failed after any earlier import of `run` (2026-09-29, the merge-queue-14 battery).
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
        for name, mod in saved.items():
            parent, _, child = name.rpartition(".")
            holder = sys.modules.get(parent) if parent else None
            if holder is not None and hasattr(holder, child) and getattr(holder, child) is not mod:
                setattr(holder, child, mod)


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
    choice = _fresh("neurobrix.cli.engine_choice")
    import inspect
    src = inspect.getsource(run.cmd_run)
    assert "declare_hardware_profile" not in src and "declare_for(execution_mode" in src, (
        "cmd_run declares the profile to the Triton surface itself instead of through the one gate "
        "(cli/engine_choice.declare_for)")
    gate = inspect.getsource(choice.declare_for)
    i = gate.rindex("declare_hardware_profile(")    # the call: the engine's declaration of the profile to the Triton surface
    assert "if mode in TRITON_ENGINES:" in gate[:i], (
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
        _fresh("nbxtensor.ops.wrappers")


def test_the_sequential_mode_makes_no_declaration_that_needs_triton(without_triton):
    """`--sequential` on a host without triton died in `declare_for` (cli/engine_choice.py) with
    `No module named 'triton'`: the declaration imports `nbxtensor.ops.wrappers`, and no PyTorch mode reads it.
    Measured 2026-11-18 on granite with triton blocked; with the declaration skipped the run answered."""
    choice = _fresh("neurobrix.cli.engine_choice")
    profile = object()          # the declaration's import comes before any read of the profile
    for mode in ("compiled", "sequential"):
        choice.declare_for(mode, profile)
    with pytest.raises(ModuleNotFoundError, match="triton"):
        choice.declare_for("triton", profile)       # the Triton modes still declare: their wrappers read it


class _Executor:
    """What `register_into_graph_executor` asks of a PyTorch-mode executor."""

    def __init__(self, mode):
        self.mode = mode
        self.interceptors = {}

    def register_op_uid_interceptors(self, interceptors, groups=(), planned=()):
        self.interceptors.update(interceptors)


@pytest.fixture
def torch_first():
    """torch imported before the block: the block's restore drops what it imported, and torch is not re-importable."""
    return pytest.importorskip("torch")


@pytest.mark.parametrize("mode", ["compiled", "sequential"])
def test_the_tiling_interceptors_of_a_pytorch_mode_build_and_run_without_triton(torch_first, without_triton, mode,
                                                                                  monkeypatch):
    """A tiled plan with in-place adds and a pixel_shuffle proxy chain imported the NBX kernels' modules (which need
    triton) before asking the tensor's type (core/module/tiling_engine.py): a PyTorch mode on a host without triton
    met a raw ModuleNotFoundError. The torch tensors take the torch path and give torch's own answer."""
    torch = torch_first
    te = _fresh("neurobrix.core.module.tiling_engine")
    plan = te.OpLevelTilingPlan("decoder")
    plan.add_inplace_add("add_0", 0)
    spec = {"expand_uid": "e", "clone_uid": "c", "view_uid": "v", "pixel_shuffle_uid": "ps", "bcast_dim": 1,
            "bcast_factor": 4, "post_clone_shape": [1, 8, 2, 2], "post_view_shape": [1, 8, 2, 2], "upscale_factor": 2}
    monkeypatch.setattr(te.OpLevelTilingEngine, "_detect_pixel_shuffle_broadcast_chains",
                        staticmethod(lambda ge: [spec]))
    monkeypatch.setattr(te.OpLevelTilingEngine, "_detect_inplace_unary_candidates",
                        staticmethod(lambda ge: []))
    ge = _Executor(mode)
    te.OpLevelTilingEngine(plan).register_into_graph_executor(ge)
    a, b = torch.arange(6.0).reshape(2, 3), torch.ones(2, 3)
    expect = a + 2.0 * b
    out = ge.interceptors["add_0"](a, b, 2.0)
    assert out is a and torch.equal(out, expect)
    x = torch.arange(32.0).reshape(1, 8, 2, 2)
    assert torch.equal(ge.interceptors["ps"](x, 2), torch.nn.functional.pixel_shuffle(x, 2))
