"""The NeuroBrix launcher: what we launch is what Triton would have launched.

Two properties decide whether this component may replace `kernel[grid]`:

1. **Our specialization equals Triton's.** We recompute the signature from the
   arguments in pure Python because Triton's binder imports torch. If our
   answer differed from its answer, we would be compiling a different kernel
   than the one the engine has always run, and every number measured since
   would be about something else. Asserted against
   `native_specialize_impl` itself over real kernels and real arguments.

2. **With no driver registered, nothing changes.** CUDA has no NeuroBrix
   driver yet; until the machine that owns CUDA activates one and re-measures
   the zoo, `kernel[grid](...)` there must resolve exactly as it did before.

The oracle in (1) is Triton's C++ specializer, which imports torch. That is
allowed: this is a test, torch is the oracle, and the launcher under test
never touches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import launcher
from neurobrix.kernels.nbx_tensor import NBXTensor


def _triton_specialize(value, specialize=True, align=True):
    """Triton's own answer — the oracle."""
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import BaseBackend

    return native_specialize_impl(BaseBackend, value, False, specialize, align)


# --- 1. the specialization agrees ------------------------------------------

def _sample_arguments():
    """Real arguments of the shapes the engine actually passes."""
    values = [
        None, True, False,
        0, 1, -1, 15, 16, 17, -16, 1024, 4096,
        2 ** 31, -(2 ** 31), 2 ** 31 - 1, 2 ** 40,
        1e-6, 0.0, 1.5, -2.5,
    ]
    tensors = [
        NBXTensor.from_numpy(np.ones((8, 64), dtype=np.float32)),
        NBXTensor.from_numpy(np.ones(64, dtype=np.float16)),
        NBXTensor.from_numpy(np.ones(64, dtype=np.int64)),
        NBXTensor.from_numpy(np.ones(64, dtype=np.int32)),
        NBXTensor.from_numpy(np.ones(64, dtype=np.uint8)),
    ]
    return values + tensors


@pytest.mark.parametrize("specialize", [True, False])
@pytest.mark.parametrize("align", [True, False])
def test_our_specialization_matches_triton(specialize, align):
    """Every argument, both flags, ours against Triton's C++."""
    mismatches = []
    for value in _sample_arguments():
        try:
            expected = _triton_specialize(value, specialize, align)
        except Exception as exc:          # Triton refuses -> we must refuse
            with pytest.raises(launcher.SpecializationError):
                launcher.specialize_argument(value, specialize, align)
            continue
        got = launcher.specialize_argument(value, specialize, align)
        if got != expected:
            label = (f"{type(value).__name__}("
                     f"{value if not hasattr(value, 'data_ptr') else 'tensor'})")
            mismatches.append(f"{label}: ours {got} != triton {expected}")
    assert not mismatches, (
        "our specialization diverges from Triton's:\n  "
        + "\n  ".join(mismatches))


def test_an_unaligned_pointer_loses_its_divisibility_marker():
    """The marker is the whole point of the alignment flag: a kernel compiled
    as if its pointer were 16-byte aligned, launched on one that is not, is
    wrong code. Checked against Triton rather than asserted."""
    tensor = NBXTensor.from_numpy(np.ones(64, dtype=np.float32))

    class _Offset:
        dtype = tensor.dtype

        def data_ptr(self):
            return tensor.data_ptr() + 4

    assert launcher.specialize_argument(_Offset()) == ("*fp32", "")
    assert launcher.specialize_argument(_Offset()) == _triton_specialize(_Offset())


def test_an_argument_we_cannot_type_is_refused_not_guessed():
    """Triton raises for these. A launcher that guessed would compile a kernel
    for a signature nobody asked for."""
    for value in ("a string", np.int64(3), object()):
        with pytest.raises(launcher.SpecializationError):
            launcher.specialize_argument(value)


def test_specialization_covers_the_real_kernel_signatures():
    """Not just scalars in isolation: every parameter of a sample of the
    engine's own kernels, with arguments of the kind that kernel receives."""
    from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel
    from neurobrix.kernels.ops.sum import sum_forward_kernel

    x = NBXTensor.from_numpy(np.ones((8, 64), dtype=np.float32))
    out = NBXTensor.from_numpy(np.zeros(8, dtype=np.float32))
    cases = [
        (sum_forward_kernel, [x, out, 8, 64, 64, 1]),
        (rms_norm_forward_kernel,
         [x, x, x, 8, 64, 64, 1, 64, 1, 1e-6]),
    ]
    for kernel, args in cases:
        for value in args:
            assert (launcher.specialize_argument(value)
                    == _triton_specialize(value)), value


# --- 2. no driver, no change ------------------------------------------------

def test_launcher_is_transparent_without_a_driver():
    """The CUDA guarantee.

    With the launcher installed but no driver registered, `kernel[grid](...)`
    must reach Triton's own path — `JITFunction.run`, with the same grid and
    the same arguments it would have received before this component existed.
    This is what keeps the machine that owns CUDA unaffected until it chooses
    to activate a driver and re-measure the zoo.

    The launcher deliberately calls the *saved original* `__getitem__`, not
    whatever `__getitem__` the object has now: a subclass that overrode it
    would otherwise be bypassed differently with and without a driver.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _noop(x_ptr, n, BLOCK: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, BLOCK), 0.0)

    saved = launcher.active_driver()
    launcher.unregister_driver()
    was_installed = launcher.is_installed()
    if not was_installed:
        launcher.install()
    try:
        seen = []
        _noop.run = lambda *a, **k: seen.append((a, k)) or "triton-path"

        assert _noop[(4,)](123, 64, BLOCK=64) == "triton-path"
        assert len(seen) == 1, "Triton's run was not reached exactly once"
        args, kwargs = seen[0]
        assert args == (123, 64), args
        assert kwargs["grid"] == (4,)
        assert kwargs["warmup"] is False
        assert kwargs["BLOCK"] == 64
    finally:
        del _noop.run
        if not was_installed:
            launcher.uninstall()
        if saved is not None:
            launcher.register_driver(saved)


def test_transparency_holds_for_a_grid_lambda_too():
    """`kernel[lambda meta: ...]` is the form most wrappers use. Without a
    driver the callable must be handed to Triton untouched — the launcher must
    not evaluate it and pass a tuple, because Triton evaluates it with the
    compilation metadata, which we do not have here."""
    import triton
    import triton.language as tl

    @triton.jit
    def _noop2(x_ptr, BLOCK: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, BLOCK), 0.0)

    saved = launcher.active_driver()
    launcher.unregister_driver()
    was_installed = launcher.is_installed()
    if not was_installed:
        launcher.install()
    try:
        seen = []
        _noop2.run = lambda *a, **k: seen.append(k) or None
        grid = lambda meta: (7,)
        _noop2[grid](123, BLOCK=32)
        assert seen[0]["grid"] is grid, (
            "the launcher evaluated the grid lambda instead of passing it on")
    finally:
        del _noop2.run
        if not was_installed:
            launcher.uninstall()
        if saved is not None:
            launcher.register_driver(saved)


def test_install_is_idempotent_and_reversible():
    was = launcher.is_installed()
    if was:
        launcher.uninstall()
    from triton.runtime.jit import KernelInterface
    original = KernelInterface.__getitem__
    launcher.install()
    launcher.install()                     # second call must not re-wrap
    assert KernelInterface.__getitem__ is not original
    launcher.uninstall()
    assert KernelInterface.__getitem__ is original
    if was:
        launcher.install()


def test_the_launcher_names_no_backend():
    """The component is one, and vendor-agnostic. A branch on a backend name
    here would be the first crack in that."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(launcher))
    names = {"metal", "cuda", "hip", "rocm", "mps", "apple", "nvidia", "amd"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.strip().lower() in names:
                offenders.append(node.value)
        elif isinstance(node, ast.Attribute) and node.attr.lower() in names:
            offenders.append(node.attr)
    assert not offenders, (
        f"the launcher names a backend in code: {offenders}. It asks the "
        f"registry which driver is active and never what it is.")
