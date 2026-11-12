"""The Metal sync counter (`NBX_SYNC_TIMING=1`) ranks kernels by the name each was loaded under, not by its type.

Before (the Mac, 2026-11-12, TinyLlama --triton): every one of 131 493 launches was counted under `MetalKernel`,
triton-ext's type name, because its kernel object keeps the function name without exposing it: no per-kernel ranking
on Metal.

What this test does if the code is wrong: a kernel loaded as `rms_norm_forward_kernel` is named `MetalKernel` — red.

Run: python -m pytest nbxtensor/tests/test_the_sync_counter_names_each_metal_kernel.py
"""


class MetalKernel:                     # triton-ext's type: no `name` attribute
    pass


class _Library:
    def get_function(self, name):
        return MetalKernel()


class _Native:
    @staticmethod
    def load_metallib(binary):
        return _Library()


def test_a_loaded_kernel_is_counted_under_its_own_name(monkeypatch):
    from nbxtensor.backend import triton_ext_driver as X
    monkeypatch.setattr(X, "_SYNC_TIMING", True)
    monkeypatch.setattr(X, "_KERNEL_NAMES", {})
    monkeypatch.setattr(X, "_native", lambda: _Native())
    drv = X.TritonExtDriver.__new__(X.TritonExtDriver)
    a, b = drv.load(b"lib", "rms_norm_forward_kernel", 0), drv.load(b"lib", "softmax_forward_kernel", 0)
    assert (X.kernel_name(a), X.kernel_name(b)) == ("rms_norm_forward_kernel", "softmax_forward_kernel")


def test_nothing_is_held_when_not_counting(monkeypatch):
    from nbxtensor.backend import triton_ext_driver as X
    monkeypatch.setattr(X, "_SYNC_TIMING", False)
    monkeypatch.setattr(X, "_KERNEL_NAMES", {})
    monkeypatch.setattr(X, "_native", lambda: _Native())
    X.TritonExtDriver.__new__(X.TritonExtDriver).load(b"lib", "k", 0)
    assert X._KERNEL_NAMES == {}
