"""The Metal sync counter (`NBX_SYNC_TIMING=1`) ranks kernels by the name each was loaded under, not by its type.

Before (the Mac, 2026-11-12, TinyLlama --triton): every one of 131 493 launches was counted under `MetalKernel`,
triton-ext's type name, because its kernel object kept the function name without exposing it: no per-kernel ranking
on Metal. Now the kernel object is nbxtensor's own pipeline (`metal_dispatch.Pipeline`), which carries its name.

What this test does if the code is wrong: a kernel loaded as `rms_norm_forward_kernel` is named `MetalKernel` — red.

Run: python -m pytest nbxtensor/tests/test_the_sync_counter_names_each_metal_kernel.py
"""


def test_a_loaded_kernel_is_counted_under_its_own_name(monkeypatch):
    from nbxtensor.backend import metal_dispatch, triton_ext_driver as X
    monkeypatch.setattr(metal_dispatch, "pipeline", lambda binary, name: metal_dispatch.Pipeline(None, name))
    drv = X.TritonExtDriver.__new__(X.TritonExtDriver)
    a, b = drv.load(b"lib", "rms_norm_forward_kernel", 0), drv.load(b"lib", "softmax_forward_kernel", 0)
    assert (X.kernel_name(a), X.kernel_name(b)) == ("rms_norm_forward_kernel", "softmax_forward_kernel")
