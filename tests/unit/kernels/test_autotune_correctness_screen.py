"""The autotuner may not choose a wrong kernel because it is fast.

An autotuner ranks configs by speed. That is safe only while every config
computes the same thing. On a backend where one does not, speed is exactly
the wrong tiebreak: a kernel that writes half its output does half the
stores, so it is genuinely faster, so it wins — silently, with a full-shape
finite plausible tensor as the result.

Measured on Apple 2026-09-07: `mm` in fp16 at [64,32]@[32,64] selected a
config that left 32 of the 64 output columns zero.

So the launcher screens for correctness BEFORE timing anything. The screen is
vendor-agnostic — it names no backend, and its tolerance is read from the
hardware profile rather than written here, because what separates "a
different summation order" from "a different answer" is a property of the
device's arithmetic.

Agreement is decided by CONSENSUS rather than against a nominated reference.
That is not a refinement of taste: the first version anchored on `configs[0]`,
and on this machine `configs[0]` was one of the broken ones, so it excluded
the seven correct configs and kept the three wrong ones. There is no way to
know in advance which config is right — that is the whole problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import launcher


def _has_gpu():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        return _detect_gpu_backend() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="no GPU backend")


def test_the_screen_names_no_backend():
    """One screen for every backend. A branch on a vendor here would be the
    first crack in that, and the defect it catches is not Apple-specific —
    any backend can have a config that computes the wrong thing."""
    import ast
    import inspect

    source = inspect.getsource(launcher)
    tree = ast.parse(source)
    names = {"metal", "cuda", "hip", "rocm", "mps", "apple", "nvidia", "amd"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith(("screen", "_screen", "_deviation",
                                     "_snapshot", "_restore", "_writable")):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                if inner.value.strip().lower() in names:
                    offenders.append((node.name, inner.value))
    assert not offenders, f"the screen names a backend: {offenders}"


def test_the_tolerance_comes_from_the_profile_not_the_code():
    """A number written here would be a hardware fact in the wrong place."""
    from neurobrix.kernels.ops._configs import active_vendor_profile

    table = active_vendor_profile().get("autotune_screen_rtol")
    assert table, "the hardware profile declares no autotune_screen_rtol"
    assert launcher._screen_rtol("float16") == table["float16"]
    # and the engine's own spelling resolves to the same entry
    got, tol = launcher._deviation(
        np.zeros(8, dtype=np.float16).tobytes(),
        np.zeros(8, dtype=np.float16).tobytes(), "fp16")
    assert tol == table["float16"], (
        "the engine spells the dtype 'fp16' and the profile 'float16'; if "
        "those do not meet, the screen silently compares floats bit-for-bit")


def test_an_integer_result_is_compared_bit_identically():
    """No valid reordering changes an integer, so nothing is tolerated."""
    a = np.arange(8, dtype=np.int32)
    b = a.copy()
    b[3] += 1
    deviation, tolerance = launcher._deviation(a.tobytes(), b.tobytes(), "i32")
    assert deviation == float("inf") and tolerance == 0.0


def test_a_deliberately_wrong_config_is_excluded_and_recorded():
    """The screen, exercised against a candidate that computes something else.

    Not a mock of the screen: the real `screen_configs`, given real buffers
    and configs whose kernel genuinely writes different bytes.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor

    launcher.clear_screened()

    out = NBXTensor.from_numpy(np.zeros(64, dtype=np.float32))
    truth = np.arange(64, dtype=np.float32)

    class _Config:
        def __init__(self, name, fill):
            self.name, self.fill = name, fill
            self.num_warps, self.num_stages, self.num_ctas = 4, 2, 1

        def all_kwargs(self):
            return {"FILL": self.fill}

        def __str__(self):
            return self.name

    class _Fn:
        def run(self, *args, **kwargs):
            # Three configs agree on the truth; one writes zeros, which is
            # what "wrote half the output" looks like from outside.
            value = truth if kwargs["FILL"] else np.zeros(64, dtype=np.float32)
            args[0].copy_from_numpy(value) if hasattr(args[0], "copy_from_numpy") \
                else _write(args[0], value)

    def _write(tensor, value):
        import ctypes

        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        host = np.ascontiguousarray(value)
        DeviceAllocator.memcpy(tensor.data_ptr(),
                               host.ctypes.data_as(ctypes.c_void_p).value,
                               host.nbytes, kind=1)

    class _Tuner:
        arg_names = ["out"]
        base_fn = type("f", (), {"__name__": "screened_probe"})()
        fn = _Fn()
        nargs = {"out": out}

    configs = [_Config("good_a", True), _Config("bad", False),
               _Config("good_b", True), _Config("good_c", True)]
    kept = launcher.screen_configs(_Tuner(), configs, ("probe",), {})

    assert [str(c) for c in kept] == ["good_a", "good_b", "good_c"], (
        f"the screen kept {[str(c) for c in kept]}")
    recorded = launcher.screened_out()
    assert len(recorded) == 1 and recorded[0].config == "bad"
    assert recorded[0].deviation > recorded[0].tolerance
    assert recorded[0].kernel == "screened_probe"


def test_no_consensus_is_refused_rather_than_timed():
    """Two configs that disagree and nothing to break the tie."""
    from neurobrix.kernels.nbx_tensor import NBXTensor

    launcher.clear_screened()
    out = NBXTensor.from_numpy(np.zeros(16, dtype=np.float32))

    class _Config:
        def __init__(self, v):
            self.v = v
            self.num_warps = self.num_stages = self.num_ctas = 1

        def all_kwargs(self):
            return {"V": self.v}

        def __str__(self):
            return f"v{self.v}"

    class _Fn:
        def run(self, *args, **kwargs):
            import ctypes

            from neurobrix.kernels.nbx_tensor import DeviceAllocator
            host = np.full(16, float(kwargs["V"]), dtype=np.float32)
            DeviceAllocator.memcpy(args[0].data_ptr(),
                                   host.ctypes.data_as(ctypes.c_void_p).value,
                                   host.nbytes, kind=1)

    class _Tuner:
        arg_names = ["out"]
        base_fn = type("f", (), {"__name__": "split_probe"})()
        fn = _Fn()
        nargs = {"out": out}

    with pytest.raises(RuntimeError, match="no majority"):
        launcher.screen_configs(_Tuner(), [_Config(1.0), _Config(2.0)],
                                ("probe",), {})


def test_the_screen_saves_fp16_mm_on_this_machine():
    """The real defect, end to end, through the engine's own wrapper.

    `mm` in fp16 at [64,32]@[32,64] is the shape where this backend's
    `BLOCK_N=32` configs write only the first N tile — 32 of 64 output
    columns left zero, deterministically, with no error raised. The autotuner
    preferred one of them because it does half the stores.

    This test asserts the OUTCOME, which is the same before and after the
    backend is fixed: the result is correct. What changes is *why*. Before
    the upstream fix it is correct because the screen excluded the broken
    configs; after it, because there are none to exclude. So the assertion on
    exclusions is not "there are some" but "whatever was excluded belongs to
    the family we know is broken" — which fails loudly if the screen ever
    starts throwing out configs for the wrong reason.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    launcher.install()
    launcher.clear_screened()

    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 32)).astype(np.float16)
    b = rng.standard_normal((32, 64)).astype(np.float16)
    got = np.asarray(wrappers.mm(NBXTensor.from_numpy(a),
                                 NBXTensor.from_numpy(b)).numpy())
    reference = a.astype(np.float64) @ b.astype(np.float64)

    zero_columns = [c for c in range(64) if (got[:, c] == 0).all()]
    assert not zero_columns, (
        f"columns {zero_columns[:8]} came back zero: a config that writes "
        f"only the first N tile was selected, which is what the screen exists "
        f"to prevent")

    finite = np.isfinite(got)
    relative = (np.abs(got[finite].astype(np.float64) - reference[finite]).max()
                / np.abs(reference).max())
    assert relative < 2e-3, f"fp16 mm relative error {relative:.2e}"

    for entry in launcher.screened_out():
        assert "BLOCK_N: 32" in entry.config, (
            f"the screen excluded a config outside the known-broken family: "
            f"{entry.config} (deviation {entry.deviation:.2e})")
