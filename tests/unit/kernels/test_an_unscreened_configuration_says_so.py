"""A configuration seated without an oracle is recorded as what it is.

The owner's ruling, 2026-09-12: the bare consensus screen may go on ranking by
speed — the engine must never refuse to run — but it may no longer produce a
verdict that reads as a validation. The distinction to hold is between *"this
configuration was validated"* and *"this configuration was the fastest among
candidates nobody verified"*. Until this record existed the two were written the
same way and read the same way.

Three things this pins, because the ruling has three parts:

* the seating still happens — nothing here refuses to run;
* the provenance is recorded and says plainly what it is;
* it cannot reach the certified directory, which is a different path filled only
  by `neurobrix autotune certify` and its own fp64 oracle.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_an_unscreened_configuration_says_so.py
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import launcher


@pytest.fixture(autouse=True)
def _clean():
    launcher.clear_screened()
    yield
    launcher.clear_screened()


def test_seating_without_an_oracle_still_returns_the_configs():
    """The engine runs. A provenance is not a refusal."""
    configs = ["a", "b", "c"]
    got = launcher._seat_unscreened("matmul_kernel", (1, 2), configs, 3, "no oracle")
    assert got is configs


def test_the_provenance_is_recorded_with_its_reason():
    launcher._seat_unscreened("conv2d_forward_kernel", (64, 64), ["a"], 7,
                              "the provider covers no oracle for this kernel")
    (entry,) = launcher.unscreened()
    assert entry.kernel == "conv2d_forward_kernel"
    assert entry.key == (64, 64)
    assert entry.candidates == 7
    assert "covers no oracle" in entry.reason


def test_the_announcement_does_not_read_as_a_validation(capsys):
    """The words are the point: this line is read by someone deciding whether a
    number can be quoted."""
    launcher._seat_unscreened("matmul_kernel", (8,), ["a"], 4, "no oracle provider")
    out = capsys.readouterr().out
    assert "UNSCREENED" in out
    assert "not a validated setting" in out.lower()
    assert "nothing verified" in out.lower()
    assert "never written to the certified directory" in out
    # And it must NOT contain the words that read as an endorsement.
    for word in ("validated setting is", "verified", "certified for"):
        assert f" {word} " not in out.lower().replace("nothing verified", "")


def test_clearing_forgets_both_registers():
    launcher._seat_unscreened("matmul_kernel", (8,), ["a"], 4, "no oracle")
    assert launcher.unscreened()
    launcher.clear_screened()
    assert launcher.unscreened() == []


def test_the_reason_names_the_provider_state():
    """Three distinguishable causes, because the remedy differs for each."""
    # Since the merge of 2026-09-13 a provider is always installed; `None` from it
    # means "no oracle for this kernel", and the reason names what IS covered.
    reason = launcher._no_oracle_reason(None)
    assert ("no oracle provider is installed" in reason) or ("covers no oracle for this kernel" in reason)
    launcher.set_screen_oracle(lambda *a: None)
    try:
        assert "covers no oracle" in launcher._no_oracle_reason(None)
        assert "produced no reference" in launcher._no_oracle_reason(object())
    finally:
        launcher.set_screen_oracle(None)


def test_the_certified_directory_is_a_different_path_entirely():
    """Structural, not a promise: the runtime cache writes under the machine's
    replay cache and the directory lives in the installed package."""
    from neurobrix.triton import autotune_cache
    assert "replay_cache" in autotune_cache._DIR
    assert "config/autotune" not in autotune_cache._DIR


def test_the_seat_is_recorded_under_the_key_the_choice_is_stored_under():
    """2026-09-13, production demonstration: three announcements, zero records
    stamped. The screen keyed its seats by the constexpr kwargs; Triton stores
    the choice under the shape key. This pins the replica of Triton's key."""
    import types
    from neurobrix.kernels.launcher import autotune_shape_key

    class T:                                    # a tensor-like: only its dtype matters here
        def __init__(self, dt): self.dtype = dt
    tuner = types.SimpleNamespace(arg_names=["a", "b", "c", "M", "N", "K"], keys=["M", "N", "K"],
                                  nargs={"a": T("fp16"), "b": T("fp16"), "c": T("fp32"), "M": 64, "N": 128, "K": 32})
    assert autotune_shape_key(tuner, {"BLOCK_M": 64}) == (64, 128, 32, "fp16", "fp16", "fp32"), (
        "the keys' values, then every argument's dtype, in argument order — Triton's own construction")
    # a constexpr kwarg that is also an arg name is part of the key only if it is in `keys`
    assert autotune_shape_key(tuner, {"M": 65}) == (65, 128, 32, "fp16", "fp16", "fp32")


def test_the_provider_sees_the_launch_kwargs_not_only_the_positional_arguments():
    """The convolution oracle needs kernel_height & co., which are constexpr
    launch kwargs absent from `tuner.nargs`. 2026-09-13: every live conv key
    was refused with the oracle's own suite green."""
    import types
    from neurobrix.kernels import launcher, screen_oracle as S

    seen = {}

    def four(tuner, key, buffers, meta):
        seen["meta"] = dict(meta or {}); return None

    def three(tuner, key, buffers):
        seen["three"] = True; return None

    launcher._call_screen_oracle(four, None, ("k",), [], {"kernel_height": 3})
    launcher._call_screen_oracle(three, None, ("k",), [], {"kernel_height": 3})
    assert seen == {"meta": {"kernel_height": 3}, "three": True}

    # and the real provider merges them: a conv key whose constexprs come only
    # through meta reaches the reference (which then refuses on the fake
    # operands — the point is that it got past the KeyError)
    calls = []
    orig = S.ORACLES["conv2d_forward_kernel"]
    S.ORACLES["conv2d_forward_kernel"] = ((lambda named: calls.append(sorted(named)) or None), orig[1])
    try:
        tuner = types.SimpleNamespace(base_fn=types.SimpleNamespace(__name__="conv2d_forward_kernel"),
                                      nargs={"input_pointer": types.SimpleNamespace(data_ptr=lambda: 1),
                                             "output_pointer": types.SimpleNamespace(data_ptr=lambda: 2)})
        S.provider(tuner, ("k",), [], {"kernel_height": 3, "groups": 1})
    finally:
        S.ORACLES["conv2d_forward_kernel"] = orig
    assert calls and "kernel_height" in calls[0] and "input_pointer" in calls[0]


def test_a_screened_seat_is_recorded_as_screened_with_its_adjudicator(tmp_path, monkeypatch):
    """The converse of the unscreened stamp: a silent entry must not read as
    verified by default. 2026-09-13: ten conv keys screened by the fp64 oracle
    left records indistinguishable from unscreened ones."""
    from neurobrix.kernels import launcher
    from neurobrix.triton import autotune_cache as atc
    launcher._ADJUDICATED.clear()
    launcher._ADJUDICATED[("matmul_kernel", repr((64, 64, 32)))] = "fp64 oracle"
    assert launcher.adjudicated() == {("matmul_kernel", "(64, 64, 32)"): "fp64 oracle"}
    launcher._ADJUDICATED.clear()


def test_the_screen_deduplicates_by_the_shape_key_not_the_constexpr_key():
    """Ten conv shapes share one constexpr tuple; keyed by it the screen ran on
    the first and silently skipped nine (2026-09-13, real-esrgan-x4 live)."""
    import inspect
    from neurobrix.kernels import launcher
    # `screen_configs` is a thin wrapper since 2026-09-17 (it releases the
    # allocator pool in a `finally` after every sweep); the dedup lives in the
    # body it delegates to. Inspecting the wrapper found none of this and the
    # test failed while the behaviour was unchanged — so it reads the body, and
    # asserts the wrapper really is only a wrapper.
    body = getattr(launcher, "_screen_configs", launcher.screen_configs)
    src = inspect.getsource(body)
    assert "if _rk in seen:" in src and "seen.add(_rk)" in src
    assert "if key in seen:" not in src
    wrapper = inspect.getsource(launcher.screen_configs)
    assert "empty_cache_pool" in wrapper, (
        "the wrapper must still release the sweep's pool; a sweep that keeps its "
        "blocks leaves the next key poorer, and the memory door reads what is "
        "available at that moment")
