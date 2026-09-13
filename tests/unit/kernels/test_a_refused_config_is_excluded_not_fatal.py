"""A backend refusal for ONE candidate config must cost that config, not the run.

Measured 2026-09-12, after the staged-dot nesting was served: hat-s-x4 and
real-esrgan-x2 both died with

    Refusing a tt.dot with tile dim 128 (> 64) on the generic per-thread path

raised from inside `triton/runtime/autotuner.py`: `benchmark` -> `_bench` ->
`kernel_call`. The refusal is correct and stays -- that path is validated only
to 64 and says so. What is wrong is where it lands. It escapes the sweep and
ends the run, when `conv2d_forward_kernel` declares EIGHTEEN configs of which
ELEVEN have every dimension at 64 or below. The tuner had eleven servable
choices and died on the first of the seven that refuse.

Triton already has this semantics: `_bench` catches `OutOfResources`,
`CompileTimeAssertionFailure` and `PTXASError` and scores that config `inf`.
A backend refusal is the same statement -- this config cannot be compiled here
-- and `MetalNonRecoverableError` merely descends from `RuntimeError` rather
than `TritonError`, so nothing catches it.

Two things must hold, and the second is what keeps this from becoming a
silent narrowing:

  * a refused config is excluded and the sweep continues;
  * the exclusion is SAID. A tuner that quietly drops configs leaves a shape
    slow for a reason nobody can find, and a shape where EVERY config refuses
    must not look like a shape that merely chose badly.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_a_refused_config_is_excluded_not_fatal.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from check_measurement_environment import owned_cache_env       # noqa: E402


@pytest.fixture(autouse=True)
def _owns_its_cache(tmp_path, monkeypatch):
    for var, value in owned_cache_env(tmp_path).items():
        monkeypatch.setenv(var, value)


@pytest.fixture(autouse=True)
def _forget_announcements():
    """The "say it once per process" is product behaviour and also makes the
    module order-dependent: the second test to use a reason would see nothing
    and could not tell that from a bug."""
    try:
        from neurobrix.kernels import autotune_refusals as R
        R.reset_announcements()
    except Exception:
        pass
    yield


def _policy():
    from neurobrix.kernels import autotune_refusals as R
    return R


def test_the_policy_module_exists_and_is_installed_by_the_launcher():
    """The exclusion must be installed by the engine, not by this test.

    A policy whose only caller is its test is the vacuous form the register
    carries; the launcher installs the screen oracle the same way and this
    joins it there.
    """
    R = _policy()
    from neurobrix.kernels import launcher
    src = Path(launcher.__file__).read_text()
    assert "autotune_refusals" in src, (
        "the launcher must install the refusal policy; a policy nothing "
        "installs excludes nothing")
    assert hasattr(R, "install"), "the policy must expose install()"


def test_a_refused_config_scores_infinite_and_the_sweep_survives():
    R = _policy()
    from triton_msl.errors import MetalNonRecoverableError

    calls = []

    def bench(config):
        calls.append(config)
        if config == "big":
            raise MetalNonRecoverableError("tile dim 128 (> 64)", op_name="tt.dot")
        return [1.0, 1.0, 1.0]

    guarded = R.exclude_refused_configs(bench)
    assert guarded("small") == [1.0, 1.0, 1.0]
    scored = guarded("big")
    assert scored == [float("inf")] * 3, (
        f"a refused config must score inf so `min` never picks it; got {scored}")
    assert calls == ["small", "big"], "the sweep must have continued"


def test_a_refusal_that_is_not_about_this_config_still_propagates():
    """The half that keeps this from swallowing real failures.

    An exception that is not a backend refusal -- an out-of-memory, a bug in
    the kernel, a driver fault -- must end the run exactly as it does today.
    Scoring it `inf` would turn every failure into a silently slower shape.
    """
    R = _policy()
    guarded = R.exclude_refused_configs(
        lambda config: (_ for _ in ()).throw(ValueError("not a refusal")))
    with pytest.raises(ValueError):
        guarded("any")


def test_the_exclusion_is_announced_once_per_reason():
    R = _policy()
    from triton_msl.errors import MetalNonRecoverableError

    said = []
    guarded = R.exclude_refused_configs(
        lambda config: (_ for _ in ()).throw(
            MetalNonRecoverableError("tile dim 128 (> 64)", op_name="tt.dot")),
        say=said.append)
    guarded("a")
    guarded("b")
    assert len(said) == 1, (
        f"one line per reason, not one per config: a sweep of eighteen would "
        f"print eighteen identical lines and the next reader would stop "
        f"reading them. Got {len(said)}")
    assert "tile dim 128" in said[0], "the line must carry the reason"


def test_every_config_refusing_is_not_reported_as_a_choice():
    """If nothing is servable, that is a refusal, not a slow shape.

    `min` over all-inf returns an arbitrary config, which then refuses at
    launch with a message about that config -- hiding that EVERY config
    refused. The policy must be able to say so.
    """
    R = _policy()
    assert R.all_refused({"a": [float("inf")] * 3, "b": [float("inf")] * 3}) is True
    assert R.all_refused({"a": [float("inf")] * 3, "b": [1.0, 1.0, 1.0]}) is False
