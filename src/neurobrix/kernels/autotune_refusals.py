"""A backend refusal for one candidate config costs that config, not the run.

Triton's autotuner already has this semantics: `_bench` catches
`OutOfResources`, `CompileTimeAssertionFailure` and `PTXASError`, scores that
config `inf`, and carries on. A backend refusal says the same thing -- this
config cannot be compiled here -- but `MetalNonRecoverableError` descends from
`RuntimeError` rather than `TritonError`, so nothing catches it and it ends
the run.

Measured 2026-09-12: hat-s-x4 and real-esrgan-x2 both died inside the sweep on

    Refusing a tt.dot with tile dim 128 (> 64) on the generic per-thread path

while `conv2d_forward_kernel` declares eighteen configs of which eleven have
every dimension at 64 or below. Eleven servable choices, and the run died on
the first of the seven that refuse.

The refusal itself is right and stays: that path is validated only to 64 and
says so. What this changes is where it lands.

Two rules, and the second is what keeps an exclusion from becoming a silent
narrowing:

  * only a REFUSAL is excluded. An out-of-memory, a kernel bug, a driver
    fault must end the run exactly as before -- scoring those `inf` would turn
    every failure into a shape that is merely slower.
  * the exclusion is SAID, once per reason. A tuner that quietly drops configs
    leaves a shape slow for a cause nobody can find.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

#: Reasons already announced this process, so a sweep of eighteen configs
#: prints one line rather than eighteen identical ones.
_ANNOUNCED: set = set()

_INF: List[float] = [float("inf"), float("inf"), float("inf")]


def reset_announcements() -> None:
    """Forget what has been said. For tests, which must not depend on order.

    The "once per process" above is deliberate product behaviour and stays;
    without a reset it also makes the behaviour order-dependent, so the second
    test to use a reason sees nothing and cannot tell that from a bug.
    """
    _ANNOUNCED.clear()


def _is_backend_refusal(exc: BaseException) -> bool:
    """Is this the backend saying it cannot compile this config correctly?

    Asked by class, not by message: matching on the text would make every
    reworded refusal a run-ending error again, silently.
    """
    try:
        from triton_msl.errors import MetalNonRecoverableError
    except Exception:                      # the backend is not installed
        return False
    return isinstance(exc, MetalNonRecoverableError)


def exclude_refused_configs(bench: Callable, say: Callable[[str], None] | None = None):
    """Wrap a per-config benchmark so a refusal scores `inf` instead of raising."""
    def _say(line: str) -> None:
        if say is not None:
            say(line)
        else:
            print(line, flush=True)

    def _run(*args, **kwargs):
        try:
            return bench(*args, **kwargs)
        except BaseException as exc:
            if not _is_backend_refusal(exc):
                raise
            reason = str(exc).strip().splitlines()[0][:160]
            if reason not in _ANNOUNCED:
                _ANNOUNCED.add(reason)
                _say(f"[AUTOTUNE_REFUSED] a candidate config is not servable on "
                     f"this backend and was excluded from the sweep: {reason}")
            return list(_INF)
    return _run


def all_refused(timings: Dict) -> bool:
    """Did EVERY config score infinite?

    `min` over all-inf returns an arbitrary config, which then refuses at
    launch with a message about that one config -- hiding that every config
    refused. A caller that can ask this can say the true thing instead.
    """
    values: Iterable = timings.values() if hasattr(timings, "values") else timings
    values = list(values)
    if not values:
        return False
    return all(all(x == float("inf") for x in (v if isinstance(v, (list, tuple)) else [v]))
               for v in values)


# ── the per-candidate time budget ──────────────────────────────────────────
#
# A third species of candidate, measured 2026-09-12 on a bf16 matmul sweep at
# M=1500: neither refused nor wrong, but PATHOLOGICALLY SLOW -- one launch of
# 2.5 minutes, which the five-launch estimate multiplied past twelve. Neither
# the refusal exclusion above nor the oracle sees it; only a campaign-level
# timeout reaped it, killing the whole run without naming the candidate.
#
# The budget is DERIVED, never a constant:
#   * first candidate of a sweep: R0 x the fp64 oracle's own CPU time for the
#     key it just computed (the one number measured before any candidate) --
#     a GPU configuration slower than the CPU float64 of the same mathematics
#     is not a configuration;
#   * later candidates: R x the fastest single-launch time completed so far
#     in the same sweep (the winner sets the scale).
#
# The ratios themselves come from measurement, and until they are measured
# they are None and the watchdog is DISARMED -- it must never guess. When the
# measurement lands, its numbers are written beside the values.
_BUDGET_RATIOS = {"first_vs_oracle_cpu": None, "later_vs_best": None}

#: The most recent oracle CPU time, in ms. The screen computes the oracle for
#: a key immediately before Triton benches that key's candidates, and sweeps
#: are serialised in-process, so "most recent" IS this sweep's key. That
#: serialisation is an assumption this module states rather than hides; a
#: parallel sweep would need the key joined explicitly.
_SWEEP: dict = {"best_ms": None, "budget_ms": None, "token": None}


def begin_sweep() -> None:
    """Reset the per-sweep state. Called when a new key's bench begins."""
    _SWEEP["best_ms"] = None
    _SWEEP["budget_ms"] = None


def current_budget_ms():
    """The budget the bench should enforce for the CURRENT candidate, or None.

    Read by the launcher's `do_bench` when its caller passed none: Triton's
    `_bench -> self.do_bench(kernel_call, quantiles=...)` chain is not ours to
    re-sign, so the wrapper publishes here and the bench consults it. Explicit
    coupling, documented at both ends.
    """
    return _SWEEP.get("budget_ms")


def _compute_budget():
    later = _BUDGET_RATIOS["later_vs_best"]
    first = _BUDGET_RATIOS["first_vs_oracle_cpu"]
    if _SWEEP["best_ms"] is not None:
        return None if later is None else later * _SWEEP["best_ms"]
    if first is None:
        return None
    try:
        from neurobrix.kernels.screen_oracle import _ORACLE_MS
        if _ORACLE_MS:
            return first * next(reversed(_ORACLE_MS.values()))
    except Exception:                                  # noqa: BLE001
        pass
    return None


def note_candidate_time(single_launch_ms: float) -> None:
    """Feed a completed candidate's probe time back into the sweep state."""
    best = _SWEEP.get("best_ms")
    if best is None or single_launch_ms < best:
        _SWEEP["best_ms"] = float(single_launch_ms)


def exclude_slow_candidates(bench, say=None):
    """Wrap a per-config benchmark so an over-budget candidate scores `inf`.

    Same shape as `exclude_refused_configs`, same two rules: only the
    dedicated exception is caught -- everything else propagates -- and the
    exclusion is SAID with both of its numbers, because a candidate scored
    out for time without its time is a silent narrowing.
    """
    def _say(line):
        (say or (lambda l: print(l, flush=True)))(line)

    def _run(*args, **kwargs):
        from neurobrix.kernels.launcher import CandidateOverTimeBudget

        _SWEEP["budget_ms"] = _compute_budget()
        try:
            out = bench(*args, **kwargs)
        except CandidateOverTimeBudget as exc:
            _say(f"[AUTOTUNE_SLOW] a candidate took {exc.took_ms:.1f} ms for "
                 f"ONE launch against a budget of {exc.budget_ms:.1f} ms "
                 f"derived from this sweep's own measurements; excluded from "
                 f"the sweep, which continues")
            return list(_INF)
        finally:
            _SWEEP["budget_ms"] = None
        return out
    return _run


def install() -> bool:
    """Wrap `Autotuner._bench` so a refused config is excluded, once.

    Installed by the launcher beside the screen oracle. Returns False when the
    autotuner is not importable, so a caller can say so rather than assume it
    took.
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except Exception:
        return False
    if getattr(Autotuner._bench, "_nbx_excludes_refusals", False):
        return True
    original = Autotuner._bench

    def _bench(self, *args, config=None, **kwargs):
        # A sweep boundary is a change of (tuner, live-arg key): Triton
        # benches one key's candidates consecutively, so the first _bench of
        # a new pair is the first candidate of a new sweep. Detected HERE
        # because nothing upstream of _bench is ours to hook -- and without
        # this call, `begin_sweep` would be machinery with no caller, the
        # register's own vacuous form.
        try:
            from neurobrix.triton import autotune_cache as _atc
            self.nargs = dict(zip(self.arg_names, args))
            _token = (id(self), str(_atc.key_of(self, args, kwargs)))
        except Exception:                              # noqa: BLE001
            _token = (id(self), None)
        if _SWEEP.get("token") != _token:
            begin_sweep()
            _SWEEP["token"] = _token
        return exclude_slow_candidates(exclude_refused_configs(
            lambda: original(self, *args, config=config, **kwargs)))()

    _bench._nbx_excludes_refusals = True
    Autotuner._bench = _bench
    return True
