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
        return exclude_refused_configs(
            lambda: original(self, *args, config=config, **kwargs))()

    _bench._nbx_excludes_refusals = True
    Autotuner._bench = _bench
    return True
