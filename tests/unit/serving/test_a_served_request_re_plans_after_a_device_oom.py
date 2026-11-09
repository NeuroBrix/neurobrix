"""A request served by `neurobrix serve` meets a device out-of-memory the way `run` does: re-planned once at the component
and re-entered there (core/runtime/replan.py, armed by cli/commands/run.py arm_replan); the daemon then serves under the
new plan (reste_apres_064 point 8; the owner's next useful version, 2026-11-09 18:58). Before: the daemon never armed the
recovery, so the same OOM answered a 500 where `run` recovered.

No card: the serving engine over a fake executor. What this test would do if the code were wrong: a request executed with
the recovery unarmed (red), an OOM of the engine not recognised by the armed recovery (red), a re-planned request leaving
the daemon on its old plan or its old warm/cold mode (red), an ordinary request changing the daemon's plan (red)."""
from __future__ import annotations

import types

from neurobrix.serving.engine import InferenceEngine


def _plan(strategy, loading_mode):
    return types.SimpleNamespace(strategy=strategy, loading_mode=loading_mode, primary_device="cuda:0", components={})


class _Executor:
    def __init__(self, plan, replans_to=None):
        self.plan, self.replan, self._persistent_mode, self._to, self.armed_at_call = plan, None, True, replans_to, None

    def execute(self, inputs, **kw):
        self.armed_at_call = self.replan
        if self._to is not None:              # what RuntimeExecutor._replan_after_oom leaves: its new plan
            self.plan = self._to
        return {"out": 1}


def _engine(executor, plan):
    e = object.__new__(InferenceEngine)
    e.mode, e.model_name, e._plan, e._device, e._warm_serving = "triton", "org/m", plan, "cuda:0", True
    e._executor, e._hw_profile, e._input_config, e._growth = executor, None, None, None
    e._container = types.SimpleNamespace(_cache_path="unused")
    return e


def test_a_request_runs_with_the_recovery_armed_for_its_engine():
    from nbxtensor.memory.allocator import DeviceOOMError
    p = _plan("single_gpu", "eager")
    ex = _Executor(p)
    e = _engine(ex, p)
    assert e._execute({"global.prompt": "x"}) == {"out": 1}
    armed = ex.armed_at_call
    assert armed is not None, "the request ran with no recovery armed"
    assert armed.is_device_oom(RuntimeError("Failed at op 12").with_traceback(None)) is False
    try:
        try:
            raise DeviceOOMError("out of memory")
        except DeviceOOMError as oom:
            raise RuntimeError("Failed at op 12") from oom
    except RuntimeError as chained:
        assert armed.is_device_oom(chained)
    assert e._plan is p and e._warm_serving is True


def test_a_re_planned_request_leaves_the_daemon_on_the_new_plan():
    old, new = _plan("single_gpu", "eager"), _plan("lazy_sequential", "lazy")
    ex = _Executor(old, replans_to=new)
    e = _engine(ex, old)
    e._execute({"global.prompt": "x"})
    assert e._plan is new and e._warm_serving is False and ex._persistent_mode is False
