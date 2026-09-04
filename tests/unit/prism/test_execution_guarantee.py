"""Prism guarantees execution: it says "slow", not "no".

Prism used to end its cascade with

    ZERO FALLBACK: No strategy can fit this model.

That contradicts the engine's own philosophy. Here a model never says it
will not run — it says it will run and that it will be slow. ZERO FALLBACK
forbids *silent defaults*; it does not forbid a slow path chosen out loud
and announced.

The ladder now ends in `cpu_streaming`: components load and release one at a
time, so the requirement drops from `sum(components)` to `max(component)` —
the same reduction that makes `lazy_sequential` viable on a GPU too small for
a whole model. A refusal survives only for real impossibility: the largest
single component does not fit even alone.

The second half of these pins is that the choice is EXPLAINED. Prism does not
take the first strategy that fits, it scores every viable one and takes the
fastest; that is invisible unless said, and an engine that decides without
saying so is indistinguishable from one that decides badly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurobrix.core.prism.solver import PrismSolver


class _Mem:
    def __init__(self, total_mb):
        self.total_mb = total_mb
        self.weight_mb = total_mb * 0.9
        self.activation_mb = total_mb * 0.1


def _profile(ram_mb: int, devices=()):
    return SimpleNamespace(
        cpu=SimpleNamespace(ram_mb=ram_mb, architecture="x86_64", cores=8),
        devices=list(devices),
        topology=None,
    )


@pytest.fixture
def solver():
    return PrismSolver()


# --- the rung exists and is reachable ---------------------------------------

def test_cpu_streaming_is_registered_in_the_cascade():
    """A rung nobody calls is not a rung."""
    import inspect

    source = inspect.getsource(PrismSolver)
    assert '("cpu_streaming", self._try_cpu_streaming)' in source, (
        "cpu_streaming must be in the strategy list, after cpu_execution"
    )
    assert '"cpu_streaming": 5' in source, (
        "it must be scored below every other strategy so it is only ever "
        "chosen last"
    )


def test_streaming_accepts_what_cpu_execution_refuses(solver):
    """The exact gap that produced 'No strategy can fit'.

    Four components of 3 GB each: 12 GB total, so `cpu_execution` refuses on
    an 8 GB machine (budget 0.7 x 8 = 5.6 GB). But the largest single
    component is 3 GB, which fits — so streaming runs it."""
    comps = [(f"c{i}", _Mem(3000)) for i in range(4)]
    profile = _profile(ram_mb=8000)

    assert solver._try_cpu_execution(comps, {}, [], {}, profile, None) is None, (
        "precondition: sum(12000MB) exceeds the 5600MB budget"
    )

    result = solver._try_cpu_streaming(comps, {}, [], {}, profile, None)
    assert result is not None, "streaming must accept: max(3000MB) fits in 5600MB"
    allocations, _devices = result
    assert set(allocations) == {"c0", "c1", "c2", "c3"}
    assert all(dev == "cpu" for dev, _shards in allocations.values())


def test_streaming_refuses_only_when_one_component_cannot_fit(solver):
    """The real bottom of the ladder: a single component larger than RAM."""
    comps = [("huge", _Mem(40000))]
    profile = _profile(ram_mb=8000)
    assert solver._try_cpu_streaming(comps, {}, [], {}, profile, None) is None


def test_missing_cpu_telemetry_does_not_cause_a_refusal(solver):
    """Refusing because we failed to measure the machine would be refusing on
    ignorance. `cpu_execution` already accepts in that case; so does this."""
    comps = [("c0", _Mem(999_999))]
    profile = _profile(ram_mb=0)
    assert solver._try_cpu_streaming(comps, {}, [], {}, profile, None) is not None


# --- the plan says what it is doing -----------------------------------------

def test_streaming_forces_lazy_loading():
    """Eager loading would restore the sum(components) requirement this rung
    exists to avoid, silently undoing it."""
    import inspect

    source = inspect.getsource(PrismSolver)
    assert 'if strategy == "cpu_streaming":' in source
    assert source.count('loading_mode = "lazy"') >= 1


def test_the_plan_carries_a_selection_reason():
    from neurobrix.core.prism.solver import ExecutionPlan

    assert "selection_reason" in ExecutionPlan.__dataclass_fields__, (
        "the reason must travel with the plan, not only be printed"
    )


def test_the_explanation_names_what_the_choice_beat(solver):
    ranked = [
        (1000.0, "single_gpu", {}, []),
        (850.0, "pipeline_parallel", {}, []),
        (100.0, "zero3", {}, []),
    ]
    reason = solver._explain_choice("single_gpu", 1000.0, ranked, _profile(8000))
    assert "single_gpu scored 1000" in reason
    assert "pipeline_parallel" in reason, "must name what it was preferred over"


def test_the_explanation_warns_that_streaming_is_slow(solver):
    """A slow run the user was told about is a decision; one they were not
    told about is a surprise."""
    ranked = [(5.0, "cpu_streaming", {}, [])]
    reason = solver._explain_choice("cpu_streaming", 5.0, ranked, _profile(8000))
    assert "slow" in reason.lower()
    assert "stream" in reason.lower()


def test_a_single_viable_strategy_is_said_to_be_the_only_one(solver):
    ranked = [(10.0, "cpu_execution", {}, [])]
    reason = solver._explain_choice("cpu_execution", 10.0, ranked, _profile(8000))
    assert "only viable" in reason


# --- the refusal that remains is honest -------------------------------------

def test_the_remaining_refusal_names_the_blocking_component(solver):
    """When it does refuse, it must say what would make it run rather than
    listing strategies the user cannot act on."""
    comps = [("small", _Mem(100)), ("enormous", _Mem(90000))]
    with pytest.raises(RuntimeError) as excinfo:
        solver._fail_error(comps, [])
    message = str(excinfo.value)
    assert "cannot run on this machine" in message
    assert "enormous" in message, "must name the component that blocks it"
    assert "90000MB" in message.replace(",", "")
    assert "cpu_streaming" in message, "must show the ladder was exhausted"
    assert "More host RAM" in message, "must say what would fix it"
