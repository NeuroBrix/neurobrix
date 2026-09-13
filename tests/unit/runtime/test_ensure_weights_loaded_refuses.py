"""The engine's real loading path must refuse, not skip.

`RuntimeExecutor._ensure_weights_loaded` is where weights are actually loaded
(87 call sites across the flows ignore its return). It had two silent exits:
an unknown component, and an executor carrying no loading params. Both left
the run going with a component that would compute on whatever was in memory.

The flag `_weights_loaded` is also fixed here: it is set by the loader itself
rather than by whichever caller remembered to set it.
"""
import pytest

from neurobrix.core.runtime.executor import RuntimeExecutor


class _Exec:
    def __init__(self, params=None, loaded=False, weights=None):
        self._weight_loading_params = params
        self._weights_loaded = loaded
        self._weights = weights or {}
        self.loaded_calls = []

    def load_weights(self, nbx_path, component, shard_map=None):
        self.loaded_calls.append((nbx_path, component, shard_map))
        self._weights_loaded = True


def _runtime(executors, strategy=None):
    rt = RuntimeExecutor.__new__(RuntimeExecutor)
    rt.executors = executors
    rt.strategy = strategy
    rt.plan = None
    return rt


def test_unknown_component_refuses_and_lists_what_exists():
    rt = _runtime({"decoder": _Exec()})
    with pytest.raises(RuntimeError) as e:
        rt._ensure_weights_loaded("vision_tower")
    msg = str(e.value)
    assert "ZERO FALLBACK" in msg and "vision_tower" in msg
    assert "decoder" in msg, "the refusal must name the components that do exist"


def test_executor_without_loading_params_refuses():
    """Every executor RuntimeFactory builds carries `_weight_loading_params`.
    One that has none was built off that path, so nobody will load it."""
    rt = _runtime({"decoder": _Exec(params=None)})
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK.*no loading\s+params"):
        rt._ensure_weights_loaded("decoder")


def test_already_loaded_returns_without_reloading():
    ex = _Exec(params={"nbx_path": "/a.nbx", "component": "decoder"}, loaded=True)
    _runtime({"decoder": ex})._ensure_weights_loaded("decoder")
    assert ex.loaded_calls == []


def test_the_ordinary_case_still_loads_once():
    ex = _Exec(params={"nbx_path": "/a.nbx", "component": "decoder", "shard_map": {}})
    rt = _runtime({"decoder": ex})
    rt._ensure_weights_loaded("decoder")
    rt._ensure_weights_loaded("decoder")          # second call is a no-op
    assert ex.loaded_calls == [("/a.nbx", "decoder", None)]


def test_shard_map_still_travels_when_present():
    sm = {"w": "cuda:1"}
    ex = _Exec(params={"nbx_path": "/a.nbx", "component": "d", "shard_map": sm})
    _runtime({"d": ex})._ensure_weights_loaded("d")
    assert ex.loaded_calls == [("/a.nbx", "d", sm)]


def test_the_loader_owns_the_loaded_flag_not_its_callers():
    """A caller that forgets to set `_weights_loaded` used to leave a fully
    loaded executor marked unloaded — and the next call loaded it again."""
    import inspect
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    src = inspect.getsource(GraphExecutor.load_weights)
    assert "self._weights_loaded = True" in src, (
        "GraphExecutor.load_weights must record that it loaded; leaving that "
        "to callers is how a component gets loaded twice")
