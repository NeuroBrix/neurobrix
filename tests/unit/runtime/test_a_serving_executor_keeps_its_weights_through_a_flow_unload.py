"""A serving daemon's executor is PERSISTENT: a flow's "unload" of its component (Janus's language model before the
image decoder) leaves it loaded. It used to mark it unloaded while `unload_weights` declined the release, so the next
request loaded a second copy and freed the first under the replay plans recorded on it: Janus-Pro-7B served by
Triton failed "replay input-copy failed at slot 0" on 2 requests in 3 (2026-11-13)."""
from types import SimpleNamespace

from neurobrix.core.runtime.executor import RuntimeExecutor


class _Exec:
    def __init__(self, persistent):
        self._persistent, self._weights_loaded, self.unloaded = persistent, True, 0

    def unload_weights(self):
        self.unloaded += 1


def _runtime(executor, uninstalled):
    rt = RuntimeExecutor.__new__(RuntimeExecutor)
    rt.executors = {"language_model": executor}
    rt.strategy = SimpleNamespace(uninstall_for_executor=lambda name, ex: uninstalled.append(name))
    rt.mode = "triton"
    return rt


def test_a_persistent_executor_stays_loaded_and_installed():
    ex, uninstalled = _Exec(persistent=True), []
    _runtime(ex, uninstalled)._unload_component_weights("language_model")
    assert ex._weights_loaded is True and ex.unloaded == 0 and uninstalled == []


def test_a_run_s_executor_is_still_unloaded(monkeypatch):
    import neurobrix.core.runtime.executor as E
    monkeypatch.setattr(E, "_release_device_memory", lambda *a, **k: None)
    ex, uninstalled = _Exec(persistent=False), []
    _runtime(ex, uninstalled)._unload_component_weights("language_model")
    assert ex._weights_loaded is False and ex.unloaded == 1 and uninstalled == ["language_model"]
