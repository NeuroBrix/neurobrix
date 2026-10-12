"""A flow releases a component through the engine executor's own unload brick, never by calling the
component executor's `unload_weights()` itself.

`RuntimeExecutor._unload_component_weights` uninstalls the strategy FIRST — the zero3 ratchet evicts
its resident blocks nowhere else — then unloads the executor. The iterative and static-graph handlers
of both engines called `executor.unload_weights()` directly. Before e7445f32 the ratchet kept nothing
resident across passes, so nothing showed; after it, mochi-1-preview on a 32 GB V100 (lazy_sequential,
transformer on zero3, 48 blocks resident = 18 938 MB) met that resident set under the post-loop VAE:
`aten.convolution::27` 757 MB, live_tracked 31 694 MB, triton rc=1 (mq25 gate, 2026-10-12 15:0x;
native survived by margin). Before the fix these handlers take no `unload_weights_fn`: red.
"""
import importlib
from types import SimpleNamespace

import pytest


class _Executor:
    """A component executor whose direct unload is the defect."""
    _weights_loaded = True

    def unload_weights(self):
        pytest.fail("the flow unloaded the executor itself, bypassing the strategy's uninstall")


def _ctx(name, executor):
    return SimpleNamespace(
        persistent_mode=False, plan=SimpleNamespace(loading_mode="lazy"),
        executors={name: executor}, primary_device="cpu", mode="compiled",
        pkg=SimpleNamespace(topology={"flow": {"pre_loop": [], "loop": {"components": [name]}}}),
    )


@pytest.mark.parametrize("module_name, cls_name, kwargs", [
    ("neurobrix.core.flow.iterative_process", "IterativeProcessHandler", dict(extract_primary_output_fn=None)),
    ("neurobrix.triton.flow.iterative_process", "TritonIterativeProcessHandler", dict(extract_primary_output_fn=None)),
    ("neurobrix.core.flow.static_graph", "StaticGraphHandler", {}),
    ("neurobrix.triton.flow.static_graph", "TritonStaticGraphHandler", {}),
])
def test_a_handler_releases_a_component_through_the_brick_it_is_given(monkeypatch, module_name, cls_name, kwargs):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "release_flow_memory", lambda device=None: None)
    released = []
    handler = getattr(module, cls_name)(ctx=_ctx("transformer", _Executor()), execute_component_fn=None,
                                        unload_weights_fn=released.append, **kwargs)
    if "iterative" in module_name:
        handler._unload_component("transformer", force=True)
    else:
        handler._unload_component("transformer")
    assert released == ["transformer"]


def test_the_executors_brick_uninstalls_the_strategy_before_the_executor_unloads():
    from neurobrix.core.runtime.executor import RuntimeExecutor
    order = []
    executor = SimpleNamespace(_weights_loaded=True, unload_weights=lambda: order.append("unload"),
                               _devices_computed=True)
    strategy = SimpleNamespace(uninstall_for_executor=lambda name, ex: order.append(f"uninstall {name}"))
    engine = SimpleNamespace(executors={"transformer": executor}, strategy=strategy)
    RuntimeExecutor._unload_component_weights(engine, "transformer")
    assert order == ["uninstall transformer", "unload"]
    assert executor._weights_loaded is False and executor._devices_computed is False
