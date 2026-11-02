"""Every compiled flow registers its handler CLASS under its flow type (core/flow/base.FLOW_REGISTRY, read by
get_flow_handler). On 86f4f630 (2026-09-09) a diagnostic function was inserted between `@register_flow("iterative_process")`
and IterativeProcessHandler: the decorator registered the function, the registry mapped the flow to it."""
import importlib

import pytest

from neurobrix.core.flow.base import COMPILED_FLOW_MODULES, FLOW_REGISTRY, FlowHandler


@pytest.mark.parametrize("flow_type", sorted(COMPILED_FLOW_MODULES))
def test_the_registered_handler_is_a_flow_handler_class(flow_type):
    importlib.import_module(COMPILED_FLOW_MODULES[flow_type])
    handler = FLOW_REGISTRY[flow_type]
    assert isinstance(handler, type) and issubclass(handler, FlowHandler), (flow_type, handler)
