"""An executor owns the state its Triton run reads: whether its constants are bound in fp32. Before, only
`load_weights` set it, so a Triton run of an executor whose weights were never loaded (a weightless graph run by a
caller) raised an unnamed AttributeError at the first run (2026-11-17).

What the test would do if the code were wrong: the state missing from a new executor (red)."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("mode", ["triton", "triton_sequential"])
def test_a_new_triton_executor_states_its_fp32_constants_are_not_bound(mode):
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor(family="video", vendor="nvidia", arch="volta", device="cuda:0", dtype="float16", mode=mode)
    assert ex._fp32_constants_bound is False
