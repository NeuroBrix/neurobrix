"""A stored positional table is read by the graph, never resized at load.

The compiled engine's loader used to hand every loaded weight dict to the component handler's
`prepare_weights`, which reshaped any weight whose name holds `pos_embed` to the request's
patch grid (bilinear, or a sincos recompute). That was a load-time compensation for traces that
froze the table, in one engine only (the Triton loader never had it: R30). On a graph that reads
the table itself — CogVideoX-5b-I2V indexes its learned 17 776-row table (226 text + 13x30x45
video rows) with `index_select` over symbolic rows — the rescale tried to view the table as a
square 133x133 grid and the compiled run died at load (2026-10-05 09:06,
"shape '[1, 3072, 133, 133]' is invalid for input of size 54607872").

What this test does if the code is wrong: a loader that still offers the weights to a handler
hook lets the hook below replace the table, and the test fails on the shape and on the call.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurobrix.core.runtime.graph_executor import GraphExecutor


class _ResizingHandler:
    """A handler that would rewrite the table, as the legacy one did."""

    def __init__(self):
        self.calls = 0

    def prepare_weights(self, weights, runtime_height, runtime_width):
        self.calls += 1
        return {k: v[:, :4] for k, v in weights.items()}


def _executor(table):
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._component_from = None
    ex.mode = "compiled"
    ex._component_handler = _ResizingHandler()
    ex._runtime_height = 480
    ex._runtime_width = 720
    ex._computable_specs = {}

    def _native(nbx_path, component, shard_map, only=None):
        ex._weights = {"patch_embed.pos_embed": table.clone()}

    ex._load_weights_native = _native
    ex._load_constants_from_graph = lambda: None
    return ex


def test_a_learned_table_reaches_the_graph_as_stored():
    table = torch.randn(1, 17776, 8)
    ex = _executor(table)
    ex.load_weights("/unused.nbx", "transformer", None)
    got = ex._weights["patch_embed.pos_embed"]
    assert tuple(got.shape) == (1, 17776, 8), got.shape
    assert torch.equal(got, table)
    assert ex._component_handler.calls == 0, "the loader offered the weights to a resize hook"
    assert ex._weights_loaded is True


if __name__ == "__main__":
    test_a_learned_table_reaches_the_graph_as_stored()
    print("ok")
