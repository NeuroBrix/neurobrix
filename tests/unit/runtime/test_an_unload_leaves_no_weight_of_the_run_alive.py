"""After `unload_weights`, no weight of the component's last compiled run is alive.

MEASURED on the rack 2026-10-05: SANA-Video_2B_720p compiled on a 16 GB V100 died in the vae's
tiled decode ("tried 578 MiB, 489 MiB free, 14.40 GiB allocated"). A CPU residency probe at the
first post_loop op found the transformer's `_weights` empty and its run context's `weights`
holding 498 tensors (7 848 MB): `_prepare_execution` builds the context on `self._weights`, the
weight-key reconcile then REBINDS `self._weights` to a new dict, and `unload_weights` cleared the
new one. The old dict — every weight of the load — lived on through `self._ctx` beside the vae.
Under layer_streaming every piece kept its last load the same way.

Injection (seen red, then restored green): with the context's rebind and the unload's
`self._ctx = None` removed, the weakref to a loaded weight stayed alive after the unload — RED.
"""
from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import pytest
import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.runtime.graph_executor import GraphExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prism"))
from test_a_streamed_component_returns_what_it_would_whole import B, D, T_TRACE  # noqa: E402
from test_a_streamed_component_returns_what_it_would_whole import _graph as graph  # noqa: E402


def _loaded():
    torch.manual_seed(0)
    # The loader's key space is NOT the graph's: every key carries a wrapper prefix the
    # reconcile strips — the case where it rebinds `self._weights` (SANA-Video's transformer).
    return {"model.layers.0.w": torch.randn(D), "model.layers.1.w": torch.randn(D),
            "model.head.weight": torch.randn(8, D)}


class _CpuExecutor(GraphExecutor):
    def load_weights(self, nbx_path, component, *args, **kwargs):
        self._weights = _loaded()
        self._weights_loaded = True


@pytest.mark.parametrize("mode", ["compiled"])
def test_no_weight_of_the_last_run_survives_the_unload(mode):
    ex = _CpuExecutor(family="llm", vendor="nvidia", arch="volta", device="cpu",
                      dtype="float32", mode=mode)
    ex.load_graph_from_dict(graph())
    ex.load_weights(None, "c")
    refs = [weakref.ref(t) for t in ex._weights.values()]
    out = ex.run({"inputs_embeds": torch.randn(B, T_TRACE, D)})
    assert out, "the run returned nothing"
    assert set(ex._weights) == {"layers.0.w", "layers.1.w", "head.weight"}, \
        f"the reconcile did not rebind the loader's keys: {sorted(ex._weights)}"
    del out
    ex.unload_weights()
    gc.collect()
    alive = [r() for r in refs if r() is not None]
    assert not alive, (f"{len(alive)} of {len(refs)} weights of the run are alive after the "
                       f"unload (held by {type(getattr(ex, '_ctx', None)).__name__})")
