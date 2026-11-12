"""A routed layer whose experts are held by several devices computes what it computes on one
(`ops.moe_multi_device.fused_moe_across_devices`; the two-device cases need two cards).

A placement that fills its cards in order can leave one layer's experts on both. Three defects of the split path,
found on a 128-expert coding model on two V100-32GB, 2026-11-11 (89 experts of a block on one card, 39 on the other:
"cudaMemcpy rc=700" on a run, a wrong text served):

  * each card held a pointer table of ITS experts only, packed in ascending order, and the kernel reads a table at
    the expert's number in the LAYER — right only for a card holding experts 0..n-1. A card's table keeps the layer's
    length (`memory.pointer_tables`), which also keeps the launch's key the layer's;
  * the kernel writes a pair's row at its flat position in the layer's routing, and the three outputs were sized by
    the card's own padded count — rows written past the buffer when a card receives few of the layer's pairs;
  * the combine read row i as the pair of sorted id i, where row p is the pair at flat position p.

What this test would do if the code were wrong: with experts {0, 2} on the first card and {1, 3} on the second, a
token routed to expert 1 is computed with expert 3's weights — O(1) against the fp64 oracle (case `in_table`); a token
routed to expert 2 or 3 reads past a two-entry table (case `every_expert`). `one_device` runs the same op with every
expert on one device — the op's own arithmetic (the routing read back, the positions mapped, the full-row outputs,
the per-token sum) against the oracle, on a machine with one card."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

from nbxtensor.memory.allocator import DeviceAllocator  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402

E, K_, H, F, T = 4, 2, 64, 32, 6


def _routing(probs):
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :K_]
    sc = np.take_along_axis(probs.astype(np.float64), idx, -1)
    return idx, sc / sc.sum(-1, keepdims=True)


def _oracle(h, idx, sc, gate, up, down):
    h = h.astype(np.float64)
    out = np.zeros((T, H))
    for t in range(T):
        for e, s in zip(idx[t], sc[t]):
            g = h[t] @ gate[e].astype(np.float64).T
            u = h[t] @ up[e].astype(np.float64).T
            out[t] += s * (((g / (1.0 + np.exp(-g))) * u) @ down[e].astype(np.float64).T)
    return out


@pytest.mark.parametrize("held", ["one_device", "in_table", "every_expert"])
def test_the_layer_is_the_sum_of_each_devices_share(held):
    from nbxtensor.memory.device_transfer import transfer_tensor
    from nbxtensor.memory.pointer_tables import build_ptr_tables
    from nbxtensor.ops.moe_multi_device import fused_moe_across_devices
    from nbxtensor.ops.moe_wrappers import MOE_BLOCK_M
    cards = DeviceAllocator.device_count()
    if cards == 0:
        pytest.skip("no accelerator on this machine")
    if held != "one_device" and cards < 2:
        pytest.skip("two cards are needed: one layer's experts on both")
    rng = np.random.default_rng(11)
    h = (rng.standard_normal((T, H)) * 0.5).astype(np.float16)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    if held == "in_table":
        logits[:, 2:] -= 30.0                          # every token routes to experts 0 and 1
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    idx, sc = _routing((probs / probs.sum(-1, keepdims=True)).astype(np.float32))
    gate = (rng.standard_normal((E, F, H)) * 0.1).astype(np.float16)
    up = (rng.standard_normal((E, F, H)) * 0.1).astype(np.float16)
    down = (rng.standard_normal((E, H, F)) * 0.1).astype(np.float16)

    here, other = 0, 1

    def on_devices(w):
        return [NBXTensor.from_numpy(w[e], device=here) if e % 2 == 0 or held == "one_device"
                else transfer_tensor(NBXTensor.from_numpy(w[e], device=here), other) for e in range(E)]
    lists = [on_devices(w) for w in (gate, up, down)]
    tables = build_ptr_tables(*lists)
    assert len(tables.device_experts) == (1 if held == "one_device" else 2)
    DeviceAllocator.set_device(here)
    DeviceAllocator.ensure_triton_device(here)
    out = fused_moe_across_devices(
        NBXTensor.from_numpy(h, device=here), NBXTensor.from_numpy(idx.reshape(-1).astype(np.int64), device=here),
        NBXTensor.from_numpy(sc.reshape(-1).astype(np.float16), device=here), tables, K_, E, F, MOE_BLOCK_M, here)
    got = np.asarray(out.numpy(), dtype=np.float64).reshape(T, H)
    ref = _oracle(h, idx, sc, gate, up, down)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 2e-2, f"{held}: max rel {rel:.3e} against the oracle"
    tables.release_pins()
