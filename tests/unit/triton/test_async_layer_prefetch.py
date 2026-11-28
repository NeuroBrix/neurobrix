"""Cardless unit tests for AsyncLayerPrefetcher and Zero3 async double/N-buffering.

100% CPU-only (no GPU required): tests `AsyncLayerPrefetcher` with both mock
tensors and real `nbxtensor.memory.NBXStream` / `NBXEvent` (backed by fake
stream/event ops), integration with Prism's `ResidencyPlan.prefetch_depth` and
`Zero3Strategy`, and enforces the zero-Torch AST invariant across
`src/neurobrix/triton/**/*.py`.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from nbxtensor.memory import NBXEvent, NBXStream
from neurobrix.core.state.residency_tier import ResidencyPlan
from neurobrix.triton.strategies.async_prefetch import (
    AsyncLayerPrefetcher,
    resolve_prefetch_depth,
)
from neurobrix.triton.strategies.base import StrategyContext
from neurobrix.triton.strategies.zero3 import Zero3Strategy


# ---------------------------------------------------------------------------
# Helpers — 100% CPU-only deterministic fakes
# ---------------------------------------------------------------------------

class FakeStreamOps:
    """Cardless backend ops for real `NBXStream` and `NBXEvent` objects."""

    def __init__(self) -> None:
        self._next_stream = 100
        self._next_event = 500
        self.current_device = 0
        self.log: List[Tuple[Any, ...]] = []

    def get_device(self) -> int:
        return self.current_device

    def set_device(self, idx: int) -> None:
        self.current_device = idx
        self.log.append(("set_device", idx))

    def restore_device(self, idx: int) -> None:
        self.current_device = idx
        self.log.append(("restore_device", idx))

    def create_stream(self) -> int:
        handle = self._next_stream
        self._next_stream += 1
        self.log.append(("create_stream", handle, self.current_device))
        return handle

    def destroy_stream(self, handle: int) -> None:
        self.log.append(("destroy_stream", handle))

    def stream_synchronize(self, handle: int) -> None:
        self.log.append(("stream_synchronize", handle))

    def create_event(self, timing: bool = True) -> int:
        handle = self._next_event
        self._next_event += 1
        self.log.append(("create_event", handle, timing))
        return handle

    def destroy_event(self, handle: int) -> None:
        self.log.append(("destroy_event", handle))

    def record_event(self, ev: int, stream: int = 0) -> None:
        self.log.append(("record_event", ev, stream))

    def stream_wait_event(self, stream: int, ev: int, flags: int = 0) -> None:
        self.log.append(("stream_wait_event", stream, ev))

    def event_synchronize(self, ev: int) -> None:
        self.log.append(("event_synchronize", ev))


class FakeHostTensor:
    """Deterministic host/device tensor recording sync vs async transfers."""

    def __init__(
        self,
        name: str,
        *,
        is_cpu: bool = True,
        pinned: bool = True,
        log: List[Tuple[Any, ...]] | None = None,
    ) -> None:
        self.name = name
        self.is_cpu = is_cpu
        self._device = "cpu" if is_cpu else "cuda"
        self._pinned = pinned
        self._nbytes = 4096
        self.log = log if log is not None else []

    def is_pinned(self) -> bool:
        return self._pinned

    def to_cuda(self, device_idx: int = 0) -> "FakeHostTensor":
        self.log.append(("to_cuda_sync", self.name, device_idx))
        return FakeHostTensor(f"{self.name}@cuda:{device_idx}", is_cpu=False, pinned=False, log=self.log)

    def to(self, device: str, *, non_blocking: bool = False, stream: Any = None) -> "FakeHostTensor":
        stream_h = int(stream) if stream is not None else 0
        self.log.append(("to", self.name, device, non_blocking, stream_h))
        return FakeHostTensor(f"{self.name}@{device}", is_cpu=False, pinned=False, log=self.log)


# ---------------------------------------------------------------------------
# 1. prefetch_depth == 0 (synchronous path, zero stream overhead)
# ---------------------------------------------------------------------------

def test_prefetch_depth_zero_is_synchronous_with_zero_stream_overhead():
    stream_calls: List[int] = []
    transfer_log: List[Tuple[Any, ...]] = []

    prefetcher = AsyncLayerPrefetcher(
        prefetch_depth=0,
        device="cuda:0",
        stream_factory=lambda idx: stream_calls.append(idx) or None,
        backend_name="cuda",
    )

    assert prefetcher.prefetch_depth == 0
    assert prefetcher.copy_stream is None
    assert prefetcher.copy_stream_handle == 0
    assert prefetcher.is_async is False
    assert stream_calls == []
    assert prefetcher.next_keys([0, 1, 2, 3], 0) == []

    w0 = FakeHostTensor("l0.w", is_cpu=True, pinned=True, log=transfer_log)
    moved = prefetcher.prefetch(0, {"w": w0})
    assert "w" in moved and moved["w"].name == "l0.w@cuda:0"
    assert transfer_log == [("to_cuda_sync", "l0.w", 0)]
    assert prefetcher.pending_keys == ()

    awaited = prefetcher.await_layer(0)
    assert awaited is moved
    prefetcher.close()


# ---------------------------------------------------------------------------
# 2. prefetch_depth == 1 (double-buffering over real NBXStream / NBXEvent)
# ---------------------------------------------------------------------------

def test_prefetch_depth_one_double_buffering_with_nbx_stream_and_event():
    ops = FakeStreamOps()
    transfer_log: List[Tuple[Any, ...]] = []

    prefetcher = AsyncLayerPrefetcher(
        prefetch_depth=1,
        device="cuda:1",
        stream_factory=lambda idx: NBXStream(idx, _ops=ops),
        backend_name="cuda",
    )

    assert prefetcher.is_async is True
    assert prefetcher.copy_stream_handle == 100
    assert prefetcher.next_keys([0, 1, 2, 3], 0) == [1]

    # Prefetch layer 1 asynchronously while layer 0 computes
    w1 = FakeHostTensor("l1.w", is_cpu=True, pinned=True, log=transfer_log)
    moved = prefetcher.prefetch(1, {"w": w1})
    assert moved["w"].name == "l1.w@cuda:1"
    assert transfer_log == [("to", "l1.w", "cuda:1", True, 100)]
    assert prefetcher.pending_keys == (1,)
    assert ("record_event", 500, 100) in ops.log

    # Compute stream waits on the readiness event without blocking host
    compute_stream = NBXStream(1, _ops=ops)
    awaited = prefetcher.await_layer(1, compute_stream=compute_stream)
    assert awaited is moved
    assert prefetcher.pending_keys == ()
    assert ("stream_wait_event", compute_stream.handle, 500) in ops.log
    assert ("destroy_event", 500) in ops.log

    # When compute_stream is None, await_layer synchronizes the event on host
    w2 = FakeHostTensor("l2.w", is_cpu=True, pinned=True, log=transfer_log)
    prefetcher.prefetch(2, {"w": w2})
    prefetcher.await_layer(2, compute_stream=None)
    assert ("event_synchronize", 501) in ops.log
    assert ("destroy_event", 501) in ops.log

    compute_stream.close()
    prefetcher.close()
    assert ("destroy_stream", 100) in ops.log
    # Idempotent close does not double-destroy
    before_count = ops.log.count(("destroy_stream", 100))
    prefetcher.close()
    assert ops.log.count(("destroy_stream", 100)) == before_count


# ---------------------------------------------------------------------------
# 3. prefetch_depth == 2 (N-buffering window + clear/close cleanup)
# ---------------------------------------------------------------------------

def test_prefetch_depth_two_window_and_cleanup():
    ops = FakeStreamOps()
    transfer_log: List[Tuple[Any, ...]] = []

    with AsyncLayerPrefetcher(
        prefetch_depth=2,
        device="cuda:0",
        stream_factory=lambda idx: NBXStream(idx, _ops=ops),
        backend_name="cuda",
    ) as prefetcher:
        assert prefetcher.next_keys([0, 1, 2, 3, 4], 0) == [1, 2]
        assert prefetcher.next_keys([0, 1, 2, 3, 4], 3) == [4]

        for bidx in (0, 1, 2):
            w = FakeHostTensor(f"l{bidx}.w", is_cpu=True, pinned=True, log=transfer_log)
            prefetcher.prefetch(bidx, {"w": w})

        assert set(prefetcher.cached_keys) == {0, 1, 2}
        assert set(prefetcher.pending_keys) == {0, 1, 2}
        assert len(prefetcher) == 3

        # Idempotent prefetch does not re-transfer an already cached layer
        prev_len = len(transfer_log)
        prefetcher.prefetch(1, {"w": FakeHostTensor("dup", log=transfer_log)})
        assert len(transfer_log) == prev_len

        # Clear keeping resident layer 0 drops layers 1 & 2 and destroys their unwaited events
        prefetcher.clear(keep={0})
        assert prefetcher.cached_keys == (0,)
        assert prefetcher.pending_keys == (0,)
        assert ("destroy_event", 501) in ops.log
        assert ("destroy_event", 502) in ops.log

    # Exiting context manager closes remaining event (500) and copy stream (100)
    assert ("destroy_event", 500) in ops.log
    assert ("destroy_stream", 100) in ops.log


# ---------------------------------------------------------------------------
# 4. Edge cases: None, non-pinned, already-on-GPU, Metal/CPU, negative depth
# ---------------------------------------------------------------------------

def test_edge_cases_none_nonpinned_device_and_metal():
    ops = FakeStreamOps()
    transfer_log: List[Tuple[Any, ...]] = []

    with pytest.raises(ValueError):
        AsyncLayerPrefetcher(prefetch_depth=-1)

    # Metal / MPS / CPU never allocate a discrete H2D copy stream
    for dev, backend in [("mps", "metal"), ("metal:0", None), ("cuda:0", "metal"), ("cpu", "cuda")]:
        called: List[int] = []
        p = AsyncLayerPrefetcher(
            prefetch_depth=2,
            device=dev,
            stream_factory=lambda idx, c=called: c.append(idx),
            backend_name=backend,
        )
        assert p.is_async is False
        assert p.copy_stream is None
        assert called == []
        p.close()

    # None tensors skipped, non-pinned transferred sync, already-GPU kept as-is
    p = AsyncLayerPrefetcher(
        prefetch_depth=1,
        device="cuda:0",
        stream_factory=lambda idx: NBXStream(idx, _ops=ops),
        backend_name="cuda",
    )
    unpinned = FakeHostTensor("unpinned.w", is_cpu=True, pinned=False, log=transfer_log)
    gpu_tensor = FakeHostTensor("already_gpu.w", is_cpu=False, pinned=False, log=transfer_log)
    moved = p.prefetch(0, {"missing": None, "u": unpinned, "g": gpu_tensor})
    assert "missing" not in moved
    assert moved["g"] is gpu_tensor
    assert moved["u"].name == "unpinned.w@cuda:0"
    assert transfer_log == [("to_cuda_sync", "unpinned.w", 0)]
    # No async DMA was enqueued -> no event recorded
    assert p.pending_keys == ()
    p.close()


# ---------------------------------------------------------------------------
# 5. Integration with Prism ResidencyPlan.prefetch_depth & Zero3Strategy
# ---------------------------------------------------------------------------

def test_resolve_prefetch_depth_and_zero3_ratchet_integration(monkeypatch):
    assert resolve_prefetch_depth(None) == 0
    assert resolve_prefetch_depth({"prefetch_depth": 2}) == 2
    rp_host = ResidencyPlan(
        residency_tier="host_pinned",
        vram_budget_mb=16384.0,
        host_pinned_budget_mb=65536.0,
        peer_spill_allowed=False,
        interconnect_bandwidth_gbps=0.0,
        prefetch_depth=2,
        reason="host pinned double/triple buffer",
    )
    assert resolve_prefetch_depth(rp_host) == 2

    # Stub device drain so _evict_block runs 100% cardless
    from nbxtensor.memory import block_upload as BU
    monkeypatch.setattr(BU, "wait_device", lambda: None)

    ops = FakeStreamOps()
    transfer_log: List[Tuple[Any, ...]] = []

    class _FakeSeq:
        def __init__(self, n_blocks: int) -> None:
            self._blocks = {
                b: {"first_op": b * 2, "last_op": b * 2 + 1, "weight_tensor_ids": [f"param::l{b}.w"]}
                for b in range(n_blocks)
            }
            self.rebound: List[Dict[str, Any]] = []

        def get_op_blocks(self):
            return self._blocks

        def rebind_partial(self, mapping):
            self.rebound.append(dict(mapping))
            return list(mapping)

        def recompute_op_devices_for_slots(self, slots):
            pass

        def materialize_slots_depending_on(self, slots):
            pass

        def mark_cpu_weighted_ops_for_transfer(self, dev):
            return 0

        def override_weightless_op_devices(self, dev):
            pass

    def _run_zero3_with_plan(residency_plan: Any, *, depth_expected: int) -> Dict[str, Any]:
        transfer_log.clear()
        seq = _FakeSeq(4)
        weights = {f"l{b}.w": FakeHostTensor(f"l{b}.w", is_cpu=True, pinned=True, log=transfer_log) for b in range(4)}
        executor = SimpleNamespace(_triton_seq=seq, _weights=weights)

        ctx = StrategyContext(
            strategy_name="zero3",
            allocations={"lm": {"device": "cuda:0", "strategy": "zero3", "resident_weight_mb": 0.0}},
            residency_plan=residency_plan,
        )
        strat = Zero3Strategy(ctx)

        # Inject fake NBXStream factory so AsyncLayerPrefetcher stays 100% cardless
        orig_init = AsyncLayerPrefetcher.__init__

        def _cardless_init(self_p, prefetch_depth=0, device="cuda:0", **kw):
            kw.setdefault("stream_factory", lambda idx: NBXStream(idx, _ops=ops))
            kw.setdefault("backend_name", "cuda")
            orig_init(self_p, prefetch_depth=prefetch_depth, device=device, **kw)

        monkeypatch.setattr(AsyncLayerPrefetcher, "__init__", _cardless_init)
        state = strat._build_ratchet_state("lm", executor)
        assert state is not None
        assert state["prefetch_depth"] == depth_expected
        state["primed"] = True

        # Enter block 0 -> should trigger lookahead prefetch of `depth_expected` blocks
        strat._ratchet_on_op(state, 0)
        cached_after_b0 = sorted(state["gpu_cache"].keys())
        strat._reset_ratchet("lm")
        strat._uninstall("lm")
        return {"cached_after_b0": cached_after_b0, "transfers": list(transfer_log)}

    # Case A: residency_plan absent -> defaults safely to prefetch_depth=0 (sync, no lookahead)
    res_none = _run_zero3_with_plan(None, depth_expected=0)
    assert res_none["cached_after_b0"] == [0]
    assert all(t[0] == "to_cuda_sync" for t in res_none["transfers"])

    # Case B: residency_plan.prefetch_depth == 2 -> prefetches blocks 1 and 2 ahead asynchronously
    res_d2 = _run_zero3_with_plan(rp_host, depth_expected=2)
    assert res_d2["cached_after_b0"] == [0, 1, 2]
    assert all(t[0] == "to" and t[3] is True for t in res_d2["transfers"])


# ---------------------------------------------------------------------------
# 6. AST zero-Torch invariant across src/neurobrix/triton/**/*.py
# ---------------------------------------------------------------------------

def test_zero_torch_imports_across_triton_tree():
    triton_root = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "triton"
    assert triton_root.is_dir()
    offenders: List[str] = []
    for py_file in sorted(triton_root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        offenders.append(f"{py_file.relative_to(triton_root)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "torch" or mod.startswith("torch."):
                    offenders.append(f"{py_file.relative_to(triton_root)}:{node.lineno}: from {mod}")
    assert offenders == [], f"Forbidden torch imports in triton/: {offenders}"
