#!/usr/bin/env python3
"""What a cross-GPU collective actually costs on this rig — measured, not costed.

The tensor-parallelism study (`docs/internal/tensor_parallel_study_2026_09_03.md`)
turns entirely on one number it did not have. Two all-reduces per layer x 48
layers = **96 collectives per decoded token**, against a measured 25.1 ms token.
At 10 us each that is 3.8 % overhead and TP4 is worth building; at 30 us it is
11.5 % and marginal; above that the chantier is a documented wall. The study
said so itself and deferred the measurement to here, before any rewriter is
written.

Four things are measured, because the estimate had blind spots:

1. **Peer access OFF vs ON.** A cross-device `cudaMemcpy(kind=D2D)` without
   peer access is STAGED THROUGH HOST MEMORY by the driver — it never touches
   NVLink, whatever `nvidia-smi topo -m` reports. Nothing in this engine has
   ever called `cudaDeviceEnablePeerAccess`, so the OFF row is also the true
   present-day cost of the `pipeline_parallel` stage-boundary transfer.

2. **Ring vs one-shot.** A ring all-reduce is a BANDWIDTH algorithm: 2(N-1)
   dependent steps, each at least one launch. At a 4 KB payload the wire time
   is ~2 us and the launches are ~7 us each, so the ring pays six latencies to
   save bandwidth that was never the constraint. The one-shot shape — every
   device pushes its buffer to every other in parallel, then each sums N
   buffers locally in fixed device order — costs ONE latency step. It is the
   only candidate for the 10-15 us scenario. Both are measured.

3. **Device time AND host-issue time, as separate columns.** The study budgets
   only the device-side latency. But a one-shot collective over 4 devices is
   ~30 ctypes calls, and 96 of them per token could cost more host time than
   the GPU work TP is supposed to save. If the host is the wall, NVLink speed
   is irrelevant — and that would decide the item.

4. **Bit-identity.** Fixed-order reduction is a product clause, not a tuning
   choice (study section 5). Every device's result must be byte-identical to
   every other's and to a fixed-order host reference, or the number above is
   measuring the wrong thing.

R33-pure: NBXTensor + DeviceAllocator + one `@triton.jit`. No torch, no NCCL.

    python3 tools/collective_latency_probe.py --devices 0,1,3 --iters 2000
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor, NBXDtype  # noqa: E402

_D2D = 3


# --------------------------------------------------------------------------
# the local reduction — fixed order, by construction
# --------------------------------------------------------------------------

@triton.jit
def _sum_fixed_order(out_ptr, p0, p1, p2, p3, n,
                     N_SRC: tl.constexpr, BLOCK: tl.constexpr):
    """Sum up to four buffers in a FIXED order, accumulating in fp32.

    The order is the argument order, which the caller fixes to ascending
    device index at plan time. That is what makes the collective
    bit-reproducible: an atomic accumulation would sum in arrival order and
    two runs of the same prompt could differ in the last bits — and then in
    the token, at any near-tie.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(p0 + offs, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(p1 + offs, mask=mask, other=0.0).to(tl.float32)
    if N_SRC > 2:
        acc += tl.load(p2 + offs, mask=mask, other=0.0).to(tl.float32)
    if N_SRC > 3:
        acc += tl.load(p3 + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(tl.float16), mask=mask)


@triton.jit
def _add_into(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    """dst += src. The ring's per-step reduction."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(dst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + offs, (a + b).to(tl.float16), mask=mask)


_BLOCK = 1024


def _grid(n):
    return ((n + _BLOCK - 1) // _BLOCK,)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

class Rig:
    """Buffers and events for one (devices, payload) configuration."""

    def __init__(self, devices, n_elems):
        self.devices = devices
        self.n = n_elems
        self.N = len(devices)
        # Each device contributes a distinct pattern so a mis-indexed ring
        # cannot pass the bit gate by symmetry.
        self.host = [
            (np.arange(n_elems, dtype=np.float32) * 0.001 + (d + 1)
             ).astype(np.float16)
            for d in range(self.N)
        ]
        self.buf, self.inbox, self.out, self.recv, self.events = [], [], [], [], []
        for i, dev in enumerate(devices):
            DeviceAllocator.set_device(dev)
            DeviceAllocator.ensure_triton_device(dev)
            self.buf.append(self._upload(self.host[i], dev))
            # one inbox slot per participating device, indexed by RANK so the
            # local sum reads them in a fixed order regardless of arrival
            self.inbox.append([self._upload(self.host[i], dev) if j == i
                               else NBXTensor.zeros((n_elems,), dtype=NBXDtype.float16,
                                                    device=f"cuda:{dev}")
                               for j in range(self.N)])
            self.out.append(NBXTensor.zeros((n_elems,), dtype=NBXDtype.float16,
                                            device=f"cuda:{dev}"))
            self.recv.append(NBXTensor.zeros((n_elems,), dtype=NBXDtype.float16,
                                             device=f"cuda:{dev}"))
            # ordering only — never measured, so timing is disabled
            self.events.append(DeviceAllocator.create_event(timing=False))
        self.timer_start = None
        self.timer_end = None
        DeviceAllocator.set_device(devices[0])
        self.timer_start = DeviceAllocator.create_event()
        self.timer_end = DeviceAllocator.create_event()

    @staticmethod
    def _upload(arr, dev):
        t = NBXTensor.zeros(arr.shape, dtype=NBXDtype.float16, device=f"cuda:{dev}")
        DeviceAllocator.set_device(dev)
        DeviceAllocator.memcpy(t.data_ptr(), arr.ctypes.data, arr.nbytes, kind=1)
        return t

    def reset(self):
        """Restore every contribution — a reduction is destructive."""
        for i, dev in enumerate(self.devices):
            DeviceAllocator.set_device(dev)
            h = self.host[i]
            DeviceAllocator.memcpy(self.buf[i].data_ptr(), h.ctypes.data,
                                   h.nbytes, kind=1)
            DeviceAllocator.memcpy(self.inbox[i][i].data_ptr(), h.ctypes.data,
                                   h.nbytes, kind=1)
        self.sync_all()

    def sync_all(self):
        for dev in self.devices:
            DeviceAllocator.set_device(dev)
            DeviceAllocator.sync_device()

    def reference(self):
        """Fixed-order host sum, in the SAME order and the same accumulation
        width as the kernel — so a mismatch is a real disagreement, not a
        difference of arithmetic convention."""
        acc = self.host[0].astype(np.float32).copy()
        for i in range(1, self.N):
            acc += self.host[i].astype(np.float32)
        return acc.astype(np.float16)

    def download(self, tensors):
        outs = []
        for i, dev in enumerate(self.devices):
            DeviceAllocator.set_device(dev)
            DeviceAllocator.sync_device()
            buf = np.empty(self.n, dtype=np.float16)
            DeviceAllocator.memcpy(buf.ctypes.data, tensors[i].data_ptr(),
                                   buf.nbytes, kind=2)
            outs.append(buf)
        return outs


# --- the two collective shapes ---------------------------------------------

def one_shot(rig):
    """Every device pushes to every other in parallel, then reduces locally.

    ONE latency step. N-1 outbound copies per device, all independent, so the
    wire transfers overlap; the only serialisation is the local sum waiting on
    the last arrival.
    """
    devs, N = rig.devices, rig.N
    for i, dev in enumerate(devs):
        DeviceAllocator.set_device(dev)
        for j in range(N):
            if j == i:
                continue
            DeviceAllocator.memcpy_async(
                rig.inbox[j][i].data_ptr(), rig.buf[i].data_ptr(),
                rig.buf[i].nbytes(), kind=_D2D, stream=0)
        DeviceAllocator.record_event(rig.events[i], 0)
    for i, dev in enumerate(devs):
        DeviceAllocator.set_device(dev)
        for j in range(N):
            if j != i:
                DeviceAllocator.stream_wait_event(0, rig.events[j])
        DeviceAllocator.ensure_triton_device(dev)
        box = rig.inbox[i]
        _sum_fixed_order[_grid(rig.n)](
            rig.out[i], box[0], box[1],
            box[2] if N > 2 else box[0], box[3] if N > 3 else box[0],
            rig.n, N_SRC=N, BLOCK=_BLOCK)


def copies_only(rig):
    """The one-shot's COMMUNICATION alone — every cross-device copy and the
    events, with no local reduction.

    This is the decomposition that makes the total readable. A collective is
    a transfer plus a reduction, and they fail for different reasons: the
    transfer is limited by the interconnect and by peer access, the reduction
    by how fast the host can issue a kernel. Reporting only the sum would hide
    which one is the wall.
    """
    devs, N = rig.devices, rig.N
    for i, dev in enumerate(devs):
        DeviceAllocator.set_device(dev)
        for j in range(N):
            if j != i:
                DeviceAllocator.memcpy_async(
                    rig.inbox[j][i].data_ptr(), rig.buf[i].data_ptr(),
                    rig.buf[i].nbytes(), kind=_D2D, stream=0)
        DeviceAllocator.record_event(rig.events[i], 0)
    for i, dev in enumerate(devs):
        DeviceAllocator.set_device(dev)
        for j in range(N):
            if j != i:
                DeviceAllocator.stream_wait_event(0, rig.events[j])


def reduce_only(rig):
    """The local reduction alone — N Triton launches, no transfer.

    Isolates the Python-side dispatch cost of a `@triton.jit` call, which the
    study's "kernel launch 7.0 us" figure does NOT cover: 7 us is the
    device-side launch, while the host spends far more deciding what to
    launch. With 96 collectives per token that difference is the whole
    question.
    """
    for i, dev in enumerate(rig.devices):
        DeviceAllocator.set_device(dev)
        DeviceAllocator.ensure_triton_device(dev)
        box, N = rig.inbox[i], rig.N
        _sum_fixed_order[_grid(rig.n)](
            rig.out[i], box[0], box[1],
            box[2] if N > 2 else box[0], box[3] if N > 3 else box[0],
            rig.n, N_SRC=N, BLOCK=_BLOCK)


def direct_peer_reduce(rig):
    """ONE kernel per device, reading the peers' memory straight over NVLink.

    The shape every distributed library uses is copy-then-reduce, because a
    library has to work over a network where a remote read is not addressable.
    On one machine with peer access enabled it IS addressable: device i's
    kernel can dereference a pointer into device j's memory, so the transfer
    and the reduction become the SAME kernel and the copies disappear.

    Measured cost of the copies they replace: ~25 us of host time each,
    `cudaMemcpyAsync` on a peer pointer, six per collective at N=3. That is
    the whole of the one-shot's 200 us. This variant issues one launch per
    device instead, and is the reason writing the primitive in-house rather
    than adopting NCCL's shape is worth doing.

    Determinism is unchanged: the kernel sums its source pointers in a fixed
    argument order, which is ascending device index, decided at plan time.
    """
    devs, N = rig.devices, rig.N
    for i, dev in enumerate(devs):
        DeviceAllocator.set_device(dev)
        DeviceAllocator.record_event(rig.events[i], 0)
    for i, dev in enumerate(devs):
        DeviceAllocator.set_device(dev)
        for j in range(N):
            if j != i:
                DeviceAllocator.stream_wait_event(0, rig.events[j])
        DeviceAllocator.ensure_triton_device(dev)
        src = rig.buf
        _sum_fixed_order[_grid(rig.n)](
            rig.out[i], src[0], src[1],
            src[2] if N > 2 else src[0], src[3] if N > 3 else src[0],
            rig.n, N_SRC=N, BLOCK=_BLOCK)


def ring(rig):
    """Fixed-order ring all-reduce: reduce-scatter then all-gather.

    2(N-1) DEPENDENT steps. Optimal in bytes moved (2(N-1)/N x payload per
    device) and therefore the right algorithm when bandwidth is the
    constraint. At a 4 KB payload it is six latencies to save two microseconds
    of wire time, which is why it is measured against the one-shot rather than
    assumed to be the answer.
    """
    devs, N = rig.devices, rig.N
    chunk = (rig.n + N - 1) // N

    def piece(t, k):
        start = k * chunk
        return t.narrow(0, start, min(chunk, rig.n - start))

    for step in range(N - 1):                       # reduce-scatter
        for i, dev in enumerate(devs):
            src_k = (i - step) % N
            DeviceAllocator.set_device(dev)
            src = piece(rig.buf[i], src_k)
            dst = piece(rig.recv[(i + 1) % N], src_k)
            DeviceAllocator.memcpy_async(dst.data_ptr(), src.data_ptr(),
                                         src.nbytes(), kind=_D2D, stream=0)
            DeviceAllocator.record_event(rig.events[i], 0)
        for i, dev in enumerate(devs):
            k = (i - 1 - step) % N
            DeviceAllocator.set_device(dev)
            DeviceAllocator.stream_wait_event(0, rig.events[(i - 1) % N])
            DeviceAllocator.ensure_triton_device(dev)
            dst, src = piece(rig.buf[i], k), piece(rig.recv[i], k)
            _add_into[_grid(dst.numel())](dst, src, dst.numel(), BLOCK=_BLOCK)

    for step in range(N - 1):                       # all-gather
        for i, dev in enumerate(devs):
            k = (i + 1 - step) % N
            DeviceAllocator.set_device(dev)
            src = piece(rig.buf[i], k)
            dst = piece(rig.buf[(i + 1) % N], k)
            DeviceAllocator.memcpy_async(dst.data_ptr(), src.data_ptr(),
                                         src.nbytes(), kind=_D2D, stream=0)
            DeviceAllocator.record_event(rig.events[i], 0)
        for i, dev in enumerate(devs):
            DeviceAllocator.set_device(dev)
            DeviceAllocator.stream_wait_event(0, rig.events[(i - 1) % N])


# --- timing -----------------------------------------------------------------

def measure(rig, fn, iters, warmup=50):
    """Device latency and host-issue cost for one collective shape.

    Device time comes from two events recorded on device 0's stream only —
    an interval between events on different devices has no defined value.
    Host time is wall-clock across the enqueue loop, read BEFORE the final
    synchronisation, so it is the cost of issuing the work rather than of
    waiting for it. The true per-collective cost is the larger of the two:
    whichever side runs out of capacity first is the one that sets the pace.
    """
    for _ in range(warmup):
        fn(rig)
    rig.sync_all()

    DeviceAllocator.set_device(rig.devices[0])
    DeviceAllocator.record_event(rig.timer_start, 0)
    host_t0 = time.perf_counter()
    for _ in range(iters):
        fn(rig)
    host_issue = time.perf_counter() - host_t0
    DeviceAllocator.set_device(rig.devices[0])
    DeviceAllocator.record_event(rig.timer_end, 0)
    DeviceAllocator.event_synchronize(rig.timer_end)
    rig.sync_all()
    host_total = time.perf_counter() - host_t0
    device_ms = DeviceAllocator.event_elapsed_ms(rig.timer_start, rig.timer_end)
    return {
        "device_us": device_ms * 1000.0 / iters,
        "host_issue_us": host_issue * 1e6 / iters,
        "host_total_us": host_total * 1e6 / iters,
    }


def correctness(rig, fn, result_tensors):
    rig.reset()
    fn(rig)
    rig.sync_all()
    outs = rig.download(result_tensors)
    ref = rig.reference()
    per_device = [bytes(o.tobytes()) == bytes(ref.tobytes()) for o in outs]
    all_agree = all(bytes(o.tobytes()) == bytes(outs[0].tobytes()) for o in outs)
    max_diff = float(max(np.max(np.abs(o.astype(np.float32)
                                       - ref.astype(np.float32))) for o in outs))
    return {"matches_fixed_order_reference": all(per_device),
            "devices_byte_identical": all_agree,
            "max_abs_diff": max_diff}


def attribution(devices, n_elems, iters=4000):
    """Where the microseconds actually go.

    The collective figure alone cannot say whether the cost is the wire, the
    API, or the fact that ONE process is driving several devices and must
    call `cudaSetDevice` between every pair of operations. Those have
    completely different consequences: a wire cost is hardware, an API cost is
    a batching problem, and a device-switch cost is an ARCHITECTURE problem —
    it disappears only if each device is driven by its own process or thread,
    which is how every distributed stack is built and is not how this engine
    executes a Prism plan.

    Three isolated costs, each measured the same way as the collectives:

      switch_only   N set_device calls, nothing else
      copy_no_switch one peer copy, device fixed outside the loop
      launch_only   one Triton launch, device fixed outside the loop
    """
    out = {}
    DeviceAllocator.set_device(devices[0])
    a = NBXTensor.zeros((n_elems,), dtype=NBXDtype.float16,
                        device=f"cuda:{devices[0]}")
    DeviceAllocator.set_device(devices[1])
    b = NBXTensor.zeros((n_elems,), dtype=NBXDtype.float16,
                        device=f"cuda:{devices[1]}")
    ev = None
    DeviceAllocator.set_device(devices[0])
    ev0, ev1 = DeviceAllocator.create_event(), DeviceAllocator.create_event()

    def timed(fn, setup_device):
        DeviceAllocator.set_device(setup_device)
        for _ in range(200):
            fn()
        DeviceAllocator.set_device(setup_device)
        DeviceAllocator.sync_device()
        DeviceAllocator.set_device(devices[0])
        DeviceAllocator.record_event(ev0, 0)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        host = time.perf_counter() - t0
        DeviceAllocator.set_device(devices[0])
        DeviceAllocator.record_event(ev1, 0)
        DeviceAllocator.event_synchronize(ev1)
        for d in devices:
            DeviceAllocator.set_device(d)
            DeviceAllocator.sync_device()
        return {"device_us": DeviceAllocator.event_elapsed_ms(ev0, ev1) * 1000 / iters,
                "host_issue_us": host * 1e6 / iters}

    cycle = list(devices)

    def switch():
        for d in cycle:
            DeviceAllocator.set_device(d)

    out["switch_only"] = timed(switch, devices[0])
    out["switch_only"]["note"] = f"{len(cycle)} set_device calls per iteration"

    DeviceAllocator.set_device(devices[0])

    def copy():
        DeviceAllocator.memcpy_async(b.data_ptr(), a.data_ptr(), a.nbytes(),
                                     kind=_D2D, stream=0)

    out["copy_no_switch"] = timed(copy, devices[0])
    out["copy_no_switch"]["note"] = (
        f"one {a.nbytes()} B peer copy {devices[0]}->{devices[1]}, "
        f"device fixed outside the loop")

    DeviceAllocator.set_device(devices[0])
    DeviceAllocator.ensure_triton_device(devices[0])

    def launch():
        _sum_fixed_order[_grid(n_elems)](a, a, a, a, a, n_elems,
                                         N_SRC=2, BLOCK=_BLOCK)

    out["launch_only"] = timed(launch, devices[0])
    out["launch_only"]["note"] = "one @triton.jit launch, device fixed"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", default="0,1,3",
                    help="GPUs to use. Default avoids GPU 2, which holds a long render.")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default="validation_outputs/tp_collective_latency_2026_09_03")
    args = ap.parse_args()

    devices = [int(d) for d in args.devices.split(",")]
    N = len(devices)
    if not 2 <= N <= 4:
        print("need 2 to 4 devices", file=sys.stderr)
        return 2

    payloads = [("decode 4 KB", 2048), ("prefill 34 MB", 34 * 1024 * 1024 // 2)]
    report = {"devices": devices, "iters": args.iters, "rows": [],
              "peer_matrix": {}, "correctness": {}}

    print("=" * 78)
    print(f"COLLECTIVE LATENCY — devices {devices}, {args.iters} iterations")
    print("=" * 78)

    for a in devices:
        for b in devices:
            if a != b:
                report["peer_matrix"][f"{a}->{b}"] = DeviceAllocator.can_access_peer(a, b)
    print("peer-capable pairs:",
          sum(report["peer_matrix"].values()), "/", len(report["peer_matrix"]))

    for peer_on in (False, True):
        if peer_on:
            for a in devices:
                for b in devices:
                    if a != b:
                        DeviceAllocator.enable_peer_access(a, b)
            print("\n--- peer access ENABLED (both directions, every pair) ---")
        else:
            print("\n--- peer access OFF (the engine's present-day state) ---")

        for label, n in payloads:
            rig = Rig(devices, n)
            for name, fn in (("copies_only", copies_only),
                             ("reduce_only", reduce_only),
                             ("one_shot", one_shot),
                             ("direct_peer_reduce", direct_peer_reduce),
                             ("ring", ring)):
                row = {"peer": peer_on, "payload": label, "bytes": n * 2,
                       "algo": name}
                if name == "direct_peer_reduce" and not peer_on:
                    # Not a limitation of the probe: Triton's own launcher
                    # refuses the argument with "Pointer argument cannot be
                    # accessed from Triton". Without peer access a peer
                    # pointer is not addressable from a kernel at all, so
                    # this shape only exists once peer access is on.
                    row["skipped"] = "peer pointer is not addressable from a kernel"
                    report["rows"].append(row)
                    print(f"  {label:<14} {name:<18} "
                          f"SKIPPED — {row['skipped']}")
                    continue
                if name in ("one_shot", "ring", "direct_peer_reduce"):
                    row.update(correctness(
                        rig, fn, rig.buf if name == "ring" else rig.out))
                rig.reset()
                row.update(measure(rig, fn, args.iters))
                report["rows"].append(row)
                exact = row.get("matches_fixed_order_reference", "-")
                agree = row.get("devices_byte_identical", "-")
                print(f"  {label:<14} {name:<18} "
                      f"device {row['device_us']:9.2f} us   "
                      f"host-issue {row['host_issue_us']:9.2f} us   "
                      f"ref-exact {str(exact):<5} agree {agree}")
            del rig

    print("\n--- attribution: where the microseconds go (peer ON, 4 KB) ---")
    report["attribution"] = attribution(devices, 2048)
    for k, v in report["attribution"].items():
        print(f"  {k:<16} device {v['device_us']:8.2f} us   "
              f"host-issue {v['host_issue_us']:8.2f} us   ({v['note']})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "collective_latency.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out/'collective_latency.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
