"""Where a real run's GPU time goes, kernel by kernel: `kernel_times()` opens a profiling session around the work the
caller runs and returns the table when it closes.

    with nbxtensor.profile.kernel_times() as table:
        ...                     # the work to rank
    print(table.format())

Inside a session every launch (and every copy, as one row 'blit') runs alone and is timed on the GPU at its own
boundaries, which changes the batching: the table RANKS kernels; it is not the production step's time, and its header
says so with the session's wall time beside the summed GPU time. Metal today (stage-boundary timestamps); another
backend is refused by name. One session per process.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple


class KernelTimes:
    """The table a session fills: per kernel name, launches, total / max GPU ms; the session's wall time and summed
    GPU time. Backend-neutral (Metal fills it today; the CUDA twin fills the same columns). Each launch and each
    copy ran alone: the figures rank kernels, they are not the production step's time."""

    def __init__(self) -> None:
        self.rows: Dict[str, List[float]] = {}          # name -> [launches, total ms, max ms]
        self.wall_s = 0.0
        self.unsampled = 0                              # launches whose samples Metal did not take

    def add(self, name: str, ms: float) -> None:
        row = self.rows.setdefault(name, [0, 0.0, 0.0])
        row[0] += 1
        row[1] += ms
        row[2] = max(row[2], ms)

    @property
    def gpu_ms(self) -> float:
        return sum(r[1] for r in self.rows.values())

    def ranked(self) -> List[Tuple[str, int, float, float]]:
        """(name, launches, total ms, max ms), the heaviest total first."""
        return sorted(((n, int(r[0]), r[1], r[2]) for n, r in self.rows.items()), key=lambda r: -r[2])

    def as_dict(self) -> dict:
        return {"wall_s": self.wall_s, "gpu_ms": self.gpu_ms, "unsampled": self.unsampled,
                "measures": "every launch and copy alone: the figures rank kernels, not the production step",
                "kernels": [{"name": n, "launches": c, "total_ms": t, "max_ms": m} for n, c, t, m in self.ranked()]}

    def report(self, path=None, top: int = 25) -> str:
        """The printed table; with `path`, the whole table also written there as JSON (said on the last line)."""
        text = self.format(top)
        if path:
            import json
            from pathlib import Path
            Path(path).write_text(json.dumps(self.as_dict(), indent=1))
            text += f"\n[kernel-times] the whole table: {path}"
        return text

    def format(self, top: int = 25) -> str:
        rows = self.ranked()
        gpu = self.gpu_ms or 1.0
        out = [f"[kernel-times] wall {self.wall_s:.3f} s, GPU {self.gpu_ms:.1f} ms summed over "
               f"{sum(r[1] for r in rows)} launches (each launch and copy timed alone: a ranking, not the step's time)",
               f"[kernel-times]   {'kernel':<48} {'launches':>8} {'total ms':>10} {'mean ms':>8} {'max ms':>8} {'share':>6}"]
        for n, c, t, m in rows[:top]:
            out.append(f"[kernel-times]   {n[:48]:<48} {c:>8} {t:>10.2f} {t / c:>8.4f} {m:>8.3f} {100 * t / gpu:>5.1f}%")
        rest = rows[top:]
        if rest:
            t = sum(r[2] for r in rest)
            out.append(f"[kernel-times]   {f'the rest ({len(rest)} kernels)':<48} {sum(r[1] for r in rest):>8} "
                       f"{t:>10.2f} {'':>8} {'':>8} {100 * t / gpu:>5.1f}%")
        return "\n".join(out)


def ranked_call(fn, path=None, out=print):
    """`fn()` inside a session; its table handed to `out` (and written to `path` as JSON); `fn`'s result returned."""
    with kernel_times() as table:
        result = fn()
    out(table.report(path))
    return result


@contextmanager
def kernel_times() -> Iterator[KernelTimes]:
    """A profiling session around the caller's work; the table is filled when the block ends (also on an error)."""
    from nbxtensor.backend.runtime import gpu_backend
    if gpu_backend() != "metal":
        raise RuntimeError(f"ZERO FALLBACK: per-kernel GPU time is measured on Metal only for now; this backend is "
                           f"{gpu_backend()!r}")
    from nbxtensor.backend.metal_device import runtime
    rt = runtime()
    table = rt.open_kernel_times()
    try:
        yield table
    finally:
        rt.close_kernel_times()
