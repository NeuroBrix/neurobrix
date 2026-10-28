# nbxtensor benchmarks

Two levels, three contenders, per device.

- **Kernel level:** every op at the shapes and dtypes the certification census
  recorded for the catalogue's models (the real shapes, not synthetic ones).
- **Model level:** the gate cells (entry point, the machine's profile).
- **Contenders:** torch eager, `torch.compile` (default mode, then
  `max-autotune`), nbxtensor.
- **Devices:** V100 16 GB and 32 GB (NVLink), Apple M4 Pro; others as they come.

Rules: the machine runs nothing else during a measurement; warm-up excluded
and reported apart; median of N with N and spread printed; outputs compared
(a faster wrong result is a red, not a win); peak memory recorded with time.
Output: one TSV per device and level under `nbxtensor/bench/results/`, and a
markdown summary naming every cell where nbxtensor loses.

This harness lives OUTSIDE the nbxtensor library on purpose: it imports
torch to measure against it, and nbxtensor never imports torch (R-NBXT 6).

Status: M4 — the harness is written by the Mac agent.
