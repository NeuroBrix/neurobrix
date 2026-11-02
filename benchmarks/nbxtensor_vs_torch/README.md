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
Output: one TSV per device and level where the run says (`--out`, the run's campaign directory), every row
also appended as it is written to the dated, cumulative registry (`--registry`: `nbxtensor_vs_torch.kernel.tsv`,
`nbxtensor_vs_torch.model.tsv`, each row with its date, device and engine commit — `registry.py`), and a markdown
summary naming every cell where nbxtensor loses. No figure is written into a document before the bench has run.

This harness lives OUTSIDE the nbxtensor library on purpose: it imports torch to measure against it, and nbxtensor
never imports torch (R-NBXT 6). The two sides run in two environments (the supervisor, 2026-11-02 11:55): torch in an
oracle environment (`venv-vendor-oracle-*`: torch, no triton), nbxtensor in the library's torch-free one
(`venv-notorch-*`) — the nbxtensor rows are also the proof that the library ran without torch.

## The harness

| file | what it does |
|---|---|
| `instances.py` | card-less: the census table's ops of every model, bound at the model's gate request the way the census binds them, each with its runtime dtypes (the census's own derivation); one instance per (op, shapes, dtypes, literal arguments, attributes), the models and op uids that run it listed |
| `kernel_bench.py` | `--side torch` (the oracle environment) then `--side nbxtensor` (the library's): each instance through the engine's own per-op entries: `NativeATenDispatcher` (torch eager), `torch.compile` of that call (each `--compile-modes`), `TritonSequentialDispatcher` (nbxtensor); first call apart, `--warmup`, then `--reps` synchronized calls; median, p10, p90; every output against the float64 reference the torch side writes (`--refs`) from the same rounded inputs, held to the narrowest dtype's bound (`TOLERANCE`); peak memory per contender |
| `model_bench.py` | the gate cells through `python -m neurobrix run`: the ATen branch (torch eager, `--python-torch`) and the Triton branch (`--python-nbxtensor`), time, `[Timings]`, `[Peak]`, output sha and agreement; `torch.compile` has no engine mode and is said so |
| `registry.py` | the dated registry every row is appended to as it is written |
| `summary.py` | the markdown table of every loss: kernel instances where nbxtensor is wrong, refused or slower than the best right torch contender (with the ratio), model cells where the Triton run fails, disagrees or is slower |

What a row does NOT say: an instance runs the op WHOLE at the request's shapes, where a run may cut it (Prism's op-level
tiling, a streamed piece, a temporal chunk) — the census keys of a cut op are the cut's; a key missing from the
certified directory is swept at the first call, which `first_ms` carries. Torch reads one floating dtype per call:
an op the Triton engine runs at mixed widths (an fp32 island beside half weights) is given to torch at the widest.
On Apple, MPS keeps no peak counter: torch's peak is the driver sampled every 2 ms (`sampled driver`), blind to an
op shorter than that; and timings under 1 ms sit inside the M4 Pro's measured noise floor (51 % p95 spread, CLAUDE.local).

## Commands

Each device runs alone (no other process), after the day's landings. `<T>` is the oracle environment's python (torch),
`<L>` the library's (no torch); `<C>` the run's campaign directory, `<REG>` the registry directory
(`<atelier>/registres/benchmarks`).

```
# Apple M4 Pro (the Mac): census apple/apple_m4_pro/18g, profile default-9f169c79
H=default-9f169c79
<L> benchmarks/nbxtensor_vs_torch/instances.py --census apple/apple_m4_pro/18g --hardware $H --out <C>/instances.jsonl
<T> benchmarks/nbxtensor_vs_torch/kernel_bench.py --side torch     --hardware $H --instances <C>/instances.jsonl \
      --out <C>/kernel.tsv --refs <C>/refs --registry <REG>
<L> benchmarks/nbxtensor_vs_torch/kernel_bench.py --side nbxtensor --hardware $H --instances <C>/instances.jsonl \
      --out <C>/kernel.tsv --refs <C>/refs --registry <REG>
<L> benchmarks/nbxtensor_vs_torch/model_bench.py --models @<models file> --out <C>/model.tsv --outputs <C>/outputs \
      --python-torch <T> --python-nbxtensor <L> --device $H --registry <REG>
<L> benchmarks/nbxtensor_vs_torch/summary.py --device apple_m4_pro --kernel <C>/kernel.tsv --model <C>/model.tsv --out <C>/summary.md

# V100 16 GB / 32 GB (the Dell): the same lines with --census nvidia/volta/16g (resp. 32g), --hardware v100-16g
# (resp. v100-32g), CUDA_VISIBLE_DEVICES=<one card>, and the Dell's oracle and library environments.
```

Both scripts append as they go and skip what is already written: a stopped run is resumed by the same command.

