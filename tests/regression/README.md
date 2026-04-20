# Regression harness

Runs every imported model from `~/.neurobrix/cache/` in each supported
runtime mode (`native`, `triton`) and checks for

1. **successful completion** (exit code 0, no timeout), and
2. **output stability** for LLMs — the generated text is compared to a
   golden file committed under `tests/regression/golden/`.

For v1 we only compare generated text for LLMs. Image / audio / video
regression is just "the run completed"; fidelity comparison for binary
outputs will come in a later iteration.

## Quick use

```bash
# Fast cells only (LLM + audio). ~10 min on a single V100:
pytest tests/regression/ -v

# Include slow families (image, video):
pytest tests/regression/ -v --runslow

# Just one model, one mode:
pytest tests/regression/ -v -k "TinyLlama and native"

# Capture / refresh goldens after an intentional change:
UPDATE_GOLDEN=1 pytest tests/regression/ -v -k "TinyLlama"
```

A missing golden file is treated as `skip` with a message, not as a
silent pass — this is deliberate so CI never records a broken output
as the new truth.

## Layout

- `conftest.py` — discovers models from `~/.neurobrix/cache/`, reads
  `manifest.json` + `topology.json` to get family / flow, sets per-model
  timeouts, and declares the `KNOWN_FAILURES` list.
- `test_all_models.py` — the parametrized pytest module. One test per
  `(model, mode)` cell.
- `golden/` — one `.txt` per `(model, mode)` cell holding the expected
  LLM greedy output. Committed so drift is visible in code review.

## Matrix

With 21 imported models × 2 modes, a full run is a 42-cell matrix. The
fast subset (excluding image + video) is ~14 models × 2 modes = 28
cells. The first pass populates the golden store and flags the cells
that crash.

## Updating `KNOWN_FAILURES`

When a cell fails for a known reason and you want pytest to treat it
as `xfail` (so regressions in other cells aren't drowned out), add a
tuple to `KNOWN_FAILURES` in `conftest.py`:

```python
KNOWN_FAILURES = [
    ("orpheus-3b-0.1-ft", None, "Repeats token indefinitely in both modes"),
    ("PixArt-Sigma-XL-2-1024-MS", "triton", "SDPA VRAM alloc fails on V100 16 GB"),
]
```

The third tuple element is free-form text. `None` in position 2 means
"both modes"; use `"native"` or `"triton"` to narrow.
