"""Run-scoped seeded RNG stream for the Triton random wrappers.

The random wrappers (`rand/randn/normal/*_like` in `kernels/wrappers.py`)
historically drew each call's Philox seed from Python's `random.randint`,
which `--seed` never seeds: Triton-mode stochastic sampling (EulerAncestral
noise, DDIM eta>0 variance) was NON-reproducible by construction, and no
run-scoped stream contract existed at all.

This module is the Triton-engine mirror of the compiled engine's run-scoped
`torch.Generator` (see `VariableResolver.sampling_generator`): ONE seeded
stream drives every stochastic draw of a run, in consumption order. The flow
handler arms it at execution start from `defaults.seed` (the same data-driven
source both engines read); each wrapper call then derives an independent
per-draw kernel seed as splitmix64(run_seed, draw_counter). Draws are
reproducible per (seed, draw order) and statistically independent across the
counter. Cross-ENGINE bit-equality is NOT the contract here (the two engines
use different RNG algorithms); when a bit-identical cross-engine noise diff
is needed, the `rng_pin` module (NBX_FORCE_RAND_SEED) remains the tool and
takes precedence in the wrappers.

Zero torch, zero device state: pure Python integers.
"""

_RUN_SEED = None
_COUNTER = 0


def set_run_seed(seed):
    """Arm (or disarm with None) the run stream; resets the draw counter."""
    global _RUN_SEED, _COUNTER
    _RUN_SEED = int(seed) if seed is not None else None
    _COUNTER = 0


def active() -> bool:
    return _RUN_SEED is not None


def next_seed() -> int:
    """Next per-draw kernel seed: splitmix64 of (run_seed, counter), 31-bit."""
    global _COUNTER
    if _RUN_SEED is None:
        raise RuntimeError("rng_stream.next_seed() called while inactive")
    x = (_RUN_SEED * 0x9E3779B97F4A7C15 + _COUNTER * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    _COUNTER += 1
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return int(x & 0x7FFFFFFF)
