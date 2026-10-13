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

THE ONE DRAW (owner, 2026-10-13 02:05): a model's initial noise is ONE draw, data-driven (the run's
seed), identical between the compiled and the Triton branch and to the vendor at equal seed. Both
engines take it from `init_draw(shape)` below — the vendor's CPU draw reproduced torch-free
(`kernels/mt_randn.py`: MT19937 + ATen's normal_fill, within 4 ulp of `torch.randn(generator=cpu)`)
— as the first draw of the run's seeded stream, then the next for every further `init: randn`
variable, in consumption order. The per-draw Philox seeds above still serve the Triton random
kernels of the hot loop (scheduler noise); the compiled branch's scheduler draws keep its generator.
Cross-engine bit-equality of the INITIAL draw is the contract from here on.
"""

_RUN_SEED = None
_COUNTER = 0
_MT = None            # the run's MT19937, armed with the seed; the init draws come from it in order


def set_run_seed(seed):
    """Arm (or disarm with None) the run stream; resets the draw counter and the init-draw generator."""
    global _RUN_SEED, _COUNTER, _MT
    _RUN_SEED = int(seed) if seed is not None else None
    _COUNTER = 0
    _MT = None


def init_draw(shape):
    """The run's next initial-noise draw as a float32 numpy array of `shape` — the vendor's CPU draw for
    the run's seed (`kernels/mt_randn.MT19937`), the same bytes in both engines. Refused when no seed is
    armed: a seedless run draws unseeded in each engine (the historical behaviour), and that is said."""
    global _MT
    if _RUN_SEED is None:
        raise RuntimeError("rng_stream.init_draw() called while inactive: the run has no seed")
    if _MT is None:
        from neurobrix.kernels.mt_randn import MT19937
        _MT = MT19937(_RUN_SEED)
    shape = tuple(int(x) for x in shape)
    n = 1
    for x in shape:
        n *= x
    return _MT.randn(n).reshape(shape)


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
