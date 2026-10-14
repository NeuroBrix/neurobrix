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

EVERY DRAW OF A SEEDED RUN (2026-10-14): the vendor threads ONE generator through every draw of a run — an
image encoder's posterior sample (`retrieve_latents(vae.encode(x), generator)`), the initial noise, each
ancestral step's noise (`EulerAncestralDiscreteScheduler.step(..., generator)`). Measured on Allegro-TI2V at
its gate request: the two engines rendered the same clip at one step (mean |a - b| under 1 of 255 per frame)
and different content at ten, the ancestral noise being the only difference (torch's device generator in the
compiled branch, Philox in the Triton branch); on CogVideoX-5b-I2V the first op that departs between the
kernel oracle and the Triton branch is the posterior sample `vae_encoder/aten.randn_like::0`. So every draw of a
seeded run — graph random ops and scheduler noise, in both engines — is the run stream's next normal
(`next_normal`) or uniform (`next_uniform`) values, in consumption order. The per-draw Philox seeds
(`next_seed`) serve only what is not a seeded run's draw.
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


def _mt():
    global _MT
    if _RUN_SEED is None:
        raise RuntimeError("rng_stream: a draw while inactive: the run has no seed")
    if _MT is None:
        from neurobrix.kernels.mt_randn import MT19937
        _MT = MT19937(_RUN_SEED)
    return _MT


def _count(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n


def next_normal(shape):
    """The run stream's next standard normals as a float32 numpy array of `shape` (torch's CPU `randn` for the
    run's seed, continued): the draw of every seeded random op of either engine."""
    shape = tuple(int(x) for x in shape)
    return _mt().randn(_count(shape)).reshape(shape)


def next_uniform(shape):
    """The run stream's next [0, 1) uniforms as a float32 numpy array of `shape` (torch's CPU `rand`, continued)."""
    shape = tuple(int(x) for x in shape)
    return _mt().uniform_f32(_count(shape)).reshape(shape)


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
