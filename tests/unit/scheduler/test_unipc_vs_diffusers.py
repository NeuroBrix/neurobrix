"""Numerical equivalence: NeuroBrix UniPCMultistepScheduler vs diffusers.

Proves the runtime-pure NeuroBrix port matches diffusers' UniPCMultistepScheduler
to max|diff| < 1e-4, both for the FLOW config (Wan2.1-T2V: use_flow_sigmas=True,
flow_shift=3.0, prediction_type='flow_prediction') and a STANDARD diffusion config
(epsilon, scaled_linear, linspace spacing).

The scheduler under test (`neurobrix.core.module.scheduler.diffusion.unipc_multistep`)
imports NO diffusers — that is the R34 contract. This DEV TEST imports diffusers
(the [adapter] extra) purely as the algorithm oracle.

VERSION NOTE — the task premise "the UniPC algorithm is stable" is true for the
predictor/corrector B(h) updates but FALSE for the flow-sigma SCHEDULE block,
which changed between diffusers 0.36 and 0.38:
  - 0.36/0.33 (the port-of-record; Wan2.1 traces under diffusers 0.33.1):
      alphas = linspace(1, 1/N, N+1); sigmas = 1 - alphas;
      sigmas = flip(flow_shift*sigmas/(1+(flow_shift-1)*sigmas))[:-1];
      timesteps = sigmas * num_train_timesteps
  - 0.38 (the version installed in this venv):
      sigmas = linspace(1, 1/N, N+1)[:-1] (no `1-alphas`, no flip);
      sigmas = flow_shift*sigmas/(1+(flow_shift-1)*sigmas); + eps[0] guard
NeuroBrix implements the 0.36/0.33 formula DELIBERATELY (it is the port source
and matches Wan2.1's production trace). Therefore:
  - STANDARD config: validated end-to-end against the LIVE installed diffusers
    (schedule + step math — version-stable for the standard path).
  - FLOW config: the step()/predictor/corrector/flow-branch MATH is validated
    against live diffusers driven on a SHARED 0.36-spec schedule; NBX's own flow
    schedule is pinned independently to the 0.36 reference values (regression
    guard). Copying the schedule still exercises the flow branches of
    _sigma_to_alpha_sigma_t (alpha=1-s) and convert_model_output (flow_prediction)
    against the live oracle — only the 6-line deterministic schedule transcription
    is pinned to spec rather than to the divergent 0.38 oracle.

Rank-agnosticism is asserted on TWO latent shapes:
  - [1, 16, 3, 8, 8]  — 5D video latent
  - [2, 4, 16, 16]    — 4D image latent

Run:
  source /home/mlops/ml/venv/bin/activate
  cd /home/mlops/NeuroBrix_System
  python -m pytest tests/unit/scheduler/test_unipc_vs_diffusers.py -x -q
or directly (no pytest needed):
  python tests/unit/scheduler/test_unipc_vs_diffusers.py
"""
from __future__ import annotations

import torch

from neurobrix.core.module.scheduler.diffusion.unipc_multistep import (
    UniPCMultistepScheduler as NBXUniPC,
)

try:
    import pytest
except ImportError:  # allow direct `python <file>` runs without pytest
    pytest = None

if pytest is not None:
    diffusers = pytest.importorskip("diffusers")
else:
    import diffusers
from diffusers import UniPCMultistepScheduler as DiffusersUniPC  # noqa: E402


def _parametrize(argnames, argvalues):
    """pytest.mark.parametrize when pytest is present, no-op decorator otherwise."""
    if pytest is not None:
        return pytest.mark.parametrize(argnames, argvalues)
    return lambda f: f


def _raises(exc):
    if pytest is not None:
        return pytest.raises(exc)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            assert et is not None, "expected an exception, none raised"
            return True

    return _Ctx()


N_STEPS = 10
SEED = 1234
TOL = 1e-4

SHAPES = [
    (1, 16, 3, 8, 8),  # 5D video latent (Wan2.1-style)
    (2, 4, 16, 16),    # 4D image latent
]

# ---------------------------------------------------------------------------
# Config builders. The diffusers kwargs are the canonical surface; the NBX
# config dict carries the same semantic values PLUS the keys the NeuroBrix
# SchedulerConfig validator requires (algorithm_type is required by the shared
# validator but never read by UniPC — it is inert here).
# ---------------------------------------------------------------------------

FLOW_DIFFUSERS_KWARGS = dict(
    num_train_timesteps=1000,
    solver_order=2,
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    flow_shift=3.0,
    solver_type="bh2",
    predict_x0=True,
    lower_order_final=True,
    # final_sigmas_type defaults to "zero" in diffusers UniPC — leave default.
)

STANDARD_DIFFUSERS_KWARGS = dict(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    solver_order=2,
    prediction_type="epsilon",
    solver_type="bh2",
    predict_x0=True,
    lower_order_final=True,
    timestep_spacing="linspace",
    # final_sigmas_type defaults to "zero" in diffusers UniPC — leave default.
)


def _nbx_config_from(diffusers_kwargs: dict) -> dict:
    """Translate diffusers kwargs into a NeuroBrix scheduler config dict."""
    cfg = dict(diffusers_kwargs)
    # NBX SchedulerConfig REQUIRED_KEYS that diffusers does not surface as a
    # UniPC kwarg. beta_* / beta_schedule default to the diffusers UniPC values.
    cfg.setdefault("beta_start", 0.0001)
    cfg.setdefault("beta_end", 0.02)
    cfg.setdefault("beta_schedule", "linear")
    cfg.setdefault("timestep_spacing", "linspace")
    # algorithm_type is required by the shared validator but inert for UniPC.
    cfg.setdefault("algorithm_type", "dpmsolver++")
    return cfg


def _build_pair(diffusers_kwargs: dict):
    nbx = NBXUniPC(_nbx_config_from(diffusers_kwargs))
    dif = DiffusersUniPC(**diffusers_kwargs)
    return nbx, dif


# 0.36/0.33-spec flow schedule reference values for N=10, flow_shift=3.0,
# num_train_timesteps=1000 (the production schedule Wan2.1 traces under). These
# pin NBX's flow set_timesteps independently of the divergent 0.38 oracle.
FLOW_REF_SIGMA0 = 0.99967  # first sigma (1 - 1/N flow value, shifted)
FLOW_REF_SIGMA_LAST = 0.0  # final_sigmas_type="zero"


# ---------------------------------------------------------------------------
# Schedule equivalence.
#   - STANDARD: NBX vs LIVE diffusers (version-stable; must be bit-equal).
#   - FLOW: NBX pinned to the 0.36/0.33 reference (the 0.38 formula diverges).
# ---------------------------------------------------------------------------

def test_standard_schedule_matches_live_diffusers():
    nbx, dif = _build_pair(STANDARD_DIFFUSERS_KWARGS)
    nbx.set_timesteps(N_STEPS)
    dif.set_timesteps(N_STEPS)

    assert nbx.num_inference_steps == dif.num_inference_steps == N_STEPS
    ts_diff = (nbx.timesteps.long() - dif.timesteps.long()).abs().max().item()
    assert ts_diff == 0, f"[standard] timesteps differ by {ts_diff}"
    sig_diff = (nbx.sigmas.float() - dif.sigmas.float()).abs().max().item()
    assert sig_diff < 1e-6, f"[standard] sigmas max|diff|={sig_diff}"


def test_flow_schedule_matches_036_reference():
    nbx, _ = _build_pair(FLOW_DIFFUSERS_KWARGS)
    nbx.set_timesteps(N_STEPS)

    assert nbx.num_inference_steps == N_STEPS
    assert len(nbx.sigmas) == N_STEPS + 1, "flow sigmas must carry a trailing entry"
    # 0.36/0.33-spec flow schedule (NOT the 0.38 formula).
    assert abs(float(nbx.sigmas[0]) - FLOW_REF_SIGMA0) < 1e-4, (
        f"[flow] first sigma {float(nbx.sigmas[0])} != 0.36 ref {FLOW_REF_SIGMA0}"
    )
    assert float(nbx.sigmas[-1]) == FLOW_REF_SIGMA_LAST, (
        f"[flow] final sigma must be 0.0 (final_sigmas_type='zero'), "
        f"got {float(nbx.sigmas[-1])}"
    )
    # Sigmas strictly descending (a flow schedule property).
    sig = nbx.sigmas[:-1]
    assert bool((sig[1:] < sig[:-1]).all()), "[flow] sigmas must be descending"


# ---------------------------------------------------------------------------
# Per-step prev_sample equivalence — the numerical proof.
# ---------------------------------------------------------------------------

def _run_step_sequence(kwargs: dict, shape: tuple, share_schedule: bool = False) -> float:
    """Run N identical steps through both schedulers, return overall max|diff|.

    A single seeded RNG pre-generates the per-step model_output and the initial
    sample so both schedulers consume bit-identical inputs. Each scheduler is fed
    a fresh clone of every input (step() mutates internal buffers, not inputs).

    share_schedule=True (FLOW path): the flow-sigma SCHEDULE differs between
    diffusers 0.36 (NBX port source) and the installed 0.38. To validate the
    version-stable predictor/corrector/flow-branch MATH against the live oracle,
    both schedulers are driven off NBX's 0.36-spec schedule. This still exercises
    the flow branches of _sigma_to_alpha_sigma_t and convert_model_output in
    diffusers; only the 6-line deterministic schedule transcription is excluded
    (it is pinned separately in test_flow_schedule_matches_036_reference).
    """
    nbx, dif = _build_pair(kwargs)
    nbx.set_timesteps(N_STEPS)
    dif.set_timesteps(N_STEPS)

    if share_schedule:
        dif.sigmas = nbx.sigmas.clone()
        dif.timesteps = nbx.timesteps.clone()
        dif.num_inference_steps = nbx.num_inference_steps

    gen = torch.Generator().manual_seed(SEED)
    init_sample = torch.randn(shape, generator=gen, dtype=torch.float32)
    model_outputs = [
        torch.randn(shape, generator=gen, dtype=torch.float32) for _ in range(N_STEPS)
    ]

    sample_nbx = init_sample.clone()
    sample_dif = init_sample.clone()

    max_diff = 0.0
    timesteps = nbx.timesteps
    # Sanity: both schedulers expose the same timestep ordering.
    assert torch.equal(nbx.timesteps.long(), dif.timesteps.long())

    for i in range(N_STEPS):
        t = timesteps[i]
        mo = model_outputs[i]

        out_nbx = nbx.step(mo.clone(), t, sample_nbx, return_dict=True)["prev_sample"]
        out_dif = dif.step(mo.clone(), t, sample_dif, return_dict=True).prev_sample

        step_diff = (out_nbx.float() - out_dif.float()).abs().max().item()
        max_diff = max(max_diff, step_diff)

        # Feed the same scheduler's own output forward (independent chains; both
        # must track identically step-by-step).
        sample_nbx = out_nbx
        sample_dif = out_dif

    return max_diff


@_parametrize("shape", SHAPES)
def test_unipc_flow_matches_diffusers(shape):
    # FLOW: step-math validated on a shared 0.36-spec schedule (see docstring).
    max_diff = _run_step_sequence(FLOW_DIFFUSERS_KWARGS, shape, share_schedule=True)
    print(f"\n[flow]     shape={shape} max|diff|={max_diff:.3e}")
    assert max_diff < TOL, f"flow config shape={shape} max|diff|={max_diff}"


@_parametrize("shape", SHAPES)
def test_unipc_standard_matches_diffusers(shape):
    # STANDARD: validated end-to-end against the LIVE installed diffusers.
    max_diff = _run_step_sequence(STANDARD_DIFFUSERS_KWARGS, shape, share_schedule=False)
    print(f"\n[standard] shape={shape} max|diff|={max_diff:.3e}")
    assert max_diff < TOL, f"standard config shape={shape} max|diff|={max_diff}"


# ---------------------------------------------------------------------------
# Factory + contract smoke checks.
# ---------------------------------------------------------------------------

def test_factory_registration():
    from neurobrix.core.module.scheduler.factory import SchedulerFactory

    for key in ("UniPCMultistepScheduler", "unipc", "unipc_multistep"):
        cfg = _nbx_config_from(FLOW_DIFFUSERS_KWARGS)
        cfg["_class_name"] = key
        sched = SchedulerFactory.create(cfg)
        assert isinstance(sched, NBXUniPC), f"factory key '{key}' returned {type(sched)}"


def test_contract_surface():
    nbx, _ = _build_pair(FLOW_DIFFUSERS_KWARGS)
    nbx.set_timesteps(N_STEPS)
    # init_noise_sigma property.
    assert nbx.init_noise_sigma == 1.0
    # scale_model_input is identity.
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(nbx.scale_model_input(x, nbx.timesteps[0]), x)
    # bad solver_type crashes (ZERO FALLBACK).
    bad = _nbx_config_from(FLOW_DIFFUSERS_KWARGS)
    bad["solver_type"] = "midpoint"
    with _raises(Exception):
        NBXUniPC(bad)


if __name__ == "__main__":
    # Allow running the file directly (in case pytest discovery is fussy).
    print(f"diffusers {diffusers.__version__}, torch {torch.__version__}")
    ok = True

    # Schedule pins.
    try:
        test_standard_schedule_matches_live_diffusers()
        print("[schedule] standard vs live diffusers: PASS (timesteps + sigmas equal)")
    except AssertionError as e:
        ok = False
        print(f"[schedule] standard: FAIL {e}")
    try:
        test_flow_schedule_matches_036_reference()
        print("[schedule] flow vs 0.36 reference: PASS")
    except AssertionError as e:
        ok = False
        print(f"[schedule] flow: FAIL {e}")

    # Per-step numerical equivalence.
    for name, kwargs, share in [
        ("standard", STANDARD_DIFFUSERS_KWARGS, False),
        ("flow", FLOW_DIFFUSERS_KWARGS, True),
    ]:
        for shape in SHAPES:
            d = _run_step_sequence(kwargs, shape, share_schedule=share)
            status = "PASS" if d < TOL else "FAIL"
            ok = ok and d < TOL
            print(f"[{name:8s}] shape={str(shape):16s} max|diff|={d:.3e} {status}")

    # Factory + contract.
    try:
        test_factory_registration()
        test_contract_surface()
        print("[smoke] factory registration + contract surface: PASS")
    except AssertionError as e:
        ok = False
        print(f"[smoke] FAIL {e}")

    print("\nOVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
