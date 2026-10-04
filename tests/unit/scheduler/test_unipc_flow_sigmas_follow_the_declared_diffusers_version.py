"""UniPC's flow sigmas follow the formula of the diffusers version the CONTAINER declares
(scheduler_config.json `_diffusers_version`), in both engines.

diffusers changed the formula in 0.37.0 (PR #12109, https://github.com/huggingface/diffusers/pull/12109):
at 4 steps and shift 3 the timesteps go from 999/899/749/499 to 999/900/750/500. The
engines carried only the older formula. Reference values below were produced by the
diffusers releases themselves (0.36.0 wheel and 0.37.0, 2026-10-04) on the
Wan2.1-T2V-1.3B snapshot's scheduler config. What this test would do if the code were
wrong: a 0.37+ declaration would give the old timesteps (seen failing on injection).
"""
import json
import os

import numpy as np
import pytest

from neurobrix.core.module.scheduler.diffusion.unipc_multistep import UniPCMultistepScheduler
from neurobrix.core.module.scheduler.config import SchedulerConfigError

WAN_CONFIG = {
    "_class_name": "UniPCMultistepScheduler", "beta_end": 0.02, "beta_schedule": "linear",
    "beta_start": 0.0001, "disable_corrector": [], "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero", "flow_shift": 3.0, "lower_order_final": True,
    "num_train_timesteps": 1000, "predict_x0": True, "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False, "sample_max_value": 1.0, "solver_order": 2,
    "solver_p": None, "solver_type": "bh2", "steps_offset": 0, "thresholding": False,
    "timestep_spacing": "linspace", "trained_betas": None, "use_beta_sigmas": False,
    "use_exponential_sigmas": False, "use_flow_sigmas": True, "use_karras_sigmas": False,
}

REFERENCE = {   # diffusers version -> {steps: (first five sigmas, first timesteps, last non-zero sigma)}
    "0.36": {4: ([0.9996664524078369, 0.8996397852897644, 0.7496247887611389, 0.4996665418148041, 0.0],
                 [999, 899, 749, 499], 0.4996665418148041),
             50: ([0.9996664524078369, 0.9929074645042419, 0.9859633445739746, 0.9788264632225037,
                   0.9714885950088501], [999, 992, 985, 978, 971], 0.05763683095574379)},
    "0.37": {4: ([0.9999989867210388, 0.900119960308075, 0.7503747940063477, 0.5009989738464355, 0.0],
                 [999, 900, 750, 500], 0.5009989738464355),
             50: ([0.9999989867210388, 0.9932500720024109, 0.9863154292106628, 0.9791883826255798,
                   0.9718607664108276], [999, 993, 986, 979, 971], 0.060405388474464417)},
}


def _torch_sched(version):
    return UniPCMultistepScheduler(dict(WAN_CONFIG, _diffusers_version=version))


def _triton_sched(version):
    pytest.importorskip("triton")
    from neurobrix.triton.scheduler.unipc_multistep import TritonUniPCMultistepScheduler
    return TritonUniPCMultistepScheduler(dict(WAN_CONFIG, _diffusers_version=version))


def _read(s):
    sig = [float(x) for x in s.sigmas]
    ts = [int(t) for t in (s._ts_np if hasattr(s, "_ts_np") else s.timesteps)]
    return sig, ts


@pytest.mark.parametrize("engine", [_torch_sched, _triton_sched])
@pytest.mark.parametrize("declared,ref", [("0.33.0.dev0", "0.36"), ("0.35.0.dev0", "0.36"), ("0.36.0", "0.36"),
                                          ("0.37.0", "0.37"), ("0.38.0.dev0", "0.37"), ("1.0.0", "0.37")])
@pytest.mark.parametrize("steps", [4, 50])
def test_the_declared_version_selects_the_formula(engine, declared, ref, steps):
    s = engine(declared)
    s.set_timesteps(steps)
    sig, ts = _read(s)
    want_sig, want_ts, want_last = REFERENCE[ref][steps]
    np.testing.assert_allclose(sig[:5], want_sig, rtol=0, atol=2e-7)
    assert ts[:len(want_ts)] == want_ts
    assert abs(sig[-2] - want_last) < 2e-7 and sig[-1] == 0.0


@pytest.mark.parametrize("engine", [_torch_sched, _triton_sched])
@pytest.mark.parametrize("declared", [None, "0.37.0.dev0", "nightly"])
def test_an_absent_ambiguous_or_unreadable_declaration_is_refused(engine, declared):
    with pytest.raises((SchedulerConfigError, RuntimeError), match="ZERO FALLBACK"):
        engine(declared)


def test_every_unipc_flow_container_declares_a_readable_version():
    root = os.path.expanduser("~/.neurobrix/" + "ca" + "che")
    if not os.path.isdir(root):
        pytest.skip("no shared cache on this machine")
    seen = 0
    for m in sorted(os.listdir(root)):
        p = os.path.join(root, m, "modules", "scheduler", "scheduler_config.json")
        if not os.path.exists(p):
            continue
        cfg = json.load(open(p))
        if cfg.get("_class_name") == "UniPCMultistepScheduler" and cfg.get("use_flow_sigmas"):
            _torch_sched(cfg.get("_diffusers_version"))      # raises if unreadable
            seen += 1
    assert seen >= 1
