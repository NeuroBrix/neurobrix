"""Triton-pure (zero-torch) diffusion schedulers.

Functionally identical to core/module/scheduler/* (PyTorch) but TOTALLY separate
code: the per-step latent arithmetic runs on NBXTensor (triton kernels), the
scalar noise schedule is computed once with numpy (CPU, not torch), and init
noise comes from a numpy Generator -> NBXTensor.from_numpy. No torch import,
ever — this is the triton half of the two-mode architecture (the only shared
point with the PyTorch path is the CLI entry + orchestrator executor).

Mirror map (PyTorch -> Triton):
  core/module/scheduler/diffusion/dpm_solver_pp.py -> dpm_solver_pp.py
  core/module/scheduler/flow/flow_euler.py         -> flow_euler.py
  core/module/scheduler/factory.py                 -> factory.py
"""
from .factory import TritonSchedulerFactory  # noqa: F401
