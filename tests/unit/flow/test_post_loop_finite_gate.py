"""The post-loop decoder gate — a non-finite VAE output is refused before
anything is written (2026-09-04: a DC-AE decoded in fp16 wrote a black PNG
while the loop-state gate had passed)."""
import pytest
import torch

from neurobrix.core.flow.iterative_process import _gate_component_outputs_finite


def test_finite_output_passes():
    resolved = {"vae.output_0": torch.randn(1, 3, 8, 8), "transformer.output_0": torch.tensor([float("nan")])}
    _gate_component_outputs_finite(resolved, "vae")   # only the named component is inspected


def test_non_finite_output_is_refused():
    resolved = {"vae.output_0": torch.tensor([[1.0, float("nan")], [float("inf"), 0.0]])}
    with pytest.raises(RuntimeError, match="non-finite output of post-loop component 'vae'"):
        _gate_component_outputs_finite(resolved, "vae")
