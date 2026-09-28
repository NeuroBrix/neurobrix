"""NBX_SAVE_INPUTS=<component>:<dir> saves a component's first inputs whole as .npy, in both tensor formats.

The hand-off instrument of 2026-09-28: the rack decodes one latent of a Metal run with the vendor's own
autoencoder, and the per-op dump keeps norms and heads only. What this does if the code is wrong: with the
variable unset nothing may be written (red on a hook that always saves); with it set, a torch bf16 input must
land as float32 with its values, and an NBX-style bf16 input (2-byte void bits) as uint16 bits AND a float32
copy whose values decode the bits — a hook that saved the void array raw would fail the value check.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from neurobrix.core.runtime import graph_executor as GE


class _NBXLike:
    """The two things the hook reads from an NBXTensor: `.numpy()` (bf16 as |V2 bits) and `.shape`."""
    def __init__(self, bits: np.ndarray):
        self._bits = bits
        self.shape = bits.shape
    def numpy(self):
        return self._bits.view(np.dtype("V2"))


def test_nothing_is_written_when_the_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("NBX_SAVE_INPUTS", raising=False)
    GE._SAVED_INPUTS.clear()
    GE._save_inputs_if_asked("vae", {"latent": torch.zeros(2, 2)}, "compiled")
    assert not os.listdir(tmp_path) and not any(k[0] == "vae" for k in GE._SAVED_INPUTS)


def test_torch_and_nbx_inputs_land_whole_with_their_values(tmp_path, monkeypatch):
    monkeypatch.setenv("NBX_SAVE_INPUTS", f"vae:{tmp_path}")
    GE._SAVED_INPUTS.clear()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 7
    xb = x.to(torch.bfloat16)
    f32 = np.arange(12, dtype=np.float32).reshape(3, 4) / 3
    bits = (f32.view(np.uint32) >> 16).astype(np.uint16)                # bf16 bits of f32 (truncation)
    GE._save_inputs_if_asked("vae", {"latent": xb, "codes": _NBXLike(bits)}, "triton")
    a = np.load(tmp_path / "vae.triton.latent.npy")
    assert a.dtype == np.float32 and a.shape == (1, 2, 3, 4)
    assert np.array_equal(a, xb.float().numpy()), "the torch bf16 input did not land with its values"
    b = np.load(tmp_path / "vae.triton.codes.npy"); bb = np.load(tmp_path / "vae.triton.codes.bf16bits.npy")
    assert bb.dtype == np.uint16 and np.array_equal(bb, bits)
    assert b.dtype == np.float32 and np.array_equal(b, (bits.astype(np.uint32) << 16).view(np.float32).reshape(3, 4))
    # the first call only
    GE._save_inputs_if_asked("vae", {"latent": torch.ones(1)}, "triton")
    assert np.load(tmp_path / "vae.triton.latent.npy").shape == (1, 2, 3, 4)
