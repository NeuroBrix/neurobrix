"""
Universal image input preprocessing (compiled mode).

DATA-DRIVEN mirror of `core.module.audio.input_processor.AudioInputProcessor`:
the preprocessing type is a topology/config-declared string, the DSP is the
vendor-free numpy core `image_dsp` (shared with the triton path, R34), and
compiled mode converts the numpy result to torch only at the boundary.
ZERO FALLBACK: an unknown preprocessing type raises, never guesses.
ZERO model-specific branches: the type string is the only discriminator.

Supported preprocessing types:
  i2v_vae_condition — video I2V conditioning clip [1,3,T,H,W] in [-1,1]
  clip_centercrop   — CLIP view [1,3,cs,cs] from the embedded
                      modules/image_processor/preprocessor_config.json

Planned (each lands WITH its consumer model, zero-fallback until then):
  native_patch_grid — dynamic-resolution patch grid + grid_thw
                      (Qwen3-VL / GLM-4.1V class)
  anyres_slice      — LLaVA-UHD adaptive slicing (MiniCPM-o class)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from neurobrix.core.module.vision import image_dsp


class ImageInputProcessor:
    """Routes to the correct image preprocessor based on declared type."""

    SUPPORTED = ("i2v_vae_condition", "clip_centercrop")

    @staticmethod
    def process(
        preprocessing_type: str,
        image_path: str,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        pad_to_num_frames: int = 0,
        preprocessor_config: Optional[dict] = None,
    ) -> torch.Tensor:
        """Preprocess an image file into a model input tensor.

        Returns a CPU float32 torch.Tensor (the runtime resolver owns the
        later device/dtype placement — identical contract to the former
        inline CLI block this replaces).
        """
        if preprocessing_type not in ImageInputProcessor.SUPPORTED:
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown image preprocessing type "
                f"'{preprocessing_type}'.\n"
                f"Supported: {', '.join(ImageInputProcessor.SUPPORTED)}. "
                f"(native_patch_grid / anyres_slice land with their "
                f"consumer models.)"
            )
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if preprocessing_type == "i2v_vae_condition":
            arr = image_dsp.i2v_vae_condition_np(
                str(image_path), height=height, width=width,
                pad_to_num_frames=pad_to_num_frames)
        else:  # clip_centercrop
            if not isinstance(preprocessor_config, dict):
                raise RuntimeError(
                    "ZERO FALLBACK: clip_centercrop requires the embedded "
                    "preprocessor_config.json contents (dict); the caller "
                    "reads modules/image_processor/preprocessor_config.json "
                    "and passes it explicitly."
                )
            arr = image_dsp.clip_centercrop_np(
                str(image_path), preprocessor_config)

        # Torch only at the compiled boundary; stays CPU float32.
        return torch.from_numpy(np.ascontiguousarray(arr))
