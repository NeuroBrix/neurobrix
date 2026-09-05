"""The PNG encoder level is a family-config product default (lossless), read
at save time — never a constant in the writer."""
import numpy as np
import torch
from PIL import Image

from neurobrix.core.config import get_family_config
from neurobrix.core.runtime import output_dispatch as OD


def test_image_family_declares_a_png_level():
    lvl = (get_family_config("image").get("output") or {}).get("png_compress_level")
    assert isinstance(lvl, int) and 0 <= lvl <= 9


class _Executor:
    def get_final_output(self, outputs):
        return outputs["image"]


class _Pkg:
    defaults = {"output_range": [0.0, 1.0]}


def test_saved_png_is_lossless_at_the_configured_level(tmp_path):
    img = torch.rand(1, 3, 32, 48)                      # [B, C, H, W] in [0, 1]
    out = OD.save_image({"image": img}, str(tmp_path / "a.png"), "image", _Executor(), _Pkg())
    a = np.asarray(Image.open(out).convert("RGB"))
    ref = tmp_path / "ref.png"                           # PIL's default level: pixel-identical
    Image.fromarray((img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(ref)
    assert np.array_equal(a, np.asarray(Image.open(ref).convert("RGB")))
