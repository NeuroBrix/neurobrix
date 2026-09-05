"""The PNG encoder level is a family-config product default (lossless), read
at save time — never a constant in the writer."""
import numpy as np
from PIL import Image

from neurobrix.core.config import get_family_config
from neurobrix.core.runtime.output_dispatch import png_save_kwargs


def test_image_family_declares_a_png_level():
    lvl = (get_family_config("image").get("output") or {}).get("png_compress_level")
    assert isinstance(lvl, int) and 0 <= lvl <= 9
    assert png_save_kwargs("image", "/x/out.png") == {"compress_level": lvl}
    assert png_save_kwargs("image", "/x/out.jpg") == {}


def test_configured_level_is_lossless(tmp_path):
    arr = (np.random.RandomState(0).rand(32, 48, 3) * 255).astype(np.uint8)
    fast, ref = tmp_path / "fast.png", tmp_path / "ref.png"
    Image.fromarray(arr).save(fast, **png_save_kwargs("image", str(fast)))
    Image.fromarray(arr).save(ref)
    assert np.array_equal(np.asarray(Image.open(fast)), np.asarray(Image.open(ref)))
