"""The size an image-conditioned video run generates at, and the fit of its conditioning image: one reader.

The container declares both (the project lead's ruling of 2026-11-23 11:13); the engine reads, it never supplies:

  runtime/defaults.json `size_policy`
      "fixed"       `height` and `width` declared: the run's size whatever the image.
      "from_image"  `max_area` (the product, an integer) and `size_multiple` declared, no height/width: the size
                    follows the conditioning image's aspect (`from_image_size`, the diffusers form).
      Absent, unknown, or both forms at once: refused by name.
  An explicit request size wins: for "from_image" when each side is a multiple of `size_multiple` (else refused,
  the multiple named); for "fixed" any value, as before.

  topology.json extracted_values.vae_encoder.condition_image_fit
      `crop` (none|center) and `resample` (lanczos|bicubic|linear), in the vendor's order of operations (the
      key order); undeclared, refused by name. What the library reproduces of them is the library's own check
      (`core/module/vision/image_dsp`).

Every door that prepares a conditioning image (`cli run`, the daemon, the plan, the executor) asks this module; the
image's own size is never the run's size (the fallback of 18a3b1bcb, withdrawn)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

POLICIES = ("fixed", "from_image")
_FORMS = {"fixed": ("height", "width"), "from_image": ("max_area", "size_multiple")}
FIT_KEYS = ("crop", "resample")
DEFAULTS_FILE = "runtime/defaults.json"
FIT_SITE = "topology.json extracted_values.vae_encoder.condition_image_fit"


def _refuse(message: str) -> None:
    raise RuntimeError(f"ZERO FALLBACK: {message}")


def _policy(defaults: Mapping[str, Any], where: str) -> str:
    """The declared policy, refused by name when absent, unknown, or declared beside the other form's keys."""
    policy = defaults.get("size_policy")
    if policy is None:
        _refuse(f"size_policy is not declared in {where}: a run with a conditioning image generates at the size its "
                f"container declares, \"fixed\" (height, width) or \"from_image\" (max_area, size_multiple)")
    if policy not in POLICIES:
        _refuse(f"size_policy = {policy!r} in {where} is not a size policy: \"fixed\" or \"from_image\"")
    other = "from_image" if policy == "fixed" else "fixed"
    stray = [k for k in _FORMS[other] if k in defaults]
    if stray:
        _refuse(f"{where} declares size_policy = {policy!r} and also {stray} of the {other!r} form: one form, "
                f"not both")
    return policy


def _positive_int(defaults: Mapping[str, Any], key: str, policy: str, where: str) -> int:
    value = defaults.get(key)
    if value is None:
        _refuse(f"size_policy = {policy!r} needs {key}, which is not declared in {where}")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(f"{key} = {value!r} in {where} is not a positive integer")
    return value


def from_image_size(max_area: int, size_multiple: int, image_hw: Tuple[int, int]) -> Tuple[int, int]:
    """(height, width) at `max_area` pixels and the image's aspect, each side floored to `size_multiple`.

    The diffusers form (the Wan I2V pipelines' documented example, `pipeline_wan_i2v.py` EXAMPLE_DOC_STRING):
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    Not checked against a vendor copy on disk: none was found under ~/Mounts/Super-HF-Snapshots on 2026-11-23 (the
    snapshots there carry weights and configs, no README, no diffusers source); this is the ruling's formula."""
    ar = image_hw[0] / image_hw[1]
    return (round(math.sqrt(max_area * ar)) // size_multiple * size_multiple,
            round(math.sqrt(max_area / ar)) // size_multiple * size_multiple)


def _from_image(defaults, image_hw, asked_h, asked_w, where) -> Tuple[int, int]:
    max_area = _positive_int(defaults, "max_area", "from_image", where)
    multiple = _positive_int(defaults, "size_multiple", "from_image", where)
    for side, asked in (("height", asked_h), ("width", asked_w)):
        if asked is not None and int(asked) % multiple:
            _refuse(f"the requested {side} {asked} is not a multiple of size_multiple = {multiple} (declared in "
                    f"{where}): ask a multiple of {multiple}")
    if asked_h is not None and asked_w is not None:
        return int(asked_h), int(asked_w)
    if not image_hw or not all(isinstance(v, int) and v > 0 for v in image_hw):
        _refuse(f"size_policy = 'from_image' in {where} sizes the run from its conditioning image, and this call "
                f"gave no image size ({image_hw!r})")
    h, w = from_image_size(max_area, multiple, tuple(image_hw))
    if h < multiple or w < multiple:
        _refuse(f"the conditioning image's aspect ({image_hw[0]}x{image_hw[1]}) leaves a side under size_multiple = "
                f"{multiple} at max_area = {max_area} ({where}): {h}x{w}")
    return (int(asked_h) if asked_h is not None else h, int(asked_w) if asked_w is not None else w)


def resolve_size(defaults: Optional[Mapping[str, Any]], image_hw: Optional[Tuple[int, int]],
                 asked_h: Optional[int], asked_w: Optional[int], where: str = DEFAULTS_FILE) -> Tuple[int, int]:
    """(height, width) a run with a conditioning image generates at, and the image is fitted to.

    `defaults` is the container's runtime/defaults.json, `image_hw` the conditioning image's own (height, width),
    `asked_h`/`asked_w` the request's (None when it names none: each side is taken on its own), `where` the file
    named in a refusal."""
    defaults = defaults or {}
    if _policy(defaults, where) == "from_image":
        return _from_image(defaults, image_hw, asked_h, asked_w, where)
    h = _positive_int(defaults, "height", "fixed", where)
    w = _positive_int(defaults, "width", "fixed", where)
    return (int(asked_h) if asked_h is not None else h, int(asked_w) if asked_w is not None else w)


def image_fit(fit: Optional[Mapping[str, Any]], container: str = "the container") -> Tuple[str, str]:
    """(crop, resample) the vae_encoder's `condition_image_fit` declares, refused by name when either is undeclared.

    The order of the two keys in the declaration is the vendor's order of operations; the library reads it from the
    declaration itself (`image_dsp.i2v_vae_condition_np`)."""
    if not isinstance(fit, Mapping):
        _refuse(f"condition_image_fit is not declared on the vae_encoder of {container} ({FIT_SITE}): the "
                f"conditioning image's crop (none|center) and resample (lanczos|bicubic|linear) are its container's")
    for key in FIT_KEYS:
        if fit.get(key) is None:
            _refuse(f"condition_image_fit.{key} is not declared on the vae_encoder of {container} ({FIT_SITE}): "
                    f"crop (none|center) and resample (lanczos|bicubic|linear), in the vendor's order")
    return fit["crop"], fit["resample"]
