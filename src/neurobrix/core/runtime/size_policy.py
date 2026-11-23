"""The size an image-conditioned video run generates at, and the fit of its conditioning image: one reader.

The container declares both; the engine reads, it never supplies (rulings of 2026-11-23: 11:13, revised by the owner
13:48 and the project lead 13:51):

  runtime/defaults.json `size_policy`
      "default"     `height` and `width` declared: the run's size when the request names none, whatever the image.
      "from_image"  `max_area` (the product, an integer) and `size_multiple` declared, no height/width: the size
                    follows the conditioning image's aspect (`from_image_size`, the diffusers form).
      An unknown policy ("fixed" included: none ships with it) or both forms at once ("default" with max_area,
      "from_image" with height/width): refused by name.
  The user's size is a value. A request naming BOTH height and width runs at them, with or without a declared
  policy: the one check is what the graph needs, the declared `size_multiple` where one is declared (refused, the
  multiple named). The refusal by name with its remedy stays only where no size comes from anywhere; a request
  naming one side to a container with no policy is refused naming the missing side.

  The vendor's statement is HELP, never a gate: `size_note` (one sentence), `size_static` (true where the vendor
  documents one size), `size_range` ({min, max, multiple}) are read and shown (`size_notice`) when the request's size
  is not the declared default; nothing is refused on them.

  topology.json extracted_values.vae_encoder.condition_image_fit
      `crop` (none|center) and `resample` (lanczos|bicubic|linear), in the vendor's order of operations (the
      key order); undeclared, refused by name. What the library reproduces of them is the library's own check
      (`core/module/vision/image_dsp`).

Every door that prepares a conditioning image (`cli run`, the daemon, the plan, the executor) asks this module; the
image's own size is never the run's size (the fallback of 18a3b1bcb, withdrawn)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

POLICIES = ("default", "from_image")
_FORMS = {"default": ("height", "width"), "from_image": ("max_area", "size_multiple")}
#: the keys that mark the OTHER form: a container declaring them beside its policy declares both forms
_OTHER_FORM = {"default": ("max_area",), "from_image": ("height", "width")}
FIT_KEYS = ("crop", "resample")
DEFAULTS_FILE = "runtime/defaults.json"
FIT_SITE = "topology.json extracted_values.vae_encoder.condition_image_fit"


def _refuse(message: str) -> None:
    raise RuntimeError(f"ZERO FALLBACK: {message}")


# The remedy, in the engine's own wording for a container whose data is incomplete (core/module/cache/factory.py):
# the value is the container's, so the fix is the corrected container, never a value supplied here.
REMEDY = (" — this container predates the declared size policy: install the corrected container "
          "(neurobrix remove <model> && neurobrix import <org>/<model>)")


def _outdated(message: str) -> None:
    """A size key the container lacks or holds wrong: refused by name, with the remedy."""
    _refuse(message + REMEDY)


def _policy(defaults: Mapping[str, Any], where: str) -> Optional[str]:
    """The declared policy, None when undeclared; refused by name when unknown or beside the other form's keys."""
    policy = defaults.get("size_policy")
    if policy is None:
        return None
    if policy not in POLICIES:
        _outdated(f"size_policy = {policy!r} in {where} is not a size policy: \"default\" or \"from_image\"")
    stray = [k for k in _OTHER_FORM[policy] if k in defaults]
    if stray:
        _outdated(f"{where} declares size_policy = {policy!r} and also {stray} of the other form: one form, "
                  f"not both")
    return policy


def _positive_int(defaults: Mapping[str, Any], key: str, policy: str, where: str) -> int:
    value = defaults.get(key)
    if value is None:
        _outdated(f"size_policy = {policy!r} needs {key}, which is not declared in {where}")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _outdated(f"{key} = {value!r} in {where} is not a positive integer")
    return value


def _check_multiple(defaults: Mapping[str, Any], asked_h, asked_w, where: str) -> None:
    """The one check on a requested size: the declared `size_multiple`, where one is declared."""
    if defaults.get("size_multiple") is None:
        return
    multiple = _positive_int(defaults, "size_multiple", str(defaults.get("size_policy")), where)
    for side, asked in (("height", asked_h), ("width", asked_w)):
        if asked is not None and int(asked) % multiple:
            _refuse(f"the requested {side} {asked} is not a multiple of size_multiple = {multiple} (declared in "
                    f"{where}): ask a multiple of {multiple}")


def _no_size(asked_h, asked_w, where: str) -> None:
    """No declared policy and the request does not name both sides: the size comes from nowhere."""
    if asked_h is None and asked_w is None:
        _outdated(f"size_policy is not declared in {where} and the request names no size: a run with a "
                  f"conditioning image generates at --height/--width, else at the size its container declares, "
                  f"\"default\" (height, width) or \"from_image\" (max_area, size_multiple)")
    given, missing = ("height", "width") if asked_h is not None else ("width", "height")
    _refuse(f"the request names its {given} but not its {missing}, and {where} declares no size_policy to give "
            f"it: pass --{missing} too")


def from_image_size(max_area: int, size_multiple: int, image_hw: Tuple[int, int]) -> Tuple[int, int]:
    """(height, width) at `max_area` pixels and the image's aspect, each side floored to `size_multiple`.

    The diffusers form: diffusers pipelines/wan/pipeline_wan_i2v.py:70-75 (EXAMPLE_DOC_STRING, with
    mod_value = vae_scale_factor_spatial * patch_size[1]) and the Wan2.1 card README L173-175:
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    The vendor lines are quoted in transfer/dell_cards/size_policy/research.md (Dell lead, 23/11/2026). A vendor that
    rounds otherwise declares its own key then; it is not a branch here.

    A daemon plans at load, before any image (serving/engine.py `_container_extent`): with the declared height and
    width, else the container's own output size (its built latent times the VAE scale). A "from_image" container
    declares neither, so it is planned at its built latent, which can hold fewer pixels than `max_area` (an open
    point of the record, not settled here)."""
    ar = image_hw[0] / image_hw[1]
    return (round(math.sqrt(max_area * ar)) // size_multiple * size_multiple,
            round(math.sqrt(max_area / ar)) // size_multiple * size_multiple)


def _from_image(defaults, image_hw, asked_h, asked_w, where) -> Tuple[int, int]:
    max_area = _positive_int(defaults, "max_area", "from_image", where)
    multiple = _positive_int(defaults, "size_multiple", "from_image", where)
    if not image_hw or not all(isinstance(v, int) and v > 0 for v in image_hw):
        _refuse(f"size_policy = 'from_image' in {where} sizes the run from its conditioning image, and this call "
                f"gave no image size ({image_hw!r})")
    h, w = from_image_size(max_area, multiple, tuple(image_hw))
    if h < multiple or w < multiple:
        _refuse(f"the conditioning image's aspect ({image_hw[0]}x{image_hw[1]}) leaves a side under size_multiple = "
                f"{multiple} at max_area = {max_area} ({where}): {h}x{w}")
    return (int(asked_h) if asked_h is not None else h, int(asked_w) if asked_w is not None else w)


def plan_size(defaults: Optional[Mapping[str, Any]], built_hw: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """The size a daemon plans a "from_image" container at, before any image: what its policy resolves for the
    container's own built aspect (`built_hw`, its built latent times the VAE scale) — the LIKELIEST request, not a
    bound (a resolved size can hold more pixels: another aspect, the vendor's round-then-floor, an explicit size;
    project lead 23/11/2026). None for a container of another policy, or with no built size: its declared or built
    size stands as before."""
    if not defaults or defaults.get("size_policy") != "from_image" or not built_hw:
        return None
    _policy(defaults, DEFAULTS_FILE)
    return from_image_size(_positive_int(defaults, "max_area", "from_image", DEFAULTS_FILE),
                           _positive_int(defaults, "size_multiple", "from_image", DEFAULTS_FILE), tuple(built_hw))


def resolve_size(defaults: Optional[Mapping[str, Any]], image_hw: Optional[Tuple[int, int]],
                 asked_h: Optional[int], asked_w: Optional[int], where: str = DEFAULTS_FILE) -> Tuple[int, int]:
    """(height, width) a run with a conditioning image generates at, and the image is fitted to.

    `defaults` is the container's runtime/defaults.json, `image_hw` the conditioning image's own (height, width),
    `asked_h`/`asked_w` the request's (None when it names none: each side is taken on its own), `where` the file
    named in a refusal. A request naming both sides runs at them, policy or none, checked only against a declared
    `size_multiple`."""
    defaults = defaults or {}
    policy = _policy(defaults, where)
    _check_multiple(defaults, asked_h, asked_w, where)
    if asked_h is not None and asked_w is not None:
        return int(asked_h), int(asked_w)
    if policy is None:
        _no_size(asked_h, asked_w, where)
    if policy == "from_image":
        return _from_image(defaults, image_hw, asked_h, asked_w, where)
    h = _positive_int(defaults, "height", "default", where)
    w = _positive_int(defaults, "width", "default", where)
    return (int(asked_h) if asked_h is not None else h, int(asked_w) if asked_w is not None else w)


def _vendor_bounds(defaults: Mapping[str, Any]) -> str:
    """What the vendor documents of the size, as read (`size_static`, `size_range`): words, never a check."""
    out = ""
    if defaults.get("size_static") is True:
        out += f" The vendor documents one size: {defaults.get('height')}x{defaults.get('width')}."
    rng = defaults.get("size_range")
    if isinstance(rng, Mapping):
        out += (f" The vendor documents {rng.get('min')} to {rng.get('max')}"
                + (f", a multiple of {rng['multiple']}." if rng.get("multiple") is not None else "."))
    return out


def size_notice(defaults: Optional[Mapping[str, Any]], asked_h: Optional[int],
                asked_w: Optional[int]) -> Optional[str]:
    """The one line shown when a request names a size other than the declared default and the container declares a
    `size_note`, else None. Help, not a gate: it never refuses (a "from_image" container declares no default, so any
    size a request names is shown the note)."""
    defaults = defaults or {}
    note = defaults.get("size_note")
    if not note or (asked_h is None and asked_w is None):
        return None
    declared = (defaults.get("height"), defaults.get("width"))
    if all(a is None or (d is not None and int(a) == int(d)) for a, d in zip((asked_h, asked_w), declared)):
        return None
    return f"[size] {note}{_vendor_bounds(defaults)}"


def image_fit(fit: Optional[Mapping[str, Any]], container: str = "the container") -> Tuple[str, str]:
    """(crop, resample) the vae_encoder's `condition_image_fit` declares, refused by name when either is undeclared.

    The order of the two keys in the declaration is the vendor's order of operations; the library reads it from the
    declaration itself (`image_dsp.i2v_vae_condition_np`)."""
    if not isinstance(fit, Mapping):
        _outdated(f"condition_image_fit is not declared on the vae_encoder of {container} ({FIT_SITE}): the "
                f"conditioning image's crop (none|center) and resample (lanczos|bicubic|linear) are its container's")
    for key in FIT_KEYS:
        if fit.get(key) is None:
            _outdated(f"condition_image_fit.{key} is not declared on the vae_encoder of {container} ({FIT_SITE}): "
                    f"crop (none|center) and resample (lanczos|bicubic|linear), in the vendor's order")
    return fit["crop"], fit["resample"]
