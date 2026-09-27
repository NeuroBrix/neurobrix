"""Resolution binning — the vendor's own mapping of a requested size to a trained size, and back.

A pipeline trained on a fixed table of sizes (diffusers' PixArt and Sana pipelines, `use_resolution_binning`,
default True) never denoises at the size it is asked for. It classifies the request to the nearest trained
aspect ratio, denoises and decodes at that bin, then restores the requested size by resizing to cover it and
cropping the centre. Without the step the engine denoises at the literal request, and a model trained at 1024
asked for 512 renders colour noise: the vendor's own output at native 512 (the Dell's vendor oracle, diffusers
0.37.0, `use_resolution_binning=False` against `True`, 2026-09-26).

Forge records what the vendor ran, at trace, under `flow.resolution_binning` (NeuroBrix_forge branch
`resolution-binning-captured`, f8e5866), and nothing here re-codes a vendor table:

    {"default": bool,
     "source": "<PipelineClass>.__call__(use_resolution_binning)",
     "bins": {"<ratio key verbatim>": [h, w], ...},
     "classify": "nearest_ratio",
     "restore": "cover_resize_center_crop",
     "interpolate": {"mode": "bilinear", "align_corners": false}}

A pipeline whose default is False and whose trace never binned carries only `default` and `source`.

This is the shared half, and it is torch-free (R33): reading the contract, classifying a request, and planning
the restore. The restore runs in each branch's flow on its own container, torch in `core/flow` and NBXTensor in
`triton/flow` (R30). The rules are the vendor's, from diffusers v0.37.0 `PixArtImageProcessor`
(src/diffusers/image_processor.py):

    classify_height_width_bin:  ar = float(height / width)
                                closest = min(ratios.keys(), key=lambda r: abs(float(r) - ar))
                                return int(ratios[closest][0]), int(ratios[closest][1])
    resize_and_crop_tensor:     if (orig_h, orig_w) != (new_h, new_w):
                                    ratio = max(new_h / orig_h, new_w / orig_w)
                                    resized_w, resized_h = int(orig_w * ratio), int(orig_h * ratio)
                                    F.interpolate(size=(resized_h, resized_w), mode="bilinear", align_corners=False)
                                    start_x = (resized_w - new_w) // 2; start_y = (resized_h - new_h) // 2
                                    samples[:, :, start_y:start_y + new_h, start_x:start_x + new_w]

and the pipeline applies them in `PixArtAlphaPipeline.__call__`: the request (or the default size) is classified
before the latents are prepared, and the decoded image is restored before post-processing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

# The vocabularies the engine executes. Forge writes what it SAW; a word outside these is a
# contract this engine cannot honour, and it is refused, never approximated (ZERO FALLBACK).
_CLASSIFY = ("nearest_ratio",)
_RESTORE = ("cover_resize_center_crop",)
_INTERPOLATE = (("bilinear", False),)

FIELD = "resolution_binning"
REQUEST_FLAG = "global.use_resolution_binning"


@dataclass(frozen=True)
class BinningContract:
    default: bool
    source: str
    bins: Tuple[Tuple[str, int, int], ...]   # (ratio key verbatim, h, w), in the vendor's table order
    mode: str
    align_corners: bool


@dataclass(frozen=True)
class RestorePlan:
    """Resize the decoded (H, W) to (resized_h, resized_w), then keep rows [top, bottom) and columns
    [left, right). The bounds are Python's own slice bounds over the resized extent, so the crop is the
    vendor's slice exactly, including its edge cases."""
    resized_h: int
    resized_w: int
    top: int
    bottom: int
    left: int
    right: int
    mode: str
    align_corners: bool


@dataclass(frozen=True)
class BinnedRequest:
    requested: Tuple[int, int]   # (h, w) the request asked for
    binned: Tuple[int, int]      # (h, w) the pipeline runs at
    contract: BinningContract


def _refuse(msg: str) -> RuntimeError:
    return RuntimeError(f"ZERO FALLBACK: flow.{FIELD}: {msg}")


def read_contract(topology: Mapping[str, Any]) -> Optional[BinningContract]:
    """The container's binning contract, or None when its flow records none.

    Refuses a field it cannot execute: an unknown classify, restore or interpolation, a default of True
    without bins, a malformed bin."""
    field = (topology.get("flow") or {}).get(FIELD)
    if field is None:
        return None
    if not isinstance(field, Mapping) or not isinstance(field.get("default"), bool):
        raise _refuse(f"expected a mapping with a boolean 'default', got {field!r}")
    source = str(field.get("source", ""))
    raw_bins = field.get("bins")
    if raw_bins is None:
        if field["default"]:
            raise _refuse(f"'default' is true but no 'bins' were recorded (source {source!r})")
        return BinningContract(False, source, (), "", False)
    classify = field.get("classify")
    restore = field.get("restore")
    interp = field.get("interpolate") or {}
    mode, align = interp.get("mode"), interp.get("align_corners")
    if classify not in _CLASSIFY:
        raise _refuse(f"classify {classify!r} is not executed by this engine (known: {_CLASSIFY})")
    if restore not in _RESTORE:
        raise _refuse(f"restore {restore!r} is not executed by this engine (known: {_RESTORE})")
    if (mode, align) not in _INTERPOLATE:
        raise _refuse(f"interpolate mode={mode!r} align_corners={align!r} is not executed by this engine "
                      f"(known: {_INTERPOLATE})")
    if not isinstance(raw_bins, Mapping) or not raw_bins:
        raise _refuse(f"'bins' must be a non-empty mapping, got {raw_bins!r}")
    bins = []
    for key, hw in raw_bins.items():
        try:
            float(key)
            h, w = hw
            bins.append((str(key), int(h), int(w)))
        except (TypeError, ValueError) as exc:
            raise _refuse(f"bin {key!r}: {hw!r} is not '<ratio>': [h, w] ({exc})") from None
    return BinningContract(field["default"], source, tuple(bins), mode, align)


def is_active(contract: Optional[BinningContract], requested: Any = None) -> bool:
    """Whether a request runs binned: the request's own `use_resolution_binning` when it carries one, else the
    vendor's default as recorded. A request asking for binning on a container that recorded no bins is refused."""
    if contract is None:
        if requested:
            raise _refuse("the request asks for resolution binning, but this container's flow records none")
        return False
    active = contract.default if requested is None else bool(requested)
    if active and not contract.bins:
        raise _refuse(f"the request asks for resolution binning, but no bins were recorded "
                      f"(source {contract.source!r})")
    return active


def classify(height: int, width: int, contract: BinningContract) -> Tuple[int, int]:
    """The vendor's `classify_height_width_bin`: the bin whose ratio key is nearest to height / width; on a tie
    the first in the table's order, as `min` over the dict's keys gives."""
    ar = float(height / width)
    best = None
    for key, h, w in contract.bins:
        d = abs(float(key) - ar)
        if best is None or d < best[0]:
            best = (d, h, w)
    return best[1], best[2]


def restore_plan(orig_h: int, orig_w: int, new_h: int, new_w: int,
                 contract: BinningContract) -> Optional[RestorePlan]:
    """The vendor's `resize_and_crop_tensor`, as a plan. None when the decoded size already is the request."""
    if orig_h == new_h and orig_w == new_w:
        return None
    ratio = max(new_h / orig_h, new_w / orig_w)
    resized_w = int(orig_w * ratio)
    resized_h = int(orig_h * ratio)
    start_x = (resized_w - new_w) // 2
    start_y = (resized_h - new_h) // 2
    top, bottom, _ = slice(start_y, start_y + new_h).indices(resized_h)
    left, right, _ = slice(start_x, start_x + new_w).indices(resized_w)
    return RestorePlan(resized_h, resized_w, top, max(top, bottom), left, max(left, right),
                       contract.mode, contract.align_corners)


def bin_request(topology: Mapping[str, Any], merged: Dict[str, Any],
                requested_flag: Any = None) -> Optional[BinnedRequest]:
    """Classify the request in `merged` (the executor's merged defaults, request applied) and rewrite its
    height and width to the bin. Returns what the flow needs to restore the requested size, or None when the
    request does not run binned."""
    contract = read_contract(topology)
    if not is_active(contract, requested_flag):
        return None
    if merged.get("height") is None or merged.get("width") is None:
        raise _refuse(f"binning is active but the request has no height/width to classify "
                      f"(height={merged.get('height')!r}, width={merged.get('width')!r})")
    req = (int(merged["height"]), int(merged["width"]))
    binned = classify(req[0], req[1], contract)
    merged["height"], merged["width"] = binned
    return BinnedRequest(req, binned, contract)


def rebind_request_size(inputs: Mapping[str, Any], binned: BinnedRequest) -> Dict[str, Any]:
    """The request's inputs with its own height/width replaced by the bin. The executor binds a request's
    inputs into the resolver as given, so a binned request must carry the bin there too, or the latent and
    the graph would be sized by the literal request while the defaults say the bin."""
    out = dict(inputs)
    for key in ("height", "global.height"):
        if key in out:
            out[key] = binned.binned[0]
    for key in ("width", "global.width"):
        if key in out:
            out[key] = binned.binned[1]
    return out
