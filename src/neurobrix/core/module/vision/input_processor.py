"""
Universal image input preprocessing (compiled mode).

DATA-DRIVEN mirror of `core.module.audio.input_processor.AudioInputProcessor`:
the preprocessing type is a topology/config-declared string, the DSP is the
vendor-free numpy core `image_dsp` (shared with the triton path, R34), and
the executor puts the arrays in the engine's container (R33: no torch here).
ZERO FALLBACK: an unknown preprocessing type raises, never guesses.
ZERO model-specific branches: the type string is the only discriminator.

Supported preprocessing types:
  i2v_vae_condition — video I2V conditioning clip [1,3,T,H,W] in [-1,1]
  clip_centercrop   — CLIP view [1,3,cs,cs] from the embedded
                      modules/image_processor/preprocessor_config.json
  native_patch_grid — dynamic-resolution flattened patch grid
                      (GLM-4.1V / Qwen-VL class); returns a dict
                      {"pixel_values": [n_patches, C*Tp*P*P] float32,
                       "image_grid_thw": [1, 3] int64} (vendor
                      model_input_names, landed with GLM-4.1V)
  minicpm_adaptive_slice — LLaVA-UHD adaptive-slice NaViT contract
                      (MiniCPM-o class, landed with MiniCPM-o-4_5);
                      returns a dict matching the traced vpm graph's
                      input names {"all_pixel_values": [1, N, C*p*p]
                      float32, "patch_attention_mask": [1, 1, N] bool,
                      "tgt_sizes": [[gh, gw]] int32}
"""

from pathlib import Path
from typing import Optional

import numpy as np

from neurobrix.core.module.vision import image_dsp


class ImageInputProcessor:
    """Routes to the correct image preprocessor based on declared type."""

    SUPPORTED = ("i2v_vae_condition", "clip_centercrop", "native_patch_grid",
                 "native_patch_grid_video", "minicpm_adaptive_slice")

    @staticmethod
    def process(
        preprocessing_type: str,
        image_path: str,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        pad_to_num_frames: int = 0,
        preprocessor_config: Optional[dict] = None,
        fps: Optional[float] = None,
    ):
        """Preprocess an image file into model input tensor(s).

        Returns a float32 array for single-tensor types (i2v_vae_condition,
        clip_centercrop — the values of the former inline CLI block), or a
        dict of named arrays for multi-tensor types (native_patch_grid →
        pixel_values + image_grid_thw, vendor model_input_names). Arrays
        are the input boundary's container: the executor puts them in the
        engine's own (torch on the ATen branch, NBXTensor on the Triton
        branch — R33: no torch here) and the resolver owns placement.
        """
        if preprocessing_type not in ImageInputProcessor.SUPPORTED:
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown image preprocessing type "
                f"'{preprocessing_type}'.\n"
                f"Supported: {', '.join(ImageInputProcessor.SUPPORTED)}. "
                f"(New preprocessing types land with their consumer models.)"
            )
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if preprocessing_type == "minicpm_adaptive_slice":
            if not isinstance(preprocessor_config, dict):
                raise RuntimeError(
                    "ZERO FALLBACK: minicpm_adaptive_slice requires the "
                    "model's preprocessing config (dict) — the "
                    "topology.flow.vlm.preprocessing block."
                )
            out = image_dsp.minicpm_adaptive_slice_np(
                str(image_path), preprocessor_config)
            # Keys mirror the traced vpm graph's input:: names; dtypes are
            # the graph contract (float32 / bool / int32 — tgt_sizes values
            # feed in-graph position arithmetic and must stay exact ints).
            return {k: np.ascontiguousarray(v) for k, v in out.items()}

        if preprocessing_type == "native_patch_grid":
            if not isinstance(preprocessor_config, dict):
                raise RuntimeError(
                    "ZERO FALLBACK: native_patch_grid requires the model's "
                    "preprocessing config (dict) — registry/topology "
                    "preprocessing block or the embedded "
                    "preprocessor_config.json contents."
                )
            flat, grid = image_dsp.native_patch_grid_np(
                str(image_path), preprocessor_config)
            return {
                "pixel_values": np.ascontiguousarray(flat),
                "image_grid_thw": np.ascontiguousarray(grid),
            }

        if preprocessing_type == "native_patch_grid_video":
            if not isinstance(preprocessor_config, dict):
                raise RuntimeError(
                    "ZERO FALLBACK: native_patch_grid_video requires the "
                    "model's preprocessing config (dict) — registry/topology "
                    "preprocessing block."
                )
            flat, grid, second_per_grid = image_dsp.native_patch_grid_video_np(
                str(image_path), preprocessor_config, fps=fps)
            return {
                "pixel_values_videos": np.ascontiguousarray(flat),
                "video_grid_thw": np.ascontiguousarray(grid),
                "video_second_per_grid": np.asarray([second_per_grid], dtype=np.float32),
            }

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

        return np.ascontiguousarray(arr)


def find_upscale_input_variable(topology: dict) -> str:
    """Return the unique ``global.*`` connection source feeding the graph.

    R34: the input variable name is never hardcoded — it is whatever
    the container's topology declares (e.g. ``global.pixel_values``).
    Shared by the CLI cold path (``cli/commands/upscale.py``) and the
    serving warm path via :func:`prepare_image_inputs`.
    """
    sources = []
    for conn in topology.get("connections", []):
        src = conn.get("from", "")
        if src.startswith("global."):
            sources.append(src)
    uniq = sorted(set(sources))
    if not uniq:
        raise RuntimeError(
            "No `global.*` input connection found in topology. "
            "The container does not declare a user-facing image input."
        )
    if len(uniq) > 1:
        raise RuntimeError(
            f"Expected exactly one global input connection, found "
            f"{uniq}. Upscalers take a single image input."
        )
    return uniq[0]


def load_upscale_image(image_path: str, cache_path: Path):
    """PIL load → NCHW float32 array with container-declared preprocessing.

    Reads ``modules/processor/preprocessor_config.json`` for the
    rescale factor and pad alignment. Falls back to the standard
    1/255 rescale + multiple-of-8 reflect pad when the processor
    config is absent (the conventional SR defaults). Returns
    ``(array, (orig_h, orig_w))`` — the executor puts the array in the
    engine's container (R33: no torch here). Shared by ``nbx upscale`` and the
    serving warm path — one preprocessing, every entry point.
    """
    import json

    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    proc_cfg = {}
    proc_file = (cache_path / "modules" / "processor"
                 / "preprocessor_config.json")
    if proc_file.exists():
        with open(proc_file) as f:
            proc_cfg = json.load(f)

    rescale_factor = (
        proc_cfg.get("rescale_factor", 1.0 / 255.0)
        if proc_cfg.get("do_rescale", True)
        else 1.0
    )
    # Pad alignment, in declaration order: the processor config (the
    # transformers-route contract), then the topology's extracted
    # window_size (the pth-route contract — HAT's window is 16, and the
    # hardcoded 8 fed it a H=200 tensor that crashed the first window
    # view at any size that is a multiple of 8 but not 16; found by the
    # 2026-08-29 odd-size smoke). The literal 8 survives only as the
    # last-resort default for containers that declare neither.
    pad_size = None
    if proc_cfg.get("do_pad", True):
        pad_size = proc_cfg.get("pad_size")
    else:
        pad_size = 1
    if pad_size is None:
        topo_file = cache_path / "topology.json"
        if topo_file.exists():
            with open(topo_file) as f:
                _topo = json.load(f)
            for _comp_vals in (_topo.get("extracted_values") or {}).values():
                if isinstance(_comp_vals, dict) and "window_size" in _comp_vals:
                    _ws = _comp_vals["window_size"]
                    if isinstance(_ws, int) and _ws >= 1:
                        pad_size = _ws
                    break
    if pad_size is None:
        pad_size = 8

    arr = np.asarray(img, dtype=np.float32) * float(rescale_factor)
    # HWC → CHW → NCHW
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    # Pad H and W up to a multiple of `pad_size` (reflect padding — the
    # mirror without its edge, the Swin2SR image processor convention).
    _, _, h, w = tensor.shape
    pad_h = (pad_size - h % pad_size) % pad_size
    pad_w = (pad_size - w % pad_size) % pad_size
    if pad_h or pad_w:
        tensor = np.pad(tensor, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    return np.ascontiguousarray(tensor), (h, w)


def prepare_image_inputs(topology: dict, model_name: Optional[str],
                         image_path: str, cache_path: Path, *,
                         height: Optional[int] = None,
                         width: Optional[int] = None,
                         num_frames: Optional[int] = None) -> dict:
    """Image file → the ``global.*`` inputs dict, data-driven from the build.

    Single source of truth shared by the CLI cold path
    (``cli/commands/run.py``) and the serving warm path
    (``serving/engine.py``) — the output_dispatch pattern.

    The preprocessing TYPE is data-driven from the build:
    family=upscaler (read from the container manifest) takes the SR
    contract — the topology's unique input variable fed with the
    rescaled/padded NCHW float image (the ``nbx upscale`` cold-path
    preparation; before this branch existed the warm path fell
    through to the video contract and the upscaler family broke on
    serve warm, both engines — S3_FINDING_SERVE_UPSCALER_WARM,
    2026-08-26). A ``topology.flow.vlm`` block declares its own
    preprocessing (dynamic-resolution VLM); otherwise the video
    contract applies: ``global.image`` = I2V VAE-conditioning clip
    [1,3,T,H,W] in [-1,1] (T>1 zero-padded only when the model's
    vae_encoder declares ``pad_image_to_num_frames`` — Wan-I2V
    temporal-VAE class), and ``global.pixel_values`` = the CLIP view
    of the SAME image when the build embeds
    ``modules/image_processor/preprocessor_config.json``.
    Every value is an array; the executor puts it in the engine's
    container and the runtime resolver owns placement.
    """
    import json

    manifest_file = cache_path / "manifest.json"
    if manifest_file.exists():
        family = (json.loads(manifest_file.read_text()) or {}).get("family")
        if family == "upscaler":
            tensor, orig_hw = load_upscale_image(image_path, cache_path)
            # The private key travels WITH the inputs so the warm path
            # can crop the output to exactly orig x scale; callers pop
            # it before the runtime sees the dict.
            return {find_upscale_input_variable(topology): tensor,
                    "_upscale_orig_hw": orig_hw}

    inputs: dict = {}
    vlm_blk = (topology.get("flow", {}) or {}).get("vlm") or {}
    vlm_input = vlm_blk.get("input", {})
    if vlm_input.get("preprocessing"):
        vis = ImageInputProcessor.process(
            vlm_input["preprocessing"], image_path,
            preprocessor_config=(vlm_blk.get("preprocessing") or {}))
        if isinstance(vis, dict):
            for k, v in vis.items():
                inputs[f"global.{k}"] = v
        else:
            inputs[vlm_input.get("image_variable",
                                 "global.pixel_values")] = vis
    else:
        from neurobrix.core.runtime.registry_flags import get_component_flag
        pad_nf = 0
        if get_component_flag(model_name, "vae_encoder",
                              "pad_image_to_num_frames", default=False):
            pad_nf = int(num_frames or 0)
        inputs["global.image"] = ImageInputProcessor.process(
            "i2v_vae_condition", image_path,
            height=height, width=width, pad_to_num_frames=pad_nf)
        proc_cfg = (cache_path / "modules" / "image_processor"
                    / "preprocessor_config.json")
        if proc_cfg.exists():
            inputs["global.pixel_values"] = ImageInputProcessor.process(
                "clip_centercrop", image_path,
                preprocessor_config=json.loads(proc_cfg.read_text()))
    return inputs
