"""Resolution binning runs the vendor's own rules, in both branches.

The reference is the vendor's code itself, quoted verbatim below from diffusers v0.37.0
(src/diffusers/image_processor.py, `PixArtImageProcessor.classify_height_width_bin` and
`resize_and_crop_tensor`), with the vendor's own 1024 table (src/diffusers/pipelines/pixart_alpha/
pipeline_pixart_alpha.py, `ASPECT_RATIO_1024_BIN`) as Forge records it: ratio keys verbatim, in order.

What these tests would do if the code were wrong: a tie resolved to the last bin, a crop off by one,
align_corners flipped, a restore skipped, or the request's own size left bound, each fails an
assertion below (seen red on each, planted one at a time, 2026-09-27).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from neurobrix.core.runtime.resolution import resolution_binning as rb

torch = pytest.importorskip("torch")
F = torch.nn.functional

# --- the vendor, verbatim (diffusers v0.37.0) ---------------------------------------------------

def _vendor_classify_height_width_bin(height: int, width: int, ratios: dict):
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


def _vendor_resize_and_crop_tensor(samples, new_width: int, new_height: int):
    orig_height, orig_width = samples.shape[2], samples.shape[3]
    if orig_height != new_height or orig_width != new_width:
        ratio = max(new_height / orig_height, new_width / orig_width)
        resized_width = int(orig_width * ratio)
        resized_height = int(orig_height * ratio)
        samples = F.interpolate(
            samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )
        start_x = (resized_width - new_width) // 2
        end_x = start_x + new_width
        start_y = (resized_height - new_height) // 2
        end_y = start_y + new_height
        samples = samples[:, :, start_y:end_y, start_x:end_x]
    return samples


ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0], "0.28": [512.0, 1856.0], "0.32": [576.0, 1792.0], "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0], "0.4": [640.0, 1600.0], "0.42": [640.0, 1536.0], "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0], "0.52": [704.0, 1344.0], "0.57": [768.0, 1344.0], "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0], "0.72": [832.0, 1152.0], "0.78": [896.0, 1152.0], "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0], "0.94": [960.0, 1024.0], "1.0": [1024.0, 1024.0], "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0], "1.21": [1088.0, 896.0], "1.29": [1152.0, 896.0], "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0], "1.67": [1280.0, 768.0], "1.75": [1344.0, 768.0], "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0], "2.4": [1536.0, 640.0], "2.5": [1600.0, 640.0], "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}


def _field(bins=ASPECT_RATIO_1024_BIN, **over):
    f = {"default": True, "source": "PixArtAlphaPipeline.__call__(use_resolution_binning)", "bins": bins,
         "classify": "nearest_ratio", "restore": "cover_resize_center_crop",
         "interpolate": {"mode": "bilinear", "align_corners": False}}
    f.update(over)
    return {"flow": {"type": "iterative_process", "resolution_binning": f}}


CONTRACT = rb.read_contract(_field())

# Requests: square, portrait, landscape, the ones PixArt was seen failing at (512, 768x1024), odd sizes a
# user types, sizes off the 8-grid, and extremes past the table's ends. The last two are there for the
# vendor's int() on the resized extent: at (300, 565) the cover width is 572.7, so int() and round() give
# different images; none of the others tells them apart (checked 2026-09-27).
REQUESTS = [(512, 512), (768, 1024), (1024, 768), (1024, 1024), (500, 500), (333, 1000), (1000, 333),
            (720, 1280), (1080, 1920), (2048, 512), (100, 2000), (3000, 200), (257, 263), (1023, 1025),
            (300, 565), (565, 300)]


# --- classify -------------------------------------------------------------------------------------

def test_classify_is_the_vendors_over_a_dense_sweep():
    for h in range(64, 2561, 24):
        for w in range(64, 2561, 40):
            assert rb.classify(h, w, CONTRACT) == _vendor_classify_height_width_bin(h, w, ASPECT_RATIO_1024_BIN), (h, w)


def test_a_tie_goes_to_the_first_bin_in_the_table_order():
    bins = {"0.5": [100, 200], "1.5": [300, 200]}          # ratio 1.0 is exactly 0.5 from both keys
    c = rb.read_contract(_field(bins))
    assert rb.classify(10, 10, c) == _vendor_classify_height_width_bin(10, 10, bins) == (100, 200)
    reordered = {"1.5": [300, 200], "0.5": [100, 200]}
    c2 = rb.read_contract(_field(reordered))
    assert rb.classify(10, 10, c2) == _vendor_classify_height_width_bin(10, 10, reordered) == (300, 200)


# --- restore, the ATen branch --------------------------------------------------------------------

def _handler(binned_request, resolved):
    return SimpleNamespace(ctx=SimpleNamespace(binned_request=binned_request,
                                               variable_resolver=SimpleNamespace(resolved=resolved)))


def _binned(req):
    merged = {"height": req[0], "width": req[1]}
    br = rb.bin_request(_field(), merged)
    assert (merged["height"], merged["width"]) == br.binned
    return br


@pytest.mark.parametrize("req", REQUESTS)
def test_the_aten_restore_is_the_vendors_bit_for_bit(req):
    from neurobrix.core.flow.iterative_process import IterativeProcessHandler
    br = _binned(req)
    g = torch.Generator().manual_seed(req[0] * 7919 + req[1])
    image = torch.randn(2, 3, *br.binned, generator=g)
    resolved = {"vae.output_0": image, "vae.last_output": image, "transformer.output_0": image}
    IterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])
    want = _vendor_resize_and_crop_tensor(image, req[1], req[0])
    got = resolved["vae.output_0"]
    assert tuple(got.shape) == tuple(want.shape), (req, tuple(got.shape), tuple(want.shape))
    assert torch.equal(got, want), (req, (got - want).abs().max().item())
    assert resolved["vae.last_output"] is got                 # one tensor under two names, restored once
    assert resolved["transformer.output_0"] is image          # another component's output is not touched


def test_a_request_already_at_its_bin_is_left_as_decoded():
    from neurobrix.core.flow.iterative_process import IterativeProcessHandler
    br = _binned((1024, 1024))
    image = torch.randn(1, 3, 1024, 1024)
    resolved = {"vae.output_0": image}
    IterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])
    assert resolved["vae.output_0"] is image


def test_a_decoder_with_no_output_at_the_bin_is_refused():
    from neurobrix.core.flow.iterative_process import IterativeProcessHandler
    br = _binned((512, 512))
    resolved = {"vae.output_0": torch.randn(1, 3, 512, 512)}
    with pytest.raises(RuntimeError, match="no output at"):
        IterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])


def _vendor_video_resize_and_crop_tensor(samples, new_width: int, new_height: int):
    """diffusers v0.37.0 src/diffusers/video_processor.py, `VideoProcessor.resize_and_crop_tensor`, verbatim."""
    orig_height, orig_width = samples.shape[3], samples.shape[4]
    if orig_height != new_height or orig_width != new_width:
        ratio = max(new_height / orig_height, new_width / orig_width)
        resized_width = int(orig_width * ratio)
        resized_height = int(orig_height * ratio)
        n, c, t, h, w = samples.shape
        samples = samples.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
        samples = F.interpolate(samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        start_x = (resized_width - new_width) // 2
        end_x = start_x + new_width
        start_y = (resized_height - new_height) // 2
        end_y = start_y + new_height
        samples = samples[:, :, start_y:end_y, start_x:end_x]
        samples = samples.reshape(n, t, c, new_height, new_width).permute(0, 2, 1, 3, 4)
    return samples


ASPECT_RATIO_720_BIN = {    # diffusers 0.38.0.dev0 pipeline_sana_video.py:61 (SANA-Video 720p, sample_size 22)
    "0.5": [672.0, 1344.0], "0.57": [704.0, 1280.0], "0.68": [800.0, 1152.0], "0.78": [832.0, 1088.0],
    "0.88": [896.0, 1024.0], "1.0": [960.0, 960.0], "1.13": [1024.0, 896.0], "1.29": [1088.0, 832.0],
    "1.46": [1152.0, 800.0], "1.75": [1280.0, 704.0], "2.0": [1344.0, 672.0],
}
VIDEO_REQUESTS = [(256, 640), (704, 1280), (480, 480)]


def _binned_video(req):
    merged = {"height": req[0], "width": req[1]}
    return rb.bin_request(_field(bins=ASPECT_RATIO_720_BIN, source="SanaVideoPipeline.__call__(use_resolution_binning)"), merged)


def test_the_fold_names_the_planes_of_an_image_and_of_a_video():
    assert rb.restore_fold((2, 3, 704, 1408), (704, 1408)) == (2, 3, 704, 1408)
    assert rb.restore_fold((1, 3, 81, 672, 1344), (672, 1344)) == (3, 81, 672, 1344)
    assert rb.restore_fold((1, 81, 3, 672, 1344), (672, 1344)) == (81, 3, 672, 1344)
    assert rb.restore_fold((3, 672, 1344), (672, 1344)) is None            # fewer than four axes
    assert rb.restore_fold((1, 3, 81, 672, 1344), (704, 1280)) is None     # not at the bin


@pytest.mark.parametrize("req", VIDEO_REQUESTS)
def test_the_aten_restore_of_a_video_is_the_vendors_frame_for_frame(req):
    """A 5-D decoder output [N, C, T, H, W] at the bin: the vendor's VideoProcessor restore, bit for bit.
    On the 4-D-only form this is refused as 'no 4-D output'."""
    from neurobrix.core.flow.iterative_process import IterativeProcessHandler
    br = _binned_video(req)
    g = torch.Generator().manual_seed(req[0] * 131 + req[1])
    video = torch.randn(1, 3, 5, *br.binned, generator=g)
    resolved = {"vae.output_0": video, "vae.last_output": video}
    IterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])
    want = _vendor_video_resize_and_crop_tensor(video, req[1], req[0])
    got = resolved["vae.output_0"]
    assert tuple(got.shape) == tuple(want.shape) == (1, 3, 5, req[0], req[1]), (req, tuple(got.shape))
    assert torch.equal(got, want), (req, (got - want).abs().max().item())
    assert resolved["vae.last_output"] is got


def test_the_aten_restore_of_a_video_in_another_axis_order_restores_every_frame():
    """[N, T, C, H, W]: the fold does not care which leading axis is which."""
    from neurobrix.core.flow.iterative_process import IterativeProcessHandler
    br = _binned_video((256, 640))
    video = torch.randn(1, 5, 3, *br.binned, generator=torch.Generator().manual_seed(7))
    resolved = {"vae.output_0": video}
    IterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])
    want = _vendor_video_resize_and_crop_tensor(video.permute(0, 2, 1, 3, 4), 640, 256).permute(0, 2, 1, 3, 4)
    assert torch.equal(resolved["vae.output_0"], want)


# --- restore, the Triton branch -----------------------------------------------------------------

def _gpu():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        return DeviceAllocator.device_count() > 0
    except Exception:                                     # pragma: no cover
        return False


@pytest.mark.skipif(not _gpu(), reason="needs a device")
@pytest.mark.parametrize("req", REQUESTS)
def test_the_triton_restore_is_the_vendors(req):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.triton.flow.iterative_process import TritonIterativeProcessHandler
    br = _binned(req)
    rng = np.random.default_rng(req[0] * 7919 + req[1])
    host = rng.standard_normal((1, 3, *br.binned)).astype(np.float32)
    image = NBXTensor.from_numpy(host)                     # on the device
    resolved = {"vae.output_0": image}
    TritonIterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])
    got = np.asarray(resolved["vae.output_0"].numpy(), dtype=np.float64)
    want = _vendor_resize_and_crop_tensor(torch.from_numpy(host), req[1], req[0]).double().numpy()
    exact = _vendor_resize_and_crop_tensor(torch.from_numpy(host).double(), req[1], req[0]).numpy()
    assert got.shape == want.shape, (req, got.shape, want.shape)
    # The yardstick is the vendor's own fp32 error, against the vendor evaluated in fp64. Measured 2026-09-27
    # on this M4 Pro over the 16 requests: the vendor in fp32 sits up to 7.4e-4 from fp64, and the Triton
    # restore sits at the same distance (ratio at most 1.002). The two fp32 paths do NOT agree with each other
    # more closely than that (up to 3.4e-4 apart): each rounds its source coordinate in fp32 at a different
    # point, and at a coordinate near 1300 one fp32 step is 1.2e-4 pixel, which moves a value by up to about
    # 6e-4. A wrong mapping (align_corners, a half-pixel offset, a crop) misses by 0.1 to 1.
    slack = 4 * float(np.spacing(np.float32(np.abs(host).max())))
    vendor_err = float(np.abs(want - exact).max())
    ours_err = float(np.abs(got - exact).max())
    assert ours_err <= 1.25 * vendor_err + slack, (req, ours_err, vendor_err)


@pytest.mark.skipif(not _gpu(), reason="needs a device")
@pytest.mark.parametrize("req", VIDEO_REQUESTS)
def test_the_triton_restore_of_a_video_is_the_vendors(req):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.triton.flow.iterative_process import TritonIterativeProcessHandler
    br = _binned_video(req)
    rng = np.random.default_rng(req[0] * 131 + req[1])
    host = rng.standard_normal((1, 3, 5, *br.binned)).astype(np.float32)
    resolved = {"vae.output_0": NBXTensor.from_numpy(host)}
    TritonIterativeProcessHandler._restore_requested_resolution(_handler(br, resolved), ["vae"])
    got = np.asarray(resolved["vae.output_0"].numpy(), dtype=np.float64)
    want = _vendor_video_resize_and_crop_tensor(torch.from_numpy(host), req[1], req[0]).double().numpy()
    exact = _vendor_video_resize_and_crop_tensor(torch.from_numpy(host).double(), req[1], req[0]).numpy()
    assert got.shape == want.shape == (1, 3, 5, req[0], req[1]), (req, got.shape, want.shape)
    slack = 4 * float(np.spacing(np.float32(np.abs(host).max())))
    vendor_err = float(np.abs(want - exact).max())
    ours_err = float(np.abs(got - exact).max())
    assert ours_err <= 1.25 * vendor_err + slack, (req, ours_err, vendor_err)


# --- the contract -------------------------------------------------------------------------------

@pytest.mark.parametrize("over, match", [
    ({"classify": "nearest_area"}, "classify"),
    ({"restore": "letterbox"}, "restore"),
    ({"interpolate": {"mode": "bicubic", "align_corners": False}}, "interpolate"),
    ({"interpolate": {"mode": "bilinear", "align_corners": True}}, "interpolate"),
    ({"bins": {}}, "non-empty"),
    ({"bins": {"one": [1, 2]}}, "not '<ratio>'"),
    ({"bins": None}, "no 'bins'"),
])
def test_a_contract_the_engine_cannot_execute_is_refused(over, match):
    with pytest.raises(RuntimeError, match=match):
        rb.read_contract(_field(**over))


def test_default_false_without_bins_runs_unbinned_and_refuses_a_request_for_binning():
    topo = {"flow": {"resolution_binning": {"default": False, "source": "X.__call__(use_resolution_binning)"}}}
    c = rb.read_contract(topo)
    assert rb.is_active(c) is False
    assert rb.bin_request(topo, {"height": 512, "width": 512}) is None
    with pytest.raises(RuntimeError, match="no bins were recorded"):
        rb.is_active(c, True)


def test_a_container_without_the_field_is_not_binned():
    assert rb.bin_request({"flow": {"type": "iterative_process"}}, {"height": 512, "width": 512}) is None
    with pytest.raises(RuntimeError, match="records none"):
        rb.is_active(None, True)


def test_the_request_can_turn_binning_off():
    merged = {"height": 512, "width": 512}
    assert rb.bin_request(_field(), merged, False) is None and merged == {"height": 512, "width": 512}


def test_a_binned_request_rebinds_its_own_size_and_keeps_the_rest():
    merged = {"height": 512, "width": 768}
    br = rb.bin_request(_field(), merged)
    assert br.requested == (512, 768) and br.binned == (832, 1216) == (merged["height"], merged["width"])
    inputs = {"global.height": 512, "global.width": 768, "global.prompt": "an apple"}
    out = rb.rebind_request_size(inputs, br)
    assert out == {"global.height": 832, "global.width": 1216, "global.prompt": "an apple"}
    assert inputs["global.height"] == 512                       # the caller's dict is not mutated


def test_binning_without_a_size_is_refused():
    with pytest.raises(RuntimeError, match="no height/width"):
        rb.bin_request(_field(), {"height": None, "width": 512})


def test_the_bins_are_integers_whatever_the_table_wrote():
    assert all(isinstance(h, int) and isinstance(w, int) for _, h, w in CONTRACT.bins)
    assert math.isclose(float(CONTRACT.bins[0][0]), 0.25)
