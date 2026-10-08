"""The TilingEngine owns every memory-split size — and moving them there changed no number.

Every size a memory split is cut by (a conv3d chunk, a conv2d band, an attention chunk, Prism's
op-level band factors and op budget, the tile overlaps, the in-place thresholds) is computed by
`core/module/tiling_sizes.py` from `config/tiling.yml`. These expectations were RECORDED on the
tree before the move (4dd80924), by calling the functions that computed the sizes there, and the
move kept every literal below: a size that moved is a size this test sees.

What this test does if the code were wrong: any value of `tiling.yml` changed, or any formula
re-written with another rounding, turns at least one expected literal red (seen red with
`conv3d.chunk_bytes` halved and `overlap.spatial_divisor` 8 -> 6, then green again once restored).
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from neurobrix.core.module import tiling_sizes as TS
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import ComponentMemory
from neurobrix.kernels import launch_keys as LK

MB = 1024 ** 2
GiB = 1024 ** 3


# --- conv3d: the one-shot peak and the chunk ------------------------------------------------------

CONV3D_CASES = [
    # (x_shape, w_shape, stride, padding, dilation, in_bytes, out_bytes) -> (need, frame)
    (([1, 128, 5, 64, 64], [128, 128, 3, 3, 3], 1, 1, 1, 2, 2), (0, 1048576)),        # under CHUNK
    (([1, 256, 13, 240, 360], [256, 256, 3, 3, 3], 1, 1, 1, 4, 4),                     # over CHUNK
     (5046009856, 88473600)),
    (([1, 128, 17, 480, 720], [512, 128, 3, 3, 3], 1, 1, 1, 4, 4),                     # fold_out > BAND
     (39727661056, 707788800)),
    (([2, 64, 9, 1024, 1024], [64, 64, 1, 3, 3], [1, 2, 2], [0, 1, 1], 1, 2, 2),       # stride 2, kt 1
     (3288334336, 268435456)),
    (([1, 16, 1, 64, 64], [16, 16, 3, 3, 3], 1, 0, 1, 2, 2), (0, 0)),                  # no output frame
]


@pytest.mark.parametrize("args,expected", CONV3D_CASES)
def test_the_conv3d_one_shot_peak_is_unchanged(args, expected):
    assert TS.conv3d_need(*args) == expected


@pytest.mark.parametrize("frame,expected", [(1, 1073741824), (1000, 1073741), (10 ** 8, 10),
                                            (2 ** 31, 1), (0, 1073741824)])
def test_the_conv3d_chunk_is_unchanged(frame, expected):
    assert TS.chunk_frames(frame) == expected


def test_the_conv_thresholds_are_unchanged():
    assert TS.conv3d_chunk_bytes() == 1073741824
    assert TS.conv2d_band_bytes() == 4294967296


# --- conv2d: the band cut -------------------------------------------------------------------------

@pytest.mark.parametrize("args,expected", [
    ((1, 256, 64, 1 << 16, 4, 4 * GiB), 32),
    ((1, 128, 4096, 4096, 2, 4 * GiB), 2048),
    ((2, 64, 8192, 8192, 4, 4 * GiB), 512),
    ((1, 512, 1000, 3000, 2, 4 * GiB), 500),
])
def test_the_conv2d_band_rows_are_unchanged(args, expected):
    assert LK.conv2d_band_rows(*args) == expected


def test_a_conv2d_row_over_the_band_is_still_refused():
    with pytest.raises(LK.ConvRowOverBand):
        LK.conv2d_band_rows(1, 256, 1, 1 << 25, 2, 4 * GiB)


# --- attention: the chunk rows, the route, the scores budget --------------------------------------

@pytest.mark.parametrize("args,expected", [
    ((2 << 30, 1, 32, 16384, 16384, 64, 32), 1024),
    ((2 << 30, 1, 32, 4096, 4096, 64, 32), 4096),
    ((1 << 30, 2, 40, 65536, 65536, 64, 32), 0),          # over the chunk ceiling
    ((2 << 30, 1, 32, 8192, 8192, 0, 32), 0),             # the arch declares no row block
])
def test_the_sdpa_chunk_rows_are_unchanged(args, expected):
    assert LK.sdpa_chunk_rows(*args) == expected


@pytest.mark.parametrize("args,expected", [
    ((1, 32, 4096, 4096, 128, 128, 2 << 30, 64, 32), ("math", 0)),
    ((1, 32, 16384, 16384, 128, 128, 2 << 30, 64, 32), ("chunked", 1024)),
    ((1, 32, 16384, 16384, 80, 80, 0, 64, 32), ("chunked", 1024)),     # non-pow2 head, no budget
    ((1, 16, 4096, 4096, 80, 80, 0, 64, 32), ("math", 0)),             # under the non-pow2 bound
    ((1, 32, 16384, 16384, 128, 128, 0, 64, 32), ("flash", 0)),        # pow2 head, no budget
    ((1, 32, 4096, 4096, 128, 64, 2 << 30, 64, 32), ("math", 0)),      # head dims differ
])
def test_the_sdpa_route_is_unchanged(args, expected):
    assert LK.sdpa_route(*args) == expected


def test_the_sdpa_scores_budget_for_a_device_is_unchanged(monkeypatch):
    from neurobrix.kernels import wrappers as W
    devs = [SimpleNamespace(index=0, memory_mb=16384), SimpleNamespace(index=2, memory_mb=32768)]
    monkeypatch.setattr(W, "get_hardware_profile", lambda: SimpleNamespace(devices=devs))
    monkeypatch.setattr(W, "_sdpa_math_scores_budget_bytes", lambda: 2 << 30)
    monkeypatch.setattr(W, "_sdpa_math_scores_device_fraction", lambda: 0.066)
    assert W._sdpa_math_scores_budget_bytes_for(0) == 1133871366
    assert W._sdpa_math_scores_budget_bytes_for(2) == 2147483648
    assert W._sdpa_math_scores_budget_bytes_for(None) == 2147483648
    assert W._sdpa_math_scores_budget_bytes_for(7) == 2147483648
    monkeypatch.setattr(W, "_sdpa_math_scores_budget_bytes", lambda: 0)
    assert W._sdpa_math_scores_budget_bytes_for(0) == 0


# --- Prism's op-level band factors and op budget --------------------------------------------------

class _Ap:
    def __init__(self, overflow_ops):
        self.overflow_ops = overflow_ops


def _t(shape):
    return {"shape": list(shape)}


def _op_level_graph():
    """An upsample feeding a conv (a fusion pair), two standalone overflowing convs, two rms_norms
    (one over the 0.20 threshold, one under), and a rank-5 conv over the op budget."""
    tensors = {
        "u_in": _t([1, 128, 1024, 1024]), "u_out": _t([1, 128, 2048, 2048]),
        "c_w": _t([128, 128, 3, 3]), "c_out": _t([1, 128, 2048, 2048]),
        "s_in": _t([1, 128, 2048, 2048]), "s_w": _t([128, 128, 3, 3]), "s_out": _t([1, 128, 2048, 2048]),
        "s2_in": _t([1, 64, 512, 512]), "s2_w": _t([64, 64, 3, 3]), "s2_out": _t([1, 64, 512, 512]),
        "r_out": _t([1, 2048, 2048, 512]), "r2_out": _t([1, 64, 64, 64]),
        "v_in": _t([1, 128, 81, 480, 832]), "v_w": _t([128, 128, 3, 3, 3]), "v_out": _t([1, 128, 81, 480, 832]),
    }
    ops = {
        "up": {"op_type": "aten::upsample_nearest2d", "input_tensor_ids": ["u_in"], "output_tensor_ids": ["u_out"]},
        "conv": {"op_type": "aten::convolution", "input_tensor_ids": ["u_out", "c_w"], "output_tensor_ids": ["c_out"],
                 "input_shapes": [[1, 128, 2048, 2048], [128, 128, 3, 3]]},
        "solo": {"op_type": "aten::convolution", "input_tensor_ids": ["s_in", "s_w"], "output_tensor_ids": ["s_out"],
                 "input_shapes": [[1, 128, 2048, 2048], [128, 128, 3, 3]]},
        "solo2": {"op_type": "aten::convolution", "input_tensor_ids": ["s2_in", "s2_w"], "output_tensor_ids": ["s2_out"],
                  "input_shapes": [[1, 64, 512, 512], [64, 64, 3, 3]]},
        "rms": {"op_type": "custom::rms_norm", "input_tensor_ids": ["s_out"], "output_tensor_ids": ["r_out"]},
        "rms2": {"op_type": "custom::rms_norm", "input_tensor_ids": ["s2_out"], "output_tensor_ids": ["r2_out"]},
        "vconv": {"op_type": "aten::convolution", "input_tensor_ids": ["v_in", "v_w"], "output_tensor_ids": ["v_out"],
                  "attributes": {"stride": [1, 1, 1], "padding": [1, 1, 1], "dilation": [1, 1, 1]}},
    }
    return {"tensors": tensors, "ops": ops, "execution_order": list(ops)}


def _op_level_plan(monkeypatch, card_mb, resident_mb, chain_bytes_fp32):
    import neurobrix.core.prism.profiler as P
    import neurobrix.core.prism.memory_estimator as ME
    seen = {}

    class _Profiler:
        def __init__(self, graph):
            self.graph = graph

        def estimate_peak_memory(self, **kw):
            seen["safety"] = kw["safety"]
            return _Ap([("conv", "aten::convolution", 2 * GiB, 3 * GiB, []),
                        ("solo", "aten::convolution", 2 * GiB, 9 * GiB, []),
                        ("solo2", "aten::convolution", 64 * MB, 300 * GiB, [])])

        def build_symbol_map(self, ic):
            return {}

        def _resolve_shape(self, meta, symbols):
            return meta["shape"]

        def _compute_size(self, shape, meta, dtype_bytes):
            n = 1
            for d in shape:
                n *= int(d)
            return n * dtype_bytes

    monkeypatch.setattr(P, "ActivationProfiler", _Profiler)
    monkeypatch.setattr(ME, "estimate_op_workspace_bytes", lambda *a, **k: 5 * GiB)
    s = PrismSolver()
    monkeypatch.setattr(s, "_identify_residual_chain_specs",
                        lambda g: [{"fork_uid": "f", "merge_uid": "m", "chain_uids": [], "halo": 2,
                                    "bytes_fp32": chain_bytes_fp32}])
    comp = SimpleNamespace(name="vae", graph=_op_level_graph())
    dev = SimpleNamespace(memory_mb=card_mb, get_device_string=lambda: "cuda:0")
    alloc = SimpleNamespace(device="cuda:0", memory_mb=resident_mb, shard_map={})
    plans = s._detect_op_level_tiling_pairs(
        container=None, components=[comp], allocations={"vae": alloc},
        profile=SimpleNamespace(devices=[dev]), input_config=None, target_dtype_str="float16")
    return plans["vae"], seen["safety"]


@pytest.mark.parametrize("card_mb,resident_mb,chain,expected", [
    (16384, 300, 20 * GiB,
     {"fusion": [("up", "conv", 2)], "tiled": [("solo", "aten::convolution", 2),
                                                ("solo2", "aten::convolution", 32),
                                                ("rms", "custom::rms_norm", 4)],
      "chain_tf": 16, "conv3d": ["vconv"]}),
    (32768, 0, 200 * GiB,                                   # chain factor at its cap
     {"fusion": [("up", "conv", 2)], "tiled": [("solo", "aten::convolution", 2),
                                                ("solo2", "aten::convolution", 16)],
      "chain_tf": 32, "conv3d": ["vconv"]}),
    (4096, 0, 1 * GiB,                                      # conv factor at its cap, rms at its cap
     {"fusion": [("up", "conv", 8)], "tiled": [("solo", "aten::convolution", 32),
                                                ("solo2", "aten::convolution", 64),
                                                ("rms", "custom::rms_norm", 8)],
      "chain_tf": 2, "conv3d": ["vconv"]}),
    (3072, 0, 3 * GiB,                                      # the band budget at its floor
     {"fusion": [("up", "conv", 8)], "tiled": [("solo", "aten::convolution", 64),
                                                ("solo2", "aten::convolution", 64),
                                                ("rms", "custom::rms_norm", 8)],
      "chain_tf": 2, "conv3d": ["vconv"]}),
])
def test_the_op_level_band_factors_are_unchanged(monkeypatch, card_mb, resident_mb, chain, expected):
    plan, safety = _op_level_plan(monkeypatch, card_mb, resident_mb, chain)
    assert safety == 0.85
    assert plan.fusion_pairs == expected["fusion"]
    assert plan.tiled_ops == expected["tiled"]
    assert [c["tile_factor"] for c in plan.residual_chains] == [expected["chain_tf"]]
    assert plan.conv3d_chunks == expected["conv3d"]


def test_the_op_budget_is_unchanged():
    assert TS.op_budget_fraction() == 0.85
    assert TS.op_budget_bytes(16384 * MB, 300 * MB) == 14288316006   # int(0.85 * card) - resident, the solver's form before the move


# --- overlaps -------------------------------------------------------------------------------------

def _sym(i, trace):
    return {"type": "symbol", "id": i, "trace": trace}


def _sop(kind, left, right, trace):
    return {"type": kind, "left": left, "right": right, "trace": trace}


def _decoder_graph():
    t, h, w = 5, 32, 48
    b, st, sh, sw = _sym("s0", 1), _sym("s1", t), _sym("s2", h), _sym("s3", w)
    return {"input_tensor_ids": ["x"], "output_tensor_ids": ["z"], "ops": {}, "execution_order": [],
            "tensors": {
                "x": {"shape": [1, 4, t, h, w], "symbolic_shape": {"dims": [b, 4, st, sh, sw]}},
                "z": {"shape": [1, 3, 4 * t, 8 * h, 8 * w],
                      "symbolic_shape": {"dims": [b, 3, _sop("mul", st, 4, 4 * t), _sop("mul", sh, 8, 8 * h),
                                                  _sop("mul", sw, 8, 8 * w)]}}}}


def _encoder_graph():
    t, h, w = 20, 112, 176
    b, st, sh, sw = _sym("s0", 1), _sym("s1", t), _sym("s2", h), _sym("s3", w)
    return {"input_tensor_ids": ["x"], "output_tensor_ids": ["z"], "ops": {}, "execution_order": [],
            "tensors": {
                "x": {"shape": [1, 3, t, h, w], "symbolic_shape": {"dims": [b, 3, st, sh, sw]}},
                "z": {"shape": [1, 5, 4, h // 8, w // 8],
                      "symbolic_shape": {"dims": [b, _sop("floordiv", st, 4, 5), 4, _sop("floordiv", sh, 8, 14),
                                                  _sop("floordiv", sw, 8, 22)]}}}}


def _component_spec(tmp_path, name, graph, config, frames, activation_bytes):
    comp = tmp_path / name / "components" / name
    comp.mkdir(parents=True)
    (comp / "graph.json").write_text(json.dumps(graph))
    (comp / "profile.json").write_text(json.dumps({"config": config}))
    s = PrismSolver()
    s._input_config = InputConfig(batch_size=1, height=720, width=1280, num_frames=frames,
                                  temporal_compression=4)
    mem = ComponentMemory(component_name=name, weight_bytes=244 * MB, activation_bytes=activation_bytes,
                          overhead_bytes=0)
    return s._spatial_component_tiling(SimpleNamespace(_cache_path=tmp_path / name), name, mem, 32768)


@pytest.mark.parametrize("frames,act_gib,expected", [
    (88, 300, {"tile_size": 351, "overlap": 43, "t_tile": 7, "t_overlap": 1}),
    (400, 3000, {"tile_size": 156, "overlap": 19, "t_tile": 16, "t_overlap": 4}),
])
def test_the_decode_tile_overlaps_are_unchanged(tmp_path, frames, act_gib, expected):
    spec = _component_spec(tmp_path, "dec", _decoder_graph(), {"decoder_block_out_channels": [1, 2, 3, 4]},
                           frames, act_gib * GiB)
    assert {k: spec[k] for k in expected} == expected


def test_the_encode_tile_overlaps_are_unchanged(tmp_path):
    spec = _component_spec(tmp_path, "enc", _encoder_graph(), {"block_out_channels": [1, 2, 3, 4]},
                           88, 319_500 * MB)
    assert {k: spec[k] for k in ("tile_size", "overlap", "t_tile", "t_overlap")} == \
        {"tile_size": 344, "overlap": 48, "t_tile": 28, "t_overlap": 4}


@pytest.mark.parametrize("trace,window,expected", [(64, 1, 8), (16, 1, 4), (64, 6, 16), (200, 1, 25)])
def test_the_tiling_engine_overlap_is_unchanged(tmp_path, trace, window, expected):
    from neurobrix.core.module.tiling_engine import TilingEngine
    graph = {"input_tensor_ids": ["x"], "output_tensor_ids": ["y"],
             "tensors": {"x": {"shape": [1, 3, trace, trace]}, "y": {"shape": [1, 3, 4 * trace, 4 * trace]}}}
    (tmp_path / "graph.json").write_text(json.dumps(graph))
    (tmp_path / "profile.json").write_text(json.dumps({"config": {"upscale": 4, "window_size": window}}))
    eng = TilingEngine.from_component_config(tmp_path / "graph.json", tmp_path / "profile.json")
    assert eng.overlap == expected


def test_the_in_place_and_chain_sizes_are_unchanged():
    assert TS.inplace_min_bytes() == 1073741824
    assert TS.residual_chain_min_base_bytes_fp32() == 1600000000
    assert TS.residual_chain_default_tile_factor() == 4
    assert TS.residual_chain_default_halo() == 2



def test_a_refused_attention_chunk_is_an_error_naming_its_key_never_a_smaller_retry(monkeypatch):
    """ZERO FALLBACK: the chunked SDPA's rows are the TilingEngine's, fixed before the launch.
    An allocation failure re-raises naming the key (op, shape, rows, bytes asked); there is no
    second launch at fewer rows (the 2026-09-02 halving retry is gone)."""
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import DeviceOOMError

    class _T:
        def __init__(self, shape):
            self.shape = shape
            self._device_idx = 0
            self.nbx_dtype = None
        def __getitem__(self, _ix):
            return self
        def contiguous(self):
            return self

    launches = []

    def _refused(q_c, k, v, attn_mask=None, is_causal=False, scale=None):
        launches.append(1)
        raise DeviceOOMError("GPU malloc failed (injected)", requested=4 * GiB, device_idx=0)

    monkeypatch.setattr(W, "NBXTensor", SimpleNamespace(empty=lambda *a, **k: _T((2, 8, 4096, 64))))
    monkeypatch.setattr(W, "_math_attention", _refused)
    monkeypatch.setattr(W, "_sdpa_math_min_chunk_rows", lambda: 128)   # a halving retry would fire
    q, kv = _T((2, 8, 4096, 64)), _T((2, 8, 4096, 64))
    with pytest.raises(DeviceOOMError) as e:
        W._math_attention_chunked(q, kv, kv, None, False, 0.125, 1024)
    msg = str(e.value)
    assert "chunked SDPA: q [2, 8, 4096, 64]" in msg and "1024 rows a chunk" in msg, msg
    assert f"{2 * 8 * 1024 * 4096 * 4} bytes of fp32 scores asked" in msg, msg
    assert e.value.requested == 4 * GiB
    assert len(launches) == 1, f"{len(launches)} launches: a refused chunk was retried"


def test_a_missing_key_is_refused_by_name(monkeypatch):
    from neurobrix.core.config import loader
    policy = loader.get_tiling_policy()
    stripped = {**policy, "conv3d": {k: v for k, v in policy["conv3d"].items() if k != "chunk_bytes"}}
    monkeypatch.setattr(loader, "get_tiling_policy", lambda: stripped)
    with pytest.raises(KeyError, match="states no conv3d.chunk_bytes"):
        TS.conv3d_chunk_bytes()
    monkeypatch.setattr(loader, "get_tiling_policy", lambda: {})
    with pytest.raises(KeyError, match="states no overlap"):
        TS.spatial_overlap(64)

# --- the sizing half imports no torch -------------------------------------------------------------

def test_the_sizing_half_imports_no_torch():
    import subprocess
    from pathlib import Path
    src = str(Path(__file__).resolve().parents[3] / "src")
    probe = ("import sys; import neurobrix.core.module.tiling_sizes as T; T.conv3d_chunk_bytes(); "
             "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                         env={"PYTHONPATH": src, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout
