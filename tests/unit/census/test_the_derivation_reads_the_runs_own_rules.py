"""The derived census binds a run's extents and dtypes through the functions the run itself
calls — never through copies. These are the rules measured against the census walk
(2026-09-29): Wan's denoiser reads its text at the FINALIZED length (512, not the encoder's
226), and the attention wrapper aligns disagreeing operands to fp32 before it launches.

Injections, each seen RED: `finalized_text_length` returning `encoded` -> the Wan and Sana
cases fail; `sdpa_operand_dtypes` returning its inputs unchanged -> the mixed case fails."""
import pytest

from neurobrix.core.components.handlers.text_encoder_handler import finalized_text_length
from neurobrix.kernels import launch_keys as LK
from neurobrix.kernels.nbx_tensor import NBXDtype

F16, F32 = NBXDtype.float16, NBXDtype.float32


def test_the_text_axis_is_finalized_as_the_handler_finalizes_it():
    wan = {"max_sequence_length": 512, "zero_pad_embeddings": True}
    sana = {"max_sequence_length": 300, "complex_human_instruction": ["..."]}
    assert finalized_text_length(wan, 226) == 512          # padded up to the design length
    assert finalized_text_length(wan, 600) == 600          # never cut by the pad flag
    assert finalized_text_length(sana, 506) == 300         # the CHI prefix sliced away
    assert finalized_text_length(sana, 120) == 120
    assert finalized_text_length({"max_sequence_length": 77}, 120) == 120   # no flag: as encoded
    assert finalized_text_length(None, 23) == 23
    with pytest.raises(RuntimeError, match="max_sequence_length"):
        finalized_text_length({"zero_pad_embeddings": True}, 226)


def test_the_handler_produces_the_length_the_rule_names():
    """`finalize_embeddings` refuses to hand on a length the rule does not name — the door
    that keeps the census's binding and the run's tensor one number."""
    torch = pytest.importorskip("torch")
    from neurobrix.core.components.handlers.text_encoder_handler import TextEncoderComponentHandler
    h = TextEncoderComponentHandler.__new__(TextEncoderComponentHandler)
    out = h.finalize_embeddings(hidden_state=torch.ones(1, 226, 8),
                                attention_mask=torch.ones(1, 226, dtype=torch.long),
                                tokenizer_config={"max_sequence_length": 512,
                                                  "zero_pad_embeddings": True})
    assert out["hidden_state"].shape[1] == 512 and out["attention_mask"].shape[1] == 512


def test_disagreeing_attention_operands_are_aligned_to_fp32():
    assert LK.sdpa_operand_dtypes(F16, F16, F32) == (F32, F32, F32, None)
    assert LK.sdpa_operand_dtypes(F16, F16, F16) == (F16, F16, F16, None)
    # a KV-cache rounding judges Q at the cache's dtype, and survives agreement
    assert LK.sdpa_operand_dtypes(F32, F16, F16, F16) == (F32, F16, F16, F16)


def test_a_conv_row_over_the_band_budget_is_refused_not_recursed():
    """`conv2d_band_rows` is the one band cut of `_conv2d_band_streamed` and of the derived
    census: a single output row over the budget cannot be banded, and the per-band recursion
    called itself on the same row until Python's stack ran out (the derivation on orpheus's codec
    and on Sana-4K's contradicting annotation, the Mac and this rack, 2026-09-29). Now a named
    refusal. Injection: the refusal removed -> the 1-row case returns 1 and the launch
    recursion ends in RecursionError, RED."""
    GiB = 1 << 30
    assert LK.conv2d_band_rows(1, 256, 64, 1 << 16, 4, 4 * GiB) < 64      # rows split
    with pytest.raises(LK.ConvRowOverBand, match="one row is"):
        LK.conv2d_band_rows(1, 256, 1, 1 << 25, 2, 4 * GiB)
    with pytest.raises(LK.ConvRowOverBand):
        LK.conv2d_launches(1, 256, 1, 1 << 25, 256, 1, 7, 1, 1, 0, 3, 1, 1, 1,
                           F16, F16, F16, 4 * GiB)


@pytest.mark.parametrize("IH,kh,sh,dh,ph,tf", [(37, 3, 1, 1, 1, 4), (64, 3, 2, 1, 1, 3), (50, 5, 1, 2, 4, 5),
                                               (29, 1, 1, 1, 0, 2)])
def test_the_tiled_conv_band_cut_rebuilds_the_whole_conv(IH, kh, sh, dh, ph, tf):
    """`launch_keys.tiled_conv2d_bands` is the band cut `_tiled_conv2d_spatial_nbx` iterates (and
    the derived census keys the tiled conv with): a convolution rebuilt band by band from it —
    each band read with its halo, padded only at the image edges, run at padding 0, its halo
    rows skipped — equals the whole convolution exactly. Injection: pad_h added again on the edge
    bands (the 2026-05-14 defect) -> the rebuilt rows shift, RED; the halo unrounded -> the
    stride-2 case, RED (the defect this test found)."""
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, IH, 11, generator=g, dtype=torch.float64)
    w = torch.randn(4, 3, kh, 3, generator=g, dtype=torch.float64)
    full = F.conv2d(x, w, stride=(sh, 1), padding=(ph, 1), dilation=(dh, 1))
    out = torch.empty_like(full)
    for o0, o1, s0, s1, pt, pb, skip in LK.tiled_conv2d_bands(IH, full.shape[2], kh, sh, dh, ph, tf):
        band = F.pad(x[:, :, s0:s1, :], (1, 1, pt, pb))
        cb = F.conv2d(band, w, stride=(sh, 1), padding=0, dilation=(dh, 1))
        n = min(o1 - o0, cb.shape[2] - skip)
        out[:, :, o0:o0 + n] = cb[:, :, skip:skip + n]
    assert torch.equal(out, full)


@pytest.mark.parametrize("sh,kh", [(1, 3), (2, 3), (2, 5), (3, 3)])
def test_the_torch_tiled_conv_equals_the_whole_conv_at_any_stride(sh, kh):
    """The compiled engine's twin (`_tiled_conv2d_spatial_torch`), whose arithmetic mirrors the
    band cut: equal to F.conv2d at stride 1, 2 and 3. Before 2026-09-29 every stride-2 internal
    band was shifted by half an output row (the halo skipped as output rows). Injection: the
    halo left unrounded -> the stride-2 cases differ, RED."""
    torch = pytest.importorskip("torch")
    from neurobrix.kernels.ops.fused_upsample_conv import _tiled_conv2d_spatial_torch
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, 41, 9, generator=g, dtype=torch.float64)
    w = torch.randn(4, 3, kh, 3, generator=g, dtype=torch.float64)
    full = torch.nn.functional.conv2d(x, w, stride=(sh, 1), padding=(kh // 2, 1))
    got = _tiled_conv2d_spatial_torch(x, w, None, sh, 1, kh // 2, 1, 1, 1, 1, 4)
    assert torch.equal(got, full)


@pytest.mark.parametrize("sh,kh", [(1, 3), (2, 3), (1, 5)])
def test_the_torch_fused_upsample_conv_equals_upsample_then_conv(sh, kh):
    """The compiled engine's fused upsample->conv (`_fused_upsample_conv2d_torch`) streams bands of
    the upsampled extent: equal to upsample-then-conv at stride 1 and 2 (the halo rounded to whole
    strides, 2026-09-29). Injection: the halo left unrounded -> the stride-2 case, RED."""
    torch = pytest.importorskip("torch")
    from neurobrix.kernels.ops.fused_upsample_conv import (FusionUpsampleProxy,
                                                           _fused_upsample_conv2d_torch)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 3, 13, 7, generator=g, dtype=torch.float64)
    w = torch.randn(4, 3, kh, 3, generator=g, dtype=torch.float64)
    up = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    full = torch.nn.functional.conv2d(up, w, stride=(sh, 1), padding=(kh // 2, 1))
    proxy = FusionUpsampleProxy(x, 2.0, 2.0, list(up.shape))
    got = _fused_upsample_conv2d_torch(proxy, w, None, (sh, 1), (kh // 2, 1), (1, 1), False, (0, 0), 1, 4)
    assert torch.equal(got, full)


def test_an_embedded_constant_is_bound_by_the_loaders_rule():
    """`constant_load_dtype` is the Triton loader's rule (GraphExecutor._load_constant_triton) and
    the width pass's: a bf16 constant decodes to the half compute dtype (canary's positional table,
    fp16 in the walk), an fp32 one stays fp32 (swin2SR's coordinates table). Injection: every
    constant cast to the compute dtype -> the fp32 case, RED."""
    from neurobrix.triton.dtype import constant_load_dtype
    assert constant_load_dtype("bfloat16", "float16") == "float16"
    assert constant_load_dtype("bfloat16", "bfloat16") == "bfloat16"
    assert constant_load_dtype("bfloat16", "float32") == "float32"
    assert constant_load_dtype("float32", "float16") == "float32"
    assert constant_load_dtype("float64", "float16") == "float32"
    assert constant_load_dtype("int64", "float16") == "int64"


def test_the_audio_towers_frames_are_pooled_to_the_projectors_width():
    """`pooled_frames_shape`: Voxtral's audio tower [1, 1500, 1280] reaches a projector reading 5120
    features as [1, 375, 5120] (the flow's reshape, the walk's 375 audio tokens); equal widths or a
    width that does not divide are left alone."""
    from neurobrix.triton.flow.audio_llm import pooled_frames_shape
    assert pooled_frames_shape([1, 1500, 1280], 5120) == [1, 375, 5120]
    assert pooled_frames_shape([1, 1502, 1280], 5120) == [1, 375, 5120]
    assert pooled_frames_shape([1, 141, 4096], 4096) is None
    assert pooled_frames_shape([1, 10, 1000], 1500) is None


def test_the_extent_bisection_evaluates_both_ends_of_a_short_range():
    """`census.bisect_extent` met no extent at all on a range of one or two values (the loop's
    first test skipped an adjacent pair before evaluating it): a shadow walked with max_tokens <= 2
    recorded nothing, and the derivation's one-shot forward stages derived nothing. Injection: the
    two endpoint calls removed -> [] here, RED."""
    from neurobrix.kernels import census
    for lo, hi, want in ((1, 1, [1]), (1, 2, [1, 2]), (3, 9, None)):
        seen = []
        census.bisect_extent(lo, hi, lambda n: seen.append(n) or frozenset({n // 4}))
        if want is not None:
            assert sorted(set(seen)) == want
        else:
            assert {lo, hi} <= set(seen)


def test_a_spatial_downscaler_reads_pixels_whatever_its_time_map():
    """`is_downscale_graph`: a rank-5 graph whose output is spatially smaller than its input reads
    PIXELS — its docstring's rule. It required a temporal class, so CogVideoX-5b-I2V's image encoder
    (time a concrete 1) and Allegro-TI2V's / Wan2.1-I2V's (a time map neither class recognises)
    were bound latent-side: priced 8 x 8 too small per frame and derived at the latent grid while
    the walk encoded 160 x 352 pixels. Injection: the temporal-class requirement restored -> RED."""
    from neurobrix.core.prism.profiler import is_downscale_graph

    def dag(t_in, t_out):
        return {"tensors": {"input::x": {"shape": [1, 3, t_in, 112, 176]},
                            "o": {"shape": [1, 16, t_out, 14, 22]}},
                "input_tensor_ids": ["input::x"], "output_tensor_ids": ["o"]}
    assert is_downscale_graph(dag(1, 1))          # an image encoder, time concrete
    assert is_downscale_graph(dag(20, 5))
    up = {"tensors": {"input::x": {"shape": [1, 16, 1, 14, 22]}, "o": {"shape": [1, 3, 1, 112, 176]}},
          "input_tensor_ids": ["input::x"], "output_tensor_ids": ["o"]}
    assert not is_downscale_graph(up)             # a decoder upsamples


def test_the_denoisers_packed_inputs_are_the_flows_own_shapes():
    """The FLUX packing and conditioning the Triton flow builds — the derived census binds the
    denoiser with the same functions: Open-Sora's [1, 16, 13, 4, 8] latent packs to 104 tokens of 64
    (the walk's 208 = 2 x 104 rows under CFG), its ids and cond follow; Flex's [1, 16, 48, 64] packs to
    768 tokens. Injection: a pack that forgets the /2 of either side -> RED."""
    from neurobrix.triton.flow.iterative_process import TritonIterativeProcessHandler as H
    from neurobrix.triton import flux_video_conditioning as FV
    assert H.packed_5d_shape([1, 16, 13, 4, 8]) == [1, 104, 64]
    assert H.packed_4d_shape([1, 16, 48, 64]) == [1, 768, 64]
    cs = FV.conditioning_shapes(1, 104, 64, 16, 13, 4, 8, 512)
    assert (cs["img_ids"], cs["txt_ids"], cs["cond"], cs["p"]) == ([1, 104, 3], [1, 512, 3], [1, 104, 68], 2)


def test_a_diffusion_prompt_is_tokenized_to_the_encoders_declared_length():
    """`diffusion_max_length` (TextProcessor's cascade): the encoder's declared input shape, then the
    tokenizer's max — Open-Sora's T5 runs at 512 where its graph was traced at 31."""
    from neurobrix.core.module.text.processor import diffusion_max_length
    topo = {"components": {"text_encoder": {"shapes": {"input_ids": [1, 512]}}, "t2": {}}}
    assert diffusion_max_length(topo, "text_encoder", {}) == 512
    assert diffusion_max_length(topo, "t2", {"max_sequence_length": 77}) == 77
    assert diffusion_max_length(topo, "t2", {}) is None


def test_a_guidance_embedding_denoiser_runs_no_cfg_batch():
    """`guidance_embedding_component` (the CFG engine's rule): a loop denoiser that takes `guidance`
    embeds the scale — no batch-2 pass (Flex's walk: 512 text rows, not 1024)."""
    from neurobrix.triton.cfg.engine import guidance_embedding_component
    topo = {"flow": {"loop": {"components": ["transformer"]}},
            "components": {"transformer": {"interface": {"inputs": ["hidden_states", "guidance"]}}}}
    assert guidance_embedding_component(topo) == "transformer"
    topo["components"]["transformer"]["interface"]["inputs"] = ["hidden_states"]
    assert guidance_embedding_component(topo) is None


def test_the_plan_is_sized_at_the_resolution_bin_the_run_uses(tmp_path):
    """`run.request_input_config` bins the request as the executor does (`bin_request`): Sana-1024
    at 320 x 512 runs — and is now planned — at its 768 x 1280 bin. Injection: the binning removed
    from request_input_config -> 320 x 512, RED."""
    import json
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps({"dtype": "float16"}))
    (tmp_path / "topology.json").write_text(json.dumps({"components": {}, "flow": {"resolution_binning": {
        "source": "test", "default": True, "classify": "nearest_ratio",
        "restore": "cover_resize_center_crop",
        "interpolate": {"mode": "bilinear", "align_corners": False},
        "bins": {"0.6": [768, 1280], "1.0": [1024, 1024]}}}}))
    args = create_parser().parse_args(["run", "--model", "x", "--prompt", "p", "--height", "320",
                                       "--width", "512"])
    ic = request_input_config(args, {"family": "image", "dtype": "float16"}, "image", tmp_path)
    assert (ic.height, ic.width) == (768, 1280)


class _WordTokenizer:
    """One id per whitespace word; a BOS id first when specials are asked for."""
    def encode(self, text, add_special_tokens=False, padding=False):
        return ([0] if add_special_tokens else []) + [len(w) for w in text.split()]


def test_the_speech_prompt_is_the_flows_own_context():
    """`speaker_prompt_ids` is the context the next-token-diffusion flow prefills, and the census
    keys the LM's prefill from its length (VibeVoice: 48 tokens for the derived prompt, the walk's
    48). Injection: the closing speech-start id dropped -> the last id and the length RED."""
    import ast
    import inspect
    from neurobrix.triton.flow import next_token_diffusion as NTD
    ids = NTD.speaker_prompt_ids(_WordTokenizer(), "a b c", 999)
    base = NTD.speaker_prompt_ids(_WordTokenizer(), "", 999)
    assert ids[0] == 0 and ids[-1] == 999
    assert len(ids) - len(base) == 3
    # the flow builds its prompt by this function, not by a copy of it
    src = inspect.getsource(NTD.TritonNextTokenDiffusionEngine.execute)
    import textwrap
    calls = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(src)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "speaker_prompt_ids" in calls


def test_the_decode_cache_is_the_sessions_own_choice():
    """`session_kv_params`: the plan's cache, and nothing else — a decoding flow's plan always
    carries one (`core.runtime.lm_facts`); the LM facts from `session_lm_config` (extracted values
    when the package has no `lm_config`). The derived census keys the decode from the same.
    Injection: the plan branch ignored -> RED; a plan without a cache accepted -> RED."""
    import pytest
    from types import SimpleNamespace
    from neurobrix.triton.flow.autoregressive import session_kv_params, session_lm_config
    lmc = session_lm_config({}, {"extracted_values": {"lm": {"num_layers": 28, "num_heads": 12,
                                                             "hidden_size": 1536}}}, "lm")
    assert (lmc["num_layers"], lmc["num_heads"], lmc["num_kv_heads"]) == (28, 12, None)
    plan = SimpleNamespace(num_layers=28, num_kv_heads=2, k_head_dim=128, v_head_dim=128,
                           max_cache_len=4096, dtype="bfloat16")
    p = session_kv_params(plan, 0, 2048)
    assert (p["num_kv_heads"], p["dtype"], p["max_cache_len"]) == (2, NBXDtype.bfloat16, 4096)
    with pytest.raises(RuntimeError, match="no KV cache"):
        session_kv_params(None, 0, 2048)
    assert session_lm_config({"lm_config": {"num_layers": 3}}, {}, "lm") == {"num_layers": 3}


def test_a_prefill_through_the_kv_interceptor_sees_no_mask():
    """The interceptor's prefill drops the graph's mask for `is_causal`; a one-token prefill (the
    CFG negative context) then takes the vector kernel, whatever a frozen mask says (VibeVoice's
    causal mask is annotated [23, 23]). Injection: the prefill rule removed -> launches, RED."""
    import collections
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import derived_census as D
    shapes = {"q": [1, 12, 1, 128], "k": [1, 12, 1, 128], "v": [1, 12, 1, 128], "m": [23, 23]}
    o = {"attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in "qkvm"]}}
    args = ("aten::scaled_dot_product_attention", "u", o, list("qkvm"), shapes.__getitem__,
            lambda _t: F16, LK, None, "float16", False, 1 << 40, 16, 64, 0,
            collections.Counter())
    assert D._op_launches(*args, None, None)                     # the graph's mask: the math route
    assert not D._op_launches(*args, None, {"prefill": True})    # through the interceptor: none


def test_an_image_ar_generation_is_keyed_by_the_strategys_own_rules():
    """The image strategy's facts the census keys Janus from (25/25 walked keys): its fixed token
    count (image / patch)^2, the guidance weight (CLI over package; above 1 the LM runs at batch
    2), the session's LM. Injections: the count as image // patch, the CLI weight ignored, the LM
    rule returning the first component — each RED."""
    import ast
    import inspect
    import textwrap
    from neurobrix.triton.flow import autoregressive as AR
    d = {"image_size": 384, "patch_size": 16, "guidance_scale": 5.0}
    assert AR.image_token_count(d) == 576
    assert AR.image_cfg_weight({}, d) == 5.0
    assert AR.image_cfg_weight({"global.guidance_scale": 1.0}, d) == 1.0
    assert AR.session_lm_name({"lm_component": "language_model"},
                              ["vision_model", "language_model"]) == "language_model"
    assert AR.session_lm_name({}, ["lm_head", "model"]) == "model"
    for fn, name in ((AR.TritonAutoregressiveHandler._create_session, "session_lm_name"),
                     (AR.TritonAutoregressiveHandler._create_strategy, "image_cfg_weight"),
                     (AR.TritonAutoregressiveHandler._tokenize, "image_cfg_weight"),
                     (AR.TritonImageStrategy.create_generator, "image_token_count")):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        assert name in {n.func.id for n in ast.walk(tree)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}, fn.__name__


class _TemplateTokenizer:
    """A chat template that writes the image placeholder as a run of `run` ids between a user
    header and the text, then an assistant header."""
    def __init__(self, run):
        self.run = run

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = messages[0]["content"][1]["text"]
        return [1, 2] + [9] * self.run + [len(w) for w in text.split()] + [3, 4]


def test_the_vlm_context_is_the_template_around_the_modality_span():
    """`tokenize_around_span` gives the flow's (prefix, suffix) around the placeholder run, whatever
    its length in the template; the LM context is prefix + vision tokens + suffix (Qwen3-VL: 4 +
    196 + 15 = 215, the walk's 224 class). Injection: the suffix taken from the first placeholder
    -> RED."""
    import ast
    import inspect
    import textwrap
    from neurobrix.triton.flow import vlm as V
    for run in (1, 5):
        pre, suf = V.tokenize_around_span(_TemplateTokenizer(run), "a bb", 9, "image")
        assert (pre, suf) == ([1, 2], [1, 2, 3, 4])
    tree = ast.parse(textwrap.dedent(inspect.getsource(V.TritonVLMEngine._tokenize_around_span)))
    assert "tokenize_around_span" in {n.func.id for n in ast.walk(tree)
                                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_a_value_bound_symbol_is_read_from_the_feeds_value():
    """A vision tower binds symbols from its grid's VALUES (`input::grid_thw::val_1`); the census
    feeds the request's own grid array and the runtime's binder reads it. Injection: the feed
    without its value -> the binder's read refused, RED."""
    import numpy as np
    from neurobrix.triton.symbols import SymbolResolver
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import derived_census as D
    ctx = {"symbols": {"s0": {"name": "val_grid_thw_1", "source": "input::grid_thw::val_1"},
                       "s1": {"name": "val_grid_thw_2_fd2", "source": "input::grid_thw::val_2_fd2"}}}
    res = SymbolResolver(ctx)
    feed = {"input::grid_thw": D._Shape(np.array([[1, 28, 32]]))}
    res.bind_from_inputs(feed, list(feed), {})
    assert dict(res.bindings) == {"s0": 28, "s1": 16}
