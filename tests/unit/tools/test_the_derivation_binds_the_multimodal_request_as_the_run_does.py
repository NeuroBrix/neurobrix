"""The derived census binds a vlm request the way `TritonVLMEngine.execute` runs it, whichever of
its three contracts the container's graphs declare: the staged splice (a vision graph declaring
`input::all_pixel_values` — MiniCPM-o), the M-RoPE masked splice (an LM graph declaring
`input::image_pos_masks` — Ming-Lite-Omni), else the splice path (GLM-4.1V, Qwen3-Omni).

Before (2026-10-03): only the splice path had sites; the two others fell to the plan's trace
binding — Ming's vision tower at its trace's 17 020 patches, its LM at 37 tokens — and every
component ran, the audio and generative legs included: 0 of Ming's 27 walked keys, 10 of
MiniCPM-o's 44. The walk's own log is the reference: 256 vision tokens and a 275-token context
(Ming), 64 and 82 (MiniCPM-o), for the census's request (apple_448.png, "Describe this image in
one sentence.", 32 tokens).

Injections, each seen RED: `_vlm_sites` returning [] for a non-splice contract (the gate as it
was) -> both site tests fail; `vlm_runs` without the projection -> the runs test fails;
`baddbmm_launches` keyed with HAS_BIAS False -> the baddbmm test fails.

The site tests bind a committed hardware profile the way the census's own entry does
(`derive_keys`: `load_profile`, `census._bind_target`, `wrappers.set_hardware_profile`): the census
path asks the profile whether the Triton branch stores float64 and refuses, rightly, under none
("ZERO FALLBACK: no hardware profile with a device"). Before 2026-10-16 these two tests bound none
and failed on that refusal whenever no earlier test in the process had left a profile bound. The
values asserted are profile-independent (token and patch counts); the binding is restored after."""
import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pytest
import nbxtensor.backend.capabilities as _nbx_capabilities  # the module the moved names live in (M2 cut of tensor.py)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as D  # noqa: E402


#: A committed profile, bound as the census entry binds `--hardware` (any profile with a device serves:
#: the asserted counts do not depend on it).
HARDWARE = "v100-32g"


@pytest.fixture
def census_hardware(monkeypatch):
    """Bind HARDWARE as `derived_census.derive_keys` does; every global it sets is restored after."""
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.kernels import census as _census
    from nbxtensor.backend import launcher as _launcher
    from nbxtensor import tensor as T
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs
    for mod, name in ((W, "_NBX_HW_PROFILE"), (W, "_NBX_HAS_NATIVE_BF16"), (_nbx_capabilities, "_BACKEND_HAS_FP64"),
                      (_launcher, "_TARGET")):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(_configs, "_ACTIVE_PROFILE", dict(_configs._ACTIVE_PROFILE))
    prof = load_profile(HARDWARE)
    _census._bind_target(HARDWARE, None)
    W.set_hardware_profile(prof)
    assert W.get_hardware_profile() is prof
    yield prof


def _request_sites(model, cache_len=None):
    """The census's own request for `model`, parsed by the CLI's parser, through `extent_sites`
    at a plan naming every component in fp16 (the dtypes only feed the chain's outputs)."""
    import json
    from trace_request import derived_request
    from neurobrix.cli import create_parser
    root = D.CACHE / model
    if not (root / "topology.json").exists():
        pytest.fail(f"{model} is not in the catalogue {D.CACHE}: the census cannot be checked")
    topo = json.loads((root / "topology.json").read_text())
    defaults = json.loads((root / "runtime" / "defaults.json").read_text())
    args = create_parser().parse_args(["run", "--model", model, *derived_request(model), "--triton"])
    plan = {"components": [{"name": c, "dtype": "float16"} for c in topo["components"]]}
    if cache_len is not None:       # the decoder's cache the plan sizes (since every decoding flow decodes on it, 3d3daa99)
        plan["kv_cache"] = {"max_cache_len": int(cache_len), "dtype": "float16"}
    return topo, D.extent_sites(model, topo, defaults, plan, args.prompt, None, args.max_tokens,
                                args.input_image)


@pytest.mark.parametrize("model,contract,vis_rows,n_modal,L0,lm,embeds", [
    ("Ming-Lite-Omni-1.5", "masked", [1024, 1176], 256, 275, "model.model", "image_embeds"),
    ("MiniCPM-o-4_5", "staged", None, 64, 82, "llm.model", "vision_hidden_states"),
])
def test_the_vlm_request_is_bound_by_its_contract(census_hardware, model, contract, vis_rows, n_modal, L0, lm,
                                                 embeds):
    topo, sites = _request_sites(model, cache_len=L0 + 32)   # a prefill leaves the cache one position to decode into
    assert D.vlm_contract(model, topo) == contract
    by_name = {s[0]: s for s in sites}
    # the decoder on its KV cache (3d3daa99): its prefill over the prompt lengths up to the cache's reach, then decode
    assert set(by_name) == {"vision tower and projection", f"{lm} prefill (prompt lengths)",
                            f"{lm} decode KV length"}, list(by_name)
    _n, lo, hi, chain = by_name[f"{lm} prefill (prompt lengths)"]
    assert (lo, hi) == (L0, L0 + 32 - 1)                 # the request's context up to the planned cache's reach
    (comp, feed), = chain(lo)
    assert comp == lm
    feed = {k: v for k, v in feed.items() if not k.startswith("__")}
    H = feed["inputs_embeds"][-1]
    assert feed["inputs_embeds"] == [1, lo, H]
    assert feed[embeds] == [n_modal, H]
    steps = by_name["vision tower and projection"][3](1)
    v = topo["flow"]["vlm"]
    assert [c for c, _f in steps] == [v["vision_component"], v["vision_projection_component"]]
    if vis_rows is not None:
        assert steps[0][1]["hidden_states"] == vis_rows    # the processor's grid, not the trace's


def test_a_vlm_request_runs_its_contracts_components_only():
    import json
    for model, want in (("Ming-Lite-Omni-1.5", {"vision", "linear_proj", "model.model"}),
                        ("MiniCPM-o-4_5", {"vpm", "resampler", "llm.model"})):
        topo = json.loads((D.CACHE / model / "topology.json").read_text())
        assert D.vlm_runs(model, topo) == want


def test_baddbmm_is_keyed_as_its_wrapper_launches(tl_dot_gemms):
    """`baddbmm_wrapper` launches IEEE_PRECISION True, PROMOTE_B False, HAS_BIAS True, its output
    batch1's dtype; the walk's key (64, 1024, 128, True, False, True, fp16, fp16, fp16, <bias>)."""
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import linalg_wrappers as _nbx_linalg_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    F16 = NBXDtype.float16
    assert LK.baddbmm_launches(64, 128, 1024, F16, F16, F16) == [
        (LK.BADDBMM, (64, 1024, 128, True, False, True, "fp16", "fp16", "fp16", "fp16"))]
    assert LK.baddbmm_launches(64, 128, 1024, F16, F16, NBXDtype.bool_)[0][1][-1] == "uint8"
    src = textwrap.dedent(inspect.getsource(_nbx_linalg_wrappers.baddbmm_wrapper))
    kw = {k.arg: k.value.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
          for k in n.keywords if k.arg in ("IEEE_PRECISION", "PROMOTE_B", "HAS_BIAS")
          and isinstance(k.value, ast.Constant)}
    assert kw == {"IEEE_PRECISION": True, "PROMOTE_B": False, "HAS_BIAS": True}


def test_a_vlm_leg_the_sites_do_not_derive_is_counted_by_name():
    """The sites derive the image+text request only; an audio recording, the image or speech leg
    (`global.mode`), or a request without an image is named, never derived as the text request.
    Injection: `vlm_legs_not_derived` returning [] -> RED."""
    from types import SimpleNamespace as N
    img = ["--input-image", "x.png"]
    assert D.vlm_legs_not_derived(img, N(mode="text", input_image="x.png")) == []
    assert D.vlm_legs_not_derived(img + ["--audio", "a.wav"], N(mode="text", input_image="x.png")) \
        == ["an audio recording"]
    assert D.vlm_legs_not_derived(img, N(mode="image", input_image="x.png")) == ["the 'image' leg"]
    assert D.vlm_legs_not_derived([], N(mode="text", input_image=None)) == ["a request with no image"]
