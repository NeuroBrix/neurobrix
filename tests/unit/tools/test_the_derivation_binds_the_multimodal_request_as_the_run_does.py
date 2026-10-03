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
`baddbmm_launches` keyed with HAS_BIAS False -> the baddbmm test fails."""
import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as D  # noqa: E402


def _request_sites(model):
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
    return topo, D.extent_sites(model, topo, defaults, plan, args.prompt, None, args.max_tokens,
                                args.input_image)


@pytest.mark.parametrize("model,contract,vis_rows,n_modal,L0,lm,embeds", [
    ("Ming-Lite-Omni-1.5", "masked", [1024, 1176], 256, 275, "model.model", "image_embeds"),
    ("MiniCPM-o-4_5", "staged", None, 64, 82, "llm.model", "vision_hidden_states"),
])
def test_the_vlm_request_is_bound_by_its_contract(model, contract, vis_rows, n_modal, L0, lm, embeds):
    topo, sites = _request_sites(model)
    assert D.vlm_contract(model, topo) == contract
    by_name = {s[0]: s for s in sites}
    assert set(by_name) == {"vision tower and projection", f"{lm} context"}, list(by_name)
    _n, lo, hi, chain = by_name[f"{lm} context"]
    assert (lo, hi) == (L0, L0 + 32 - 1)                 # the walk's context, 32 tokens
    (comp, feed), = chain(lo)
    assert comp == lm
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


def test_baddbmm_is_keyed_as_its_wrapper_launches():
    """`baddbmm_wrapper` launches IEEE_PRECISION True, PROMOTE_B False, HAS_BIAS True, its output
    batch1's dtype; the walk's key (64, 1024, 128, True, False, True, fp16, fp16, fp16, <bias>)."""
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXDtype
    F16 = NBXDtype.float16
    assert LK.baddbmm_launches(64, 128, 1024, F16, F16, F16) == [
        (LK.BADDBMM, (64, 1024, 128, True, False, True, "fp16", "fp16", "fp16", "fp16"))]
    assert LK.baddbmm_launches(64, 128, 1024, F16, F16, NBXDtype.bool_)[0][1][-1] == "uint8"
    src = textwrap.dedent(inspect.getsource(W.baddbmm_wrapper))
    kw = {k.arg: k.value.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
          for k in n.keywords if k.arg in ("IEEE_PRECISION", "PROMOTE_B", "HAS_BIAS")
          and isinstance(k.value, ast.Constant)}
    assert kw == {"IEEE_PRECISION": True, "PROMOTE_B": False, "HAS_BIAS": True}
