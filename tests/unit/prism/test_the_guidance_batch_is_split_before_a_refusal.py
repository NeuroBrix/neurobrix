"""The guidance batch is split before the card refuses — Prism plans each branch at its own batch.

Classifier-free guidance runs a loop component on [uncond, cond] concatenated on the batch axis
(`CFGEngine._execute_batched_cfg`), and the plan priced it at that batch (`run.request_input_config`
doubles `batch_size`). Every op of a denoiser is per-sample, so the two halves are independent: the
same component run once per branch returns the same values, at half the activations. Prism never
offered that cut. SANA-Video_2B_720p at its derived request on the rack's 16 GB card (rung 16 384,
15 565 MB, 2026-10-05, merge-queue-18 953cac5b) had NO placement in any mode: the transformer's
linear-attention region holds four [2, 155 232, 2 240] fp32 tensors at once (residual, q, k, v),
18 774.5 MB of activations alone against the 15 072.4 MB layer_streaming window, so no cut between
ops serves it, and the census wrote no row ("the derivation cannot place it completely").

The rule (`PrismSolver._split_guidance_batch`): when no candidate holds every component on an
accelerator, the loop components' guidance branches are priced one pass each
(`FlowBindings.split_guidance`, `InputConfig.guidance_passes`); the split is kept only if it places
on the accelerator, and the plan says so (`ExecutionPlan.cfg_split_components`). Both CFG engines
read that field and run the branches one pass each, joining the halves where the single pass put
them; the derived census binds the split components' batch at one branch, as the plan priced it.

Allegro and Allegro-TI2V were refused on this rung for another cause — their `vae_encoder` froze the
time axis — which Forge's re-trace (2026-10-04) removed; they are held here at the same rung.

SEEN RED (2026-10-05): `_split_guidance_batch` made to return None -> SANA-Video refuses in every
mode and the split cells fail; the split rule removed from `FlowBindings.overrides` -> the binding
cell fails; the per-branch pass in the compiled CFG engine made to run the whole batch -> the
values cell fails on its pass count. Triton's CFG engine mirrors the compiled one line for line
(R33: no Triton on this CPU) — its values are the GPU proof's.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.flow_bindings import FlowBindings
from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, impose_rung,
                                              pin_dedicated_card, pin_host, profile)
from tests.unit.prism.test_the_plan_binds_what_the_flow_runs import _container

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import trace_request as TR  # noqa: E402

MODES = ["triton", "triton_sequential", "compiled"]
FLAG = {"triton": "--triton", "triton_sequential": "--triton-sequential", "compiled": "--compiled"}


def _plan(monkeypatch, model, mode):
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "the rack's V100-16GB card 0 as measured")
    impose_rung(monkeypatch, 16384)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    c = NBXContainer.load(str(container_root(model)))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", model, *TR.derived_request(model), FLAG[mode]])
    ic = request_input_config(args, man, man.get("family"), c.cache_path)
    return PrismSolver().solve_smart(c, profile(V100_16GB), ic, mode=mode)


def _off_the_card(plan):
    return {n: a.devices for n, a in plan.components.items()
            if not a.devices or any(not str(d).startswith("cuda") for d in a.devices)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", ["SANA-Video_2B_720p_diffusers", "Allegro", "Allegro-TI2V"])
def test_the_video_models_place_on_the_16gb_card_at_its_top_rung(monkeypatch, model, mode):
    plan = _plan(monkeypatch, model, mode)
    assert plan.strategy not in ("cpu_execution", "cpu_streaming"), plan.strategy
    assert not _off_the_card(plan), f"{model} [{mode}] put components on the host: {_off_the_card(plan)}"


@pytest.mark.parametrize("mode", MODES)
def test_sana_video_places_by_splitting_its_guidance_batch(monkeypatch, mode):
    plan = _plan(monkeypatch, "SANA-Video_2B_720p_diffusers", mode)
    assert plan.cfg_split_components == ["transformer"], plan.cfg_split_components
    assert "guidance branches of transformer run one pass each" in plan.selection_reason


def test_a_plan_that_places_whole_keeps_its_guidance_batch(monkeypatch):
    """The split is a cut taken only where the batch does not fit: Allegro places as it is."""
    assert _plan(monkeypatch, "Allegro", "triton").cfg_split_components == []


def test_a_split_component_binds_one_branchs_batch(tmp_path):
    topo, _te, tr = _container(tmp_path)
    fb = FlowBindings(topo, tmp_path)
    ic = InputConfig(batch_size=2, guidance_passes=2, dtype="float16", flow=fb)
    assert ActivationProfiler(tr).build_symbol_map(ic)["s0"] == 2
    split = InputConfig(batch_size=2, guidance_passes=2, dtype="float16",
                        flow=fb.split_guidance(["transformer"]))
    assert ActivationProfiler(tr).build_symbol_map(split)["s0"] == 1
    assert fb.cfg_split_components == frozenset(), "split_guidance changed the bindings it was asked on"
    no_guidance = InputConfig(batch_size=2, dtype="float16", flow=fb.split_guidance(["transformer"]))
    with pytest.raises(ValueError, match="ZERO FALLBACK"):
        ActivationProfiler(tr).build_symbol_map(no_guidance)


# ───────────────── values: one pass per branch == the batched pass (compiled CFG engine, CPU) ─────

class _Resolver:
    def __init__(self, values):
        self.resolved = dict(values)
        self.loop_state = {}

    def get(self, key, default=None):
        return self.resolved.get(key, default)

    def set(self, key, value):
        self.resolved[key] = value


def _cfg_run(split: bool):
    from neurobrix.core.cfg.engine import CFGEngine
    g = torch.Generator().manual_seed(0)
    state = torch.randn(1, 4, 6, generator=g)
    pos, neg = torch.randn(1, 5, 6, generator=g), torch.randn(1, 5, 6, generator=g)
    image = torch.randn(1, 3, 6, generator=g)
    mask = torch.ones(1, 5, dtype=torch.long)
    topo = {"flow": {"type": "iterative_process", "pre_loop": ["text_encoder", "image_encoder"],
                     "loop": {"components": ["transformer"], "state_variable": "global.latents"}},
            "connections": [{"from": "text_encoder.hidden_state", "to": "transformer.encoder_hidden_states"},
                            {"from": "image_encoder.image_embeds", "to": "transformer.encoder_hidden_states_image"}]}
    res = _Resolver({"text_encoder.hidden_state": pos, "text_encoder.negative_hidden_state": neg,
                     "image_encoder.image_embeds": image, "global.attention_mask": mask,
                     "global.latents": state})
    ctx = SimpleNamespace(variable_resolver=res, pkg=SimpleNamespace(topology=topo), loop_id="loop",
                          plan=SimpleNamespace(cfg_split_components=["transformer"] if split else []))
    passes = []

    def execute(comp, label):
        # a per-sample nonlinear denoiser: every op reads its own sample only, as a denoiser's do
        x = res.get("global.latents")
        txt = res.get("text_encoder.hidden_state")
        img = res.get("image_encoder.image_embeds")
        t = res.loop_state["loop"].reshape(-1, 1, 1)
        attn = torch.softmax(x @ torch.cat([txt, img], 1).transpose(1, 2), -1) @ torch.cat([txt, img], 1)
        passes.append((label, x.shape[0], txt.shape[0], img.shape[0]))
        return {"out": torch.tanh(attn + x) * (1 + t)}

    eng = CFGEngine(ctx, execute, lambda comp, out: out["out"], guidance_scale=4.5)
    out = eng.execute_component_with_cfg("transformer", state, torch.tensor(0.7), 4.5, torch.float32)
    after = {k: res.get(k) for k in ("text_encoder.hidden_state", "global.latents",
                                     "image_encoder.image_embeds")}
    return out["output_0"], passes, after, (pos, state, image)


def test_one_pass_per_branch_returns_the_batched_values():
    whole, whole_passes, _, _ = _cfg_run(split=False)
    split, split_passes, after, (pos, state, image) = _cfg_run(split=True)
    assert [p[1:] for p in whole_passes] == [(2, 2, 2)], whole_passes
    assert split_passes == [("cfg_uncond", 1, 1, 1), ("cfg_cond", 1, 1, 1)], split_passes
    assert torch.allclose(split, whole, rtol=0, atol=1e-6), (split - whole).abs().max()
    # every input the passes rebound is the caller's again
    assert after["text_encoder.hidden_state"] is pos and after["global.latents"] is state
    assert after["image_encoder.image_embeds"] is image
