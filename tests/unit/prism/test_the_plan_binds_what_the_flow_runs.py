"""`core.prism.flow_bindings.FlowBindings` — the per-component bindings no symbol NAME carries, which
the plan prices with and the derived census keys with (one map for both):

* the loop denoiser runs the CFG batch (2 under guidance), nothing else does — unless it embeds the
  guidance scale (a `guidance` input: no batch-2 pass);
* a diffusion encoder runs at its tokenized length (its declared input shape), not its trace;
* the denoiser's text axis is bound by SOURCE (the text input and its mask) at the FINALIZED length —
  never by symbol name (Flex names its image-token symbol `seq_len` too).

Measured before: Wan2.1-T2V's denoiser priced at batch 1 while the CFG engine runs 2; Open-Sora's T5
priced at 31 tokens while it runs 512. Injections: the CFG block removed from `overrides` -> the batch
case RED; the name-based text rebinding restored -> the image-token case RED."""
import json

from neurobrix.core.prism.flow_bindings import FlowBindings
from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig


def _graph(name, syms):
    return {"component_name": name, "tensors": {}, "ops": {}, "execution_order": [],
            "symbolic_context": {"symbols": syms}}


def _container(tmp_path, guidance_input=False):
    topo = {
        "flow": {"type": "iterative_process", "pre_loop": ["text_encoder"],
                 "loop": {"components": ["transformer"], "state_input": "hidden_states"},
                 "post_loop": ["vae"]},
        "components": {"text_encoder": {"shapes": {"input_ids": [1, 512]}},
                       "transformer": {"interface": {"inputs": ["hidden_states", "encoder_hidden_states"]
                                                     + (["guidance"] if guidance_input else [])}}},
        "connections": [{"from": "text_encoder.last_hidden_state",
                         "to": "transformer.encoder_hidden_states"}],
        "extracted_values": {"tokenizer": {}},
    }
    te = _graph("text_encoder", {"s0": {"name": "batch", "trace_value": 1, "source": "input::input_ids::dim_0"},
                                 "s1": {"name": "seq_len", "trace_value": 31, "source": "input::input_ids::dim_1"}})
    tr = _graph("transformer", {
        "s0": {"name": "batch", "trace_value": 1, "source": "input::hidden_states::dim_0"},
        "s1": {"name": "seq_len", "trace_value": 60, "source": "input::hidden_states::dim_1"},
        "s2": {"name": "seq_len", "trace_value": 31, "source": "input::encoder_hidden_states::dim_1"}})
    for name, g in (("text_encoder", te), ("transformer", tr)):
        (tmp_path / "components" / name).mkdir(parents=True)
        (tmp_path / "components" / name / "graph.json").write_text(json.dumps(g))
    return topo, te, tr


def test_the_loop_denoiser_runs_the_cfg_batch_and_the_encoder_its_tokenized_length(tmp_path):
    topo, te, tr = _container(tmp_path)
    ic = InputConfig(batch_size=2, seq_len=None, dtype="float16", flow=FlowBindings(topo, tmp_path))
    m_te = ActivationProfiler(te).build_symbol_map(ic)
    m_tr = ActivationProfiler(tr).build_symbol_map(ic)
    assert m_te["s1"] == 512 and m_te["s0"] == 1          # the encoder: its length, no CFG batch
    assert m_tr["s0"] == 2                                 # the loop denoiser: the CFG batch
    assert m_tr["s2"] == 512                               # its text axis, by source
    assert m_tr["s1"] == 60                                # its image tokens NOT rebound by name


def test_a_guidance_embedding_denoiser_runs_no_cfg_batch(tmp_path):
    topo, te, tr = _container(tmp_path, guidance_input=True)
    ic = InputConfig(batch_size=2, seq_len=None, dtype="float16", flow=FlowBindings(topo, tmp_path))
    assert ActivationProfiler(tr).build_symbol_map(ic)["s0"] == 1


def test_without_a_flow_the_map_is_the_name_driven_one(tmp_path):
    topo, te, tr = _container(tmp_path)
    ic = InputConfig(batch_size=2, seq_len=None, dtype="float16")
    assert ActivationProfiler(te).build_symbol_map(ic)["s1"] == 31
    assert ActivationProfiler(tr).build_symbol_map(ic)["s0"] == 1


def test_the_vace_control_encoder_runs_the_pair(monkeypatch):
    """Under the loop component's `vace_control_conditioning` flag the CLI feeds the component the
    image input reaches the (inactive, reactive) pair — batch 2, `image_dsp.VACE_CONTROL_CLIPS` —
    and the plan was sized at 1 (the retraced VACE's encoder at 162 = 2 x 81 frames in its walk).
    No flag, no pair. Injection: the rule removed from `overrides` -> RED."""
    import neurobrix.core.runtime.registry_flags as RF
    flags = {("transformer", "vace_control_conditioning"): True}
    monkeypatch.setattr(RF, "get_component_flag",
                        lambda m, c, f, default=None, env_override=None: flags.get((c, f), default))
    topo = {"flow": {"type": "iterative_process", "pre_loop": ["text_encoder", "vae_encoder"],
                     "loop": {"components": ["transformer"]}, "post_loop": ["vae"]},
            "connections": [{"from": "global.image", "to": "vae_encoder.args"}]}
    fb = FlowBindings(topo, None, "m")
    enc = _graph("vae_encoder", {"s0": {"name": "batch", "source": "input::args::dim_0"},
                                 "s1": {"name": "time", "source": "input::args::dim_2"}})
    ic = InputConfig(batch_size=1)
    assert fb.overrides(enc, ic) == {"s0": 2}
    assert fb.vace_control_batch("vae") is None
    flags.clear()
    assert fb.overrides(enc, ic) == {}
