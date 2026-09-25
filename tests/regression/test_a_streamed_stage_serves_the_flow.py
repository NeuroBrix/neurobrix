"""A language model streamed in pieces serves its FLOW as it does whole: same tokens, bit for bit.

The Mac counted 30 streamed runs of VLM and audio-LLM stages refused before their first piece
(df2588e7), six models, both triton engines:

    ZERO FALLBACK: Audio-LLM stage 'language_model' requires embed_tokens weight.
    ZERO FALLBACK: vlm stage 'model.language_model' requires embed_tokens weight.

A flow reads the token embedding BY NAME from its LM's executor, outside the graph (the graph
takes `inputs_embeds`); a whole executor holds it because the loader keeps every non-block key for
exactly that reader. Under `layer_streaming` the base executor held nothing — and every PIECE
loaded every non-block key with every run instead, the embedding into pieces that never read it,
in no plan's budget (granite-speech: 8 pieces of 48-53 weights for ~46 they read). Every earlier
streaming gate ran the component, never the flow that reads it by name, so none could see it.

This runs the real flow through the CLI, twice per cell on one card: WHOLE (the plan this card
gets) and STREAMED (`layer_streaming` forced — on a discrete card `lazy_sequential` outscores it;
the Mac, where lazy is not viable, gets it by elimination — at the Mac's 8 192 MB rung), greedy,
a bounded decode, and requires the decoded token ids identical (`NBX_DECODE_PROGRESS`).

Seen failing on main (96288dfc, card 3): the streamed run refused with the Mac's exact
"Audio-LLM stage 'language_model' requires embed_tokens weight."

Sizes (rule 1): the sequence the LM runs at is the prompt plus what the stage before it hands
over; two prompts of different lengths per engine for granite-speech.

Needs a card: `CUDA_VISIBLE_DEVICES=<n>`. Without one this FAILS.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SHORT = "Transcribe the audio."
LONG = ("Listen carefully to the recording and write down, word for word, exactly what the "
        "speaker says, without adding any comment of your own.")

# (model, extra CLI args, prompt)
FLOWS = [
    ("granite-speech-3.3-8b", ["--audio", "test_speech_ref.wav"], SHORT),
    ("granite-speech-3.3-8b", ["--audio", "test_speech_ref.wav"], LONG),
    ("GLM-4.1V-9B-Thinking", ["--input-image", "test_upscale_input.png"], "Describe the image."),
]
CELLS = ([(m, a, p, e) for m, a, p in FLOWS for e in ("triton", "triton-sequential")]
         # R30: the compiled flows read the base the same way, through the native loader.
         + [(FLOWS[0][0], FLOWS[0][1], FLOWS[0][2], e) for e in ("sequential", "compiled")])
_ENGINE_FLAG = {"triton": ["--triton"], "triton-sequential": ["--triton-sequential"],
                "sequential": ["--sequential"], "compiled": []}


def _tokens(model, args, prompt, engine, out: Path, streamed: bool):
    # Each flow writes one of the two per-step decode instruments: the token trajectory
    # (`NBX_DECODE_PROGRESS`, the generators) or the top-4 ids AND logit values per step
    # (`NBX_DECODE_TOPK`, the vlm flows). Both are armed; whichever is written is compared.
    topk = out.with_suffix(".topk.jsonl")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "NBX_DECODE_PROGRESS": str(out),
           "NBX_DECODE_TOPK": str(topk)}
    env.pop("NBX_PRISM_BUDGET_MB", None)
    env.pop("NBX_FORCE_STRATEGY", None)
    if streamed:
        env.update(NBX_PRISM_BUDGET_MB="8192", NBX_FORCE_STRATEGY="layer_streaming")
    proc = subprocess.run(
        [os.environ["NEUROBRIX_PYTHON"], "-m", "neurobrix", "run", "--model", model, *_ENGINE_FLAG[engine],
         *args, "--prompt", prompt, "--max-tokens", "8", "--temperature", "0", "--seed", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, (("streamed" if streamed else "whole"), proc.stdout[-2500:],
                                  proc.stderr[-2500:])
    log = proc.stdout + proc.stderr
    if streamed:
        assert "Strategy: layer_streaming" in log, "the streamed arm did not stream: nothing proved"
    if out.exists():
        toks = [f.split("=", 1)[1] for line in out.read_text().splitlines()
                for f in line.split() if f.startswith("last=")]
    elif topk.exists():
        import json
        toks = [(r["ids"], r["vals"]) for r in map(json.loads, topk.read_text().splitlines())]
    else:
        toks = []
    assert toks, f"neither decode instrument was written ({out}, {topk})"
    return toks


@pytest.mark.slow
@pytest.mark.parametrize("model,args,prompt,engine", CELLS,
                         ids=[f"{m}-{e}-{'long' if p == LONG else 'short'}" for m, _, p, e in CELLS])
def test_the_streamed_stage_decodes_what_the_whole_one_decodes(model, args, prompt, engine, tmp_path):
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        pytest.fail("this cell runs a model on a card: set CUDA_VISIBLE_DEVICES=<n>. A streaming "
                    "gate that did not run is not a streaming gate that passed.", pytrace=False)
    whole = _tokens(model, args, prompt, engine, tmp_path / "whole.txt", streamed=False)
    streamed = _tokens(model, args, prompt, engine, tmp_path / "streamed.txt", streamed=True)
    assert streamed == whole, f"{model} {engine}: whole {whole} vs streamed {streamed}"


@pytest.mark.slow
def test_the_base_holds_the_flow_reads_and_a_piece_borrows_them(tmp_path):
    """The mechanism under the tokens, executed on the card with real weights: the base holds
    exactly the component's non-block weights (once: a second load loads nothing) and leaves the
    whole-load arguments unset; every piece loads only what its ops consume, and the piece that
    consumes a non-block weight (granite-speech's tied head reads the token embedding) holds the
    BASE's tensor, not a second copy."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        pytest.fail("this cell loads weights on a card: set CUDA_VISIBLE_DEVICES=<n>.", pytrace=False)
    import json
    from types import SimpleNamespace
    from neurobrix.core.prism import InputConfig, PrismSolver
    from neurobrix.core.prism.autodetect import load_default_profile
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.core.strategies.base import StrategyContext
    from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy
    from neurobrix.nbx import NBXContainer
    from neurobrix.triton.weight_loader import is_block_key
    from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung,
                                                  pin_host, profile)
    model, comp = "granite-speech-3.3-8b", "language_model"
    mp = pytest.MonkeyPatch()
    try:
        pin_host(mp, 24576, 18186, "the Mac, idle")
        impose_rung(mp, 16384)
        root = container_root(model)
        plan = PrismSolver().solve_smart(NBXContainer.load(str(root)), profile(APPLE_M4_PRO),
                                         InputConfig(batch_size=1), mode="triton")
    finally:
        mp.undo()
    assert plan.strategy == "layer_streaming" and plan.layer_stream_plan.get(comp), plan.strategy
    family = json.loads((root / "manifest.json").read_text())["family"]
    here = load_default_profile().devices[0]
    base = GraphExecutor(family=family, vendor=str(here.brand).split(".")[-1].lower(),
                         arch=str(here.architecture), device=here.get_device_string(),
                         dtype=plan.components[comp].dtype, mode="triton")
    base.load_graph(root / "components" / comp / "graph.json")
    ctx = StrategyContext(strategy_name="layer_streaming", allocations={comp: (base.device, {})},
                          component_executors={comp: base},
                          runtime_package=SimpleNamespace(cache_path=root),
                          layer_segments={comp: plan.layer_stream_plan[comp]},
                          layer_graphs=dict(plan.layer_stream_graph),
                          layer_moe=dict(plan.layer_stream_moe))
    strategy = LayerStreamingStrategy(ctx, "layer_streaming")
    pieces = strategy._build_segment_executors(comp)
    strategy._ensure_flow_reads(comp, base)
    index = json.loads((root / "components" / comp / "weights_index.json").read_text())["tensors"]
    non_block = {k for k in index if not is_block_key(k)}
    held = {k for k in base._weights if k != "_arenas"}
    assert non_block <= held, ("the base does not hold what the flow reads", non_block - held)
    assert base.load_flow_read_weights(str(root), comp) == 0, "a second load loaded again"
    assert getattr(base, "_load_args", None) is None, "a later rewrite would load the component"
    borrowed_any = False
    dt = "float16" if str(base.dtype) in ("float16", "torch.float16") else "float32"
    whole_contract = base.precision_contract(dt)
    for piece in pieces:
        assert piece._flow_reads_weights is False and piece._borrow_from is base
        assert piece._contract_from is base and piece.precision_contract(dt) == whole_contract, (
            "a piece resolves its own contract")
        wanted = piece._consumed_in_loader_space(piece.consumed_weight_names(), str(root), comp)
        consumed = piece.consumed_weight_names() or set()      # graph names == index keys here
        stray = [k for k in wanted if not is_block_key(k) and k not in consumed]
        assert not stray, f"a piece loads non-block weights its ops never read: {stray}"
        if any(not is_block_key(k) for k in wanted):
            piece.load_weights(str(root), comp)
            for k in wanted:
                if not is_block_key(k):
                    assert piece._weights[k] is base._weights[k], f"{k}: a second copy"
                    borrowed_any = True
            piece.unload_weights()
    assert borrowed_any, "precondition: some piece consumes a non-block weight (the tied head)"
