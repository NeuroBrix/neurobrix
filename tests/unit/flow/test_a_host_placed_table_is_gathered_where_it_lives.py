"""A table the plan placed on the host is gathered where it lives; its rows join the context.

granite-speech-3.3-8b, compiled, on a 16 GB card (2026-10-04): Prism maps the 8B language
model to `zero3:cuda:0`, and on the compiled engine every weight of such a component is
pinned on the host — the token embedding the audio-LLM flow reads by name included. The
flow built the index of each decoded token on the card and looked it up in the host table:

    Expected all tensors to be on the same device, but got index is on cuda:0, different
    from other tensors on cpu (... wrapper_CUDA__index_select)

The prompt's two lookups had been moved to the table's device on 2026-09-06; the decode
step, the third copy of the same line, had not. `core/flow/table_gather.gather_rows` is now
where a compiled flow's embedding lookups run, and the door at the bottom refuses an
`embedding(...)` call anywhere else under core/flow (a lookup written as a subscript is
beyond what it can see).

What each test would do if the code were wrong:
  * the card tests raise torch's cross-device error (seen: they are red on the tree before
    the fix, and red again when the direct lookup is injected back);
  * the door names the file and line of any flow that calls an embedding lookup itself.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest
import torch

from neurobrix.core.flow import audio_llm as AL
from neurobrix.core.flow import audio_utils
from neurobrix.core.flow import table_gather
from neurobrix.core.flow.autoregressive import GraphLMSession
from neurobrix.core.flow.table_gather import gather_rows

HOST = "cpu"
# A second device beside the host: a CUDA card, or Apple's (the cross-device lookup and the
# mixed-device cat are refused on both).
CARD = ("cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else None)
needs_card = pytest.mark.skipif(
    CARD is None, reason="needs two devices: a table on the host and a context on a card")
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reads the CUDA allocator's peak")


def _on(tensor, device) -> bool:
    return tensor.device.type == torch.device(device).type


def _table(vocab=16, hidden=8, device=HOST, dtype=torch.float32):
    # Row r is r*100 + column: a gathered row names the row it came from.
    rows = torch.arange(vocab, dtype=torch.float32).unsqueeze(1) * 100.0
    return (rows + torch.arange(hidden, dtype=torch.float32)).to(device=device, dtype=dtype)


# ── the brick ────────────────────────────────────────────────────────────────

def test_on_one_device_the_gather_is_the_plain_embedding():
    table = _table()
    for ids in ([[3, 1, 4]], [5], [[9]]):
        want = torch.nn.functional.embedding(torch.tensor(ids, dtype=torch.long), table)
        got = gather_rows(table, ids, device=HOST)
        assert got.dtype == table.dtype and got.shape == want.shape
        assert torch.equal(got, want)
    index = torch.tensor([[2, 7]], dtype=torch.long)
    assert torch.equal(gather_rows(table, index, device=HOST),
                       torch.nn.functional.embedding(index, table))


def test_the_rows_are_delivered_in_the_dtype_the_caller_names():
    table = _table(dtype=torch.float16)
    got = gather_rows(table, [[3, 1]], device=HOST, dtype=torch.float32)
    assert got.dtype == torch.float32
    assert torch.equal(got, table[[3, 1]].unsqueeze(0).to(torch.float32))
    assert table.dtype == torch.float16, "the table itself is never cast"


@needs_card
@pytest.mark.parametrize("ids", [
    [[3, 1, 4]],                                             # the flow's own list
    torch.tensor([[3, 1, 4]], dtype=torch.long),             # an index on the host
    "card",                                                  # an index already on the card
])
def test_a_host_table_is_gathered_on_the_host_and_delivered_on_the_card(ids):
    table = _table(device=HOST)
    if isinstance(ids, str):
        ids = torch.tensor([[3, 1, 4]], dtype=torch.long, device=CARD)
    got = gather_rows(table, ids, device=CARD)
    assert _on(got, CARD)
    assert torch.equal(got.cpu(), table[[3, 1, 4]].unsqueeze(0))
    assert _on(table, HOST), "the table stays where its plan placed it"


@needs_cuda
def test_only_the_rows_travel_never_the_table():
    """The card was budgeted for the activations: a lookup must not put the table on it."""
    table = torch.zeros(4096, 4096, dtype=torch.float16)            # 32 MiB on the host
    table_bytes = table.numel() * table.element_size()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(CARD)
    before = torch.cuda.memory_allocated(CARD)
    rows = gather_rows(table, [[7]], device=CARD)
    torch.cuda.synchronize()
    grown = torch.cuda.max_memory_allocated(CARD) - before
    assert rows.device == torch.device(CARD)
    assert grown < table_bytes // 100, (
        f"the lookup put {grown} bytes on the card for one row of a {table_bytes}-byte table")


# ── the audio-LLM decode loop, the site that failed ──────────────────────────

class _Resolver:
    def __init__(self, resolved):
        self.resolved = resolved

    def get(self, name, default=None):
        return self.resolved.get(name, default)

    def resolve_all(self):
        return dict(self.resolved)


def _decode(monkeypatch, table, context_device, script):
    """Run AudioLLMEngine.execute over a scripted language model: the graph is replaced by
    a function that answers step k with the token script[k]; the table and the context are
    real tensors on the devices the plan would give them. Returns (token ids, contexts)."""
    vocab, hidden = table.shape
    resolved = {"projector.output_0": torch.full((1, 3, hidden), -1.0, device=context_device)}
    contexts = []

    def execute_component(name, phase, inputs):
        context = resolved["inputs_embeds"]
        contexts.append(context)
        logits = torch.full((1, context.shape[1], vocab), -1.0, device=context.device)
        logits[0, -1, script[len(contexts) - 1]] = 1.0
        resolved[f"{name}.output_0"] = logits

    stages = [{"component": "projector", "execution": "forward"},
              {"component": "language_model", "execution": "autoregressive",
               "logits_source": "self"}]
    ctx = types.SimpleNamespace(
        pkg=types.SimpleNamespace(
            topology={"flow": {"audio": {"stages": stages}}},
            defaults={"temperature": 0.0, "eos_token_id": 0, "max_tokens": 8,
                      "stt_prefix_ids": [1, 2], "stt_suffix_ids": [3]},
            manifest={}),
        variable_resolver=_Resolver(resolved),
        executors={"language_model": types.SimpleNamespace(
            _weights={"model.token_embed.weight": table}, _dag=None)},
        primary_device=context_device,
        persistent_mode=True,
        compute_dtype=lambda: torch.float32)
    monkeypatch.delenv("NBX_DECODE_BOUND", raising=False)
    monkeypatch.delenv("NBX_DECODE_PROGRESS", raising=False)
    monkeypatch.setattr(audio_utils, "preprocess_audio_input", lambda *a, **k: None)
    monkeypatch.setattr(audio_utils, "postprocess_text_output", lambda ctx: None)
    engine = AL.AudioLLMEngine(ctx, execute_component, lambda *a, **k: {},
                               lambda name: None, lambda name: None)
    engine.execute()
    return resolved["global.generated_token_ids"], contexts


def _assert_the_decode_is_the_scripted_one(table, context_device, tokens, contexts):
    assert tokens == [5, 7, 4, 0], "the scripted tokens, eos last"
    # prompt = 2 prefix rows + 3 audio rows + 1 suffix row, then one row per decoded token
    assert [c.shape[1] for c in contexts] == [6, 7, 8, 9]
    host_table = table.cpu()
    for context in contexts:
        assert _on(context, context_device), "every row joined the context"
    last = contexts[-1].cpu()
    assert torch.equal(last[0, :2], host_table[[1, 2]]), "the prefix rows"
    assert torch.equal(last[0, 5], host_table[3]), "the suffix row"
    assert torch.equal(last[0, 6:], host_table[[5, 7, 4]]), "each decoded token's own row"


def test_the_decode_loop_on_one_device(monkeypatch):
    table = _table(device=HOST)
    tokens, contexts = _decode(monkeypatch, table, HOST, script=[5, 7, 4, 0])
    _assert_the_decode_is_the_scripted_one(table, HOST, tokens, contexts)


@needs_card
@pytest.mark.parametrize("table_device", [HOST, CARD])
def test_the_decode_loop_with_the_context_on_the_card(monkeypatch, table_device):
    """table_device=HOST is granite-speech on a 16 GB card: red before the fix at the first
    decoded token ("index is on cuda:0, different from other tensors on cpu")."""
    table = _table(device=table_device)
    tokens, contexts = _decode(monkeypatch, table, CARD, script=[5, 7, 4, 0])
    _assert_the_decode_is_the_scripted_one(table, CARD, tokens, contexts)
    assert _on(table, table_device), "the table stays where its plan placed it"


# ── the language-model session every other flow decodes through ──────────────

@needs_card
def test_the_lm_session_embeds_a_card_token_from_a_host_table():
    table = _table(device=HOST)
    session = types.SimpleNamespace(
        executor=types.SimpleNamespace(get_embed_tokens=lambda: table))
    ids = torch.tensor([[6]], dtype=torch.long, device=CARD)
    rows = GraphLMSession._embed_from_ids(session, ids)
    assert rows.device == ids.device
    assert torch.equal(rows.cpu(), table[[6]].unsqueeze(0))


@needs_card
def test_a_session_without_a_kv_cache_grows_its_context_where_the_context_lives():
    """The O(n) path: prefill left the accumulated context on the table's device (the host);
    a token decoded on the card is looked up in the host table and its row joins that
    context — neither the lookup nor the cat meets two devices."""
    table = _table(device=HOST)
    ran = []
    session = types.SimpleNamespace(
        kv_wrapper=None,
        _accumulated_embeds=table[[1, 2]].unsqueeze(0).clone(),     # as prefill leaves it
        graph_inputs=set(),
        hidden_dim=table.shape[1],
        _add_visual_stubs=lambda run_inputs: None,
        executor=types.SimpleNamespace(
            get_embed_tokens=lambda: table,
            run=lambda run_inputs: ran.append(run_inputs["inputs_embeds"]),
            get_hidden_states=lambda **kw: torch.zeros(1, 3, table.shape[1])))
    session._embed_from_ids = lambda ids: GraphLMSession._embed_from_ids(session, ids)
    GraphLMSession.decode_step(session, torch.tensor([[6]], dtype=torch.long, device=CARD))
    assert len(ran) == 1 and _on(ran[0], HOST)
    assert torch.equal(ran[0], table[[1, 2, 6]].unsqueeze(0))


# ── the door: no compiled flow looks rows up by itself ───────────────────────

_FLOWS = Path(AL.__file__).parent
_BRICK = Path(table_gather.__file__)


def _embedding_calls(source: str):
    """Line numbers of every `<anything>.embedding(...)` call: `F.embedding`,
    `torch.nn.functional.embedding`, `torch.embedding`."""
    return [node.lineno for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "embedding"]


def test_the_door_sees_a_direct_lookup():
    """The scanner on a line it must refuse, and on the brick it must find: a door that
    cannot see either is no door."""
    assert _embedding_calls("import torch.nn.functional as F\nx = F.embedding(i, t)\n") == [2]
    assert _embedding_calls("x = torch.nn.functional.embedding(i, t).to(d)\n") == [1]
    assert len(_embedding_calls(_BRICK.read_text())) == 1, "the brick holds the one lookup"


def test_no_compiled_flow_looks_rows_up_outside_the_brick():
    offenders = []
    scanned = 0
    for path in sorted(_FLOWS.rglob("*.py")):
        if path == _BRICK:
            continue
        scanned += 1
        offenders += [f"{path.relative_to(_FLOWS)}:{line}"
                      for line in _embedding_calls(path.read_text())]
    assert scanned >= 15, f"mis-globbed: {scanned} flow modules under {_FLOWS}"
    assert not offenders, (
        "a flow looks rows up in a table by itself — under a host placement of that table "
        "the index and the table sit on two devices. Use core/flow/table_gather.gather_rows: "
        + ", ".join(offenders))
