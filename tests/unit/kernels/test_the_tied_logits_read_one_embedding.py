"""The weight-tied logits of an encoder-decoder (`triton/flow/encoder_decoder.tied_embedding_name`) read ONE weight by
one rule, for the flow and the derived census alike: the first `token_embed` name, else the 2-D `embed` weight with the
most rows. Injection: the rows comparison reversed -> red on the second case."""
from neurobrix.triton.flow.encoder_decoder import tied_embedding_name


def test_the_token_embedding_wins():
    assert tied_embedding_name({"a.weight": (4, 4), "model.token_embed.weight": (51866, 1280),
                                "pos_embed": (448, 1280)}) == "model.token_embed.weight"


def test_else_the_widest_two_d_embedding():
    assert tied_embedding_name({"embed_positions.weight": (448, 1280), "embed_tokens.weight": (51866, 1280),
                                "embed_scale": (1,)}) == "embed_tokens.weight"
    assert tied_embedding_name({"proj.weight": (8, 8)}) is None


def test_an_entry_without_a_shape_is_never_the_embedding():
    """The flow hands every weight entry by name, a shapeless one with an empty shape (`_get_embed_weight`): it is never
    taken for the 2-D embedding (whisper's decoder, the chantier 1 gate 2026-10-22: 'dict' has no attribute 'shape')."""
    assert tied_embedding_name({"embed_meta": (), "embed_positions.weight": (448, 1280),
                                "embed_tokens.weight": (51866, 1280)}) == "embed_tokens.weight"


def test_the_flow_finds_the_embedding_among_entries_without_a_shape():
    """`_get_embed_weight` over a decoder whose weights hold an entry that is not a tensor (a dict): the token
    embedding is still found. Injection (seen red before the fix of 2026-10-22): every entry's `.shape` read."""
    from types import SimpleNamespace
    from neurobrix.triton.flow.encoder_decoder import _get_embed_weight
    emb = SimpleNamespace(shape=(51866, 1280), ndim=2)
    ex = SimpleNamespace(_weights={"meta": {"k": 1}, "embed_positions.weight": SimpleNamespace(shape=(448, 1280), ndim=2),
                                   "embed_tokens.weight": emb})
    ctx = SimpleNamespace(executors={"model.decoder": ex})
    assert _get_embed_weight(ctx, "model.decoder") is emb
