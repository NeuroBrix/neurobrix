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
