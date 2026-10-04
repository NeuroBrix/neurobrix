"""The audio-LLM flow runs every token-table op where the table lives and joins
the result where the context lives.

Under a host placement of the language model (granite-speech-3.3-8b compiled on
a 16 GB card) the embedding table is on the CPU while the context is on the GPU.
The prompt's lookup was fixed on 2026-09-05 and its join on 2026-09-21, but the
decode step still built its index on the GPU and torch refused the lookup
("index is on cuda:0, different from other tensors on cpu", 2026-10-04): the
first generated token killed the run. One pair of helpers now serves the
prompt, the decode step and the logits.
"""
import pytest
import torch

from neurobrix.core.flow.audio_llm import embed_ids, project_on_table


def _old_decode_lookup(token, table, like, dtype):
    # The decode step before the fix: the index on the context's device.
    index = torch.tensor([[token]], dtype=torch.long, device=like.device)
    return torch.nn.functional.embedding(index, table).to(dtype=dtype)


def test_the_lookup_returns_the_tables_rows_in_the_requested_dtype():
    table = torch.randn(11, 6)
    like = torch.zeros(1, 3, 6)
    out = embed_ids([4, 0, 10], table, like, torch.float16)
    assert out.shape == (1, 3, 6) and out.dtype == torch.float16
    assert torch.equal(out, table[[4, 0, 10]].unsqueeze(0).to(torch.float16))


def test_the_projection_is_computed_in_the_hidden_dtype():
    # The logits were computed as hidden @ table.T with the table cast to the
    # hidden dtype; the helper keeps exactly that arithmetic.
    hidden = torch.randn(1, 1, 6, dtype=torch.float32)
    table = torch.randn(9, 6, dtype=torch.float16)
    expect = torch.matmul(hidden, table.to(torch.float32).T)
    out = project_on_table(hidden, table)
    assert out.dtype == torch.float32 and torch.equal(out, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU beside the host table")
def test_a_host_table_feeds_a_gpu_context_at_every_step():
    table = torch.randn(13, 8)                       # the host placement
    context = torch.randn(1, 2, 8, device="cuda")    # the context on the card
    with pytest.raises(RuntimeError, match="same device"):
        _old_decode_lookup(5, table, context, torch.float32)   # the defect, reproduced
    step = embed_ids([5], table, context, torch.float32)
    assert step.device == context.device
    joined = torch.cat([context, step], dim=1)                 # the join torch refused on mps
    assert torch.equal(joined[0, 2].cpu(), table[5])
    logits = project_on_table(joined[:, -1:, :], table)
    assert logits.device == context.device
    assert torch.allclose(logits.cpu(), torch.matmul(table[5], table.T).view(1, 1, -1), atol=1e-5)
