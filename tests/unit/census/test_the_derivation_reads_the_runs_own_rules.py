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
