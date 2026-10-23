"""Workstream 1: the softmax kernel's rows per program and warps are certified per key, the key derived from the graph
(`launch_keys.softmax_launches`, read by the census and by the wrapper's autotuner key) — and the candidates a key
sweeps never hold more than the kernel already holds for one row (a wider tile compiled for minutes per candidate)."""
from neurobrix.kernels import launch_keys as LK
from neurobrix.kernels.ops import softmax as S


def test_a_short_row_is_keyed_by_its_rows_bucket_padded_length_log_and_dtypes():
    (kern, key), = LK.softmax_launches(30720, 1500, False, "fp32", "fp32")
    assert kern == LK.SOFTMAX and key == (LK.bucket_of("M", 30720), 2048, False, "fp32", "fp32")


def test_every_length_that_pads_to_one_tile_shares_its_key():
    """A decoder's attention row grows by one position per token: keyed by the exact length it made one key per
    position (11 504 softmax keys on 16 GB); the program's tile is the padded row, so the class is the padded row."""
    keys = {LK.softmax_launches(8, n, False, "fp32", "fp32")[0][1] for n in range(1025, 2049)}
    assert len(keys) == 1 and LK.softmax_feat_bucket(1024) == 1024 and LK.softmax_feat_bucket(1) == 1


def test_a_row_past_the_fused_kernel_forms_the_looped_kernel_s_key():
    """Chantier 1: the looped kernel is certified too — a long row forms its key, keyed as the fused one's."""
    (kernel, key), = LK.softmax_launches(8, LK.SOFTMAX_FUSED_MAX_FEAT + 1, False, "fp32", "fp32")
    assert kernel == LK.SOFTMAX_LOOPED and key[1] == 2 * LK.SOFTMAX_FUSED_MAX_FEAT


def test_the_looped_candidates_stay_under_the_cap_and_the_row():
    keep = S._prune_looped_softmax(S._looped_softmax_configs(), {"feat_dim": 20000})
    assert keep and all(c.kwargs["TILE_FEAT"] <= 32768 for c in keep)
    assert any(c.kwargs["TILE_FEAT"] == 4096 and c.num_warps == 4 for c in keep)   # the tile it ran before


def test_the_candidates_hold_at_most_one_longest_row_and_never_more_rows_than_exist():
    keep = S._prune_softmax(S._softmax_configs(), {"feat_dim": 1500, "batch_dim": 30720})
    assert keep
    assert all(c.kwargs["BLOCK_SIZE_BATCH"] * 2048 <= LK.SOFTMAX_FUSED_MAX_FEAT for c in keep)
    few = S._prune_softmax(S._softmax_configs(), {"feat_dim": 16, "batch_dim": 3})
    assert all(c.kwargs["BLOCK_SIZE_BATCH"] <= 3 for c in few)
