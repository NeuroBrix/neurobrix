"""The certifier takes a key list (`--keys`): the coverage check's missing keys, certified exactly (the supervisor,
2026-10-26 12:35). What this test would do if the code were wrong: a parser that skipped lines or accepted an empty
file would make a run that certified nothing read green — the refusals below go RED."""
import pytest
from neurobrix.kernels.autotune_certify import keys_file


def test_a_key_list_gives_each_kernel_its_keys(tmp_path):
    f = tmp_path / "k.txt"
    f.write_text("a.b.softmax_forward_kernel::(8, 64, False, 'fp16', 'fp16')\n"
                 "a.b.softmax_forward_kernel::(8, 64, False, 'fp16', 'fp16')\n"
                 "a.b.gemv_vec_kernel::(1024, 256, 'fp16', 'fp32', 'fp16')\n")
    got = keys_file(str(f))
    assert got == {"a.b.softmax_forward_kernel": [(8, 64, False, "fp16", "fp16")],
                   "a.b.gemv_vec_kernel": [(1024, 256, "fp16", "fp32", "fp16")]}


@pytest.mark.parametrize("text", ["", "no separator here\n", "a.b.k::not a tuple(\n", "a.b.k::[1, 2]\n"])
def test_an_empty_or_malformed_list_is_refused_by_name(tmp_path, text):
    f = tmp_path / "k.txt"
    f.write_text(text)
    with pytest.raises(RuntimeError, match="--keys"):
        keys_file(str(f))
