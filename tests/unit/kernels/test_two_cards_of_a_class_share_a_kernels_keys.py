"""Two cards of one memory class share a kernel's keys: `certify --shard K/N`.

The loops split a class's work by KERNEL; when one kernel is all that is left (the convolutions,
2026-09-29 09:30: GEMM and depthwise complete in both classes), the second card of the class had
nothing to take. A shard is a stable hash of the kernel and the key's text: the N shards of a key
list are disjoint and cover it, and a key keeps its shard across passes and processes.

Injections: the hash reading the key alone (not the kernel) -> the kernel-independence case RED;
`shard_spec` accepting K >= N -> RED."""
import pytest

from neurobrix.kernels import autotune_certify as AC


def _keys():
    return [(m, n, k, True, False, "fp16") for m in (1, 7, 64, 208) for n in (128, 3072) for k in (64, 640)]


def test_the_shards_partition_the_keys():
    q = "neurobrix.kernels.ops.conv.conv2d_forward_kernel"
    keys = _keys()
    parts = [[k for k in keys if AC.shard_of(q, k, 2) == s] for s in range(2)]
    assert sorted(parts[0] + parts[1]) == sorted(keys)
    assert not set(parts[0]) & set(parts[1])
    assert parts[0] and parts[1]
    assert [AC.shard_of(q, k, 2) for k in keys] == [AC.shard_of(q, k, 2) for k in keys]


def test_a_key_s_shard_depends_on_its_kernel_too():
    keys = _keys()
    a = [AC.shard_of("k.a", k, 4) for k in keys]
    b = [AC.shard_of("k.b", k, 4) for k in keys]
    assert a != b


@pytest.mark.parametrize("bad", ["2/2", "3/2", "x/2", "1"])
def test_a_malformed_shard_is_refused(bad):
    with pytest.raises(RuntimeError, match="--shard"):
        AC.shard_spec(bad)
