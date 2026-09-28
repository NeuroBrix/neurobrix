"""The certifier prices a key by what it will allocate, phase by phase, and the price is checked
against measured peaks — never a constant per element.

Measured 2026-09-28 on the Mac (Apple M4 Pro, unified memory, engine d1996830 / 3af26e8d, one key per
process, `--out` scratch, `/usr/bin/time -l` peak memory footprint in MiB; the runs, their logs and
the malloc/phase traces are under nbx-atelier/campagnes/2026_09_22_apple/results/certifier_price):
the old price (10 bytes per element of every operand, `_unified_bytes_per_element`) was 12 % under a
small depthwise key and 40 % over a large one, refused a depthwise key whose neighbour certified,
and 2x over on a 1x1 convolution whose bulk is its output. `price_key` computes the phases the
trace showed: the host draws, the device copies, the fp64 oracle whole or windowed, the readback.

What this file does if the code is wrong: (1) the table test fails on any point outside
[0.85, 1.65] x measured — with the conv windows shrunk to 1x1 (the injection recorded below) the
depthwise points predict 0.48 x, 0.66 x and 0.82 x their measured peaks: red, seen;
(2) the refusal test fails if a key priced over the budget is drawn or is refused without its
phases and the budget in the message.

Points: (kernel, key, measured peak MiB). The floor is the process before the key: 445 MiB,
the peak of a 0-element key on this machine (p_baddbmm_0).
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as AC
from neurobrix.triton import autotune_cache as atc

FLOOR_MIB = 445
MM = "neurobrix.kernels.ops.matmul.matmul_kernel"
ADDMM = "neurobrix.kernels.ops.matmul.addmm_kernel"
BADD = "neurobrix.kernels.ops.baddbmm_op.baddbmm_kernel"
CONV = "neurobrix.kernels.ops.conv2d.conv2d_forward_kernel"
DW = "neurobrix.kernels.ops.depthwise_conv2d.depthwise_conv2d_kernel"
MEASURED = [
    (DW, (4096, 256, 320, 256, 320, 3, 3, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16'), 6840),
    (DW, (4096, 384, 512, 384, 512, 3, 3, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16'), 10423),
    (DW, (3072, 64, 80, 64, 80, 5, 5, 1, 1, 2, 2, False, 'bf16', 'bf16', 'bf16'), 1009),
    (DW, (3072, 192, 256, 192, 256, 5, 5, 1, 1, 2, 2, False, 'bf16', 'bf16', 'bf16'), 3081),
    (CONV, (1, 256, 770, 2048, 256, 770, 2048, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16'), 3562),
    (CONV, (1, 32, 64, 80, 1024, 64, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, False, 'bf16', 'bf16', 'bf16'), 556),
    (CONV, (1, 128, 1027, 2562, 3, 1025, 2560, 3, 3, 1, 1, 0, 0, 1, 1, 1, False, 'bf16', 'bf16', 'bf16'), 2933),
    (CONV, (1, 512, 512, 512, 4096, 512, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1, False, 'bf16', 'bf16', 'bf16'), 3307),
    (ADDMM, (608, 2240, 2240, False, False, 'bf16', 'bf16', 'bf16', 'bf16'), 549),
    (ADDMM, (10240, 2240, 2240, False, False, 'fp32', 'fp32', 'bf16', 'fp32'), 972),
    (ADDMM, (32768, 2240, 2240, False, False, 'fp32', 'fp32', 'bf16', 'fp32'), 1326),
    (BADD, (5120, 320, 112, False, False, False, 'fp32', 'fp32', 'fp32', 'fp32'), 550),
    (BADD, (33, 32, 262144, False, False, False, 'fp32', 'fp32', 'fp32', 'fp32'), 948),
    (MM, (512, 1024, 2304, False, False, 'bf16', 'bf16', 'bf16'), 474),
    (MM, (16384, 1024, 2048, False, False, 'bf16', 'bf16', 'bf16'), 1139),
    (MM, (262144, 512, 1024, False, False, 'bf16', 'bf16', 'bf16'), 2414),
]


@pytest.fixture(scope="module")
def tuners():
    return {q: t for q, t in atc._autotuners()}


def _check(tuners, lo=0.85, hi=1.65):
    off = []
    for qual, key, measured in MEASURED:
        pr = AC.price_key(qual, tuners[qual], key)
        assert pr is not None, qual
        ratio = (FLOOR_MIB + pr["peak"] / 2 ** 20) / measured
        if not lo <= ratio <= hi:
            off.append(f"{AC.C.kernel_short(qual)} {key[:4]}: predicted {FLOOR_MIB + pr['peak'] / 2**20:.0f} MiB "
                       f"vs measured {measured} ({ratio:.2f}x)")
    return off


def test_the_price_is_within_the_measured_peaks(tuners):
    off = _check(tuners)
    assert not off, "the price left the measured band on:\n  " + "\n  ".join(off)


def test_the_price_is_the_phases_not_a_constant(tuners):
    """Two keys of the same kernel and dtype with different operand balance price differently per
    element (the 1x1 conv whose bulk is its output vs the 3x3 whose bulk is its input), and the
    windowed conv oracle is what makes the large depthwise key expensive."""
    conv_in = AC.price_key(CONV, tuners[CONV], MEASURED[4][1])
    conv_out = AC.price_key(CONV, tuners[CONV], MEASURED[7][1])
    e_in = 2 * 256 * 770 * 2048; e_out = 512 * 512 * 512 + 4096 * 512 * 512
    assert conv_in["peak"] / e_in > 1.5 * conv_out["peak"] / e_out
    dw = AC.price_key(DW, tuners[DW], MEASURED[0][1])
    assert dw["oracle"] > dw["draws"], "the windowed conv oracle is the depthwise key's cost, not its draws"


def test_a_key_priced_over_the_budget_is_refused_by_name_before_any_draw(tuners):
    class _NoDraw:
        def __getattr__(self, name):
            raise AssertionError(f"a value was drawn ({name}) for a key refused by its price")
    key = MEASURED[1][1]                                       # 10 423 MiB measured
    with pytest.raises(AC.KeyTooLargeForClass) as e:
        AC.synthesize(DW, tuners[DW], key, _NoDraw(), budget_bytes=8000 * 2 ** 20, floor_bytes=445 * 2 ** 20)
    msg = str(e.value)
    for word in ("oracle", "draws", "budget", "8000"):
        assert word in msg, f"the refusal does not name {word!r}: {msg}"
    # and the same key under a budget it fits is drawn
    made = AC.synthesize(DW, tuners[DW], key, np.random.default_rng(0), values=False,
                         budget_bytes=16000 * 2 ** 20, floor_bytes=445 * 2 ** 20)
    assert made is not None
