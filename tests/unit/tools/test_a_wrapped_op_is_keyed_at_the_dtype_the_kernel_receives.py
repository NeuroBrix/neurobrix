"""An op the dtype engine wraps is keyed at the operands the KERNEL receives: an AMP fp32 op's half operands are cast
to fp32 before the wrapper (`runtime_widths`' casts), for every keyed kind — the census read them for the norms only,
and canary-qwen-2.5b's perception softmax was keyed (3072, 512, bf16, bf16) on Apple where the run hands it fp32:
KeyNotCertified on the Mac's certified-only matrix (2026-10-23 13:56).

What this test would do if the code were wrong: the softmax keyed at its stored bf16 — red at the dtype assertion.
Derived on the Mac's own profile (default-9f169c79), from the shared container, without executing anything."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derived_census as D  # noqa: E402
import certified_census as CC  # noqa: E402

MODEL, HW = "canary-qwen-2.5b", "default-9f169c79"


def test_the_amp_softmax_is_keyed_in_fp32_on_a_bf16_device(tmp_path):
    if not (D.CACHE / MODEL).exists() or not (Path(D.REPO) / "src/neurobrix/config/hardware" / f"{HW}.yml").exists():
        pytest.skip("needs the shared container and the Mac's hardware profile")
    req = CC.census_requests(MODEL, CC._family(MODEL), [], None)[0]
    derived, _plan, _P, unhandled = D.derive_keys(MODEL, HW, "triton", None, req)
    soft = [(u, k) for u, q, k in derived if q.endswith("softmax_forward_kernel") and str(u).startswith("aten._softmax")]
    assert soft, "the perception's softmaxes formed no key"
    assert all(str(k).endswith("'fp32', 'fp32')") for _, k in soft), sorted({str(k) for _, k in soft})
