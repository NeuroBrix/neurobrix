"""The ENGINE's own table builder, driven through the wrapper the model uses.

The kernel-level proof (test_the_moe_table_reads_through_a_pinned_scope) pinned
and captured by hand. This test asserts the engine now does the same thing
itself: `_build_ptr_tables` on Metal must produce tables a shader can read
(device capture under a pin whose lifetime is the table's), and
`w.invoke_fused_moe` — the exact launch the model path takes — must match the
fp64 oracle through them. Same granite geometry, same instrument.

Red before the metal branch existed: the data_ptr tables read as silent zeros
(measured, demo_moe_table_pinned.py). Green by construction now — and the pin
assertions keep it honest: a table whose pin died is an accident, not a pass.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import (  # noqa: E402
    NBXDtype, NBXTensor, bf16_carrier_to_float32,
)

E, TOP_K, K, N = 32, 8, 1024, 512


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the pinned-table contract is this driver's")
        from neurobrix.triton import triton_ext_driver as d
        return d
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


def _bf16(a32):
    return ((np.asarray(a32, dtype=np.float32).view(np.uint32) + 0x7FFF)
            >> 16).astype(np.uint16)


def _val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def test_the_engine_table_matches_fp64_through_the_wrapper():
    drv = _metal_or_skip()
    from neurobrix.kernels import wrappers as w
    from neurobrix.triton import moe as M

    rng = np.random.default_rng(11)

    a_bits = _bf16(rng.standard_normal((1, K)) * 0.05)
    a = NBXTensor.from_numpy(a_bits, dtype=NBXDtype.bfloat16)

    # Engine convention: experts carry torch Linear layout [N, K] (out, in);
    # _build_ptr_tables hands the kernel stride_bk=stride(1)=1,
    # stride_bn=stride(0)=K for exactly that layout.
    def _experts():
        nbx, f64 = [], []
        for _ in range(E):
            wm = _bf16(rng.standard_normal((N, K)) * 0.05)
            f64.append(_val(wm))
            nbx.append(NBXTensor.from_numpy(wm, dtype=NBXDtype.bfloat16))
        return nbx, f64

    gate_w, gate64 = _experts()

    tables = M._build_ptr_tables(gate_w, gate_w, gate_w)

    # The pin half of the contract: held now, for the table's lifetime.
    assert tables.pins, "metal tables built without a pin are an accident"
    assert drv._PIN_COUNTS, "the pin holds no addresses"

    chosen = np.array([3, 17, 4, 29, 11, 0, 25, 8], dtype=np.int32)
    weights = rng.random(TOP_K).astype(np.float32)
    weights /= weights.sum()

    # One sorted block per assignment, exactly as the kernel-level proof.
    BM = 16
    EM = TOP_K * BM
    sorted_ids = np.full(EM, TOP_K, dtype=np.int32)
    for s in range(TOP_K):
        sorted_ids[s * BM] = s

    t_w = NBXTensor.from_numpy(weights)
    t_sid = NBXTensor.from_numpy(sorted_ids)
    t_eid = NBXTensor.from_numpy(chosen)
    t_np = NBXTensor.from_numpy(np.array([EM], dtype=np.int32))

    a64 = _val(a_bits)
    oracle = np.stack([weights[s] * (a64[0] @ gate64[chosen[s]].T)
                       for s in range(TOP_K)])

    out = NBXTensor.from_numpy(_bf16(np.zeros((TOP_K, N))),
                               dtype=NBXDtype.bfloat16)
    dev = 0
    w.invoke_fused_moe(
        a, tables.gate_ptrs[dev], out, t_w, t_sid, t_eid, t_np,
        N, K,
        tables.gate_stride_bk[dev], tables.gate_stride_bn[dev],
        TOP_K, mul_routed_weight=True,
    )

    got = bf16_carrier_to_float32(out.numpy()).astype(np.float64)
    live = int((np.abs(got) > 0).sum())
    assert live == TOP_K * N, (
        f"only {live}/{TOP_K * N} outputs nonzero: the ENGINE's table did not "
        f"resolve — the silent-zeros defect through _build_ptr_tables")
    rel = np.abs(got - oracle) / np.maximum(np.abs(oracle), 1e-3)
    assert float(rel.max()) < 5e-2, (
        f"engine table diverges from fp64 oracle: max rel {rel.max():.3e}")

    tables.release_pins()
    assert not drv._PIN_COUNTS, "release_pins left addresses pinned"


def test_the_capability_answers_from_the_selected_driver():
    _metal_or_skip()
    from neurobrix.kernels.nbx_tensor import backend_loads_pointers_from_memory
    from neurobrix.triton.metal_backend import selected_metal_backend
    assert selected_metal_backend() == "triton_ext"
    assert backend_loads_pointers_from_memory() is True, (
        "on triton_ext the capability must now say True — the refusal in "
        "execute_moe_fused lifts through this door and no other")


def test_an_address_without_a_pin_is_refused_not_guessed():
    drv = _metal_or_skip()
    t = NBXTensor.from_numpy(np.zeros(16, dtype=np.float32))
    with pytest.raises(RuntimeError, match="not under any pinned_addresses"):
        drv.pinned_gpu_address(t.data_ptr())


def test_the_pinned_authority_reads_across_command_buffers():
    """The decisive arm the capture design failed: a D2H sync between the
    table build and the read splits them into different command buffers,
    and an address from a transient wrap then reads zeros. The pin's
    kept-wrap authority (a pinned lifetime) must survive exactly this."""
    drv = _metal_or_skip()
    import triton
    import triton.language as tl
    from neurobrix.kernels.launcher import launch

    @triton.jit
    def _read_first(table_ptr, out_ptr, E: tl.constexpr):
        e = tl.arange(0, E)
        a = tl.load(table_ptr + e)
        p = tl.cast(a, tl.pointer_type(tl.float32), bitcast=True)
        tl.store(out_ptr + e, tl.load(p))

    vals = np.zeros((8, 64, 32), dtype=np.float32)
    for e in range(8):
        vals[e, 0, 0] = 100.0 + e
    W = NBXTensor.from_numpy(vals)
    views = [W.select(0, e) for e in range(8)]
    with drv.pinned_addresses(*views):
        tab = NBXTensor.from_numpy(np.array(
            [drv.pinned_gpu_address(v.data_ptr()) for v in views],
            dtype=np.int64))
        # force a command-buffer boundary between build and read
        _ = tab.numpy()
        out = NBXTensor.from_numpy(np.zeros(8, dtype=np.float32))
        launch(_read_first, (1,), tab, out, E=8)
        got = out.numpy()
    assert list(got) == [100.0 + e for e in range(8)], got


def test_the_resolver_slices_stacked_experts_by_the_traced_law():
    """gate = rows 0:F of W_in[e], up = rows F:2F, down = W_out[e] — the
    exact halves the traced granite block splits at (aten.split at F on
    the mm output's last dim)."""
    from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
    E, F, H = 4, 3, 5
    w_in = NBXTensor.from_numpy(
        np.arange(E * 2 * F * H, dtype=np.float32).reshape(E, 2 * F, H))
    w_out = NBXTensor.from_numpy(
        np.arange(E * H * F, dtype=np.float32).reshape(E, H, F) * -1.0)
    attrs = {"stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out",
                                 "ffn_dim": F},
             "num_experts": E,
             "expert_gate_weight_ids": [], "expert_up_weight_ids": [],
             "expert_down_weight_ids": []}
    lut = {"in": w_in, "out": w_out}
    g, u, d = expert_weight_lists(attrs, lut.get)
    for e in range(E):
        base = w_in.numpy()[e]
        assert np.array_equal(g[e].numpy(), base[:F]), f"gate half, expert {e}"
        assert np.array_equal(u[e].numpy(), base[F:]), f"up half, expert {e}"
        assert np.array_equal(d[e].numpy(), w_out.numpy()[e]), f"down, expert {e}"
