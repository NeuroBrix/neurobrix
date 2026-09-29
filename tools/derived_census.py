#!/usr/bin/env python3
"""The DERIVED census (the owner's ruling, 2026-09-29 01:37): the autotune keys a model's runs form,
computed from its graphs and the hardware profile — never by executing it.

    python tools/derived_census.py compare --model M --hardware PROFILE --rung MB --mode triton \\
        --walked <census logs dir>

For each component the tool reads, and computes nothing a run would compute differently:
  * the PLAN the runtime would receive — `neurobrix run --explain-plan --json` under the census door
    (no card visible, NBX_CENSUS=1) at the rung's budget: the engine's own planning path;
  * the SYMBOLS at the request — bound with the runtime's `SymbolResolver`; an autoregressive
    prefill's length is the prompt's token count through the container's own tokenizer;
  * the RUNTIME DTYPES — `runtime_widths.runtime_dtypes` under the plan-time contract;
  * the KEYS — `kernels.launch_keys`, the functions the wrappers themselves decide with.

`compare` sets the derived (op, key) pairs against a walked census's `.ops` pairs for the same
model, mode and rung: reproduced / missed / extra. This is the acceptance test's unit.

Stage: the matmul family and the attention math route on an autoregressive LM's prefill. Every
op kind or phase not yet derived is COUNTED and named, never skipped in silence.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

# The engine's own answer (NEUROBRIX_CACHE, then the configured home, then the default) — the
# catalogue the run reads is the catalogue the derivation reads (the Mac's NAS catalogue, 03:52).
from neurobrix.core.paths import cache_dir as _cache_dir  # noqa: E402

CACHE = _cache_dir()
MODE_FLAGS = {"triton": "--triton", "triton-sequential": "--triton-sequential"}


def plan_record(model: str, request: list, mode: str, hardware: str, rung: int) -> dict:
    """The plan the runtime would receive, from the engine's own `--explain-plan --json`."""
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "NBX_CENSUS": "1", "NBX_CENSUS_DEVICES": "1",
                "PYTHONPATH": str(REPO / "src"),
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    if rung is not None:
        env["NBX_PRISM_BUDGET_MB"] = str(int(rung))
    r = subprocess.run([sys.executable, "-m", "neurobrix", "run", "--model", model, *request,
                        MODE_FLAGS[mode], "--hardware", hardware, "--explain-plan", "--json"],
                       env=env, capture_output=True, text=True, timeout=600, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"{model}: the plan could not be read (rc {r.returncode}): {r.stderr[-800:]}")
    return json.loads(r.stdout)


def container_tokenizer(model: str):
    """The container's tokenizer module, where the executor loads it from."""
    from neurobrix.core.module.tokenizer.sp_tokenizer import load_tokenizer_from_path
    root = CACHE / model
    topo = json.loads((root / "topology.json").read_text())
    tok = (topo.get("modules") or {}).get("tokenizer")
    path = ((tok.get("path") if isinstance(tok, dict) else tok) or "modules/tokenizer")   # the executor's default
    if not (root / str(path).rstrip("/")).exists():
        raise SystemExit(f"{model}: no tokenizer module at {path}")
    return load_tokenizer_from_path(root / str(path).rstrip("/"), None)


def prompt_tokens(model: str, prompt: str) -> int:
    """The prefill length: the prompt through the container's own tokenizer by the autoregressive
    flow's own rule (`triton/flow/autoregressive.prompt_token_ids`: SFT format / chat template /
    basic encode with specials, unpadded)."""
    from neurobrix.triton.flow.autoregressive import prompt_token_ids
    root = CACHE / model
    topo = json.loads((root / "topology.json").read_text())
    defaults = json.loads((root / "runtime" / "defaults.json").read_text()) if (root / "runtime" / "defaults.json").exists() else {}
    tokenizer = container_tokenizer(model)
    gen_type = ((topo.get("flow") or {}).get("generation") or {}).get("type")
    return len(prompt_token_ids(tokenizer, prompt, defaults, gen_type == "autoregressive_image",
                                bool(defaults.get("chat_mode", False))))


def derive_component(model: str, comp: str, cdtype: str, mode: str, symbols: dict,
                     has_native_bf16: bool, sdpa_budget_bytes: int, sdpa_min_rows: int,
                     sdpa_max_chunks: int, unhandled: collections.Counter, tiling=None,
                     tiled_tf=None, decode_kv=None):
    """[(op uid, kernel qual, key tuple)] for one component at one symbol binding. `tiling` is
    the plan's `TilingView` for the component (its op-level tiling, empty when it has none)."""
    from neurobrix.core.prism import runtime_widths as RW
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.symbols import SymbolResolver
    g = runtime_graph(model, comp)
    res = SymbolResolver(g.get("symbolic_context") or {})
    for sid, v in symbols.items():
        res._bind(sid, int(v))
    T, ops = g["tensors"], g["ops"]

    # Tensors computed from parameters and constants alone cannot depend on the request: their
    # shape is the traced one, whatever their annotation says (Flex's transposed context-embedder
    # weight annotates T5's 4096 width with the image-token symbol — 4096 at the trace).
    fixed = set(t for t, m in T.items() if m.get("is_parameter") or m.get("constant")
                or t.startswith(("param::", "buffer::")))
    for uid_ in g["execution_order"]:
        o_ = ops[uid_]
        ins_ = o_.get("input_tensor_ids") or []
        if ins_ and all(t in fixed for t in ins_):
            fixed.update(o_.get("output_tensor_ids") or [])

    def shape(tid):
        ss = T[tid].get("symbolic_shape")
        if tid in fixed:
            # A parameter is its stored shape — no request moves it (an annotation on it that
            # names a symbol is a collision: real-esrgan's [64, 192, 3, 3] weight read H + W).
            return list(T[tid]["shape"])
        if isinstance(ss, dict) and ss.get("dims"):
            conc = ss.get("concrete") or T[tid].get("shape") or []
            for i, d in enumerate(ss["dims"]):
                # An expression whose own recorded trace value is not the tensor's extent at the
                # trace contradicts itself (a symbol on a broadcast dim, a product chain that
                # multiplied a dim into itself): nothing correct can be derived from it.
                if isinstance(d, dict) and "trace" in d and i < len(conc) and d["trace"] != conc[i]:
                    raise AnnotationContradiction(f"{comp}: {tid} dim {i} ({d.get('type')}) records "
                                                  f"trace {d['trace']} for an extent of {conc[i]}")
            return [res.resolve(d) for d in ss["dims"]]
        return list(T[tid]["shape"])

    contract = RW.plan_time_contract(CACHE / model, comp, g, cdtype)
    engine = "triton" if mode == "triton" else "triton_sequential"
    try:
        rt = RW.runtime_dtypes(g, cdtype, engine, has_native_bf16=has_native_bf16, contract=contract,
                               tiling=tiling, shape_of=shape)
    except AnnotationContradiction as e:
        # The width pass reads the shapes too (the matmul store rule reads M): a component whose
        # annotation contradicts itself is not derivable as a whole — named, never guessed.
        unhandled[f"component not derivable, annotation contradicts its trace — {e}"] += 1
        return []
    dt0 = lambda tid: NBXDtype[rt[tid]] if rt.get(tid) in NBXDtype.__members__ else NBXDtype[{"float16": "float16", "float32": "float32", "bfloat16": "bfloat16"}[rt[tid]]]
    from neurobrix.kernels import wrappers as _W
    conv_band_bytes = _W._NBX_CONV2D_BAND_BYTES
    out = []
    half = cdtype in ("float16", "bfloat16")
    for uid in g["execution_order"]:
        o = ops[uid]
        kind = o["op_type"]
        ins = [t for t in o["input_tensor_ids"]]
        # A contract island (half compute): `TritonDtypeEngine._wrap_fp32` casts every float
        # operand to fp32 before the wrapper — weights included — whatever the op's class.
        dt = (lambda tid: NBXDtype.float32 if dt0(tid).name in ("float16", "bfloat16") else dt0(tid)) \
            if (half and uid in contract.fp32_op_uids) else dt0
        try:
            launches = _op_launches(kind, uid, o, ins, shape, dt, LK, contract, cdtype, has_native_bf16,
                                    sdpa_budget_bytes, sdpa_min_rows, sdpa_max_chunks, conv_band_bytes,
                                    unhandled, (tiled_tf or {}).get(uid), decode_kv)
        except AnnotationContradiction as e:
            unhandled[f"annotation contradicts its trace — {str(e).split(' dim ')[0]}"] += 1
            continue
        except LK.ConvRowOverBand as e:
            unhandled[f"conv row over the band budget — {e}"] += 1
            continue
        if launches is None:
            continue
        for q_, key in launches:
            out.append((uid, q_, key))
    return out


def _mask_numel(o: dict, shape):
    """The element count of an SDPA op's attn_mask argument (positional 3 or the kwarg), None
    when it has none — what `_try_decode_vec` tests."""
    a_ = (o.get("attributes") or {})
    args = a_.get("args") or []
    m = args[3] if len(args) > 3 else (a_.get("kwargs") or {}).get("attn_mask")
    if not isinstance(m, dict) or m.get("type") != "tensor":
        return None
    n = 1
    for d in shape(m["tensor_id"]):
        n *= d
    return n


def _contraction(k_act: int, k_weight: int, ins, o, uid, unhandled) -> int:
    """A matmul's contraction length: the activation's last dim must equal the weight's first —
    the op cannot run otherwise. When the two annotations disagree at the request, the weight's
    (a parameter's stored extent, never moved by a request) is the op's contract, and the
    activation's annotation collided with a trace value (mochi's 3072 features annotated as a
    spatial expression that gives 3168): derived from the contract, the collision named."""
    if k_act == k_weight:
        return k_act
    unhandled[f"NOTE matmul contraction from the weight ({k_weight}), the activation's annotation "
              f"says {k_act} — an extent collision in the container"] += 1
    return k_weight


_RUNTIME_GRAPHS = {}


def runtime_graph(model: str, comp: str) -> dict:
    """The component's graph as the executor RUNS it — its own load-time passes, in its order
    (GraphExecutor.__init__): the SDPA scaling normalised, the K/V layout marked, the MoE experts
    fused (`detect_and_fuse_moe`, every mode; declared for a MoE LM packaged under another family,
    `set_moe_config`). The calibration record is signed on this graph (DeepSeek-Coder's record
    matches it and not the raw one), and a fused expert launches no autotuned matmul."""
    key = (model, comp)
    if key not in _RUNTIME_GRAPHS:
        from neurobrix.core.runtime.graph_executor import GraphExecutor
        from neurobrix.core.runtime.graph.moe_fusion import detect_and_fuse_moe
        g = json.loads((CACHE / model / "components" / comp / "graph.json").read_text())
        stub = _Stub(_dag=g, _SDPA_LAYOUT_PASSTHROUGH=GraphExecutor._SDPA_LAYOUT_PASSTHROUGH,
                     _SDPA_OP_TYPES=GraphExecutor._SDPA_OP_TYPES)
        GraphExecutor._normalize_sdpa_scaling(stub)
        GraphExecutor._mark_sdpa_k_layout(stub)
        manifest = json.loads((CACHE / model / "manifest.json").read_text())
        g = detect_and_fuse_moe(stub._dag, manifest.get("family"), norm_topk_prob=True)
        if comp == declared_moe_lm(model):
            # the flow's `set_moe_config` on its LM (lm_config.num_experts > 1): fused, declared
            g = detect_and_fuse_moe(g, manifest.get("family"), norm_topk_prob=True, declared=True) \
                if not any(o.get("op_type") == "custom::moe_fused"
                           for o in (g.get("ops") or {}).values()) else g
        _RUNTIME_GRAPHS[key] = g
    return _RUNTIME_GRAPHS[key]


def declared_moe_lm(model: str):
    """The component a flow declares a MoE LM (`set_moe_config`, when the package's lm_config
    names num_experts > 1): the vlm flow's `lm_component`, the decode session's LM."""
    root = CACHE / model
    dp = root / "runtime" / "defaults.json"
    lmc = (json.loads(dp.read_text()) if dp.exists() else {}).get("lm_config") or {}
    if not (lmc.get("num_experts") or 0) > 1:
        return None
    topo = json.loads((root / "topology.json").read_text())
    flow = topo.get("flow") or {}
    if flow.get("type") == "vlm":
        return (flow.get("vlm") or {}).get("lm_component")
    if flow.get("type") == "autoregressive_generation":
        from neurobrix.triton.flow.autoregressive import session_lm_name
        return session_lm_name(flow.get("generation") or {}, list(topo.get("components") or {}))
    return None


class AnnotationContradiction(ValueError):
    """A container's symbolic dim contradicts its own trace extent (a Forge annotation defect)."""


def _op_launches(kind, uid, o, ins, shape, dt, LK, contract, cdtype, has_native_bf16,
                 sdpa_budget_bytes, sdpa_min_rows, sdpa_max_chunks, conv_band_bytes, unhandled,
                 tile_factor=None, decode_kv=None):
    """The launches of one op; None when the op launches no autotuned kernel."""
    from neurobrix.kernels.nbx_tensor import NBXDtype
    if True:  # noqa: SIM108 — the dispatch reads as the wrapper table it mirrors
        if kind in ("aten::mm",):
            (M, K), (Kb, N) = shape(ins[0]), shape(ins[1])
            K = _contraction(K, Kb, ins, o, uid, unhandled)
            launches = LK.mm_launches(M, K, N, dt(ins[0]), dt(ins[1]), has_native_bf16)
        elif kind == "aten::bmm":
            (_, M, K), (_, _, N) = shape(ins[0]), shape(ins[1])
            launches = LK.bmm_launches(M, K, N, dt(ins[0]), dt(ins[1]), has_native_bf16)
        elif kind in ("aten::scaled_dot_product_attention", "aten::_scaled_dot_product_attention",
                      "aten::_scaled_dot_product_efficient_attention",
                      "aten::_scaled_dot_product_flash_attention",
                      "aten::_scaled_dot_product_flash_attention_for_cpu"):
            # every SDPA spelling reaches `scaled_dot_product_attention_wrapper` with (q, k, v)
            # first (dispatch.py: the fused-backend spellings through `_meta_sdpa_efficient`)
            q, k, v = (shape(t) for t in ins[:3])
            B, H, Tq, D = q
            at_ = o.get("attributes") or {}
            # K's and V's layout is the graph's own answer (`GraphExecutor._mark_sdpa_k_layout`,
            # run on this graph by derive_component), never read from shapes; the head dims are
            # Q's and V's feature axes (a pre-transposed K's head axis may carry a colliding
            # annotation: mochi's K^T annotates its 128 as a spatial expression).
            if at_.get("nbx_k_pre_transposed"):
                k = [k[0], k[1], k[3], k[2]]
            if at_.get("nbx_v_pre_transposed"):
                v = [v[0], v[1], v[3], v[2]]
            Hk, Tk, Dv = k[1], k[2], v[3]
            qd, kd, vd = dt(ins[0]), dt(ins[1]), dt(ins[2])
            mask_n = _mask_numel(o, shape)
            q_round = None
            if decode_kv is not None and "uids" in decode_kv and uid not in decode_kv["uids"]:
                decode_kv = None      # a cross-attention: K/V the encoder's, never the cache
            if decode_kv is not None and decode_kv.get("prefill"):
                # A prefill through the KV interceptor (`intercept`, `_is_prefill`): the graph's
                # mask is dropped for `is_causal` — the SDPA sees no mask at any length.
                mask_n = None
            elif decode_kv is not None:
                # A decode step through the KV interceptor (triton/kv_cache.py `intercept`): K and V
                # are the cache's — `len` positions, its KV heads (GQA un-expanded), its dtype; Q is
                # read in the cache's dtype (`q_dtype_of_kv`); a mask of the prefill length is
                # dropped as stale.
                Tk, kd, vd = int(decode_kv["len"]), decode_kv["dtype"], decode_kv["dtype"]
                Hk = int(decode_kv.get("heads") or Hk)
                q_round = kd if qd != kd else None
                mask_n = mask_n if mask_n == Tk else None
            route, rows = LK.sdpa_route(B, H, Tq, Tk, D, Dv, sdpa_budget_bytes, sdpa_min_rows, sdpa_max_chunks)
            qd, kd, vd, _qr = LK.sdpa_operand_dtypes(qd, kd, vd, q_round)
            if _qr is not None:
                qd = _qr          # every route but the vector kernel casts Q once to the cache dtype
            if route == "flash":
                launches = []
            elif route == "chunked":
                launches = LK.chunked_math_attention_launches(B, H, Hk, Tq, Tk, D, Dv, qd, kd, vd,
                                                             has_native_bf16, rows)
            elif Tq == 1 and LK.decode_vec_takes(D, Dv, mask_n, Tk):
                launches = []                     # the vector decode kernel: no autotuned key
            else:
                launches = LK.math_attention_launches(B, H, Hk, Tq, Tk, D, Dv, qd, kd, vd, has_native_bf16)
        elif kind == "aten::addmm":
            bias_s, a_s, b_s = shape(ins[0]), shape(ins[1]), shape(ins[2])
            K = _contraction(a_s[-1], b_s[0], ins[1:], o, uid, unhandled)
            M = 1
            for d in a_s[:-1]:
                M *= d
            launches = LK.addmm_launches(M, K, b_s[-1], dt(ins[1]), dt(ins[2]), dt(ins[0]), has_native_bf16)
        elif kind in ("aten::convolution", "aten::conv1d"):
            at = o.get("attributes") or {}
            x_s, w_s = shape(ins[0]), shape(ins[1])
            nd = len(w_s) - 2
            groups_ = int(at.get("groups", 1))
            if not bool(at.get("transposed", False)) and x_s[1] != w_s[1] * groups_:
                # The weight fixes a convolution's input channels (the op cannot run otherwise);
                # an annotation that makes them a function of a spatial symbol collides with a
                # trace value (real-esrgan-x4: 192 channels annotated as an expression that
                # gives 896 at 448 px). Derived from the contract; the collision is named.
                unhandled[f"NOTE conv input channels from the weight ({w_s[1] * groups_}), the "
                          f"annotation says {x_s[1]} — a channel/extent collision in the container"] += 1
                x_s = [x_s[0], w_s[1] * groups_, *x_s[2:]]
            stride = at.get("stride", [1] * nd)
            padding = at.get("padding", [0] * nd)
            dilation = at.get("dilation", [1] * nd)
            transposed = bool(at.get("transposed", False))
            groups = int(at.get("groups", 1))
            # aten::convolution is a lower-precision op for the dtype engine: its float operands
            # are cast to the compute dtype C, and C is the store dtype (`_NBX_COMPUTE_DTYPE`);
            # an fp32 island runs it all in fp32. A compute dtype that is not half leaves the
            # operands at their producers' dtypes.
            C_ = NBXDtype.float16 if cdtype == "float16" else (NBXDtype.bfloat16 if cdtype == "bfloat16" else NBXDtype.float32)
            if uid in contract.fp32_op_uids:
                xd = wd = comp_d = NBXDtype.float32
            elif C_ in (NBXDtype.float16, NBXDtype.bfloat16):
                xd = wd = comp_d = C_
            else:
                xd, wd, comp_d = dt(ins[0]), dt(ins[1]), C_
            if tile_factor is not None and nd == 2 and not transposed:
                # Prism's op-level tiled conv (`tiled_conv2d_spatial`, self-managed: no AMP cast —
                # its per-band conv2d_wrapper narrows and stores at the compute dtype).
                launches = LK.tiled_conv2d_launches(
                    x_s[0], x_s[1], x_s[2], x_s[3], w_s[0], w_s[2], w_s[3], stride[0], stride[1],
                    padding[0], padding[1], dilation[0], dilation[1], groups, dt(ins[0]), dt(ins[1]),
                    C_ if C_ in (NBXDtype.float16, NBXDtype.bfloat16) else None, conv_band_bytes,
                    tile_factor)
            else:
                launches = LK.conv_launches(x_s, w_s, stride, padding, dilation, transposed, groups,
                                            xd, wd, comp_d, conv_band_bytes)
        elif kind == "aten::lstm":
            # aten::lstm(input, (h0, c0), params, has_biases, num_layers, dropout, train,
            # bidirectional, batch_first) — `lstm_wrapper`'s signature, the args as traced
            a_ = (o.get("attributes") or {}).get("args") or []
            val = lambda i: a_[i].get("value") if i < len(a_) else None
            x_s = shape(ins[0])
            batch_first = bool(val(8))
            if len(x_s) == 2:
                B_, T_, I_ = 1, x_s[0], x_s[1]
            elif batch_first:
                B_, T_, I_ = x_s
            else:
                T_, B_, I_ = x_s
            H_ = shape(ins[1])[-1]
            launches = LK.lstm_launches(B_, T_, I_, H_, int(val(4)), bool(val(7)), dt(ins[0]),
                                        has_native_bf16)
        elif kind in ("aten::_fft_r2c", "aten::_fft_c2r"):
            # _fft_r2c(x, dim, normalization, onesided); _fft_c2r(x, dim, normalization,
            # last_dim_size) — the transform over one dim, every other dim flattened into M
            a_ = (o.get("attributes") or {}).get("args") or []
            x_s = shape(ins[0])
            dims = a_[1].get("value")
            d = (dims[0] if isinstance(dims, list) else dims) % len(x_s)
            M = 1
            for i, e in enumerate(x_s):
                if i != d:
                    M *= e
            if kind == "aten::_fft_r2c":
                launches = LK.dft_r2c_launches(M, x_s[d], bool(a_[3].get("value")), has_native_bf16)
            else:
                launches = LK.dft_c2r_launches(M, x_s[d], int(a_[3].get("value")), has_native_bf16)
        elif kind in ("aten::linear", "aten::matmul"):
            # AMP_FP16 ops: `_wrap_lower_precision` casts the float operands to the compute dtype
            # when it is half (an island's operands are fp32 — `dt` answers that); then
            # `matmul_wrapper` routes by rank (`linear_wrapper` passes the weight transposed).
            a_s, w_s = shape(ins[0]), shape(ins[1])
            b_s = [w_s[1], w_s[0]] if kind == "aten::linear" and len(w_s) == 2 else w_s
            C_ = {"float16": NBXDtype.float16, "bfloat16": NBXDtype.bfloat16}.get(cdtype)
            ad, bd = dt(ins[0]), dt(ins[1])
            if C_ is not None and not (ad == bd == NBXDtype.float32 and uid in contract.fp32_op_uids):
                ad = C_ if ad.name in ("float16", "bfloat16", "float32") else ad
                bd = C_ if bd.name in ("float16", "bfloat16", "float32") else bd
            launches = LK.matmul_launches(a_s, b_s, ad, bd, has_native_bf16)
        elif kind in ("aten::baddbmm", "aten::stft", "aten::istft"):
            unhandled[f"{kind} (not yet derived)"] += 1
            launches = []
        else:
            return None
        return launches


def walked_pairs(walked: Path, model: str, mode: str, rung):
    """{(op uid or None, kernel, key repr)} from a walked census's `<rec>.ops` + `<rec>` files: the
    rung's `<model>.<mode>.r<rung>[.walk].keys`, or — `rung` None, a census taken at the profile's
    own budget (the Mac's 18 GB walks) — `<model>.<mode>[.walk].keys`. A `.probe.` file holds the
    probe request's keys, not the walk's, and is never read here."""
    pairs = set()
    pats = ([f"{model}.{mode}.r{rung}.keys", f"{model}.{mode}.r{rung}.walk.keys"] if rung is not None
            else [f"{model}.{mode}.keys", f"{model}.{mode}.walk.keys"])
    for rec in sorted(p for pat in pats for p in walked.glob(pat)):
        ops = Path(str(rec) + ".ops")
        with_op = set()
        if ops.exists():
            for line in ops.read_text().splitlines():
                op, _, kl = line.partition("\t")
                q_, _, k = kl.partition("::")
                pairs.add((op, q_, k)); with_op.add(kl)
        for kl in rec.read_text().splitlines():
            if kl not in with_op:
                q_, _, k = kl.partition("::")
                pairs.add((None, q_, k))
    # Logs walked before the conv key's fp16 flag was fixed (aaad1c48) carry False for fp16 inputs:
    # read them as the migration re-keyed the table (tools/migrate_conv_fp16_key.py `migrated`).
    import migrate_conv_fp16_key as _MG
    from neurobrix.kernels import autotune_certified as _C
    fixed = set()
    for op, q_, k in pairs:
        m = _MG.migrated(_C.parse_key(k)) if q_ == _MG.KERNEL else None
        fixed.add((op, q_, _C.key_repr(m) if m else k))
    return fixed


def tile_bindings(g: dict, syms: dict, spec: dict) -> list:
    """The symbol bindings a component runs at under the plan's COMPONENT tiling (`plan
    .component_tiling[comp]`, the spec the executor builds its `TilingEngine` from): the
    executor tiles the FIRST 4-D/5-D input (`_find_spatial_input`) when `should_tile` says so;
    every spatial tile is padded to tile_size x tile_size (`_extract_tile`), and a temporal tile
    is t_tile frames (edge tiles snapped by position, `_compute_axis_positions`) or the whole
    extent below it. The tile is fed under that input's name, so the runtime resolver binds the
    input's symbols from the tile's shape — so does this."""
    from neurobrix.triton.symbols import SymbolResolver
    table = (g.get("symbolic_context") or {}).get("symbols") or {}
    res = SymbolResolver(g.get("symbolic_context") or {})
    for sid, v in syms.items():
        res._bind(sid, int(v))
    T = g["tensors"]
    for tid in g.get("input_tensor_ids") or [t for t, m in T.items() if m.get("input_name")]:
        ss = T[tid].get("symbolic_shape")
        full = ([res.resolve(d) for d in ss["dims"]] if isinstance(ss, dict) and ss.get("dims")
                else list(T[tid]["shape"]))
        if len(full) in (4, 5):
            break
    else:
        return [syms]
    ts, t_tile = int(spec["tile_size"]), spec.get("t_tile")
    temporal = t_tile is not None and len(full) == 5
    if not (full[-2] > ts or full[-1] > ts or (temporal and full[2] > t_tile)):
        return [syms]
    tile = list(full)
    tile[-2] = tile[-1] = ts
    if temporal:
        tile[2] = min(int(t_tile), full[2])
    prefix = f"{tid}::dim_"
    out = dict(syms)
    for sid, info in table.items():
        src = info.get("source") or ""
        if src.startswith(prefix):
            out[sid] = tile[int(src[len(prefix):])]
    return [out]


def plan_tiling(plan: dict, comp: str):
    """(TilingView, {tiled op uid: tile factor}) of a component — the plan's op-level cut as the
    width pass and the tiled launches read it (empty when the component has none)."""
    from neurobrix.core.prism import runtime_widths as RW_
    ot = (plan.get("op_level_tiling_ops") or {}).get(comp) or {}
    fus = ot.get("fusion_pairs", [])
    til = ot.get("tiled_ops", [])
    view = RW_.TilingView(fusion_convs={f[1]: f[0] for f in fus},
                          tiled_ops=frozenset(t[0] for t in til))
    # A fused upsample->conv streams the same band cut over the upsampled extent — the conv's own
    # input in the graph — so its conv derives through the same tiled launches.
    return view, {**{f[1]: int(f[2]) for f in fus}, **{t[0]: int(t[1]) for t in til}}


class _Shape:
    """A shape the runtime's binder reads (`SymbolResolver.bind_from_inputs` reads `.shape`);
    an array feed also answers `.numpy()`, where the binder reads a value-sourced symbol."""
    def __init__(self, shape):
        self._value = shape if hasattr(shape, "dtype") else None
        self.shape = tuple(int(d) for d in (shape.shape if self._value is not None else shape))

    def numpy(self):
        if self._value is None:
            raise SystemExit("a value-sourced symbol read from a shape-only feed")
        return self._value


def run_at_inputs(model: str, comp: str, cdtype: str, mode: str, inputs: dict, has_native_bf16: bool,
                  sdpa: tuple, unhandled: collections.Counter, tiling=None, tiled_tf=None,
                  decode_kv=None):
    """(launches, {output tensor id: shape}) of one component fed `inputs` {input name: shape}:
    its symbols bound by the RUNTIME's binder from those shapes (`bind_from_inputs`), its keys
    derived, its outputs' shapes resolved at that binding — what the next stage of a flow reads."""
    from neurobrix.triton.symbols import SymbolResolver
    g = json.loads((CACHE / model / "components" / comp / "graph.json").read_text())
    res = SymbolResolver(g.get("symbolic_context") or {})
    feed = {f"input::{k}": _Shape(v) for k, v in inputs.items()}
    res.bind_from_inputs(feed, list(feed), g.get("tensors") or {})
    syms = dict(res.bindings)
    launches = derive_component(model, comp, cdtype, mode, syms, has_native_bf16, *sdpa, unhandled,
                                tiling=tiling, tiled_tf=tiled_tf, decode_kv=decode_kv)
    outs = {}
    for i, tid in enumerate(g.get("output_tensor_ids") or []):
        meta = g["tensors"].get(tid) or {}
        ss = meta.get("symbolic_shape")
        shp = ([res.resolve(d) for d in ss["dims"]] if isinstance(ss, dict) and ss.get("dims")
               else list(meta["shape"]))
        # the executor stores an output under its name AND its position (`output_<i>`), and the
        # topology's connections use either
        outs[f"output_{i}"] = shp
        if meta.get("output_name"):
            outs[meta["output_name"]] = shp
    return launches, outs


def extent_sites(model: str, topo: dict, defaults: dict, plan: dict, prompt: str = "",
                 audio_path=None, max_tokens_req=None, image_path=None):
    """The flows' value-derived extents — each site of `census.walk_extent` in the Triton flows —
    as (name, lo, hi, chain) with chain(n) = [(component, {input: shape} | callable of the
    previous stage's outputs)], built from the flows' own functions and bounds."""
    from neurobrix.core.runtime.decode_bound import decode_bound
    from neurobrix.core.runtime_values import require_max_tokens
    comps = {c["name"] for c in plan["components"]}
    flow = topo.get("flow") or {}
    sites = []
    # dual_ar (triton/flow/dual_ar.py `_decode_codes`): the acoustic code grid [1, codebooks, n]
    # through codec.quantizer, whose features feed every later codec stage; n in 1..max_tokens.
    if flow.get("type") == "dual_ar" and "codec.quantizer" in comps:
        mt = decode_bound(require_max_tokens(defaults))
        gq = json.loads((CACHE / model / "components" / "codec.quantizer" / "graph.json").read_text())
        cb = gq["tensors"]["input::indices"]["shape"][1]
        later = [st["component"] for st in ((flow.get("audio") or {}).get("stages") or [])[1:]
                 if st.get("component") in comps]

        def chain(n, cb=cb, later=later):
            return [("codec.quantizer", {"indices": [1, cb, n]})] + [
                (c, lambda outs: {"x": next(iter(outs.values()))}) for c in later]
        sites.append(("codec frames", 1, int(mt), chain))
        # The slow model runs on the whole grid [1, rows, clen] every step: clen from the prompt
        # column count P (the flows' own `dual_ar_prompt_ids` over `tts_llm_token_ids`) to
        # P + max_tokens - 1 in a live run (the shadow's pace stops at max_tokens — a walk
        # bound, named where it differs).
        if "model" in comps:
            from neurobrix.core.flow.audio_utils import (apply_tts_template, dual_ar_prompt_ids,
                                                         tts_llm_token_ids)
            from neurobrix.core.module.tokenizer.sp_tokenizer import load_tokenizer_from_path
            root = CACHE / model
            tok = load_tokenizer_from_path(root / "modules" / "tokenizer", None)
            special_p = root / "modules" / "tokenizer" / "special_tokens.json"
            special = json.loads(special_p.read_text()) if special_p.exists() else {}
            tpl = defaults.get("tts_prompt_template")
            text_ids = tts_llm_token_ids(tok, apply_tts_template(prompt, tpl), templated=tpl is not None)
            P = len(dual_ar_prompt_ids(text_ids, defaults.get("bos_token_id"),
                                       special.get("<|interleave|>"), tok))
            gm = json.loads((root / "components" / "model" / "graph.json").read_text())
            rows = gm["tensors"]["input::inp"]["shape"][1]

            def grid(n, rows=rows):
                return [("model", {"inp": [1, rows, n]})]
            sites.append(("slow model grid columns", P, P + int(mt) - 1, grid))
    # autoregressive SNAC codec (triton/flow/autoregressive.py `_run_snac_codec_decoder`): the
    # n generated tokens redistributed into three codebooks by the flow's own function.
    if defaults.get("audio_output_type") == "snac_tokens" and "codec.decoder" in comps:
        from neurobrix.core.flow.audio_utils import redistribute_snac_codes
        mt = decode_bound(require_max_tokens(defaults))
        start, vocab = defaults["audio_token_start"], defaults["vocab_size"]

        def chain(n, start=start, vocab=vocab):
            codes = redistribute_snac_codes([start] * n, start, vocab)
            if codes is None:
                return []
            return [("codec.decoder", {f"c{i}": [1, len(codes[i])] for i in range(3)})]
        sites.append(("codec.decoder audio tokens", 1, int(mt), chain))
    # autoregressive text (triton/flow/autoregressive.py): after the prefill, each decode step runs
    # the LM at ONE position against the KV cache — its length P + 1 .. P + max_tokens, K/V the
    # cache's (the plan's kv_cache: dtype, KV heads), through the interceptor's attention call.
    # An image-AR generation (the VQ image strategy) runs its fixed token count, the LM on
    # [cond, uncond] under guidance, and per step the head once per branch, the codebook embedding
    # and the aligner on one token; the VQ decoder on the whole code sequence at the end.
    if flow.get("type") == "autoregressive_generation" and plan.get("kv_cache") and prompt:
        from neurobrix.core.runtime.decode_bound import decode_bound as _db
        from neurobrix.triton.flow import autoregressive as AR
        gen = flow.get("generation") or {}
        lm = AR.session_lm_name(gen, [c for c in (topo.get("components") or {}) if c in comps])
        P = prompt_tokens(model, prompt)
        image_ar = gen.get("type") == "autoregressive_image"
        B = 2 if image_ar and AR.image_cfg_weight({}, defaults) > 1.0 else 1
        mt = _db(AR.image_token_count(defaults) if image_ar else
                 max_tokens_req if max_tokens_req is not None else require_max_tokens(defaults))
        kv = plan["kv_cache"]

        def decode(n, lm=lm, kv=kv, B=B):
            return [(lm, {"__decode__": {"len": n, "dtype": kv.get("dtype"), "batch": B,
                                         "heads": kv.get("num_kv_heads")}})]
        sites.append(("decode KV length", P + 1, P + int(mt), decode))
        if image_ar:
            def one(comp, lead):
                """The component's single input at its trace shape, the leading dim `lead`."""
                g = json.loads((CACHE / model / "components" / comp / "graph.json").read_text())
                sp = next(t for t in g["tensors"].values() if t.get("input_name"))
                return {sp["input_name"]: [lead, *sp["shape"][1:]]}
            steps = [(gen["head_component"], one(gen["head_component"], 1)),
                     (gen["embed_component"], one(gen["embed_component"], 1)),
                     (gen["aligner_component"], one(gen["aligner_component"], 1)),
                     (gen["decoder_component"], one(gen["decoder_component"], int(mt)))]
            sites.append(("image-AR step and VQ decode", 1, 1, lambda _n, st=steps: st))
    # next_token_diffusion (triton/flow/next_token_diffusion.py `_generate_kv`): the LM prefills
    # the speaker prompt (the flow's own `speaker_prompt_ids`) and, under CFG, a negative context
    # of the speech start token alone; then every step decodes ONE position per context against
    # its KV cache — the prompt context P + 1 .. P + max_steps, the negative 2 .. 1 + max_steps.
    if flow.get("type") == "next_token_diffusion" and prompt:
        from types import SimpleNamespace
        from neurobrix.triton.flow import next_token_diffusion as NTD
        from neurobrix.triton.flow.autoregressive import session_kv_params, session_lm_config
        lm = NTD.TritonNextTokenDiffusionEngine.LM
        root = CACHE / model
        tok = container_tokenizer(model)
        start_id = NTD.special_token_id(tok, NTD.SPEECH_START_TOKEN)
        if start_id is None:
            raise SystemExit(f"{model}: its tokenizer has no {NTD.SPEECH_START_TOKEN}")
        P = len(NTD.speaker_prompt_ids(tok, prompt, start_id))
        mt = int(max_tokens_req if max_tokens_req is not None else require_max_tokens(defaults))
        cfg = float(NTD._require_default(defaults, "cfg_scale"))
        gl = json.loads((root / "components" / lm / "graph.json").read_text())
        dim = gl["tensors"]["input::inputs_embeds"]["shape"][-1]
        # The cache the session builds: the plan's (a plan without one is refused by the session's
        # own function, named here as the census row that should hold its keys).
        lmc = session_lm_config(defaults, topo, lm)
        if not plan.get("kv_cache"):
            raise SystemExit(f"{model}: the plan carries no KV cache for {lm} — the session refuses "
                             f"it; the census cannot key a decode the run would not reach")
        kvp = session_kv_params(SimpleNamespace(**plan["kv_cache"]),
                                int(lmc.get("max_position_embeddings") or 0), decode_bound(mt))

        def prefill(n, lm=lm, dim=dim):
            return [(lm, {"inputs_embeds": [1, n, dim], "position_ids": [1, n],
                          "__kv_prefill__": True})]

        def decode(n, lm=lm, kvp=kvp):
            return [(lm, {"__decode__": {"len": n, "dtype": kvp["dtype"].name,
                                         "heads": kvp["num_kv_heads"]}})]
        sites.append((f"{lm} prompt prefill", P, P, prefill))
        lo = P + 1
        if cfg != 1.0:
            sites.append((f"{lm} negative prefill", 1, 1, prefill))
            lo = 2
        sites.append((f"{lm} decode KV length", lo, P + mt, decode))
    # vlm (triton/flow/vlm.py, the splice path): the image through the request's own preprocessing,
    # the vision tower on its patch grid (value-bound symbols from the grid itself), then the LM
    # over the WHOLE context every step — the chat-templated prompt around the modality span
    # (`tokenize_around_span`) with the vision tokens spliced in: L0 .. L0 + max_tokens - 1.
    if flow.get("type") == "vlm" and image_path and prompt:
        sites.extend(_vlm_sites(model, topo, defaults, plan, image_path, prompt, max_tokens_req))
    # encoder_decoder (triton/flow/encoder_decoder.py `_decode_one_window`): the decoder runs ONE
    # token per step on the encoder's frames — its self-attentions through the KV cache (step 1 a
    # prefill of one token, then the cache's 2 .. max_tokens - 1), its cross-attentions native
    # (`decoder_self_attention_plan`, the flow's own split).
    if flow.get("type") == "encoder_decoder":
        sites.extend(_encoder_decoder_sites(model, topo, defaults, plan))
    # audio_llm (triton/flow/audio_llm.py): the recording's features through the forward stages
    # (the frontend's own preprocessing choice and fit), their embeddings between the declared
    # prefix and suffix ids, then the language model over the WHOLE context every step — its
    # length from L0 to L0 + max_tokens - 1.
    if flow.get("type") == "audio_llm" and audio_path:
        sites.extend(_audio_llm_sites(model, topo, defaults, plan, audio_path, max_tokens_req))
    return sites


class _Stub:
    """The attributes the frontend helpers read from a flow context — nothing else."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _encoder_decoder_sites(model, topo, defaults, plan):
    from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan
    from neurobrix.core.runtime.decode_bound import decode_bound
    stages = (topo.get("flow") or {}).get("stages") or []
    enc = next((st["component"] for st in stages if st.get("execution") == "forward"), None)
    dec = next((st["component"] for st in stages if st.get("execution") == "autoregressive"), None)
    if not enc or not dec:
        return []
    kvp = decoder_self_attention_plan(runtime_graph(model, dec))
    if kvp is None or not (kvp["arange_uids"] or kvp.get("position_slice_uids")):
        return []                 # the flow's recompute path: the whole sequence every step
    ge = json.loads((CACHE / model / "components" / enc / "graph.json").read_text())
    frames = ge["tensors"][ge["output_tensor_ids"][0]]["shape"]      # the window's encoder frames
    gd = json.loads((CACHE / model / "components" / dec / "graph.json").read_text())["tensors"]
    ids_in = next(k[len("input::"):] for k in gd if k.startswith("input::") and "ids" in k)
    enc_in = next(k[len("input::"):] for k in gd if k.startswith("input::") and k != f"input::{ids_in}")
    dtype = {c["name"]: c["dtype"] for c in plan["components"]}[dec]
    mt = decode_bound(defaults.get("max_tokens"))
    self_uids = set(kvp["self_attn_uids"])

    def step(n):
        kv = ({"prefill": True, "uids": self_uids} if n == 1 else
              {"len": n, "dtype": dtype, "heads": kvp["num_heads"], "uids": self_uids})
        return [(dec, {ids_in: [1, 1], enc_in: list(frames), "__kv__": kv})]
    return [(f"{dec} self-attention KV length", 1, int(mt) - 1, step)]


def vlm_splice_path(model: str, topo: dict) -> bool:
    """Whether the vlm flow takes its splice path (the flow's own contract detection: neither a
    staged vision graph nor a masked-splice LM graph)."""
    v = (topo.get("flow") or {}).get("vlm") or {}
    g = lambda c: json.loads((CACHE / model / "components" / c / "graph.json").read_text())
    return (bool(v.get("vision_component")) and bool(v.get("lm_component"))
            and "input::all_pixel_values" not in g(v["vision_component"])["tensors"]
            and "input::image_pos_masks" not in g(v["lm_component"])["tensors"])


def _vlm_sites(model, topo, defaults, plan, image_path, prompt, max_tokens_req):
    import numpy as np
    from neurobrix.core.module.vision.input_processor import prepare_image_inputs
    from neurobrix.core.runtime.decode_bound import decode_bound
    from neurobrix.triton.flow.vlm import tokenize_around_span
    if not vlm_splice_path(model, topo):
        return []
    v = topo["flow"]["vlm"]
    vis, lm = v["vision_component"], v["lm_component"]
    in_cfg = v.get("input") or {}
    ins = prepare_image_inputs(topo, model, str(image_path), CACHE / model)
    pix = ins[in_cfg.get("image_variable", "global.pixel_values")]
    grid = np.asarray(ins[in_cfg.get("grid_variable", "global.image_grid_thw")])
    dtypes = {c["name"]: c["dtype"] for c in plan["components"]}
    step = (vis, {"hidden_states": list(pix.shape), "grid_thw": grid})
    _l, outs = run_at_inputs(model, vis, dtypes[vis], "triton", step[1], False, (0, 1, 1),
                             collections.Counter())
    emb = outs["output_0"]
    n_modal = int(emb[-2]) if len(emb) >= 2 else int(emb[0])
    t, h, w_ = (int(x) for x in grid.reshape(-1)[:3])
    merge = int(v["spatial_merge_size"])
    if n_modal != t * (h // merge) * (w_ // merge):
        raise SystemExit(f"{model}: the vision tower gives {n_modal} tokens, the grid "
                         f"{t}x{h}x{w_} / {merge} names {t * (h // merge) * (w_ // merge)}")
    pre, suf = tokenize_around_span(container_tokenizer(model), prompt, v["image_token_id"], "image")
    L0 = len(pre) + n_modal + len(suf)
    gl = json.loads((CACHE / model / "components" / lm / "graph.json").read_text())["tensors"]
    H = gl["input::inputs_embeds"]["shape"][-1]
    ds = sorted((k[len("input::"):] for k in gl if k.startswith("input::deepstack_visual_embeds.")),
                key=lambda n: int(n.rsplit(".", 1)[1]))
    mask_rank = len((gl.get("input::visual_pos_masks") or {}).get("shape") or [])
    mt = decode_bound(max_tokens_req if max_tokens_req is not None else defaults.get("max_tokens"))

    def context(n):
        feed = {"inputs_embeds": [1, n, H], "position_ids": [3, 1, n]}
        if mask_rank:
            feed["visual_pos_masks"] = [1, n] if mask_rank == 2 else [1, n, H]
        for i, name in enumerate(ds):
            feed[name] = outs[f"output_{i + 1}"]
        return [(lm, feed)]
    return [("vision tower", 1, 1, lambda _n, st=[step]: st),
            (f"{lm} context", L0, L0 + int(mt) - 1, context)]


def _audio_llm_sites(model, topo, defaults, plan, audio_path, max_tokens_req):
    from neurobrix.core.module.audio.mel_dsp import extract_features_np, fixed_window_mels
    from neurobrix.core.runtime.decode_bound import decode_bound
    from neurobrix.triton import audio_frontend as AF
    root = CACHE / model
    flow = topo.get("flow") or {}
    audio = flow.get("audio") or {}
    stages = audio.get("stages") or []
    fwd = [st["component"] for st in stages if st.get("execution", "forward") != "autoregressive"]
    lm = next(st["component"] for st in stages if st.get("execution") == "autoregressive")
    graph = lambda c: json.loads((root / "components" / c / "graph.json").read_text())
    ctx = _Stub(executors={fwd[0]: _Stub(_dag=graph(fwd[0]))}, nbx_path_str=str(root))
    input_shape = AF._component_input_shape(ctx, fwd[0])
    prep = AF.resolve_preprocessing((audio.get("input") or {}).get("preprocessing"), input_shape)
    cfg = AF._model_config_path(ctx)
    wins = None
    if prep == "mel_spectrogram" and os.environ.get("NBX_DISABLE_STT_CHUNKING") != "1":
        wins = fixed_window_mels(str(audio_path), Path(cfg), input_shape)
    feats = (AF.fit_features(wins[0][None], input_shape) if wins is not None
             else AF.fit_features(extract_features_np(prep, str(audio_path), Path(cfg), input_shape),
                                  input_shape))
    n_win = len(wins) if wins is not None else 1
    variable = (audio.get("input") or {}).get("variable", "global.input_features")
    conns = topo.get("connections") or []

    def feeds(comp, produced):
        """{input name: shape} of `comp` from the topology's connections: the features variable,
        a previous stage's named output, a length scalar."""
        out = {}
        for c in conns:
            src, dst = c.get("from", ""), c.get("to", "")
            if not dst.startswith(comp + "."):
                continue
            inp = dst[len(comp) + 1:]
            if src in (variable, variable.split(".")[-1], "global." + variable.split(".")[-1]):
                out[inp] = list(feats.shape)
            elif src in produced:
                # the flow's frame pooling to the target's feature width (`pooled_frames_shape`)
                from neurobrix.triton.flow.audio_llm import pooled_frames_shape
                tg = graph(comp)
                tfeat = next((sp["shape"][-1] for sp in tg["tensors"].values()
                              if sp.get("input_name") == inp and len(sp.get("shape", [])) >= 3), None)
                out[inp] = (pooled_frames_shape(produced[src], tfeat) if tfeat else None) or produced[src]
            elif src.endswith("length") or "length" in inp:
                out[inp] = [1]
        return out

    # The forward stages once (every window has the same fitted shape), fed through the topology's
    # connections; their last output gives the embeddings count per window.
    sites = []
    produced = {}
    fwd_steps = []
    for comp in fwd:
        fwd_steps.append((comp, feeds(comp, produced)))
        _l, outs = run_at_inputs(model, comp, {c["name"]: c["dtype"] for c in plan["components"]}[comp],
                                 "triton", fwd_steps[-1][1], False, (0, 1, 1), collections.Counter())
        for name, shp in outs.items():
            produced[f"{comp}.{name}"] = shp
    last_out = next(iter(v for k, v in produced.items() if k.startswith(fwd[-1] + ".")))
    A = int(last_out[1]) * n_win
    dim = int(last_out[-1])
    L0 = len(defaults.get("stt_prefix_ids", [1])) + A + len(defaults.get("stt_suffix_ids", []))
    mt = decode_bound(max_tokens_req if max_tokens_req is not None else defaults.get("max_tokens"))
    sites.append(("audio forward stages", 1, 1, lambda _n, st=list(fwd_steps): st))
    sites.append((f"{lm} context", L0, L0 + int(mt) - 1,
                  lambda n, dim=dim: [(lm, {"inputs_embeds": [1, n, dim], "position_ids": [1, n]})]))
    return sites


def derive_extents(model, mode, topo, defaults, plan, has_native_bf16, sdpa, unhandled, prompt="",
                   audio_path=None, max_tokens_req=None, image_path=None):
    """Every key class of every value-derived extent, by the census's own bisection
    (`census.bisect_extent`) over the derived keys — nothing runs."""
    from neurobrix.kernels import census as _census
    from neurobrix.core.prism import runtime_widths as RW_
    dtypes = {c["name"]: c["dtype"] for c in plan["components"]}
    found = set()
    for name, lo, hi, chain in extent_sites(model, topo, defaults, plan, prompt, audio_path,
                                            max_tokens_req, image_path):
        seen = {}

        def at(n):
            if n not in seen:
                keys = set()
                outs = None
                for comp, feed in chain(n):
                    view, tiled_tf = plan_tiling(plan, comp)
                    if isinstance(feed, dict) and "__decode__" in feed:
                        # one position: every batch / sequence symbol at 1, the cache's K and V
                        from neurobrix.kernels.nbx_tensor import NBXDtype
                        g = json.loads((CACHE / model / "components" / comp / "graph.json").read_text())
                        d = dict(feed["__decode__"])
                        syms = {sid: (int(d.get("batch") or 1) if i.get("name") == "batch" else 1)
                                for sid, i in ((g.get("symbolic_context") or {})
                                               .get("symbols") or {}).items()}
                        d["dtype"] = NBXDtype[str(d["dtype"]).replace("torch.", "")]
                        launches = derive_component(model, comp, dtypes[comp], mode, syms,
                                                    has_native_bf16, *sdpa, unhandled, tiling=view,
                                                    tiled_tf=tiled_tf, decode_kv=d)
                        keys |= set(launches)
                        continue
                    inputs = dict(feed(outs) if callable(feed) else feed)
                    kv = inputs.pop("__kv__", None)
                    if inputs.pop("__kv_prefill__", False):
                        kv = {"prefill": True}
                    if kv is not None and "dtype" in kv:
                        from neurobrix.kernels.nbx_tensor import NBXDtype
                        kv = dict(kv, dtype=NBXDtype[str(kv["dtype"]).replace("torch.", "")])
                    launches, outs = run_at_inputs(model, comp, dtypes[comp], mode, inputs,
                                                   has_native_bf16, sdpa, unhandled, tiling=view,
                                                   tiled_tf=tiled_tf, decode_kv=kv)
                    keys |= set(launches)
                found.update(keys)
                seen[n] = frozenset((q_, k) for _u, q_, k in keys)
            return seen[n]
        _census.bisect_extent(lo, hi, at)
        print(f"[derived] extent {name} {lo}..{hi}: {len(set(seen.values()))} key class(es) "
              f"in {len(seen)} derivation(s)")
    return found


def compare(a) -> int:
    from trace_request import derived_request
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.kernels import autotune_certified as C
    from neurobrix.kernels import census as _census
    request = derived_request(a.model)
    rung = None if a.rung == "profile" else int(a.rung)
    plan = plan_record(a.model, request, a.mode, a.hardware, rung)
    prof = load_profile(a.hardware)
    _census._bind_target(a.hardware, None)     # the vendor ladders the keys are bucketed with
    P = None
    unhandled: collections.Counter = collections.Counter()
    derived = set()
    from neurobrix.kernels import wrappers as W
    W.set_hardware_profile(prof)
    budget = W._sdpa_math_scores_budget_bytes_for(0) or 0
    min_rows, max_chunks = W._sdpa_math_min_chunk_rows(), W._sdpa_math_max_chunks()
    topo = json.loads((CACHE / a.model / "topology.json").read_text())
    flow = topo.get("flow") or {}
    gen = flow.get("generation") or {}
    # The autoregressive text flow (triton/flow/autoregressive.py): the LM component runs the
    # prefill over the prompt's tokens; the HEAD component runs on the last position only —
    # `TritonTextStrategy.get_logits` selects hidden[:, T-1] before `_head.run` — so its sequence
    # symbol is 1. Both names are the topology's own (`lm_component`, `head_component`).
    autoregressive_text = flow.get("type") == "autoregressive_generation"
    head = gen.get("head_component", "lm_head") if autoregressive_text else None
    lm_batch, runs = 1, None
    if autoregressive_text:
        P = prompt_tokens(a.model, request[request.index("--prompt") + 1])
        if gen.get("type") == "autoregressive_image":
            # the image strategy runs the LM on [cond, uncond] under guidance, and only the
            # components its generation names (the understanding tower never runs)
            from neurobrix.triton.flow import autoregressive as AR_
            _d = json.loads((CACHE / a.model / "runtime" / "defaults.json").read_text())
            lm_batch = 2 if AR_.image_cfg_weight({}, _d) > 1.0 else 1
            runs = {gen[k] for k in ("lm_component", "head_component", "embed_component",
                                     "aligner_component", "decoder_component") if gen.get(k)}
    # Every other flow: the component's symbols as Prism binds them for THIS request — the run's own
    # InputConfig (`run.request_input_config`, from the request parsed by the CLI's own parser)
    # through `ActivationProfiler.build_symbol_map` at the request (no placement floor).
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism.profiler import ActivationProfiler
    from neurobrix.core.prism import runtime_widths as RW_
    if "op_level_tiling_ops" not in plan:
        raise SystemExit("the plan record names no per-op tiling (`op_level_tiling_ops`): this "
                         "tree's --explain-plan predates it, and the widths need the plan's cut")
    manifest = json.loads((CACHE / a.model / "manifest.json").read_text())
    args = create_parser().parse_args(["run", "--model", a.model, *request, MODE_FLAGS[a.mode],
                                       "--hardware", a.hardware])
    ic = request_input_config(args, manifest, manifest.get("family"), CACHE / a.model)
    loop_comps = set((flow.get("loop") or {}).get("components") or [])
    if flow.get("type") == "rnnt":
        # the rnnt flow executes its forward stages' graphs; the greedy decoder and the joint run
        # on the host from their weights (`_run_lstm_np`, `_run_joint_np`) — no launch
        runs = {st["component"] for st in ((flow.get("audio") or {}).get("stages") or [])
                if st.get("execution", "forward") == "forward"}
    if flow.get("type") == "vlm" and vlm_splice_path(a.model, topo):
        _v = flow["vlm"]
        runs = {_v["vision_component"], _v["lm_component"]}
    tilings = plan.get("component_tiling") or {}
    _prompt = args.prompt or ""                              # the parser's answer: the LAST --prompt
    _defaults = json.loads((CACHE / a.model / "runtime" / "defaults.json").read_text()) \
        if (CACHE / a.model / "runtime" / "defaults.json").exists() else {}
    # A component an extent site drives is derived at every class of that extent, not at the
    # plan's one binding (which binds a value-derived axis to its trace).
    _audio = (REPO / request[request.index("--audio") + 1]) if "--audio" in request else None
    _mt_req = args.max_tokens if getattr(args, "max_tokens", None) is not None else None
    _image = getattr(args, "input_image", None)
    covered = {comp for _n, lo, _h, chain in extent_sites(a.model, topo, _defaults, plan, _prompt,
                                                           _audio, _mt_req, _image)
               for comp, _f in chain(lo)
               if not (isinstance(_f, dict) and "__decode__" in _f)}   # decode ADDS to the prefill
    for c in plan["components"]:
        comp = c["name"]
        if comp in covered or (runs is not None and comp not in runs):
            continue
        g = json.loads((CACHE / a.model / "components" / comp / "graph.json").read_text())
        table = (g.get("symbolic_context") or {}).get("symbols") or {}
        syms = {}
        if autoregressive_text:
            for sid, info in table.items():
                name = info.get("name")
                if name == "batch":
                    syms[sid] = 1 if comp == head else lm_batch
                elif name in ("seq_len", "sequence_length"):
                    syms[sid] = 1 if comp == head else P
                else:
                    unhandled[f"{comp}: symbol {sid} '{name}' has no binding yet"] += 1
        else:
            bound = ActivationProfiler(g).build_symbol_map(ic, placement_floor=False)
            for sid, info in table.items():
                if bound.get(sid) is not None:
                    syms[sid] = int(bound[sid])
                else:
                    unhandled[f"{comp}: symbol {sid} '{info.get('name')}' unbound by the plan's map"] += 1
            # The flow's per-component bindings (CFG batch, the encoders' tokenized length, the
            # denoiser's text axis, a FLUX denoiser's packed inputs) are IN this map: the request's
            # InputConfig carries `core.prism.flow_bindings.FlowBindings`, the plan's own rules.
        if len(syms) < len(((g.get("symbolic_context") or {}).get("symbols") or {})):
            continue
        for b in (tile_bindings(g, syms, tilings[comp]) if comp in tilings else [syms]):
            view, tiled_tf = plan_tiling(plan, comp)
            for uid, q_, key in derive_component(a.model, comp, c["dtype"], a.mode, b,
                                                prof.has_native_bf16, budget, min_rows, max_chunks,
                                                unhandled, tiling=view, tiled_tf=tiled_tf):
                derived.add((uid, q_, C.key_repr(key)))
    defaults = json.loads((CACHE / a.model / "runtime" / "defaults.json").read_text()) \
        if (CACHE / a.model / "runtime" / "defaults.json").exists() else {}
    for uid, q_, key in derive_extents(a.model, a.mode, topo, defaults, plan, prof.has_native_bf16,
                                       (budget, min_rows, max_chunks), unhandled, prompt=_prompt,
                                       audio_path=_audio, max_tokens_req=_mt_req,
                                       image_path=_image):
        derived.add((uid, q_, C.key_repr(key)))
    if flow.get("type") == "rnnt":
        # the greedy decoder's LSTM runs through `lstm_wrapper` directly, one step at a time in
        # float32 (`TritonRNNTEngine._run_lstm_np`: [T=1, B=1, I], the weights cast once) — its
        # launches outside any op, its geometry the component's own traced lstm
        from neurobrix.kernels import launch_keys as LK_
        from neurobrix.kernels.nbx_tensor import NBXDtype as _Dt
        for st in (flow.get("audio") or {}).get("stages") or []:
            if st.get("execution") != "rnnt_greedy":
                continue
            g = runtime_graph(a.model, st["component"])
            for o in (g.get("ops") or {}).values():
                if o.get("op_type") != "aten::lstm":
                    continue
                ins_ = o.get("input_tensor_ids") or []
                args_ = (o.get("attributes") or {}).get("args") or []
                I_ = g["tensors"][ins_[0]]["shape"][-1]
                H_ = g["tensors"][ins_[1]]["shape"][-1]
                for q_, key in LK_.lstm_launches(1, 1, I_, H_, int(args_[4].get("value")),
                                                 bool(args_[7].get("value")), _Dt.float32,
                                                 prof.has_native_bf16):
                    derived.add((None, q_, C.key_repr(key)))
    walked = walked_pairs(Path(a.walked), a.model, a.mode, rung)
    w_keys = {(q_, k) for _, q_, k in walked}
    d_keys = {(q_, k) for _, q_, k in derived}
    w_op = {p for p in walked if p[0] is not None}
    print(f"[derived] {a.model} {a.mode} r{a.rung}: prompt {P} tokens; plan {plan['strategy']}, "
          f"components {[(c['name'], c['dtype']) for c in plan['components']]}")
    print(f"[derived] KEYS  walked {len(w_keys)} derived {len(d_keys)}: reproduced {len(w_keys & d_keys)}, "
          f"missed {len(w_keys - d_keys)}, extra {len(d_keys - w_keys)}")
    print(f"[derived] PAIRS walked-with-op {len(w_op)}: reproduced {len(w_op & derived)}, "
          f"missed {len(w_op - derived)}")
    for q_, k in sorted(w_keys - d_keys)[:20]:
        print(f"   MISSED {q_.split('.')[-1]} {k}")
    for q_, k in sorted(d_keys - w_keys)[:20]:
        print(f"   EXTRA  {q_.split('.')[-1]} {k}")
    for why, n in unhandled.most_common():
        print(f"   NOT YET: {n:4d} x {why}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--model", required=True)
    c.add_argument("--hardware", required=True)
    c.add_argument("--rung", required=True, help="the rung in MB, or 'profile' for a census at the profile's own budget")
    c.add_argument("--mode", choices=sorted(MODE_FLAGS), required=True)
    c.add_argument("--walked", required=True, help="a census logs directory (<model>.<mode>.r<rung>*.keys + .ops)")
    a = ap.parse_args(argv)
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
