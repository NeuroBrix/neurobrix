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

CACHE = Path.home() / ".neurobrix" / ("ca" "che")
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


def prompt_tokens(model: str, prompt: str) -> int:
    """The prefill length: the prompt through the container's own tokenizer and chat template,
    as the autoregressive flow's `_tokenize` does for a chat-mode model."""
    from neurobrix.core.module.tokenizer.sp_tokenizer import load_tokenizer_from_path
    root = CACHE / model
    topo = json.loads((root / "topology.json").read_text())
    mods = topo.get("modules") or {}
    tok = mods.get("tokenizer")
    if not tok:
        raise SystemExit(f"{model}: no tokenizer module in its topology")
    path = (tok.get("path") if isinstance(tok, dict) else tok) or "modules/tokenizer"   # the executor's default
    path = str(path).rstrip("/")
    defaults = json.loads((root / "runtime" / "defaults.json").read_text()) if (root / "runtime" / "defaults.json").exists() else {}
    tokenizer = load_tokenizer_from_path(root / path, None)
    if bool(defaults.get("chat_mode", False)) and hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    else:
        ids = tokenizer.encode(prompt)
    ids = ids.get("input_ids", ids) if isinstance(ids, dict) else ids
    while isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def derive_component(model: str, comp: str, cdtype: str, mode: str, symbols: dict,
                     has_native_bf16: bool, sdpa_budget_bytes: int, sdpa_min_rows: int,
                     sdpa_max_chunks: int, unhandled: collections.Counter):
    """[(op uid, kernel qual, key tuple)] for one component at one symbol binding."""
    from neurobrix.core.prism import runtime_widths as RW
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.triton.symbols import SymbolResolver
    g = json.loads((CACHE / model / "components" / comp / "graph.json").read_text())
    res = SymbolResolver(g.get("symbolic_context") or {})
    for sid, v in symbols.items():
        res._bind(sid, int(v))
    T, ops = g["tensors"], g["ops"]

    def shape(tid):
        ss = T[tid].get("symbolic_shape")
        if isinstance(ss, dict) and ss.get("dims"):
            return [res.resolve(d) for d in ss["dims"]]
        return list(T[tid]["shape"])

    contract = RW.plan_time_contract(CACHE / model, comp, g, cdtype)
    engine = "triton" if mode == "triton" else "triton_sequential"
    rt = RW.runtime_dtypes(g, cdtype, engine, has_native_bf16=has_native_bf16, contract=contract,
                           shape_of=shape)
    dt = lambda tid: NBXDtype[rt[tid]] if rt.get(tid) in NBXDtype.__members__ else NBXDtype[{"float16": "float16", "float32": "float32", "bfloat16": "bfloat16"}[rt[tid]]]
    from neurobrix.kernels import wrappers as _W
    conv_band_bytes = _W._NBX_CONV2D_BAND_BYTES
    out = []
    for uid in g["execution_order"]:
        o = ops[uid]
        kind = o["op_type"]
        ins = [t for t in o["input_tensor_ids"]]
        if kind in ("aten::mm",):
            (M, K), (_, N) = shape(ins[0]), shape(ins[1])
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
            if k[2] == D and k[3] == Tq and k[2] != Tq:        # the wrapper reads a pre-transposed K by its shape
                k = [k[0], k[1], k[3], k[2]]
            Hk, Tk, Dv = k[1], k[2], v[3]
            route, rows = LK.sdpa_route(B, H, Tq, Tk, D, Dv, sdpa_budget_bytes, sdpa_min_rows, sdpa_max_chunks)
            qd, kd, vd = dt(ins[0]), dt(ins[1]), dt(ins[2])
            if route == "flash":
                launches = []
            elif route == "chunked":
                launches = LK.chunked_math_attention_launches(B, H, Hk, Tq, Tk, D, Dv, qd, kd, vd,
                                                             has_native_bf16, rows)
            elif Tq == 1:
                unhandled["sdpa: Tq=1 decode-vec gate (not yet derived)"] += 1
                launches = []
            else:
                launches = LK.math_attention_launches(B, H, Hk, Tq, Tk, D, Dv, qd, kd, vd, has_native_bf16)
        elif kind == "aten::addmm":
            bias_s, a_s, b_s = shape(ins[0]), shape(ins[1]), shape(ins[2])
            K = a_s[-1]
            M = 1
            for d in a_s[:-1]:
                M *= d
            launches = LK.addmm_launches(M, K, b_s[-1], dt(ins[1]), dt(ins[2]), dt(ins[0]), has_native_bf16)
        elif kind in ("aten::convolution", "aten::conv1d"):
            at = o.get("attributes") or {}
            x_s, w_s = shape(ins[0]), shape(ins[1])
            nd = len(w_s) - 2
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
            launches = LK.conv_launches(x_s, w_s, stride, padding, dilation, transposed, groups,
                                        xd, wd, comp_d, conv_band_bytes)
        elif kind in ("aten::baddbmm", "aten::linear", "aten::matmul",
                      "aten::lstm", "aten::stft", "aten::istft"):
            unhandled[f"{kind} (not yet derived)"] += 1
            launches = []
        else:
            continue
        for q_, key in launches:
            out.append((uid, q_, key))
    return out


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
    if autoregressive_text:
        P = prompt_tokens(a.model, request[request.index("--prompt") + 1])
    # Every other flow: the component's symbols as Prism binds them for THIS request — the run's own
    # InputConfig (`run.request_input_config`, from the request parsed by the CLI's own parser)
    # through `ActivationProfiler.build_symbol_map` at the request (no placement floor).
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism.profiler import ActivationProfiler
    manifest = json.loads((CACHE / a.model / "manifest.json").read_text())
    args = create_parser().parse_args(["run", "--model", a.model, *request, MODE_FLAGS[a.mode],
                                       "--hardware", a.hardware])
    ic = request_input_config(args, manifest, manifest.get("family"), CACHE / a.model)
    for c in plan["components"]:
        comp = c["name"]
        g = json.loads((CACHE / a.model / "components" / comp / "graph.json").read_text())
        table = (g.get("symbolic_context") or {}).get("symbols") or {}
        syms = {}
        if autoregressive_text:
            for sid, info in table.items():
                name = info.get("name")
                if name == "batch":
                    syms[sid] = 1
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
        if len(syms) < len(((g.get("symbolic_context") or {}).get("symbols") or {})):
            continue
        for uid, q_, key in derive_component(a.model, comp, c["dtype"], a.mode, syms,
                                            prof.has_native_bf16, budget, min_rows, max_chunks, unhandled):
            derived.add((uid, q_, C.key_repr(key)))
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
