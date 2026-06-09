"""Symbolic Promotion — shared between compiled and sequential triton modes.

Promotes trace-time seq_len constants to symbolic references so that ops
like ones([23, 23]) become ones([s1, s1]) → ones([actual_seq_len, actual_seq_len]).

Called by TritonSequence.compile() and _run_triton_sequential().
"""

from typing import Dict, Optional


def promote_seq_len_scalars(dag: dict, tensors: dict, ops_meta: dict):
    """Promote trace-time seq_len constants to symbolic references.

    Mutates ops_meta args in-place. Safe to call multiple times
    (idempotent — already-promoted args are not re-promoted).

    Args:
        dag: Full graph DAG dict (for symbolic_context)
        tensors: dag["tensors"] dict
        ops_meta: dag["ops"] dict
    """
    sym_ctx = dag.get("symbolic_context", {})
    symbols = sym_ctx.get("symbols", {})

    # Find seq_len symbols
    seq_len_syms: Dict[str, int] = {}
    for sid, sinfo in symbols.items():
        if sinfo.get("name") == "seq_len":
            tv = sinfo.get("trace_value")
            if tv is not None:
                seq_len_syms[sid] = tv

    # Initialize variables shared with the spatial pass below. When there
    # are no seq_len symbols (diffusion VAE / image decoder) we skip the
    # entire RoPE/seq_len logic but MUST still fall through to the
    # spatial-promotion pass at the bottom of this function.
    rope_chain_op_uids: set = set()
    rope_arange_uids: set = set()

    if not seq_len_syms:
        # Skip the seq_len loop entirely; jump to the spatial-symbol pass
        # at the end of this function.
        return _spatial_promotion_pass(
            dag, tensors, ops_meta, symbols,
            rope_chain_op_uids, rope_arange_uids,
        )

    # Collision check
    weight_dims: set = set()
    for tid, tdata in tensors.items():
        if tid.startswith("param::") or tid.startswith("buffer::"):
            wname = tdata.get("weight_name", "")
            if wname.startswith("constant_T_"):
                continue
            for d in tdata.get("shape", []):
                if isinstance(d, int):
                    weight_dims.add(d)

    safe: Dict[str, int] = {}
    for sid, tv in seq_len_syms.items():
        if tv not in weight_dims:
            safe[sid] = tv

    # Handle ambiguous trace values — keep one if all same name
    tv_counts: Dict[int, list] = {}
    for sid, tv in safe.items():
        tv_counts.setdefault(tv, []).append(sid)
    for tv, sids in tv_counts.items():
        if len(sids) > 1:
            names = {symbols.get(s, {}).get("name") for s in sids}
            if len(names) == 1:
                for sid in sids[1:]:
                    del safe[sid]
            else:
                for sid in sids:
                    if sid in safe:
                        del safe[sid]

    if not safe:
        return _spatial_promotion_pass(
            dag, tensors, ops_meta, symbols,
            rope_chain_op_uids, rope_arange_uids,
        )

    def _promote_int(val: int):
        for sid, tv in safe.items():
            offset = val - tv
            if 0 <= offset <= 1:
                return {"type": "symbol", "symbol_id": sid,
                        "trace_value": val, "offset": offset}
        return None

    def _promote_scalar_dict(arg: dict):
        if arg.get("type") != "scalar":
            return None
        val = arg.get("value")
        if isinstance(val, int):
            return _promote_int(val)
        return None

    # Detect RoPE-style slices: aten::slice whose output is consumed by
    # aten::index with input::position_ids. Narrowing these to runtime seq_len
    # breaks absolute position indexing in decode (the index reads OOB →
    # garbage cos/sin → corrupts Q/K → NaN).
    #
    # The source of the slice can be either:
    #  - A constant/weight-like buffer (TinyLlama pattern — cos/sin_cached
    #    registered as buffer).
    #  - An intermediate produced by aten::cos / aten::sin over an inline
    #    position-based computation (DeepSeek-MoE pattern — RoPE recomputed
    #    per forward; source shape's seq dim equals trace_seq_len).
    # In both cases, the slice's trace-time `end` argument is trace_seq_len
    # and must stay pinned to the full source dim so position_ids > runtime
    # seq_len can still index correctly.
    consumers: Dict[str, list] = {}
    for _u, _op in ops_meta.items():
        for _t in _op.get("input_tensor_ids", []):
            consumers.setdefault(_t, []).append((_u, _op))

    def _consumed_by_pos_index(out_tid: str) -> bool:
        for _cuid, _cop in consumers.get(out_tid, []):
            if _cop.get("op_type") != "aten::index":
                continue
            if "input::position_ids" in _cop.get("input_tensor_ids", []):
                return True
        return False

    # trace_seq_len values we might see as slice end
    trace_seq_lens = set(seq_len_syms.values())

    def _tensor_is_weightlike(tid: str) -> bool:
        td = tensors.get(tid, {})
        return bool(td.get("constant")) or bool(td.get("weight_name"))

    rope_slice_full_size: Dict[str, int] = {}
    rope_seq_slice_inputs: set = set()  # intermediate-source rope slices
    # feeding from a computed chain — backward-traced to pin the arange too.
    for _uid, _op_data in ops_meta.items():
        if _op_data.get("op_type") != "aten::slice":
            continue
        _ins = _op_data.get("input_tensor_ids", [])
        if not _ins:
            continue
        _outs = _op_data.get("output_tensor_ids", [])
        if not _outs or not _consumed_by_pos_index(_outs[0]):
            continue
        _a = _op_data.get("attributes", {}).get("args", [])
        if len(_a) < 2:
            continue
        _dim_arg = _a[1]
        _dim = _dim_arg.get("value") if isinstance(_dim_arg, dict) else _dim_arg
        if not isinstance(_dim, int):
            continue
        _src_shape = tensors.get(_ins[0], {}).get("shape", [])
        if not (0 <= _dim < len(_src_shape)
                and isinstance(_src_shape[_dim], int)):
            continue
        # A RoPE slice feeds position_ids-based indexing. Protect in two
        # complementary cases:
        #  (a) Source is a constant/weight buffer (TinyLlama pattern —
        #      cos/sin_cached pre-registered at max_pos; slice trims to
        #      runtime seq_len, index reads arbitrary positions).
        #  (b) Source is an intermediate whose seq-axis dim equals
        #      trace_seq_len (DeepSeek-MoE pattern — RoPE recomputed per
        #      forward from arange(seq_len); the seq dim shrinks to
        #      runtime seq_len without pinning).
        # Either case: pin the slice end to the static source dim so the
        # downstream index with position_ids stays in bounds during decode.
        _is_weight = _tensor_is_weightlike(_ins[0])
        _is_seq_dim = _src_shape[_dim] in trace_seq_lens
        if not (_is_weight or _is_seq_dim):
            continue
        rope_slice_full_size[_uid] = _src_shape[_dim]
        # Only case (b) needs backward-tracing to pin the generating arange
        # + shape args on intermediate ops — case (a) uses a pre-loaded
        # constant whose size never shrinks.
        if _is_seq_dim and not _is_weight:
            rope_seq_slice_inputs.add(_ins[0])

    # Pin the aten::arange at the root of the RoPE chain. DeepSeek-MoE
    # recomputes RoPE per forward via arange(seq_len) → mul(inv_freq) →
    # cat → cos/sin → slice → index(position_ids). If arange shrinks to
    # runtime seq_len=1 during decode, cos/sin become (1, head_dim) and
    # position_ids > 0 read OOB → garbage RoPE → gibberish tokens from
    # the second decode step onward. Pinning arange keeps the chain at
    # trace_seq_len so positions up to trace_seq_len-1 index correctly.
    # Backward-trace from each rope slice's input through transparent-shape
    # ops (mul/add/sub/cos/sin/cat/view/reshape/unsqueeze/expand/outer/etc.)
    # until the aten::arange that generated the axis.
    _producer: Dict[str, str] = {}
    for _u, _op in ops_meta.items():
        for _out in _op.get("output_tensor_ids", []):
            _producer[_out] = _u
    rope_arange_uids: set = set()
    # Also track every intermediate op in the RoPE chain — we need to
    # un-promote their shape args (e.g. view([23, 1]) got promoted to
    # view([s1, 1]) which shrinks to (1, 1) at decode and propagates).
    rope_chain_op_uids: set = set()
    _shape_passthrough = {
        "aten::mul", "aten::add", "aten::sub", "aten::div", "aten::neg",
        "aten::cos", "aten::sin", "aten::cat", "aten::view",
        "aten::_unsafe_view", "aten::reshape", "aten::unsqueeze",
        "aten::expand", "aten::expand_as", "aten::repeat", "aten::to",
        "aten::_to_copy", "aten::contiguous", "aten::t", "aten::transpose",
        "aten::outer",
    }
    for _start_tid in rope_seq_slice_inputs:
        _stack = [_start_tid]
        _visited: set = set()
        while _stack:
            _tid = _stack.pop()
            if _tid in _visited:
                continue
            _visited.add(_tid)
            _pu = _producer.get(_tid)
            if _pu is None:
                continue
            _pop = ops_meta.get(_pu, {})
            _pt = _pop.get("op_type", "")
            if _pt == "aten::arange":
                rope_arange_uids.add(_pu)
                continue
            if _pt in _shape_passthrough:
                rope_chain_op_uids.add(_pu)
                for _pi in _pop.get("input_tensor_ids", []):
                    _ptid = tensors.get(_pi, {})
                    # Skip weight/constant inputs — they don't carry seq_len
                    if (_ptid.get("constant") or _ptid.get("weight_name")):
                        continue
                    _stack.append(_pi)

    def _un_promote_rope_shape(arg):
        """Replace any symbolic seq_len factor in a shape list with its
        static trace_value. Used inside the RoPE chain only, to prevent
        view/reshape/expand/cat shape args from shrinking at decode."""
        if isinstance(arg, dict):
            t = arg.get("type")
            if t == "symbol":
                tv = arg.get("trace_value")
                if isinstance(tv, int):
                    return {"type": "scalar", "value": tv}
            elif t == "product":
                # Resolve each factor, then collapse to a scalar if static.
                factors = arg.get("factors", [])
                new_factors = [_un_promote_rope_shape(f) for f in factors]
                resolved = 1
                all_static = True
                for f in new_factors:
                    if isinstance(f, dict) and f.get("type") == "scalar":
                        resolved *= f["value"]
                    elif isinstance(f, int):
                        resolved *= f
                    else:
                        all_static = False
                        break
                if all_static:
                    return {"type": "scalar", "value": resolved}
                arg = dict(arg)
                arg["factors"] = new_factors
                return arg
            elif t == "list":
                items = arg.get("value", [])
                new_items = [_un_promote_rope_shape(x) for x in items]
                arg = dict(arg)
                arg["value"] = new_items
                return arg
        return arg

    for op_uid, op_data in ops_meta.items():
        op_type = op_data.get("op_type", "")
        attrs = op_data.get("attributes", {})
        args = attrs.get("args", [])

        if op_type == "aten::slice" and len(args) >= 4:
            # RoPE slice: set end to full source dim (static). This turns the
            # narrow slice into effective identity so downstream aten::index
            # with position_ids can read any row, not just [0:seq_len).
            if op_uid in rope_slice_full_size:
                args[3] = {"type": "scalar",
                           "value": rope_slice_full_size[op_uid]}
            elif isinstance(args[3], dict):
                r = _promote_scalar_dict(args[3])
                if r:
                    args[3] = r

        elif op_type == "aten::arange" and len(args) >= 1:
            # If this arange feeds the RoPE chain (cos/sin → slice → index
            # with position_ids), pin it at trace_seq_len so RoPE cos/sin
            # stay sized for absolute position indexing at decode. Without
            # this, arange shrinks to runtime seq_len=1 → cos/sin become
            # (1, head_dim) → position_ids > 0 read OOB.
            # The tracer may have already replaced arg[0] with a {'type':
            # 'symbol', 'symbol_id': 's1', 'trace_value': N} node — undo
            # that back to a static scalar for RoPE aranges only.
            if op_uid in rope_arange_uids:
                a0 = args[0]
                if isinstance(a0, dict) and a0.get("type") == "symbol":
                    tv = a0.get("trace_value")
                    if isinstance(tv, int):
                        args[0] = {"type": "scalar", "value": tv}
                # Plain-int args stay as-is.
            elif isinstance(args[0], dict):
                r = _promote_scalar_dict(args[0])
                if r:
                    args[0] = r

        elif op_type in ("aten::full", "aten::zeros", "aten::ones",
                         "aten::new_zeros", "aten::new_ones"):
            shape_idx = 1 if op_type.startswith("aten::new_") else 0
            if len(args) > shape_idx:
                shape_arg = args[shape_idx]
                is_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
                items = shape_arg.get("value", []) if is_wrapped else shape_arg
                if isinstance(items, (list, tuple)):
                    items = list(items)
                    changed = False
                    for i, elem in enumerate(items):
                        if isinstance(elem, dict):
                            r = _promote_scalar_dict(elem)
                            if r:
                                items[i] = r
                                changed = True
                        elif isinstance(elem, int):
                            r = _promote_int(elem)
                            if r:
                                items[i] = r
                                changed = True
                    if changed:
                        if is_wrapped:
                            args[shape_idx] = {"type": "list", "value": items}
                        else:
                            args[shape_idx] = items

        elif op_type == "aten::expand" and len(args) >= 2:
            # Handle BOTH the raw-list and the wrapped {"type":"list"} forms.
            # The wrapped form was previously skipped (the isinstance(list) check
            # fails on the dict), leaving a seq_len element concrete — the same
            # bug already handled for view/reshape just below and for zeros/full
            # (compiled `84e05e1`). Kokoro's bert token_type expand uses the
            # wrapped form `{"type":"list","value":[1,23]}`, so the seq dim stayed
            # 23 at a runtime seq_len of 14 → aten.add broadcast failure in triton.
            size_arg = args[1]
            is_wrapped = isinstance(size_arg, dict) and size_arg.get("type") == "list"
            size_list = size_arg.get("value", []) if is_wrapped else size_arg
            if isinstance(size_list, (list, tuple)):
                size_list = list(size_list)
                changed = False
                for i, elem in enumerate(size_list):
                    if isinstance(elem, dict):
                        r = _promote_scalar_dict(elem)
                        if r:
                            size_list[i] = r
                            changed = True
                    elif isinstance(elem, int):
                        r = _promote_int(elem)
                        if r:
                            size_list[i] = r
                            changed = True
                if changed:
                    args[1] = ({"type": "list", "value": size_list}
                               if is_wrapped else size_list)

        elif op_type in ("aten::view", "aten::reshape", "aten::_unsafe_view") and len(args) >= 2:
            shape_arg = args[1]
            is_wrapped = isinstance(shape_arg, dict) and shape_arg.get("type") == "list"
            items = shape_arg.get("value", []) if is_wrapped else shape_arg
            if isinstance(items, (list, tuple)):
                has_symbols = any(isinstance(e, dict) and e.get("type") == "symbol"
                                 for e in items)
                if not has_symbols:
                    items = list(items)
                    changed = False
                    for i, elem in enumerate(items):
                        if isinstance(elem, dict):
                            r = _promote_scalar_dict(elem)
                            if r:
                                items[i] = r
                                changed = True
                        elif isinstance(elem, int):
                            r = _promote_int(elem)
                            if r:
                                items[i] = r
                                changed = True
                    if changed:
                        if is_wrapped:
                            args[1] = {"type": "list", "value": items}
                        else:
                            args[1] = items

        elif op_type == "aten::narrow" and len(args) >= 4:
            if isinstance(args[3], dict):
                r = _promote_scalar_dict(args[3])
                if r:
                    args[3] = r

    # Post-pass: un-promote symbolic seq_len factors inside shape args of
    # RoPE-chain ops (view/reshape/expand/cat/…). The main promotion loop
    # above turns `view([23, 1])` into `view([s1, 1])`; at decode with
    # runtime seq_len=1 this shrinks the RoPE position basis to a single
    # row and position_ids > 0 index OOB on the downstream cos/sin.
    # Leaving these shape args at trace_seq_len keeps the chain sized for
    # absolute position indexing during decode.
    for _uid in rope_chain_op_uids:
        _op_data = ops_meta.get(_uid)
        if _op_data is None:
            continue
        _ot = _op_data.get("op_type", "")
        _attrs = _op_data.get("attributes", {})
        _args = _attrs.get("args", [])
        if not _args:
            continue
        # view/reshape/_unsafe_view: arg[1] is the target shape list
        if _ot in ("aten::view", "aten::_unsafe_view", "aten::reshape"):
            if len(_args) >= 2:
                _args[1] = _un_promote_rope_shape(_args[1])
        # expand / expand_as: arg[1] is the target shape
        elif _ot == "aten::expand" and len(_args) >= 2:
            _args[1] = _un_promote_rope_shape(_args[1])
        # repeat: arg[1] is the repeat-spec list
        elif _ot == "aten::repeat" and len(_args) >= 2:
            _args[1] = _un_promote_rope_shape(_args[1])

    # Layer 8 — spatial-symbol promotion for diffusion VAEs.
    _spatial_promotion_pass(
        dag, tensors, ops_meta, symbols,
        rope_chain_op_uids, rope_arange_uids,
    )


def _spatial_promotion_pass(dag, tensors, ops_meta, symbols,
                            rope_chain_op_uids, rope_arange_uids):
    """Spatial (height/width) promotion for diffusion components.

    The seq_len pass only promotes symbols named "seq_len" — by design,
    that's the LLM contract. Image-diffusion graphs use named symbols
    "height", "width" (and occasionally "depth"); they were left out.
    For Sana 1024 / PixArt 1024 the trace size happens to equal the
    runtime size, so missing rebind is a silent no-op. For Sana 4Kpx
    (trace 64×64 latent → runtime 128×128 latent) the missing rebind
    surfaces as a literal `4096` (= trace H*W) / `64` (= trace H or W)
    baked into `aten::expand` / `aten::view` shape args that should
    have been `s_h * s_w` / `s_h` / `s_w` respectively → the VAE
    crashes with `Cannot broadcast (1,1024,64,64) and (1,1024,128,128)`
    on the very first attention block of the decoder. Native PyTorch
    tolerates the mismatch via implicit broadcast/interpolate semantics
    on tensors; the triton dispatcher dispatches each op literally and
    has no such smoothing. So this is an engine-side regression masked
    by the LLM-only filter, surfaced after Layer 6.3 disabled the
    parasitic TilingEngine that previously tiled the VAE down to the
    trace 64×64 size.

    Strategy: position-aware promotion on view/expand/reshape/
    _unsafe_view/ones/zeros/full/new_zeros/new_ones shape args. The
    collision check used for seq_len (skip values that match any
    param/buffer dim) cannot be reused — `64` and `4096` are also
    legitimate channel/feature widths in the VAE, so a value-only
    filter would over- or under-promote depending on which fired
    first. Instead, we use the OPERATOR'S SHAPE LAYOUT as the
    disambiguating context: in a 2D-or-higher shape arg, position [-1]
    is canonically the W axis and position [-2] the H axis
    (channels-first images, which is the standard NeuroBrix tensor
    layout). A value matching the H/W trace at that position is the
    spatial dim; a value matching H*W at any position is a flattened-
    spatial dim (B*HW or B*C*HW). Other matches stay literal.

    Bit-perfect for trace == runtime models (Sana 1024, PixArt 1024,
    every LLM): the resolver substitutes the symbol with its own
    trace_value, identical Python int output. Verified by the LLM
    harness 14/14 + Sana 1024 + PixArt × 4 modes regression.
    """
    h_sym = None
    w_sym = None
    for _sid, _info in symbols.items():
        _name = _info.get("name")
        _tv = _info.get("trace_value")
        if _tv is None or _tv <= 1:
            continue
        if _name == "height" and h_sym is None:
            h_sym = (_sid, _tv)
        elif _name == "width" and w_sym is None:
            w_sym = (_sid, _tv)

    if h_sym is None and w_sym is None:
        return  # not a spatial graph; nothing to do

    hw_product = (h_sym[1] * w_sym[1]) if (h_sym and w_sym) else None

    # DC-AE / cascade decoders double the spatial size at each up-stage,
    # so the VAE graph references both the input-scale H/W AND every
    # scaled-up version (2H, 4H, 8H, ..., 32H for Sana 4Kpx). Each
    # appears as a literal int in shape args (or in arithmetic
    # expressions). Pre-compute the scaled triplets here.
    _SPATIAL_SCALES = (1, 2, 4, 8, 16, 32, 64, 128)

    def _h_sym_dict(val):
        return {"type": "symbol", "symbol_id": h_sym[0], "trace_value": val}

    def _w_sym_dict(val):
        return {"type": "symbol", "symbol_id": w_sym[0], "trace_value": val}

    def _scaled_h_dict(val, factor):
        if factor == 1:
            return _h_sym_dict(val)
        return {"type": "mul",
                "left": factor,
                "right": {"type": "symbol", "symbol_id": h_sym[0],
                          "trace_value": h_sym[1]},
                "trace": val}

    def _scaled_w_dict(val, factor):
        if factor == 1:
            return _w_sym_dict(val)
        return {"type": "mul",
                "left": factor,
                "right": {"type": "symbol", "symbol_id": w_sym[0],
                          "trace_value": w_sym[1]},
                "trace": val}

    def _hw_product_dict(val):
        return {"type": "product",
                "factors": [
                    {"type": "symbol", "symbol_id": h_sym[0],
                     "trace_value": h_sym[1]},
                    {"type": "symbol", "symbol_id": w_sym[0],
                     "trace_value": w_sym[1]},
                ],
                "trace_value": val}

    def _scaled_hw_product_dict(val, factor):
        # Area scale = factor**2 (since both H and W double).
        if factor == 1:
            return _hw_product_dict(val)
        return {"type": "product",
                "factors": [
                    factor * factor,
                    {"type": "symbol", "symbol_id": h_sym[0],
                     "trace_value": h_sym[1]},
                    {"type": "symbol", "symbol_id": w_sym[0],
                     "trace_value": w_sym[1]},
                ],
                "trace_value": val}

    def _find_spatial_pair(items):
        """Return (h_idx, w_idx, factor) if an adjacent (H, W) pair at
        ANY supported scale is found.

        Both channels-first ([B, C, H, W]) and channels-last
        ([B, H, W, C]) layouts have H and W adjacent. The DC-AE
        decoder cascade makes the same op type appear at multiple
        scales (64×64 → 128×128 → 256×256 → ... → 2048×2048), so we
        search across the same scale set used elsewhere in this pass.
        An isolated channel dim that coincidentally equals a scaled
        H/W (e.g. 1024-channel weight at the same time as 1024×1024
        spatial in Sana 4Kpx) is left alone — promotion only fires
        when BOTH adjacent positions match the SAME scale.
        """
        if not h_sym or not w_sym:
            return None
        n = len(items)
        for i in range(n - 1):
            a, b = items[i], items[i + 1]
            if not (isinstance(a, int) and isinstance(b, int)):
                continue
            for s in _SPATIAL_SCALES:
                if a == h_sym[1] * s and b == w_sym[1] * s:
                    return (i, i + 1, s)
        return None

    def _find_hw_product_factor(val):
        """If val is a scaled H*W product at any supported scale, return
        the scale factor (1, 2, 4, ...). Else return None.
        """
        if hw_product is None:
            return None
        for s in _SPATIAL_SCALES:
            if val == hw_product * (s * s):
                return s
        return None

    def _promote_int_value(val):
        """Promote a plain int to a symbolic dict if it matches a
        scaled H*W product. Returns (new_value, changed). Used for
        both top-level items (no surrounding pair) and nested
        arithmetic operands.
        """
        if not isinstance(val, int):
            return val, False
        s = _find_hw_product_factor(val)
        if s is not None and h_sym and w_sym:
            return _scaled_hw_product_dict(val, s), True
        return val, False

    def _expr_trace_value(expr):
        """Get the trace-time value of an expression (for trace_value lookups)."""
        if isinstance(expr, int):
            return expr
        if isinstance(expr, dict):
            if "trace" in expr:
                return expr.get("trace")
            if "trace_value" in expr:
                return expr.get("trace_value")
            if expr.get("type") == "symbol":
                tv = expr.get("trace") if expr.get("trace") is not None else expr.get("trace_value")
                return tv
        return None

    def _override_misattributed_arith(expr):
        """Override Forge tracer mistakes that bind spatial dims to the
        wrong symbol.

        Sana 4Kpx VAE shape args contain expressions like `mul(s_batch=1, 64)`
        for spatial dims (where 64 is the trace H/W). At runtime the batch
        symbol still resolves to 1, so the mul evaluates to 64 — i.e., the
        spatial dim STAYS at trace size and never rebinds. The fix:
        when an arithmetic expression's trace value equals a single spatial
        trace (h_trace or w_trace) AND the expression isn't already
        anchored to a height/width symbol, REPLACE it with the canonical
        spatial symbol. This corrects the tracer's mis-attribution at
        engine load time without touching graph.json or Forge.

        Returns (new_expr, changed). If the expr already references
        height/width symbols, it's left unchanged.
        """
        if not isinstance(expr, dict):
            return expr, False
        t = expr.get("type")
        if t not in ("mul", "add", "sub", "floordiv", "mod", "product", "neg"):
            return expr, False

        # Don't override if any operand already references a spatial symbol.
        def _references_spatial(e):
            if isinstance(e, dict):
                if e.get("type") == "symbol":
                    sid = e.get("symbol_id") or e.get("id")
                    return sid in ((h_sym[0] if h_sym else None),
                                   (w_sym[0] if w_sym else None))
                # Recurse into nested
                for k in ("left", "right", "operand"):
                    if k in e and _references_spatial(e[k]):
                        return True
                for f in e.get("factors", []):
                    if _references_spatial(f):
                        return True
            return False

        if _references_spatial(expr):
            return expr, False

        # Look up trace value
        tv = _expr_trace_value(expr)
        if tv is None or not isinstance(tv, int):
            return expr, False

        if h_sym and tv == h_sym[1]:
            return _h_sym_dict(tv), True
        if w_sym and tv == w_sym[1]:
            return _w_sym_dict(tv), True
        # Multi-scale: tv == k * h_trace?
        for s in _SPATIAL_SCALES:
            if h_sym and tv == h_sym[1] * s:
                return _scaled_h_dict(tv, s), True
            if w_sym and tv == w_sym[1] * s:
                return _scaled_w_dict(tv, s), True
            if h_sym and w_sym and tv == h_sym[1] * w_sym[1] * (s * s):
                return _scaled_hw_product_dict(tv, s), True
        return expr, False

    def _walk_arith_expr(expr):
        """Recursively promote int operands inside arithmetic expr dicts.

        Handles `mul`, `add`, `sub`, `floordiv`, `mod`, `neg`, `product`.
        Returns (new_expr, changed). After recursion, attempts to override
        the entire expr if its trace value matches a spatial dim and
        no spatial symbol is referenced inside (Forge tracer mis-attribution).
        """
        if not isinstance(expr, dict):
            return _promote_int_value(expr)
        # Try whole-expression override first (the common Forge tracer mistake).
        overridden, was_overridden = _override_misattributed_arith(expr)
        if was_overridden:
            return overridden, True
        t = expr.get("type")
        if t in ("mul", "add", "sub", "floordiv", "mod"):
            new = dict(expr)
            ch = False
            for key in ("left", "right"):
                if key in new:
                    new[key], _c = _walk_arith_expr(new[key])
                    ch = ch or _c
            return new, ch
        if t == "neg":
            new = dict(expr)
            if "operand" in new:
                new["operand"], ch = _walk_arith_expr(new["operand"])
                return new, ch
            return new, False
        if t == "product":
            new = dict(expr)
            factors = new.get("factors", [])
            new_factors = []
            any_ch = False
            for f in factors:
                nf, fch = _walk_arith_expr(f)
                new_factors.append(nf)
                if fch:
                    any_ch = True
            if any_ch:
                new["factors"] = new_factors
            return new, any_ch
        # symbol / scalar / unknown — leave alone
        return expr, False

    def _walk_shape_list(items):
        """Apply spatial promotion to a flat shape list.

        Three complementary signals:
        1. Any top-level int equal to trace `H*W` is promoted to a
           product symbol. Safe at any position because the product
           value is uniquely identifiable.
        2. An adjacent (H, W) pair of top-level ints is promoted to
           two single symbols. Adjacency disambiguates from isolated
           channel dims that coincidentally equal trace H/W.
        3. Arithmetic expressions (`mul(s0, 4096)`) recurse into their
           int operands to promote `H*W` products. Required for view
           shape args where the leading B*H*W collapse is encoded as
           `mul(batch_sym, h*w_literal)`.

        Returns (items, changed).
        """
        n = len(items)
        out = list(items)
        changed = False

        # Pass 1: scaled H*W product anywhere.
        if hw_product is not None and h_sym and w_sym:
            for i, elem in enumerate(out):
                if not isinstance(elem, int):
                    continue
                s = _find_hw_product_factor(elem)
                if s is not None:
                    out[i] = _scaled_hw_product_dict(elem, s)
                    changed = True

        # Pass 2: adjacent (H, W) pair at any supported scale.
        pair = _find_spatial_pair(out)
        if pair is not None:
            h_idx, w_idx, factor = pair
            if isinstance(out[h_idx], int):
                out[h_idx] = _scaled_h_dict(out[h_idx], factor)
                changed = True
            if isinstance(out[w_idx], int):
                out[w_idx] = _scaled_w_dict(out[w_idx], factor)
                changed = True

        # Pass 3: arithmetic operands inside expression dicts.
        for i, elem in enumerate(out):
            if isinstance(elem, dict) and elem.get("type") in (
                    "mul", "add", "sub", "floordiv", "mod", "neg",
                    "product"):
                new, ch = _walk_arith_expr(elem)
                if ch:
                    out[i] = new
                    changed = True

        return out, changed

    _spatial_targets = (
        "aten::view", "aten::_unsafe_view", "aten::reshape", "aten::expand",
        "aten::expand_as", "aten::repeat",
        "aten::ones", "aten::zeros", "aten::full",
        "aten::new_zeros", "aten::new_ones", "aten::empty",
    )
    for _uid, _op_data in ops_meta.items():
        _ot = _op_data.get("op_type", "")
        if _ot not in _spatial_targets:
            continue
        # Skip ops that the seq_len pass already handled (RoPE chain).
        if _uid in rope_chain_op_uids or _uid in rope_arange_uids:
            continue
        _attrs = _op_data.get("attributes", {})
        _args = _attrs.get("args", [])
        # Shape arg index: 1 for most, 0 for ones/zeros/full/empty (no self).
        _shape_idx = 1
        if _ot in ("aten::ones", "aten::zeros", "aten::full", "aten::empty"):
            _shape_idx = 0
        if len(_args) <= _shape_idx:
            continue
        _shape_arg = _args[_shape_idx]
        _is_wrapped = isinstance(_shape_arg, dict) and \
            _shape_arg.get("type") == "list"
        _items = _shape_arg.get("value", []) if _is_wrapped else _shape_arg
        if not isinstance(_items, (list, tuple)):
            continue
        _new_items, _changed = _walk_shape_list(_items)
        if _changed:
            if _is_wrapped:
                _args[_shape_idx] = {"type": "list", "value": _new_items}
            else:
                _args[_shape_idx] = _new_items
