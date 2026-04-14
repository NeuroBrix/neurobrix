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

    if not seq_len_syms:
        return

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
        return

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
            size_arg = args[1]
            if isinstance(size_arg, (list, tuple)):
                size_list = list(size_arg)
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
                    args[1] = size_list

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
