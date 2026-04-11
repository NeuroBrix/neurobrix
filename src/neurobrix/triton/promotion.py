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

    # Detect RoPE-style slices: aten::slice on a weight/constant tensor whose
    # output is consumed by aten::index with input::position_ids. Narrowing
    # these to runtime seq_len breaks absolute position indexing in decode
    # (the index reads OOB → garbage cos/sin → corrupts Q/K → NaN).
    consumers: Dict[str, list] = {}
    for _u, _op in ops_meta.items():
        for _t in _op.get("input_tensor_ids", []):
            consumers.setdefault(_t, []).append((_u, _op))

    def _tensor_is_weightlike(tid: str) -> bool:
        td = tensors.get(tid, {})
        return bool(td.get("constant")) or bool(td.get("weight_name"))

    def _consumed_by_pos_index(out_tid: str) -> bool:
        for _cuid, _cop in consumers.get(out_tid, []):
            if _cop.get("op_type") != "aten::index":
                continue
            if "input::position_ids" in _cop.get("input_tensor_ids", []):
                return True
        return False

    rope_slice_full_size: Dict[str, int] = {}
    for _uid, _op_data in ops_meta.items():
        if _op_data.get("op_type") != "aten::slice":
            continue
        _ins = _op_data.get("input_tensor_ids", [])
        if not _ins or not _tensor_is_weightlike(_ins[0]):
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
        if 0 <= _dim < len(_src_shape) and isinstance(_src_shape[_dim], int):
            rope_slice_full_size[_uid] = _src_shape[_dim]

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
            if isinstance(args[0], dict):
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
