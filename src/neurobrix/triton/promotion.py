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

    for op_uid, op_data in ops_meta.items():
        op_type = op_data.get("op_type", "")
        attrs = op_data.get("attributes", {})
        args = attrs.get("args", [])

        if op_type == "aten::slice" and len(args) >= 4:
            if isinstance(args[3], dict):
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
