"""List-batched denoiser I/O & flow-matching step convention (NextDiT / Lumina2 / Z-Image).

Pure, torch-free brick (R33): works identically on torch.Tensor (compiled branch)
and NBXTensor (Triton branch) through their shared tensor methods (`narrow`,
`unsqueeze`, `squeeze`, `contiguous`, `__neg__`, `__mul__`, `__add__`, `__sub__`).

100 % data-driven — zero model-name branching:
1. `list_batched_spec(ctx, comp_name)` inspects the denoiser graph / topology inputs
   for dot-indexed list slots (`x.0`, `x.1`, ..., `cap_feats.0`, `cap_feats.1`, ...).
   Returns None for every standard batched denoiser (Flux, SD3, SDXL, Wan, Sana, ...).
2. `flow_step_convention(ctx, comp_name, driver, list_spec)` resolves the denoiser
   timestep transform (`invert_model_timestep`: t_model = 1.0 - sigma), velocity sign
   (`velocity_sign`: -1.0 or +1.0), and CFG formula (`pos_plus_scale_diff` vs `standard`)
   from explicit container declarations first, then from the structural NextDiT +
   FlowEuler(invert_sigmas=False) contract.
3. `trim_caption_embedding`, `build_list_batched_passes`, `unpack_list_batched_output`
   handle unbatched `[C, 1, H, W]` state slices, mask-trimmed `[L_i, D]` captions,
   1-slot or 2-slot graphs, with or without CFG, and output reassembly to `[B, C, H, W]`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from neurobrix.core.runtime.tensor_compat import is_torch_tensor

_INDEXED_RE = re.compile(r"^(.+)\.(\d+)$")
_FLOW_EULER_CLASSES = ("FlowEulerScheduler", "TritonFlowEulerScheduler")


def _component_dag(ctx: Any, comp_name: str) -> Optional[Dict[str, Any]]:
    """Return the component DAG dict (from executor._dag or on-disk graph.json)."""
    executors = getattr(ctx, "executors", None) or {}
    executor = executors.get(comp_name)
    dag = getattr(executor, "_dag", None) if executor is not None else None
    if isinstance(dag, dict):
        return dag
    pkg = getattr(ctx, "pkg", None)
    cache_path = getattr(pkg, "cache_path", None) if pkg is not None else None
    if cache_path is not None:
        gp = cache_path / "components" / comp_name / "graph.json"
        if gp.exists():
            try:
                return json.loads(gp.read_text())
            except Exception:
                return None
    return None


def _input_shapes_for(ctx: Any, comp_name: str, dag: Optional[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Collect input_name -> shape from the DAG tensors, falling back to topology shapes."""
    shapes: Dict[str, List[int]] = {}
    comp_info = ctx.pkg.topology.get("components", {}).get(comp_name, {})
    for name, shp in (comp_info.get("shapes") or {}).items():
        if isinstance(shp, (list, tuple)):
            shapes[str(name)] = [int(d) for d in shp]
    if dag:
        tensors = dag.get("tensors") or {}
        for tid in dag.get("input_tensor_ids") or []:
            spec = tensors.get(tid) or {}
            iname = spec.get("input_name")
            if not iname and isinstance(tid, str) and tid.startswith("input::"):
                iname = tid[len("input::"):]
            if iname:
                shp = spec.get("shape")
                if isinstance(shp, (list, tuple)):
                    shapes[str(iname)] = [int(d) for d in shp]
                elif str(iname) not in shapes:
                    shapes[str(iname)] = []
    return shapes


def _collect_indexed_groups(all_names: List[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for name in all_names:
        m = _INDEXED_RE.match(str(name))
        if m:
            grouped.setdefault(m.group(1), []).append(int(m.group(2)))
    for indices in grouped.values():
        indices.sort()
    return grouped


def _select_state_prefix(
    grouped: Dict[str, List[int]], shapes: Dict[str, List[int]], state_input: Optional[str],
) -> Optional[str]:
    if state_input and state_input in grouped:
        return str(state_input)
    for prefix, indices in grouped.items():
        if 0 in indices and len(shapes.get(f"{prefix}.0", [])) >= 3:
            return prefix
    return None


def _resolve_list_output_names(dag: Optional[Dict[str, Any]], slot_count: int) -> List[str]:
    output_names: List[str] = []
    if dag:
        tensors = dag.get("tensors") or {}
        for oid in dag.get("output_tensor_ids") or []:
            oname = (tensors.get(oid) or {}).get("output_name")
            if oname:
                output_names.append(str(oname))
    if len(output_names) < slot_count:
        return [f"output_{i}" for i in range(slot_count)]
    return output_names[:slot_count]


def _resolve_timestep_input_name(loop_def: Dict[str, Any], all_names: List[str]) -> Optional[str]:
    index_var = str(loop_def.get("index_variable") or "")
    ts_candidate = index_var.split(".", 1)[1] if index_var.startswith("global.") else index_var
    if ts_candidate in all_names:
        return ts_candidate
    for cand in ("t", "timestep", "timesteps"):
        if cand in all_names:
            return cand
    return None


def list_batched_spec(ctx: Any, comp_name: str) -> Optional[Dict[str, Any]]:
    """Detect dot-indexed list-batched denoiser inputs (e.g. x.0 / cap_feats.0)."""
    dag = _component_dag(ctx, comp_name)
    shapes = _input_shapes_for(ctx, comp_name, dag)
    comp_info = ctx.pkg.topology.get("components", {}).get(comp_name, {})
    all_names = list(shapes.keys()) + [
        n for n in (comp_info.get("interface", {}).get("inputs", []) or []) if n not in shapes
    ]
    grouped = _collect_indexed_groups(all_names)
    if not grouped:
        return None
    loop_def = ctx.pkg.topology.get("flow", {}).get("loop", {})
    state_prefix = _select_state_prefix(grouped, shapes, loop_def.get("state_input"))
    if state_prefix is None:
        return None
    state_indices = grouped[state_prefix]
    slot_count = len(state_indices)
    if state_indices != list(range(slot_count)):
        raise RuntimeError(
            f"ZERO FALLBACK: list-batched denoiser '{comp_name}' has non-contiguous "
            f"state slots for '{state_prefix}': {state_indices}."
        )
    state0_shape = shapes.get(f"{state_prefix}.0", [])
    frame_axis: Optional[int] = 1 if (len(state0_shape) == 4 and state0_shape[1] == 1) else None
    cond_prefix = next(
        (p for p, idx in grouped.items() if p != state_prefix and idx == list(range(slot_count))),
        None,
    )
    return {
        "state_prefix": state_prefix,
        "cond_prefix": cond_prefix,
        "slot_count": slot_count,
        "frame_axis": frame_axis,
        "timestep_input": _resolve_timestep_input_name(loop_def, all_names),
        "output_names": _resolve_list_output_names(dag, slot_count),
        "state0_shape": state0_shape,
    }


def flow_step_convention(
    ctx: Any,
    comp_name: str,
    driver: Any,
    list_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve timestep inversion, velocity sign, and CFG formula for a loop component.

    Precedence:
      1. Explicit container declarations (topology.flow.loop / component attributes /
         extracted_values.<comp_name>).
      2. Structural NextDiT contract: list-batched `[C, 1, H, W]` + caption denoiser
         paired with FlowEulerScheduler(invert_sigmas=False) ->
         invert_model_timestep=True, velocity_sign=-1.0, cfg_formula="pos_plus_scale_diff".
      3. Standard defaults (invert_model_timestep=False, velocity_sign=1.0, cfg_formula="standard").
    """
    topo = ctx.pkg.topology
    loop_def = topo.get("flow", {}).get("loop", {}) or {}
    comp_attrs = (topo.get("components", {}).get(comp_name, {}) or {}).get("attributes", {}) or {}
    extracted = (topo.get("extracted_values", {}).get(comp_name, {}) or {})

    def _lookup(key: str):
        for src in (loop_def, comp_attrs, extracted):
            if key in src and src[key] is not None:
                return src[key]
        return None

    invert_ts = _lookup("invert_model_timestep")
    vel_sign = _lookup("velocity_sign")
    if vel_sign is None:
        neg_flag = _lookup("negate_model_output")
        if neg_flag is not None:
            vel_sign = -1.0 if bool(neg_flag) else 1.0
    cfg_formula = _lookup("cfg_formula")

    is_nextdit_flow = (
        list_spec is not None
        and list_spec.get("cond_prefix") is not None
        and list_spec.get("frame_axis") == 1
        and type(driver).__name__ in _FLOW_EULER_CLASSES
        and not getattr(driver, "invert_sigmas", False)
    )

    if invert_ts is None:
        invert_ts = bool(is_nextdit_flow)
    else:
        invert_ts = bool(invert_ts)

    if vel_sign is None:
        vel_sign = -1.0 if is_nextdit_flow else 1.0
    else:
        vel_sign = float(vel_sign)

    if cfg_formula is None:
        cfg_formula = "pos_plus_scale_diff" if is_nextdit_flow else "standard"
    else:
        cfg_formula = str(cfg_formula)
        if cfg_formula not in ("standard", "pos_plus_scale_diff"):
            raise RuntimeError(
                f"ZERO FALLBACK: unknown cfg_formula {cfg_formula!r} on component {comp_name}."
            )

    return {
        "invert_model_timestep": invert_ts,
        "velocity_sign": vel_sign,
        "cfg_formula": cfg_formula,
    }


def apply_model_timestep_convention(timestep: Any, convention: Dict[str, Any]) -> Any:
    """Transform scheduler timestep into the denoiser input convention (e.g. 1.0 - sigma)."""
    if not convention.get("invert_model_timestep"):
        return timestep
    if isinstance(timestep, (int, float)):
        return 1.0 - float(timestep)
    return 1.0 - timestep


def apply_velocity_sign(model_output: Any, convention: Dict[str, Any]) -> Any:
    """Apply the declared velocity sign convention before scheduler.step."""
    sign = float(convention.get("velocity_sign", 1.0))
    if sign == 1.0:
        return model_output
    if sign == -1.0:
        return -model_output
    return model_output * sign


def resolve_state_variable_alias(ctx: Any, state_key: str) -> Any:
    """Resolve `state_key`, aliasing to the declared `state_tensor` variable when absent.

    Some topologies name `flow.loop.state_variable` after the denoiser argument
    (e.g. `global.x` or `global.img`) while `runtime/variables.json` allocates the
    latent under `global.latents` (`role: "state_tensor"`).
    """
    vr = ctx.variable_resolver
    try:
        val = vr.get(state_key)
        if val is not None:
            return val
    except Exception:
        pass

    variables = getattr(ctx.pkg, "variables", None) or {}
    for var_name, spec in variables.items():
        if isinstance(spec, dict) and spec.get("role") == "state_tensor" and var_name != state_key:
            val = vr.get(var_name)
            if val is not None:
                vr.set(state_key, val)
                return val

    # Final lookup raises the original ZERO FALLBACK error from VariableResolver.
    return vr.get(state_key)


def _is_nbx_tensor(t: Any) -> bool:
    return hasattr(t, "nbx_dtype") and not is_torch_tensor(t)


def _mask_to_bool_numpy(mask_row: Any) -> Any:
    """Convert a 1D mask (NBXTensor, torch.Tensor, or numpy/sequence) to a 1D bool ndarray."""
    import numpy as np

    if _is_nbx_tensor(mask_row):
        arr = mask_row.numpy()
    elif is_torch_tensor(mask_row):
        arr = mask_row.detach().cpu().numpy()
    else:
        arr = np.asarray(mask_row)
    return np.asarray(arr).astype(bool).reshape(-1)


def _mask_active_span(mask_row: Any) -> Tuple[int, int, bool]:
    """Return (start, length, is_contiguous) of non-zero entries in a 1D mask."""
    arr_bool = _mask_to_bool_numpy(mask_row)
    nz = [int(i) for i, v in enumerate(arr_bool) if v]
    if not nz:
        return 0, 0, True
    start, end = nz[0], nz[-1] + 1
    return start, end - start, (end - start == len(nz))


def trim_caption_embedding(emb: Any, mask: Optional[Any] = None) -> List[Any]:
    """Convert batched `[B, L_pad, D]` (or unbatched `[L_pad, D]`) embeddings into a
    list of unbatched `[L_i, D]` tensors with padding tokens stripped via `mask`."""
    if emb is None:
        return []
    if len(emb.shape) == 2:
        items = [emb]
        masks = [mask] if mask is not None else [None]
    else:
        batch = int(emb.shape[0])
        items = [emb.narrow(0, i, 1).squeeze(0) for i in range(batch)]
        if mask is not None and len(getattr(mask, "shape", ())) >= 2:
            if hasattr(mask, "narrow"):
                masks = [mask.narrow(0, i, 1).squeeze(0) for i in range(batch)]
            else:
                masks = [mask[i] for i in range(batch)]
        else:
            masks = [mask for _ in range(batch)]

    trimmed: List[Any] = []
    for row, m in zip(items, masks):
        if m is None:
            trimmed.append(row.contiguous())
            continue
        if _is_nbx_tensor(row):
            start, length, contiguous = _mask_active_span(m)
            if not contiguous:
                raise RuntimeError(
                    "ZERO FALLBACK: non-contiguous attention_mask cannot be trimmed with narrow() on NBXTensor."
                )
            if length == 0:
                trimmed.append(row.narrow(0, 0, 1).contiguous())
            else:
                trimmed.append(row.narrow(0, start, length).contiguous())
        else:
            if is_torch_tensor(m):
                bool_mask = m.bool()
            else:
                bool_mask = row.new_tensor(_mask_to_bool_numpy(m)).bool()
            if int(bool_mask.sum().item()) == 0:
                trimmed.append(row.narrow(0, 0, 1).contiguous())
            else:
                trimmed.append(row[bool_mask].contiguous())
    return trimmed


def resolve_caption_source_key(
    ctx: Any,
    enc_name: str,
    comp_name: Optional[str] = None,
    cond_prefix: Optional[str] = None,
) -> Optional[str]:
    """Return the topology-declared hidden-state output key for `enc_name` (e.g. `text_encoder.output_35`)."""
    topo = getattr(ctx.pkg, "topology", None) or {}
    conns = topo.get("connections") or []
    prefix = f"{enc_name}."
    loop_comps = set((topo.get("flow", {}) or {}).get("loop", {}).get("components", []) or [])

    if comp_name and cond_prefix:
        slot_prefix = f"{comp_name}.{cond_prefix}."
        exact_target = f"{comp_name}.{cond_prefix}"
        for conn in conns:
            src = str(conn.get("from") or "")
            dst = str(conn.get("to") or "")
            if src.startswith(prefix) and (dst.startswith(slot_prefix) or dst == exact_target):
                return src

    for conn in conns:
        src = str(conn.get("from") or "")
        dst = str(conn.get("to") or "")
        if not src.startswith(prefix) or "pooled" in src.lower():
            continue
        dst_comp = dst.split(".", 1)[0]
        if not loop_comps or dst_comp in loop_comps:
            return src
    return None


def resolve_encoder_hidden_key(ctx: Any, enc_name: str) -> str:
    """Return the resolved variable key holding `enc_name` hidden states (declared connection or `last_hidden_state`)."""
    decl = resolve_caption_source_key(ctx, enc_name)
    if decl and decl in ctx.variable_resolver.resolved:
        return decl
    return f"{enc_name}.last_hidden_state"


def extract_caption_embeddings(
    ctx: Any,
    do_cfg: bool,
    comp_name: Optional[str] = None,
    cond_prefix: Optional[str] = None,
) -> Tuple[List[Any], Optional[List[Any]]]:
    """Locate pre-loop text encoder outputs + masks and trim them for list-batched denoisers."""
    vr = ctx.variable_resolver
    pre_loop = ctx.pkg.topology.get("flow", {}).get("pre_loop", []) or []

    pos_emb = None
    enc_name = None
    for comp in pre_loop:
        candidates: List[str] = []
        declared = resolve_caption_source_key(ctx, comp, comp_name=comp_name, cond_prefix=cond_prefix)
        if declared:
            candidates.append(declared)
        for suffix in ("last_hidden_state", "output_0"):
            cand = f"{comp}.{suffix}"
            if cand not in candidates:
                candidates.append(cand)
        for cand in candidates:
            if cand in vr.resolved and vr.resolved[cand] is not None:
                pos_emb = vr.resolved[cand]
                enc_name = comp
                break
        if pos_emb is not None:
            break

    if pos_emb is None:
        raise RuntimeError(
            "ZERO FALLBACK: list-batched denoiser found no pre-loop text encoder hidden state."
        )

    pos_mask = vr.resolved.get("global.attention_mask")
    pos_caps = trim_caption_embedding(pos_emb, pos_mask)

    neg_caps: Optional[List[Any]] = None
    if do_cfg and enc_name is not None:
        neg_emb = vr.resolved.get(f"{enc_name}.negative_hidden_state")
        if neg_emb is None:
            raise RuntimeError(
                f"ZERO FALLBACK: CFG enabled for list-batched denoiser, but {enc_name}.negative_hidden_state is missing."
            )
        neg_mask = vr.resolved.get(f"{enc_name}.negative_attention_mask")
        neg_caps = trim_caption_embedding(neg_emb, neg_mask)

    return pos_caps, neg_caps


def _nbx_device_arg(tensor: Any) -> Any:
    """Return a device specifier accepted by NBXTensor.from_numpy (int index or 'cuda:N')."""
    dev_idx = getattr(tensor, "_device_idx", None)
    if isinstance(dev_idx, int) and not isinstance(dev_idx, bool) and dev_idx >= 0:
        return dev_idx
    dev = getattr(tensor, "_device", None)
    if dev == "cuda":
        return None
    return dev


def _nbx_from_numpy(arr: Any, reference_tensor: Any) -> Any:
    """Construct an NBXTensor on `reference_tensor`'s device without importing torch."""
    dev = _nbx_device_arg(reference_tensor)
    cls = type(reference_tensor)
    if hasattr(cls, "from_numpy"):
        return cls.from_numpy(arr, device=dev)
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(arr, device=dev)


def _repeat_timestep(timestep: Any, count: int, state: Any = None) -> Any:
    """Produce a 1D `[count]` timestep tensor from a scalar or `[1]` tensor."""
    if _is_nbx_tensor(timestep):
        t1 = (
            timestep.unsqueeze(0)
            if len(timestep.shape) == 0
            else (timestep if int(timestep.shape[0]) == 1 else timestep.narrow(0, 0, 1))
        )
        if count == 1:
            return t1
        if hasattr(t1, "repeat"):
            return t1.repeat(count)
        import numpy as np
        arr = np.repeat(np.asarray(t1.numpy(), dtype=np.float32).reshape(1), count)
        return _nbx_from_numpy(arr, timestep)

    if hasattr(timestep, "reshape") and hasattr(timestep, "repeat"):
        t1 = timestep.reshape(1) if len(timestep.shape) == 0 else timestep.reshape(-1)[:1]
        return t1 if count == 1 else t1.repeat(count)

    if _is_nbx_tensor(state):
        import numpy as np
        arr = np.full((count,), float(timestep), dtype=np.float32)
        return _nbx_from_numpy(arr, state)

    if hasattr(state, "new_full"):
        if hasattr(state, "new_empty"):
            return state.new_empty(0).float().new_full((count,), float(timestep))
        return state.new_full((count,), float(timestep))

    raise RuntimeError(
        "ZERO FALLBACK: _repeat_timestep requires a tensor timestep or tensor state to determine backend."
    )


def _prepare_state_slot(state_item: Any, frame_axis: Optional[int]) -> Any:
    """Convert `[C, H, W]` -> `[C, 1, H, W]` when the graph expects a unit frame axis."""
    if frame_axis is not None and len(state_item.shape) == 3:
        return state_item.unsqueeze(frame_axis).contiguous()
    return state_item.contiguous()


def build_list_batched_passes(
    state: Any,
    timestep: Any,
    pos_caps: List[Any],
    neg_caps: Optional[List[Any]],
    spec: Dict[str, Any],
    do_cfg: bool,
) -> List[Tuple[Dict[str, Any], Any]]:
    """Build one or two denoiser call input dicts for a list-batched denoiser.

    Returns a list of `(staged_inputs, pass_timestep)` pairs:
      - slot_count == 2, do_cfg == True:  1 pass with slot 0 = pos, slot 1 = neg
      - slot_count == 2, do_cfg == False: 1 pass with slot 0 = slot 1 = pos
      - slot_count == 1, do_cfg == False: 1 pass with slot 0 = pos
      - slot_count == 1, do_cfg == True:  2 passes (pass 0 = pos, pass 1 = neg)
    """
    state_prefix = spec["state_prefix"]
    cond_prefix = spec.get("cond_prefix")
    slot_count = int(spec["slot_count"])
    frame_axis = spec.get("frame_axis")
    ts_input = spec.get("timestep_input")

    if len(state.shape) < 4:
        raise RuntimeError(
            f"ZERO FALLBACK: list-batched denoiser expects 4D batched state [B, C, H, W], got {tuple(state.shape)}."
        )
    batch = int(state.shape[0])
    if batch != 1:
        raise RuntimeError(
            f"ZERO FALLBACK: list-batched denoiser currently supports batch=1, got batch={batch}."
        )

    state_item = _prepare_state_slot(state.narrow(0, 0, 1).squeeze(0), frame_axis)
    pos_cap = pos_caps[0] if pos_caps else None
    neg_cap = neg_caps[0] if (do_cfg and neg_caps) else None

    def _pack_slots(caps_for_slots: List[Any]) -> Tuple[Dict[str, Any], Any]:
        pass_t = _repeat_timestep(timestep, slot_count, state=state)
        staged: Dict[str, Any] = {}
        for idx in range(slot_count):
            staged[f"{state_prefix}.{idx}"] = state_item
            if cond_prefix is not None and caps_for_slots[idx] is not None:
                staged[f"{cond_prefix}.{idx}"] = caps_for_slots[idx]
        if ts_input:
            staged[ts_input] = pass_t
        return staged, pass_t

    if do_cfg:
        if slot_count >= 2:
            slots = [pos_cap, neg_cap] + [pos_cap] * (slot_count - 2)
            return [_pack_slots(slots)]
        # slot_count == 1 -> sequential CFG (pos pass, then neg pass)
        return [_pack_slots([pos_cap]), _pack_slots([neg_cap])]

    # Non-CFG: duplicate slot 0 into any extra slots required by a multi-slot graph
    return [_pack_slots([pos_cap] * slot_count)]


def _extract_slot_tensor(raw_output: Any, output_name: str, fallback_idx: int) -> Any:
    if isinstance(raw_output, dict):
        if output_name in raw_output:
            return raw_output[output_name]
        # Fall back to positional non-T_ semantic key
        sem = [v for k, v in raw_output.items() if not str(k).startswith("T_")]
        if fallback_idx < len(sem):
            return sem[fallback_idx]
        raise RuntimeError(
            f"ZERO FALLBACK: list-batched output {output_name} missing from {list(raw_output.keys())}."
        )
    return raw_output


def _restore_batched_state(slot_tensor: Any, frame_axis: Optional[int]) -> Any:
    """Restore `[C, 1, H, W]` -> `[1, C, H, W]` (or `[C, H, W]` -> `[1, C, H, W]`)."""
    t = slot_tensor
    if frame_axis is not None and len(t.shape) == 4 and int(t.shape[frame_axis]) == 1:
        t = t.squeeze(frame_axis)
    if len(t.shape) == 3:
        t = t.unsqueeze(0)
    return t


def unpack_list_batched_output(
    pass_outputs: List[Any],
    spec: Dict[str, Any],
    do_cfg: bool,
    guidance_scale: float,
    cfg_formula: str,
) -> Any:
    """Unpack list-batched denoiser output(s) back to `[1, C, H, W]` and apply CFG."""
    out_names = spec["output_names"]
    frame_axis = spec.get("frame_axis")
    slot_count = int(spec["slot_count"])

    if not do_cfg:
        pos = _restore_batched_state(_extract_slot_tensor(pass_outputs[0], out_names[0], 0), frame_axis)
        return pos

    if slot_count >= 2:
        pos = _restore_batched_state(_extract_slot_tensor(pass_outputs[0], out_names[0], 0), frame_axis)
        neg = _restore_batched_state(_extract_slot_tensor(pass_outputs[0], out_names[1], 1), frame_axis)
    else:
        pos = _restore_batched_state(_extract_slot_tensor(pass_outputs[0], out_names[0], 0), frame_axis)
        neg = _restore_batched_state(_extract_slot_tensor(pass_outputs[1], out_names[0], 0), frame_axis)

    diff = pos - neg
    if cfg_formula == "pos_plus_scale_diff":
        return pos + diff * float(guidance_scale)
    return neg + diff * float(guidance_scale)
