"""Weight presence — the container's statement of what it holds, and the refusal of a weight absent.

Torch-free and triton-free: the rule both engines' loaders, both engines' weight binds and the
container validator apply, kept beside the container format it guards (one rule, one place).

2026-10-04, Allegro on a staged container whose directory was replaced under the run: the Triton
loader globbed the shards it found, never compared them with `weights_index.json`, and handed back
nothing for `post_quant_conv.weight`/`.bias`; the first reader was `aten.convolution::0`, which met
None. A weight is never absent silently: the index states what the container holds, and a load or
a bind that cannot deliver what the graph reads refuses by name — component, key, the shard the
index expects it in — before any op runs.
"""

import json
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple


def read_safetensors_header_from(f) -> Tuple[dict, int]:
    """The header of the safetensors stream `f` (a file, or an archive member) and the offset
    its data starts at, reading nothing past the header."""
    header_size = struct.unpack('<Q', f.read(8))[0]
    return json.loads(f.read(header_size)), 8 + header_size


def read_safetensors_header(path: str) -> Tuple[dict, int]:
    """A safetensors file's header and the offset its data starts at, without reading the data."""
    with open(path, 'rb') as f:
        return read_safetensors_header_from(f)


class AbsentWeightError(RuntimeError):
    """A weight a load needs, or the graph binds, is absent from the container — refused by
    name (component, key, the shard the index expects it in) before any op runs.

    2026-10-04, Allegro on a staged container whose directory was replaced under the run: the
    loader globbed the shards it found, never compared them with the index, and handed back
    nothing for `post_quant_conv.weight`/`.bias`; the first reader was `aten.convolution::0`,
    which met None (`'NoneType' object has no attribute 'ndim'`). A weight is never absent
    silently: the index is the container's statement of what it holds, and a load that cannot
    deliver what the index says refuses here, in both engines (one function, R30)."""


def read_weights_index(component: str, comp_dir) -> Optional[dict]:
    """The component's index tensor table (`weights_index.json` → `tensors`): key → entry, each
    entry naming its `shard`. None for a component that stores no weights (no index AND no
    shard on disk). Shards without an index, or an index that cannot be read, are refused: no
    weight could then be verified present. Torch-free; both engines' loaders read it."""
    comp_dir = Path(comp_dir)
    index_path = comp_dir / "weights_index.json"
    if not index_path.exists():
        weights_dir = comp_dir / "weights"
        shards = sorted(p.name for p in weights_dir.glob("*.safetensors")) \
            if weights_dir.is_dir() else []
        if shards:
            raise AbsentWeightError(
                f"component '{component}': {len(shards)} weight shard(s) in {weights_dir} "
                f"(e.g. {shards[0]}) but no index at {index_path} — refused: without the index "
                f"no weight the graph binds can be verified present.")
        return None
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        raise AbsentWeightError(
            f"component '{component}': its weights index {index_path} cannot be read ({e}) — "
            f"refused: without it no weight can be verified present.") from e
    tensors = index.get("tensors") if isinstance(index, dict) else None
    if not isinstance(tensors, dict):
        raise AbsentWeightError(
            f"component '{component}': its weights index {index_path} carries no `tensors` "
            f"table — refused: without it no weight can be verified present.")
    return tensors


def absent_weights(index_tensors: dict, wanted, header_of) -> list:
    """The wanted keys a load cannot deliver, as (key, shard, reason), in sorted key order.

    `header_of(shard)` returns that shard file's safetensors header, or None when the file is
    absent, or True when the file is present but has no header to read before loading (a
    `.bin`: its keys are judged by the post-load door); an exception it raises (a truncated or
    unreadable file) is the reason. The three
    absences, each named:

      * the index lists no such key (shard None) — the graph or the flow asks for a weight the
        container does not declare;
      * the index places the key in a shard that is absent on disk;
      * the shard is there but its header holds no such key.
    """
    headers: dict = {}
    out = []
    for key in sorted(wanted):
        entry = index_tensors.get(key)
        shard = entry.get("shard") if isinstance(entry, dict) else None
        if not shard:
            out.append((key, None, "the weights index lists no such key"))
            continue
        if shard not in headers:
            try:
                headers[shard] = header_of(shard)
            except Exception as e:      # named below, never swallowed
                headers[shard] = e
        h = headers[shard]
        if h is None:
            out.append((key, shard, "the shard file is absent on disk"))
        elif isinstance(h, Exception):
            out.append((key, shard, f"the shard's header cannot be read ({h})"))
        elif h is True:
            continue        # present; its keys are checked after the load
        elif key not in h:
            out.append((key, shard, "the shard's header holds no such key"))
    return out


def refuse_absent_weights(component: str, weights_dir: str, index_path: str,
                          absent: list, limit: int = 20) -> None:
    """Raise AbsentWeightError naming each absent weight (`absent_weights`' rows) — the
    component, the key, and the shard path the index expects it in. No-op when nothing is
    absent."""
    if not absent:
        return
    lines = []
    for key, shard, reason in absent[:limit]:
        where = f"{weights_dir.rstrip('/')}/{shard}" if shard else index_path
        lines.append(f"  {key}: expected in {where} — {reason}")
    more = f"\n  ... and {len(absent) - limit} more" if len(absent) > limit else ""
    shards = sorted({s for _, s, _ in absent if s})
    raise AbsentWeightError(
        f"component '{component}': {len(absent)} weight(s) the load needs are absent from the "
        f"container — refused before execution (index: {index_path}):\n"
        + "\n".join(lines) + more
        + (f"\nShard(s) involved: {', '.join(shards)}. A container whose files changed under "
           f"the run is restaged, never run partially." if shards else ""))


def loader_weight_consumers(dag: dict) -> Dict[str, str]:
    """Every `param::`/`buffer::` tensor an op in the graph's execution order reads whose value
    comes from the container's weights → the first op that reads it.

    A graph constant (its bytes are in graph.json), a computed buffer and a folded constant
    are filled by their own paths, each with its own refusal, and are left out. What remains
    must be bound when the sequence binds its weights: a slot left empty there is a weight the
    graph binds and the load did not deliver, and the first op to read it would meet None.
    Both engines' sequences compute it once at compile, before their elimination passes
    rewire consumers (an eliminated `detach`/`t` hands its weight to the op after it, which
    still reads it)."""
    tensors = dag.get("tensors") or {}
    ops = dag.get("ops") or {}
    if isinstance(ops, list):
        ops = {op.get("op_uid"): op for op in ops}
    out: Dict[str, str] = {}
    for uid in dag.get("execution_order") or []:
        for tid in (ops.get(uid) or {}).get("input_tensor_ids") or []:
            if tid in out or not (tid.startswith("param::") or tid.startswith("buffer::")):
                continue
            meta = tensors.get(tid) or {}
            if meta.get("constant") or meta.get("is_computable") or meta.get("folded_const"):
                continue
            out[tid] = uid
    return out


def refuse_unbound_weights(component, unbound: list, tensors: dict, limit: int = 20) -> None:
    """Raise AbsentWeightError for weights the graph binds that reached the bind empty —
    `unbound` is [(tid, first consumer op uid)]. No-op when empty. Shared by both engines'
    sequences (R30)."""
    if not unbound:
        return
    lines = []
    for tid, uid in unbound[:limit]:
        name = (tensors.get(tid) or {}).get("weight_name") or tid.split("::", 1)[-1]
        lines.append(f"  {tid} (read by {uid}): no loaded tensor under '{name}' nor under "
                     f"any trailing suffix of it")
    more = f"\n  ... and {len(unbound) - limit} more" if len(unbound) > limit else ""
    raise AbsentWeightError(
        f"component '{component}': {len(unbound)} weight(s) the graph binds are absent at the "
        f"weight bind — refused before the first op reads None:\n" + "\n".join(lines) + more
        + "\nThe container's weights_index.json does not provide them, or their load did not "
          "deliver them.")
