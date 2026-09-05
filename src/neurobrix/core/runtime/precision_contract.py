"""Precision contract of a component — ONE resolver for every execution path.

A component's contract answers one vendor fact: "does the vendor run this
component's forward in plain fp16, and which modules does it keep in fp32?"
Sources, in precedence order:

  1. the per-model registry flag ``activations_fp16_safe`` (env override
     ``NBX_ACTIVATIONS_FP16_SAFE``) and list ``keep_in_fp32_modules``;
  2. the per-BACKBONE contract in ``config/dtype_contracts.yml``
     (``activations_fp16_safe``, ``keep_in_fp32_modules``);
  3. absent everywhere → False, no pins — the conservative behaviour every
     undeclared component has always had.

Both readers are keyed by the MANIFEST's ``model_name`` (the vendor's), never
by the cache directory: a hub-slug install (``Sana-1600M-MultiLing``) has a
directory name that matches no registry key and used to default every flag
silently (2026-09-04).

Consumers:
  * compiled / sequential (``core/dtype/engine.DtypeEngine``): the flag AND
    the per-op fp32 pins (``fp32_op_uids``).
  * triton / triton_sequential (``TritonDtypeEngine``): the REGISTRY flag only
    (``use_contract=False``) — that engine stores an fp32 matmul output unless
    the flag is set and has no per-op pin yet, so consuming the backbone
    contract there would drop the T5 ``ffn.down`` protection. Tracked as
    D-PRECISION-CONTRACT-TRITON-PARITY; the R30 closure is pins in the
    triton dispatcher, then ``use_contract=True`` at both triton sites.

Pins name MODULES (the vendor casts a region, not an op): a plain suffix
(``ffn.down`` = that module and nothing below it) or an fnmatch pattern on the
full ``parent_module`` path (``block.[0-9]*``). Every COMPUTE op the module
owns is pinned; view/metadata ops (no kernel) and in-place ops (the fp32
wrapper would write into a copy) are never pinned.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

from neurobrix.core.config.loader import get_backbone_dtype_contract
from neurobrix.core.runtime import registry_flags
from neurobrix.kernels.classification import is_metadata_op

FLAG = "activations_fp16_safe"
FLAG_ENV = "NBX_ACTIVATIONS_FP16_SAFE"
PIN_LIST = "keep_in_fp32_modules"


def read_manifest(cache_path: Optional[str]) -> Dict[str, Any]:
    """The extracted container's manifest, or {} when there is no cache path.
    A manifest that EXISTS but cannot be read RAISES (present-but-broken
    class, engine audit #2) — a silent {} would default every flag."""
    if cache_path is None:
        return {}
    p = Path(cache_path) / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def registry_model_name(cache_path: Optional[str]) -> Optional[str]:
    """The name registry flags are keyed by: the manifest's ``model_name``,
    falling back to the directory name only when the manifest carries none."""
    if cache_path is None:
        return None
    name = read_manifest(cache_path).get("model_name")
    return name or Path(cache_path).name


def component_backbone(cache_path: Optional[str], component_name: str) -> Optional[str]:
    comp = (read_manifest(cache_path).get("components") or {}).get(component_name)
    return comp.get("backbone") if isinstance(comp, dict) else None


def _module_hit(parent: str, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatchcase(parent, pattern)
    return parent == pattern or parent.endswith("." + pattern)


def _pinnable(op_type: str) -> bool:
    name = op_type[6:] if op_type.startswith("aten::") else op_type
    if name.endswith("_"):          # in-place: the fp32 wrapper would write a copy
        return False
    return not is_metadata_op(name)  # view / shape / creation ops carry no kernel


# Ops whose INPUT precision the vendor protects even in a plain-fp16 forward
# (sinusoids of timestep x frequency, exponentials of scaled ranges): an
# fp32-computed value feeding one of these is a vendor island that an fp32
# trace elided, so it is never narrowed on the way in. PyTorch's autocast
# leaves sin/cos in the input dtype; the vendor's code writes `.float()`.
# Not in this set: the half-IO kernels (layer_norm, group_norm, softmax —
# fp16 in/out with fp32 inside, the vendor's own kernels) and the fp16
# activations (gelu, silu, tanh, sigmoid, erf) the vendor runs in fp16: under
# the contract they cast their own input, so a value reaching them ends its
# island there, like a matmul.
_PRECISION_CONSUMERS = frozenset({
    "sin", "cos", "exp", "expm1", "log", "log1p", "pow", "rsqrt", "sqrt",
    "reciprocal", "div", "cumsum", "sum", "mean", "var", "std", "norm",
})


# Ops that cast their own inputs (FP16-class): an fp32 value reaching one of
# these is narrowed there whatever we do — the island ends.
_CASTING_CONSUMERS = frozenset({
    "mm", "bmm", "addmm", "baddbmm", "addbmm", "matmul", "linear", "mv",
    "addmv", "conv1d", "conv2d", "conv3d", "convolution", "_convolution",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_attention_math",
    # half-IO kernels and fp16 activations: cast their own input under the contract
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    "softmax", "_softmax", "log_softmax", "_log_softmax",
    "gelu", "silu", "relu", "tanh", "sigmoid", "erf", "mish", "hardswish",
})


def narrowable_op_uids(dag: Optional[Dict[str, Any]]) -> FrozenSet[str]:
    """Ops whose fp32-computed output may be STORED in compute dtype under
    the contract.

    An output belongs to a vendor fp32 island when, following the graph
    forward through pass-through ops (elementwise arithmetic, views, cat…),
    it reaches a precision-class op (sin/cos/exp/log/pow/rsqrt/div/softmax/
    norms — see _PRECISION_CONSUMERS) before reaching an op that casts its
    inputs anyway (matmul / conv / attention). The vendor's plain-fp16
    forward writes `.float()` in front of such chains (timestep sinusoids,
    hand-rolled norms); an fp32 trace elides that cast, so the island is
    reconstructed from the consumers. A component output with no consumer
    in the graph is also kept fp32. Everything else — an fp32-computed
    value consumed only by pass-through ops that end in a casting op — is
    stored in compute dtype, exactly where the vendor's fp16 forward holds
    it in fp16. Computed to a fixpoint (backward propagation through the
    pass-through consumers)."""
    ops = (dag or {}).get("ops") or {}
    consumers: Dict[str, list] = {}
    for uid, op in ops.items():
        for tid in op.get("input_tensor_ids") or []:
            consumers.setdefault(tid, []).append(uid)

    def kind(uid: str) -> str:
        t = ops[uid].get("op_type", "")
        n = t[6:] if t.startswith("aten::") else t
        if n in _PRECISION_CONSUMERS:
            return "precision"
        if n in _CASTING_CONSUMERS:
            return "casting"
        return "passthrough"

    island = set()          # ops whose OUTPUT must stay fp32
    for uid, op in ops.items():
        outs = op.get("output_tensor_ids") or []
        cons = [c for tid in outs for c in consumers.get(tid, [])]
        if not outs or not cons:
            island.add(uid)  # component output (or dead): keep fp32
            continue
        if any(kind(c) == "precision" for c in cons):
            island.add(uid)
    changed = True
    while changed:
        changed = False
        for uid, op in ops.items():
            if uid in island:
                continue
            cons = [c for tid in (op.get("output_tensor_ids") or [])
                    for c in consumers.get(tid, [])]
            # a pass-through consumer that is itself an island carries the
            # value onward in fp32 → this output is part of the island
            if any(kind(c) == "passthrough" and c in island for c in cons):
                island.add(uid)
                changed = True
    return frozenset(uid for uid in ops if uid not in island)


def resolve(cache_path: Optional[str], component_name: str,
            dag: Optional[Dict[str, Any]], *, compute_is_fp16: bool,
            use_contract: bool = True) -> Tuple[bool, FrozenSet[str], FrozenSet[str]]:
    """(activations_fp16_safe, fp32_op_uids, narrow_op_uids) for one component.

    Only meaningful when the component computes in fp16: on bf16 or fp32
    compute no engine upcasts matmuls, so nothing is read and the default
    is returned."""
    if not compute_is_fp16:
        return False, frozenset(), frozenset()
    model_name = registry_model_name(cache_path)
    backbone = component_backbone(cache_path, component_name)
    contract = get_backbone_dtype_contract(backbone) if use_contract else {}

    safe = registry_flags.get_component_flag(model_name, component_name, FLAG,
                              default=None, env_override=FLAG_ENV)
    if safe is None:
        safe = bool(contract.get(FLAG, False))
    safe = bool(safe)

    patterns = tuple(contract.get(PIN_LIST) or ())
    registry_list = registry_flags.get_component_flag(model_name, component_name, PIN_LIST, default=None)
    if isinstance(registry_list, (list, tuple)):
        patterns = patterns + tuple(str(x) for x in registry_list)

    pinned = set()
    if patterns:
        for op_uid, op_data in ((dag or {}).get("ops") or {}).items():
            if not _pinnable(op_data.get("op_type", "")):
                continue
            parent = op_data.get("parent_module") or ""
            if any(_module_hit(parent, pat) for pat in patterns):
                pinned.add(op_uid)
    narrow = narrowable_op_uids(dag) if safe else frozenset()
    if safe or pinned:
        print(f"[DtypeEngine] {component_name}: precision contract "
              f"{FLAG}={safe} (backbone={backbone}), "
              f"{len(pinned)} op(s) pinned fp32 by the vendor's keep-in-fp32 list, "
              f"{len(narrow)} narrowable")
    return safe, frozenset(pinned), narrow
