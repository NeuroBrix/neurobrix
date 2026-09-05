"""Precision contract of a component — ONE resolver for every execution path.

Which ops a component keeps in fp32 on half-precision hardware is a
MEASUREMENT: the component's calibration record (core/dtype/calibration.py —
the largest magnitude every op produced on the conservative reference path
over a whole request) and the engine's policy (config/precision_calibration.yml)
decide, for the compute dtype at hand. Sources, in precedence order:

  1. ``NBX_ACTIVATIONS_FP16_SAFE`` (diagnostic): ``0`` = the conservative
     reference path whatever the record says (the path the census runs
     on); ``1`` = the contract with no island (a vendor-style plain fp16
     forward, the differential for "is it the islands or the store?").
  2. the calibration record in the engine store
     ``~/.neurobrix/calibration/<model>/<component>.json`` (how a record
     travels with a distributed artifact is the owner's format decision,
     DETTE D-PRECISION-CONTRACT-DEPLOYMENT-SPLIT) — read only when its
     graph signature matches the loaded graph;
  3. no record → the conservative default every uncalibrated component
     has always had (fp32 matmul store on fp16 hardware).

No model name, no module name and no hand-written list take part: the
2026-09-04 backbone contract file and the per-model registry flags were
retired by the 2026-09-05 census (validation_outputs/precision_census_2026_09_05).

Consumers:
  * compiled / sequential (``core/dtype/engine.DtypeEngine``): the flag AND
    the per-op islands (``fp32_op_uids``) AND the structural narrowing set.
  * triton / triton_sequential (``TritonDtypeEngine``): the flag only — that
    engine has no per-op island yet (D-PRECISION-CONTRACT-TRITON-PARITY), so
    it takes the contract only when the record shows NO island; a component
    whose record needs islands stays on the conservative path there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple


from neurobrix.core.config.loader import get_precision_calibration_policy
from neurobrix.core.dtype import calibration as _cal

FLAG = "activations_fp16_safe"
FLAG_ENV = "NBX_ACTIVATIONS_FP16_SAFE"


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
    """The name records and registry flags are keyed by: the manifest's
    ``model_name``, falling back to the directory name only when the
    manifest carries none."""
    if cache_path is None:
        return None
    name = read_manifest(cache_path).get("model_name")
    return name or Path(cache_path).name


def _env_force() -> Optional[bool]:
    v = os.environ.get(FLAG_ENV)
    if v is None:
        return None
    v = v.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"ZERO FALLBACK: {FLAG_ENV}={v!r} is neither on nor off")


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
# TERMINAL precision-sensitive consumers only. `pow`, `exp`, `sqrt` are not
# listed: the vendor's fp16 forward runs them in fp16 inside GELU / SiLU
# chains (T5's gated GELU: x + 0.044715·pow(x,3) → tanh), and when they open
# a real island (a sinusoid: exp → mul → sin; a hand-rolled norm: pow → mean)
# the terminal op at its end names it — the fixpoint below walks back to them.
_PRECISION_CONSUMERS = frozenset({
    "sin", "cos", "log", "log1p", "rsqrt", "reciprocal", "div",
    "cumsum", "sum", "mean", "var", "std", "norm",
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


def load_calibration(cache_path: Optional[str], component_name: str,
                     dag: Optional[Dict[str, Any]]) -> Optional[_cal.CalibrationRecord]:
    """The component's calibration record when one exists AND was measured
    on this very graph (signature match); None otherwise. A record measured
    on another trace is reported and ignored — never applied."""
    model_name = registry_model_name(cache_path)
    record = _cal.load_record(model_name, component_name)
    if record is None or dag is None:
        return record
    if record.passes < 1 or not record.max_abs:
        # A record that observed nothing (the component ran on a path the
        # census does not see) carries no information: the contract would
        # switch on blind. Refused loudly; the component stays conservative.
        import sys
        print(f"[DtypeEngine] REFUSED RECORD {component_name}: calibration "
              f"{record.graph_signature} observed no op ({record.passes} pass(es), "
              f"{len(record.max_abs)} op(s) recorded) — its execution path is not "
              f"observed by the census. Not applied — conservative path.",
              file=sys.stderr, flush=True)
        return None
    if not record.matches(dag):
        # Present-but-inconsistent: said loudly on stderr at every load, and
        # the record is NOT applied. The component runs the conservative
        # path (correct, slower). Whether this must refuse to run instead
        # is the owner's call (doctrine review 2026-09-05).
        import sys
        print(f"[DtypeEngine] REFUSED RECORD {component_name}: calibration "
              f"{record.graph_signature} was measured on another graph than "
              f"{_cal.graph_signature(dag)} (re-traced artifact or engine "
              f"rewrite). Not applied — conservative path. Re-measure: "
              f"`neurobrix calibrate --model {model_name}`", file=sys.stderr, flush=True)
        return None
    return record


def resolve(cache_path: Optional[str], component_name: str,
            dag: Optional[Dict[str, Any]], *, compute_dtype,
            supports_op_pins: bool = True) -> Tuple[bool, FrozenSet[str], FrozenSet[str]]:
    """(activations_fp16_safe, fp32_op_uids, narrow_op_uids) for one component.

    Only meaningful when the component computes in fp16: on bf16 or fp32
    compute no engine upcasts matmuls, so nothing is read and the default
    is returned. ``supports_op_pins=False`` (an engine without per-op
    islands) takes the contract only when the record needs none."""
    # A dtype name or a torch dtype; compared by name so the Triton branch
    # resolves the same contract without torch (R33).
    compute_dtype = str(compute_dtype).replace("torch.", "")
    if compute_dtype != "float16":
        return False, frozenset(), frozenset()
    forced = _env_force()
    if forced is False:
        return False, frozenset(), frozenset()
    record = load_calibration(cache_path, component_name, dag)
    if record is None:
        if forced is True:
            print(f"[DtypeEngine] {component_name}: {FLAG_ENV}=1 — contract forced "
                  f"with no island (no calibration record)")
            return True, frozenset(), narrowable_op_uids(dag)
        print(f"[DtypeEngine] {component_name}: no calibration record — conservative "
              f"path (fp32 matmul store); `neurobrix calibrate --model "
              f"{Path(cache_path).name if cache_path else '<model>'}` measures one")
        return False, frozenset(), frozenset()
    policy = get_precision_calibration_policy()
    # Diagnostic (default off): NBX_PRECISION_HEADROOM_BITS=<n> overrides the
    # policy's headroom for one run — the differential for "is the bound the
    # knob?" when a calibrated arm drifts from the reference.
    headroom = os.environ.get(HEADROOM_ENV)
    bound = _cal.island_bound(compute_dtype, int(headroom) if headroom else policy["headroom_bits"])
    pinned = _cal.islands_from_calibration(dag or {}, record.max_abs, bound)
    if pinned and not supports_op_pins:
        print(f"[DtypeEngine] {component_name}: calibration needs {len(pinned)} fp32 "
              f"island(s) this engine cannot pin per op — conservative path "
              f"(D-PRECISION-CONTRACT-TRITON-PARITY)")
        return False, frozenset(), frozenset()
    narrow = narrowable_op_uids(dag)
    unrecorded = sum(1 for uid, op in ((dag or {}).get("ops") or {}).items()
                     if _cal.pinnable(op.get("op_type", "")) and uid not in record.max_abs)
    print(f"[DtypeEngine] {component_name}: precision contract from calibration "
          f"{record.graph_signature} ({record.passes} pass(es), reference "
          f"{record.reference}): {len(pinned)} op(s) islanded fp32 above "
          f"{bound:.6g}, {len(narrow)} narrowable, {unrecorded} compute op(s) "
          f"without an observation (treated as in range)")
    _dump_islands(component_name, dag or {}, record, pinned)
    return True, frozenset(pinned), narrow


ISLANDS_ENV = "NBX_PRECISION_ISLANDS"
HEADROOM_ENV = "NBX_PRECISION_HEADROOM_BITS"


def _dump_islands(component_name: str, dag: Dict[str, Any],
                  record: _cal.CalibrationRecord, pinned: FrozenSet[str]) -> None:
    """``NBX_PRECISION_ISLANDS=<tsv>``: append the engine's own island
    decision (component, op_uid, op_type, parent_module, calibrated
    magnitude) — the activation proof of a calibration, read by
    tools/precision_islands_report.py. Default off, one line per pinned op."""
    path = os.environ.get(ISLANDS_ENV)
    if not path:
        return
    ops = dag.get("ops") or {}
    with open(path, "a") as fh:
        for uid in dag.get("execution_order") or list(ops):
            if uid in pinned:
                op = ops.get(uid) or {}
                fh.write(f"{component_name}\t{uid}\t{op.get('op_type', '')}\t"
                         f"{op.get('parent_module') or ''}\t{record.max_abs.get(uid, float('nan')):.6g}\n")
