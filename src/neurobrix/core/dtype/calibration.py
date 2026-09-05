"""Precision calibration — the DtypeEngine fp32-island detector.

On fp16 hardware the engine keeps a matrix multiply in fp32 by default: the
product accumulates in fp32 inside the library whatever the input dtype, but
the STORED fp16 output overflows at 65504 — a property of the model's
activations, not of the arithmetic. Which ops need that protection is a
MEASUREMENT, never a declaration: this module records, once per artifact,
the largest magnitude every op produced on the conservative reference path
over a whole request (every diffusion step, every decoded token), and the
engine derives its fp32 islands from that record for the compute dtype at
hand. No model name, no module name, no hand-written list takes part.

Three pieces:

  * ``RangeCensus`` — the observer a sequence installs on its op loop while
    a calibration is active: per-op max|out| accumulated ON DEVICE (one
    reduction per op, no host sync until ``finalize``).
  * ``CalibrationRecord`` — the per-component record: ``max_abs`` per op_uid,
    the stimulus it was measured on, the number of passes, and the graph
    signature (execution order + op types) so a re-traced artifact never
    reads a stale record. Home: the engine store
    ``~/.neurobrix/calibration/<model>/<component>.json``. Where the record
    travels with a distributed artifact (a container field under R18, or a
    hub side channel) is the owner's format decision, tracked in DETTE
    D-PRECISION-CONTRACT-DEPLOYMENT-SPLIT — no container field is read here.
  * ``islands_from_calibration`` — the rule: a compute op is pinned fp32
    when its output, or any input it reads, exceeds the bound
    ``finfo(compute_dtype).max / 2**headroom_bits`` (policy in
    ``config/precision_calibration.yml``). Only FINITE magnitudes count:
    ±inf and NaN are representable in every float dtype (a -inf mask fill,
    log(0) in a position bucket) and are no overflow. View/metadata ops
    carry no kernel and are never pinned; they carry the value onward to
    the reader that is. On bf16 / fp32 compute the bound is astronomically
    high and no island exists — the record is hardware-universal by
    construction.

Lever record: validation_outputs/precision_census_2026_09_05/RECORD.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
    import torch
from neurobrix.core.runtime.tensor_compat import is_torch_tensor

from neurobrix.kernels.classification import is_metadata_op

FORMAT = "nbx-precision-calibration/1"
STORE_ROOT = Path(os.path.expanduser("~")) / ".neurobrix" / "calibration"


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

# Largest finite value of each compute dtype (IEEE 754 / bfloat16). Read
# here rather than through torch.finfo so the Triton branch, which applies
# the same islands, never imports torch (R33).
_FINITE_MAX = {
    "float16": 65504.0,
    "bfloat16": 3.3895313892515355e38,
    "float32": 3.4028234663852886e38,
    "float64": 1.7976931348623157e308,
}


def island_bound(compute_dtype, headroom_bits: int) -> float:
    """Magnitude above which a value cannot be stored in ``compute_dtype``
    (a dtype name, or a torch dtype) with ``headroom_bits`` of margin
    (2 bits = a quarter of the range)."""
    name = str(compute_dtype).replace("torch.", "")
    if name not in _FINITE_MAX:
        raise RuntimeError(f"island_bound: no finite range known for compute dtype {compute_dtype!r}")
    return _FINITE_MAX[name] / float(2 ** int(headroom_bits))


def pinnable(op_type: str) -> bool:
    """Compute ops only: in-place ops (the fp32 wrapper would write a copy)
    and view / shape / creation ops (no kernel) are never pinned."""
    name = op_type[6:] if op_type.startswith("aten::") else op_type
    if name.endswith("_"):
        return False
    return not is_metadata_op(name)


def _exceeds(value: Optional[float], bound: float) -> bool:
    """A recorded FINITE magnitude above the bound. A non-finite entry (a
    record written before finite/non-finite were told apart) never pins."""
    return value is not None and math.isfinite(value) and value > bound


def islands_from_calibration(dag: Dict[str, Any], max_abs: Dict[str, float],
                             bound: float) -> FrozenSet[str]:
    """op_uids to compute AND store in fp32 under the contract.

    Walks the graph in execution order. A tensor is over range when its
    producer's recorded magnitude exceeds the bound; a metadata op with an
    over-range input passes that state to its outputs even when it was not
    itself recorded. A compute op is pinned when its own output exceeds the
    bound or when any input it reads is over range (casting that input to
    the compute dtype would overflow before the op even runs)."""
    ops = dag.get("ops") or {}
    order = dag.get("execution_order") or list(ops)
    over: set = set()
    pinned: set = set()
    for uid in order:
        op = ops.get(uid)
        if op is None:
            continue
        op_type = op.get("op_type", "")
        outs = op.get("output_tensor_ids") or []
        ins = op.get("input_tensor_ids") or []
        reads_over = any(t in over for t in ins)
        own = _exceeds(max_abs.get(uid), bound)
        if own or (reads_over and not pinnable(op_type)):
            over.update(outs)
        if pinnable(op_type) and (own or reads_over):
            pinned.add(uid)
    return frozenset(pinned)


def graph_signature(dag: Dict[str, Any]) -> str:
    """Identity of the traced graph the record was measured on: execution
    order and op types. Values, shapes and weights do not take part — a
    record follows the trace, not the request."""
    ops = dag.get("ops") or {}
    h = hashlib.sha256()
    for uid in dag.get("execution_order") or list(ops):
        h.update(uid.encode())
        h.update(b"|")
        h.update(str((ops.get(uid) or {}).get("op_type", "")).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# The observer
# ---------------------------------------------------------------------------

class RangeCensus:
    """Per-op largest finite |out| accumulated on device while a sequence runs,
    and whether the op ever produced a non-finite value.

    ``observe`` costs a few device kernels per floating output and no host
    synchronisation; ``finalize`` reads the accumulators back once. Integer
    outputs, empty tensors and non-tensor results are not recorded."""

    def __init__(self) -> None:
        self._acc: Dict[str, torch.Tensor] = {}       # largest finite magnitude
        self._nonfinite: Dict[str, torch.Tensor] = {}  # any non-finite value seen
        self.passes = 0
        self.dag: Optional[Dict[str, Any]] = None
        self.cache_path: Optional[str] = None
        self.signature: Optional[str] = None

    def bind(self, dag: Dict[str, Any], cache_path: Optional[str]) -> None:
        """Bind the census to the graph at the point the precision contract
        is resolved — and freeze the graph signature THERE: the executor
        rewrites its graph afterwards (attention rescale normalisation,
        expert fusion), and the record must carry the identity the next
        load compares against, not the identity of the rewritten graph."""
        self.dag = dag
        self.cache_path = cache_path
        self.signature = graph_signature(dag)

    @staticmethod
    def _magnitude(result: Any):
        """(largest finite |x|, any non-finite) as device scalars, or None.

        The inf-norm reads the tensor once with no copy; only when it comes
        back non-finite (an inf or NaN somewhere) is the finite maximum
        recomputed through a masked copy — a 4K render's 8 GiB activations
        cannot afford a copy per op (Sana 4K OOM under the census, 2026-09-05)."""
        if is_torch_tensor(result):
            import torch
            if result.numel() == 0 or not (result.is_floating_point() or result.is_complex()):
                return None
            x = result.detach()
            m = torch.linalg.vector_norm(x, ord=math.inf).float()
            nonfinite = ~torch.isfinite(m)
            finite_max = torch.where(nonfinite, RangeCensus._finite_max_masked(x), m)
            return finite_max, nonfinite
        if isinstance(result, (tuple, list)):
            parts = [m for m in (RangeCensus._magnitude(r) for r in result) if m is not None]
            if not parts:
                return None
            mx, nf = parts[0]
            import torch
            for m, f in parts[1:]:
                mx = torch.maximum(mx, m.to(mx.device))
                nf = nf | f.to(nf.device)
            return mx, nf
        return None

    @staticmethod
    def _finite_max_masked(x: torch.Tensor) -> torch.Tensor:
        """max|x| over the finite elements — the copy path, taken lazily."""
        import torch
        a = x.abs()
        if a.dtype != torch.float32:
            a = a.float()
        a.masked_fill_(~torch.isfinite(a), 0.0)
        return a.amax()

    def observe(self, op_uid: str, result: Any) -> None:
        m = self._magnitude(result)
        if m is None:
            return
        mx, nf = m
        acc = self._acc.get(op_uid)
        if acc is None:
            self._acc[op_uid] = mx.clone()
            self._nonfinite[op_uid] = nf.clone()
        else:
            import torch
            torch.maximum(acc, mx.to(acc.device), out=acc)
            flag = self._nonfinite[op_uid]
            torch.logical_or(flag, nf.to(flag.device), out=flag)

    def pass_done(self) -> None:
        self.passes += 1

    def finalize(self) -> Dict[str, float]:
        """Largest finite magnitude per op (one host read per op)."""
        return {uid: float(acc.item()) for uid, acc in self._acc.items()}

    def non_finite_ops(self) -> list:
        return sorted(uid for uid, f in self._nonfinite.items() if bool(f.item()))


# Process-wide calibration session: {component_name: RangeCensus}. None when
# no calibration runs — the sequences' op loops read it once per run.
_ACTIVE: Optional[Dict[str, RangeCensus]] = None


def begin_calibration() -> Dict[str, RangeCensus]:
    global _ACTIVE
    _ACTIVE = {}
    return _ACTIVE


def end_calibration() -> Optional[Dict[str, RangeCensus]]:
    global _ACTIVE
    session, _ACTIVE = _ACTIVE, None
    return session


def active_census(component_name: Optional[str]) -> Optional[RangeCensus]:
    """The census of one component while a calibration is active, else None."""
    if _ACTIVE is None or not component_name:
        return None
    census = _ACTIVE.get(component_name)
    if census is None:
        census = _ACTIVE[component_name] = RangeCensus()
    return census


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    model_name: str
    component: str
    graph_signature: str
    max_abs: Dict[str, float]
    stimulus: Dict[str, Any] = field(default_factory=dict)
    passes: int = 0
    reference: str = "conservative"
    engine_version: str = ""
    created_at: str = ""
    non_finite: list = field(default_factory=list)   # ops that produced ±inf / NaN on the reference

    @classmethod
    def build(cls, model_name: str, component: str, dag: Dict[str, Any],
              max_abs: Dict[str, float], *, stimulus: Dict[str, Any], passes: int,
              reference: str, non_finite: Optional[list] = None,
              graph_signature: Optional[str] = None) -> "CalibrationRecord":
        try:
            from neurobrix import __version__ as engine_version
        except Exception:  # pragma: no cover — version module absent in a bare checkout
            engine_version = ""
        return cls(model_name=model_name, component=component,
                   graph_signature=graph_signature or globals()["graph_signature"](dag),
                   max_abs=dict(max_abs),
                   stimulus=dict(stimulus), passes=int(passes), reference=reference,
                   engine_version=str(engine_version),
                   created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   non_finite=sorted(non_finite or []))

    def matches(self, dag: Dict[str, Any]) -> bool:
        return self.graph_signature == graph_signature(dag)

    def to_dict(self) -> Dict[str, Any]:
        return {"format": FORMAT, "model_name": self.model_name, "component": self.component,
                "graph_signature": self.graph_signature, "stimulus": self.stimulus,
                "passes": self.passes, "reference": self.reference,
                "engine_version": self.engine_version, "created_at": self.created_at,
                "non_finite": list(self.non_finite), "max_abs": self.max_abs}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationRecord":
        if d.get("format") != FORMAT:
            raise ValueError(f"precision calibration: unknown record format {d.get('format')!r}")
        return cls(model_name=d["model_name"], component=d["component"],
                   graph_signature=d["graph_signature"],
                   max_abs={k: float(v) for k, v in (d.get("max_abs") or {}).items()},
                   stimulus=dict(d.get("stimulus") or {}), passes=int(d.get("passes") or 0),
                   reference=str(d.get("reference") or ""),
                   engine_version=str(d.get("engine_version") or ""),
                   created_at=str(d.get("created_at") or ""),
                   non_finite=list(d.get("non_finite") or []))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "CalibrationRecord":
        return cls.from_dict(json.loads(Path(path).read_text()))


def store_path(model_name: str, component: str) -> Path:
    return STORE_ROOT / model_name / f"{component}.json"


def load_record(model_name: Optional[str], component: str) -> Optional[CalibrationRecord]:
    """The component's record from the engine store; None when there is
    none. A record that EXISTS but cannot be read raises — a silent None
    would put the component on the conservative path without a word."""
    if not model_name:
        return None
    p = store_path(model_name, component)
    if not p.exists():
        return None
    return CalibrationRecord.load(p)
