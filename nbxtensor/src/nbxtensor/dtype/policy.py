"""Precision by policy (DESIGN rule 12): an explicit, immutable object that says in which dtype each op computes.

Modelled on JAX's jmp (`Policy(param_dtype, compute_dtype, output_dtype)`, `get_policy("params=float32,
compute=float16,output=float32")`), with the op classes of PyTorch's autocast as its default (`policy.yml`): a
policy is plain data — three dtype names, a flag, and the op classes as a mapping of names — handed to the calls that
use it. No hidden state, no context manager, no global switch: a call computes in the dtype the policy it was given
says (rule 11: what a Rust core would take as a struct).

- `param`: the dtype weights are kept in; `compute`: the dtype the "lower" class computes in; `output`: the dtype a
  caller's results are returned in.
- `compute_dtype(op, input_dtypes)`: the dtype op `op` computes in, from its class: `float32` (whatever the compute
  dtype; or the compute dtype for a `float32_half_io` op when `float32_ops_half_io` is set), `lower` (the compute
  dtype), `promote` (the widest floating input), none (its inputs promoted among themselves).
- `fill_value(op, value, dtype)`: a scalar fill of a `scalar_fill` / `creation_fill` op clamped to `dtype`'s finite
  range (a mask sentinel written for bfloat16 does not overflow float16).

An application builds the policy from what it knows (NeuroBrix: its DtypeEngine, from the hardware profile and the
container) and passes it; the library knows no model and no profile.
"""
from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from nbxtensor.dtype.itemsize import canonical_name, finite_max, kind
from nbxtensor.dtype.promotion import binary_result

_FILE = Path(__file__).resolve().parent / "policy.yml"
CLASSES = ("float32", "lower", "promote", "float32_half_io", "scalar_fill", "creation_fill")

__all__ = ["Policy", "default_classes", "get_policy", "CLASSES"]


@functools.lru_cache(maxsize=1)
def _classes_from_file() -> Tuple[Tuple[str, FrozenSet[str]], ...]:
    import yaml
    data = (yaml.safe_load(_FILE.read_text()) or {}).get("classes") or {}
    unknown = sorted(set(data) - set(CLASSES))
    missing = sorted(set(CLASSES) - set(data))
    if unknown or missing:
        raise ValueError(f"{_FILE}: classes {unknown} unknown, {missing} missing (known: {', '.join(CLASSES)})")
    return tuple((c, frozenset(str(o) for o in data[c])) for c in CLASSES)


def default_classes() -> Dict[str, FrozenSet[str]]:
    """The shipped op classes (`policy.yml`): {class: frozenset of op names}."""
    return dict(_classes_from_file())


def _result(names: Sequence[str]) -> str:
    """The dtype inputs of `names` compute in together: the type promotion of every pair (`binary_result`)."""
    return functools.reduce(binary_result, names)


def _floating(dtype: str) -> bool:
    return kind(dtype) == "float"


@dataclasses.dataclass(frozen=True)
class Policy:
    param: str = "float32"
    compute: str = "float32"
    output: str = "float32"
    float32_ops_half_io: bool = False
    classes: Tuple[Tuple[str, FrozenSet[str]], ...] = dataclasses.field(default_factory=_classes_from_file)

    def __post_init__(self):
        for field in ("param", "compute", "output"):
            name = canonical_name(getattr(self, field))
            if not _floating(name):
                raise ValueError(f"Policy.{field}: {name!r} is not a floating dtype")
            object.__setattr__(self, field, name)
        names = [c for c, _ in self.classes]
        if sorted(names) != sorted(CLASSES):
            raise ValueError(f"Policy.classes: {names} — a policy names every class ({', '.join(CLASSES)})")
        object.__setattr__(self, "classes", tuple((c, frozenset(ops)) for c, ops in self.classes))

    def _ops(self, cls: str) -> FrozenSet[str]:
        return dict(self.classes)[cls]

    def ops(self, cls: str) -> FrozenSet[str]:
        """The op names of class `cls`."""
        if cls not in CLASSES:
            raise KeyError(f"Policy.ops: {cls!r} is not a class ({', '.join(CLASSES)})")
        return self._ops(cls)

    def op_class(self, op: str) -> Optional[str]:
        """The compute class of `op` (`float32`, `lower`, `promote`), or None when it is in none."""
        for cls in ("float32", "lower", "promote"):
            if op in self._ops(cls):
                return cls
        return None

    def compute_dtype(self, op: str, input_dtypes: Sequence[str]) -> Optional[str]:
        """The dtype `op` computes in, given its inputs' dtypes; None when it has no input (a creation op takes the
        dtype it is asked for). An op with no floating input computes in its inputs' promoted dtype."""
        names = [canonical_name(d) for d in input_dtypes]
        if not names:
            return None
        if not any(_floating(n) for n in names):
            return _result(names)
        cls = self.op_class(op)
        if cls == "float32":
            if self.float32_ops_half_io and op in self._ops("float32_half_io"):
                return self.compute
            return "float32"
        if cls == "lower":
            return self.compute
        return _result([n for n in names if _floating(n)] if cls == "promote" else names)

    def fill_value(self, op: str, value, dtype: str):
        """`value`, a scalar a fill op converts to `dtype`, clamped to that dtype's finite range when `op` is a fill
        op (a value of another kind, or of an op in no fill class, is returned unchanged)."""
        if op not in self._ops("scalar_fill") and op not in self._ops("creation_fill"):
            return value
        if not isinstance(value, float):
            return value
        name = canonical_name(dtype)
        if not _floating(name):
            return value
        top = finite_max(name)
        return max(-top, min(top, value))

    def replace(self, **changes) -> "Policy":
        """A new policy with `changes` (the policy itself never changes)."""
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> Dict[str, object]:
        """The policy as plain data (names and lists only)."""
        return {"param": self.param, "compute": self.compute, "output": self.output,
                "float32_ops_half_io": self.float32_ops_half_io,
                "classes": {c: sorted(ops) for c, ops in self.classes}}


def get_policy(spec: str, classes: Optional[Mapping[str, Sequence[str]]] = None) -> Policy:
    """A policy from jmp's spelling: `"params=float32,compute=float16,output=float32"` (also `p=`, `c=`, `o=`, and
    `"float16"` alone for all three); `classes` replaces the shipped op classes."""
    spec = spec.strip()
    fields: Dict[str, str] = {}
    if "=" not in spec:
        fields = {"param": spec, "compute": spec, "output": spec}
    else:
        alias = {"p": "param", "params": "param", "param": "param", "c": "compute", "compute": "compute",
                 "o": "output", "output": "output"}
        for part in spec.split(","):
            key, _, value = part.partition("=")
            key = key.strip().lower()
            if key not in alias:
                raise ValueError(f"get_policy: {key!r} names no field (params, compute, output)")
            fields[alias[key]] = value.strip()
    if classes is not None:
        fields["classes"] = tuple((c, frozenset(classes[c])) for c in CLASSES)
    return Policy(**fields)
