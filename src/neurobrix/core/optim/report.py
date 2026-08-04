"""Optimization report — the audit trail of the OptimizationEngine.

The optimization never lies about what it did (or, in report-only
mode, about what it FOUND). Every finding carries the op_uids
involved, the category, the owning pass and its gate class, and enough
detail to re-locate the site in the graph. Reports serialize to JSON
so a finding recorded today is byte-comparable with the same analysis
re-run after any engine change.

Phase 1 reports findings only (no transformation exists). From Phase 2
on, applied transformations reuse the same structure with the applied
flag — one report format for the whole life of the engine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .policy import PASS_REGISTRY

REPORT_SCHEMA_VERSION = 1


def graph_fingerprint(raw_bytes: bytes) -> str:
    """sha256 of the graph.json bytes — the cache-invalidation anchor.

    The optimization cache key is (graph_fingerprint, pass name, pass
    version): any re-trace or pass upgrade invalidates by construction.
    """
    return hashlib.sha256(raw_bytes).hexdigest()


@dataclass
class Finding:
    category: str  # a PASS_REGISTRY key
    op_uids: list[str]  # the site (1..n ops)
    detail: str  # human-readable, e.g. "mul by scalar 1.0"
    ops_removable: int = 0  # ops this site would eliminate
    symbolic: bool = False  # site involves symbolic extents
    meta: bool = False  # metadata-class site (no launch saved)


@dataclass
class AnalysisReport:
    model: str
    component: str
    graph_fingerprint: str
    n_ops: int
    n_tensors: int
    findings: list[Finding] = field(default_factory=list)
    # Build-side bug signatures surfaced by the analysis (e.g. a slice
    # end frozen at a symbolic dim's trace extent). NEVER optimization
    # sites — they are reported for a fix at the source (the tracer),
    # per the frozen-dim doctrine.
    suspects: list[Finding] = field(default_factory=list)
    schema_version: int = REPORT_SCHEMA_VERSION

    def add(self, finding: Finding) -> None:
        if finding.category not in PASS_REGISTRY:
            raise ValueError(f"unknown pass category: {finding.category}")
        self.findings.append(finding)

    def add_suspect(self, op_uids: list[str], detail: str) -> None:
        self.suspects.append(Finding(
            category="forge_suspect", op_uids=op_uids, detail=detail,
        ))

    # -- summaries -----------------------------------------------------

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.category] = out.get(f.category, 0) + 1
        return out

    def ops_removable(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.category] = out.get(f.category, 0) + f.ops_removable
        return out

    # -- serialization -------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["counts"] = self.counts()
        d["ops_removable"] = self.ops_removable()
        d["n_suspects"] = len(self.suspects)
        d["gate_classes"] = {
            name: PASS_REGISTRY[name].gate_class.value
            for name in sorted(self.counts())
        }
        return d

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=False) + "\n"
        )
