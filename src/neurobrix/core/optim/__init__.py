"""OptimizationEngine — load-time graph optimization (report-only in Phase 1).

The .nbx graph is the canonical truth; optimization is an execution
view applied at load, engine-side, never persisted into the container.
Analysis and transformations are engine-neutral (they work on the
shared graph contract); only lowering differs per engine.

Phase 1 ships analysis only: the analyzer maps optimization
opportunities across the zoo and reports them — it transforms nothing.
"""

from .policy import GateClass, PassPolicy, PASS_REGISTRY, OPTIM_LEVELS
from .report import AnalysisReport, Finding
from .analyzer import GraphAnalyzer

__all__ = [
    "GateClass",
    "PassPolicy",
    "PASS_REGISTRY",
    "OPTIM_LEVELS",
    "AnalysisReport",
    "Finding",
    "GraphAnalyzer",
]
