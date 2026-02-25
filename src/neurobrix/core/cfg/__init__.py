# core/cfg/__init__.py
"""
CFG Module for Classifier-Free Guidance

Components:
- CFGEngine: Unified CFG engine (replaces CFGExecutor + CFGStrategy)
- CFGMode: Execution mode (disabled/batched/sequential)
"""

from .engine import CFGEngine, CFGMode

__all__ = [
    "CFGEngine",
    "CFGMode",
]
