"""What the compiled CFG engine (core/cfg/engine.CFGEngine) and the Triton one (triton/cfg/engine.TritonCFGEngine) ran
alike, body for body — written once. Torch-free at import (the Triton engine inherits it, R33; the core/cfg package
loads torch); the tensors `apply_guidance` combines are each branch's own.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable


class CFGMode(Enum):
    """CFG execution mode."""
    DISABLED = auto()     # No CFG (scale <= threshold)
    BATCHED = auto()      # Batch [cond, uncond] together (default)
    SEQUENTIAL = auto()   # Run cond/uncond separately (TP strategy)


class CFGSteps:
    """The CFG engine's state and the steps both branches share; each engine inherits it."""

    def __init__(
        self,
        ctx,
        execute_component_fn: Callable[[str, str], Any],
        extract_primary_output_fn: Callable[[str, Any], Any],
        guidance_scale: float = 7.5,
        mode: CFGMode = CFGMode.BATCHED,
    ):
        self._ctx = ctx
        self._execute_component = execute_component_fn
        self._extract_primary_output = extract_primary_output_fn
        self.guidance_scale = guidance_scale
        self.mode = mode

    @property
    def is_enabled(self) -> bool:
        """Check if CFG is enabled (guidance_scale above threshold)."""
        return self.mode != CFGMode.DISABLED

    def _split_components(self) -> frozenset:
        """The loop components the plan runs one guidance branch at a time — the plan's decision
        (`ExecutionPlan.cfg_split_components`), never this engine's."""
        return frozenset(getattr(self._ctx.plan, "cfg_split_components", None) or ())

    @staticmethod
    def apply_guidance(
        cond_output,
        uncond_output,
        guidance_scale: float,
    ):
        """Pure CFG formula: uncond + scale * (cond - uncond)."""
        return uncond_output + guidance_scale * (cond_output - uncond_output)
