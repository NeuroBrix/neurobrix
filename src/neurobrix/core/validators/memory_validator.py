"""
NeuroBrix Validators - Memory Validator.

Validates memory plans and allocation feasibility.

ZERO FALLBACK: Memory violations raise explicit MemoryPlanError.
"""

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseValidator, SafetyLevel, ValidationResult, MemoryPlanError
from .config import get_config


class MemoryValidator(BaseValidator):
    """
    Validates memory allocation plans.

    Checks:
    - Total memory fits in available VRAM
    - Peak memory with safety margin
    - Tensor size limits
    - Memory fragmentation risks
    - Offloading feasibility
    """

    @property
    def name(self) -> str:
        return "MemoryValidator"

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        super().__init__(safety_level)
        self.config = get_config()

    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate memory plan for the graph.

        Args:
            graph: Execution graph
            context: Must contain:
                - 'memory_plan': Dict with allocation info
                - 'available_memory': Available VRAM in bytes
                - 'device_info': Device specifications

        Returns:
            ValidationResult with memory validation status
        """
        warnings = []
        errors = []
        metrics = {}

        # Extract context
        memory_plan = context.get('memory_plan', {})
        available_memory = context.get('available_memory', 0)
        device_info = context.get('device_info', {})

        if not memory_plan:
            if self.safety_level == SafetyLevel.PARANOID:
                warnings.append("No memory plan provided")
            return ValidationResult(
                passed=True,
                validator_name=self.name,
                message="No memory plan to validate",
                warnings=warnings,
            )

        # Calculate totals
        total_required = self._calculate_total_memory(memory_plan, metrics)
        peak_required = self._calculate_peak_memory(memory_plan, metrics)

        # Validate against available memory
        if available_memory > 0:
            self._validate_memory_fits(
                total_required, peak_required, available_memory,
                warnings, errors, metrics
            )

        # Validate individual allocations
        allocations = memory_plan.get('allocations', [])
        for alloc in allocations:
            self._validate_allocation(alloc, warnings, errors)

        # Check fragmentation risk
        if self.safety_level >= SafetyLevel.STANDARD:
            self._check_fragmentation_risk(memory_plan, warnings, metrics)

        # Check offloading plan if present
        offload_plan = memory_plan.get('offload_plan', {})
        if offload_plan:
            self._validate_offload_plan(offload_plan, context, warnings, errors)

        if errors:
            return ValidationResult(
                passed=False,
                validator_name=self.name,
                message="\n".join(errors),
                warnings=warnings,
                metrics=metrics,
            )

        return ValidationResult(
            passed=True,
            validator_name=self.name,
            message=f"Memory validated: {peak_required / 1024**3:.2f} GB peak",
            warnings=warnings,
            metrics=metrics,
        )

    def _calculate_total_memory(
        self,
        memory_plan: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> int:
        """Calculate total memory required."""
        total = 0

        # Sum all allocations
        allocations = memory_plan.get('allocations', [])
        for alloc in allocations:
            size = alloc.get('size_bytes', 0)
            total += size

        # Add workspace memory
        workspace = memory_plan.get('workspace_bytes', 0)
        total += workspace

        metrics['total_memory_bytes'] = total
        metrics['total_memory_gb'] = total / (1024**3)

        return total

    def _calculate_peak_memory(
        self,
        memory_plan: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> int:
        """Calculate peak memory usage."""
        # If peak is explicitly provided, use it
        if 'peak_bytes' in memory_plan:
            peak = memory_plan['peak_bytes']
        else:
            # Estimate from allocations (conservative)
            peak = self._calculate_total_memory(memory_plan, {})

        metrics['peak_memory_bytes'] = peak
        metrics['peak_memory_gb'] = peak / (1024**3)

        return peak

    def _validate_memory_fits(
        self,
        total_required: int,
        peak_required: int,
        available_memory: int,
        warnings: List[str],
        errors: List[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Validate memory requirements fit in available memory."""
        config = self.config

        # Calculate usable memory with safety margin
        usable_memory = int(available_memory * (1.0 - config.memory_safety_margin))
        metrics['usable_memory_bytes'] = usable_memory
        metrics['usable_memory_gb'] = usable_memory / (1024**3)

        # Check peak fits
        if peak_required > usable_memory:
            errors.append(
                f"Peak memory {peak_required / 1024**3:.2f} GB exceeds "
                f"usable memory {usable_memory / 1024**3:.2f} GB "
                f"(with {config.memory_safety_margin:.0%} safety margin)"
            )

        # Warning threshold check
        usage_ratio = peak_required / available_memory
        metrics['memory_usage_ratio'] = usage_ratio

        if usage_ratio > config.warn_memory_usage:
            warnings.append(
                f"High memory usage: {usage_ratio:.1%} of available memory"
            )

    def _validate_allocation(
        self,
        alloc: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """Validate a single allocation."""
        config = self.config

        size = alloc.get('size_bytes', 0)
        name = alloc.get('name', 'unknown')

        # Check max tensor size
        if size > config.max_tensor_bytes:
            errors.append(
                f"Tensor '{name}' size {size / 1024**3:.2f} GB exceeds "
                f"max {config.max_tensor_bytes / 1024**3:.2f} GB"
            )

        # Check for zero-size allocations
        if size == 0:
            warnings.append(f"Zero-size allocation for '{name}'")

        # Check alignment (should be 256-byte aligned for efficiency)
        alignment = alloc.get('alignment', 256)
        if size % alignment != 0:
            warnings.append(
                f"Allocation '{name}' not aligned to {alignment} bytes"
            )

    def _check_fragmentation_risk(
        self,
        memory_plan: Dict[str, Any],
        warnings: List[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Check for memory fragmentation risks."""
        allocations = memory_plan.get('allocations', [])

        if len(allocations) < 2:
            return

        # Count size distribution
        sizes = [a.get('size_bytes', 0) for a in allocations]
        sizes.sort()

        # Large variance in sizes can cause fragmentation
        if sizes:
            min_size = sizes[0]
            max_size = sizes[-1]

            if min_size > 0 and max_size / min_size > 1000:
                warnings.append(
                    "High size variance in allocations may cause fragmentation. "
                    f"Min: {min_size / 1024:.1f} KB, Max: {max_size / 1024**2:.1f} MB"
                )

        metrics['allocation_count'] = len(allocations)
        metrics['min_allocation_bytes'] = min(sizes) if sizes else 0
        metrics['max_allocation_bytes'] = max(sizes) if sizes else 0

    def _validate_offload_plan(
        self,
        offload_plan: Dict[str, Any],
        context: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """Validate CPU/disk offload plan."""
        # Check CPU memory for offloading
        cpu_available = context.get('cpu_memory', 0)
        cpu_required = offload_plan.get('cpu_bytes', 0)

        if cpu_required > cpu_available:
            errors.append(
                f"CPU offload requires {cpu_required / 1024**3:.2f} GB, "
                f"only {cpu_available / 1024**3:.2f} GB available"
            )

        # Check transfer bandwidth impact
        transfer_bytes = offload_plan.get('transfer_bytes', 0)
        if transfer_bytes > 1024**3:  # > 1GB transfer
            warnings.append(
                f"Large CPU-GPU transfer: {transfer_bytes / 1024**3:.2f} GB. "
                f"This may impact performance."
            )

    def validate_single_allocation(
        self,
        size_bytes: int,
        available_bytes: int,
        name: str = "",
    ) -> None:
        """
        Validate a single memory allocation.

        Raises MemoryPlanError if invalid.
        """
        config = self.config

        if size_bytes > config.max_tensor_bytes:
            raise MemoryPlanError(
                f"Allocation '{name}' too large",
                required_bytes=size_bytes,
                suggestion=f"Max tensor size is {config.max_tensor_bytes / 1024**3:.0f} GB. "
                          f"Consider chunking or offloading.",
            )

        usable = int(available_bytes * (1.0 - config.memory_safety_margin))
        if size_bytes > usable:
            raise MemoryPlanError(
                f"Allocation '{name}' exceeds available memory",
                required_bytes=size_bytes,
                available_bytes=available_bytes,
                suggestion="Consider gradient checkpointing or model sharding.",
            )

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """Memory validation runs at all levels."""
        return True
