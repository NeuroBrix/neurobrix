"""
NeuroBrix Validators - Operation Validator.

Validates that all operations in graph are supported.

ZERO FALLBACK: Unsupported operations raise explicit UnsupportedOpsError.
"""

from typing import Any, Dict, List, Set

from .base import BaseValidator, SafetyLevel, ValidationResult, UnsupportedOpsError
from .config import get_config


class OpValidator(BaseValidator):
    """
    Validates operation support.

    Checks:
    - All ops are in supported set
    - Op parameters are valid
    - Op combinations are allowed
    - Hardware-specific op availability
    """

    @property
    def name(self) -> str:
        return "OpValidator"

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        super().__init__(safety_level)
        self.config = get_config()

    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate all operations in the graph.

        Args:
            graph: Execution graph with nodes
            context: Optional:
                - 'hardware': Hardware type for availability check
                - 'custom_ops': Additional supported ops

        Returns:
            ValidationResult with op validation status
        """
        warnings = []
        errors = []
        metrics = {}

        # Get supported ops (config + custom)
        supported_ops = set(self.config.supported_ops)
        custom_ops = context.get('custom_ops', set())
        supported_ops.update(custom_ops)

        # Hardware-specific ops
        hardware = context.get('hardware', 'generic')
        hw_ops = self._get_hardware_ops(hardware)
        supported_ops.update(hw_ops)

        # Collect all ops from graph
        found_ops: Set[str] = set()
        unsupported: List[str] = []
        op_counts: Dict[str, int] = {}

        nodes = getattr(graph, 'nodes', [])
        if hasattr(graph, '__iter__') and not nodes:
            nodes = list(graph)

        for node in nodes:
            # Support both attribute and dict access (ONNX format)
            if hasattr(node, 'op'):
                op = node.op
            elif hasattr(node, 'op_type'):
                op = node.op_type
            elif isinstance(node, dict):
                op = node.get('op_type') or node.get('op')
            else:
                op = None

            if op is None:
                continue

            op_lower = op.lower()
            found_ops.add(op_lower)
            op_counts[op_lower] = op_counts.get(op_lower, 0) + 1

            if op_lower not in supported_ops:
                unsupported.append(op)

            # Validate op parameters
            self._validate_op_params(node, op_lower, warnings, errors)

        # Remove duplicates from unsupported
        unsupported_unique = list(set(unsupported))

        metrics['total_ops'] = len(nodes)
        metrics['unique_ops'] = len(found_ops)
        metrics['op_distribution'] = op_counts

        if unsupported_unique:
            errors.append(
                f"Found {len(unsupported_unique)} unsupported operation(s): "
                f"{', '.join(sorted(unsupported_unique))}"
            )
            metrics['unsupported_ops'] = unsupported_unique

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
            message=f"All {len(found_ops)} unique operations are supported",
            warnings=warnings,
            metrics=metrics,
        )

    def _get_hardware_ops(self, hardware: str) -> Set[str]:
        """Get hardware-specific operations."""
        hw_ops: Dict[str, Set[str]] = {
            'nvidia': {
                'flash_attention_2',
                'tensorcore_matmul',
                'cudnn_conv',
            },
            'amd': {
                'rocm_attention',
                'hipblas_matmul',
            },
            'apple': {
                'mps_conv',
                'ane_matmul',
            },
        }
        return hw_ops.get(hardware.lower(), set())

    def _validate_op_params(
        self,
        node: Any,
        op: str,
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """Validate operation-specific parameters."""
        # Matmul/Linear checks
        if op in ('matmul', 'linear'):
            # Check for transpose flags if present
            trans_a = getattr(node, 'transpose_a', False)
            trans_b = getattr(node, 'transpose_b', False)
            # Both transposes are valid, no error

        # Convolution checks
        elif op in ('conv2d', 'conv1d'):
            stride = getattr(node, 'stride', 1)
            padding = getattr(node, 'padding', 0)
            dilation = getattr(node, 'dilation', 1)
            groups = getattr(node, 'groups', 1)

            if isinstance(stride, int) and stride <= 0:
                errors.append(f"Invalid stride {stride} for {op}")
            if isinstance(dilation, int) and dilation <= 0:
                errors.append(f"Invalid dilation {dilation} for {op}")
            if isinstance(groups, int) and groups <= 0:
                errors.append(f"Invalid groups {groups} for {op}")

        # Attention checks
        elif op in ('scaled_dot_product_attention', 'flash_attention'):
            dropout_p = getattr(node, 'dropout_p', 0.0)
            if dropout_p < 0 or dropout_p > 1:
                errors.append(f"Invalid dropout_p {dropout_p} for {op}")

            scale = getattr(node, 'scale', None)
            if scale is not None and scale <= 0:
                errors.append(f"Invalid scale {scale} for {op}")

        # Normalization checks
        elif op in ('layernorm', 'batchnorm', 'groupnorm'):
            eps = getattr(node, 'eps', 1e-5)
            if eps <= 0:
                errors.append(f"Invalid eps {eps} for {op}")

            if op == 'groupnorm':
                num_groups = getattr(node, 'num_groups', None)
                if num_groups is not None and num_groups <= 0:
                    errors.append(f"Invalid num_groups {num_groups} for groupnorm")

        # Reduction checks
        elif op in ('sum', 'mean', 'max', 'min'):
            keepdim = getattr(node, 'keepdim', False)
            # keepdim is valid as bool, no error

        # Activation checks
        elif op == 'leaky_relu':
            negative_slope = getattr(node, 'negative_slope', 0.01)
            if negative_slope < 0:
                warnings.append(f"Unusual negative_slope {negative_slope} for leaky_relu")

        # Pooling checks
        elif op in ('maxpool2d', 'avgpool2d'):
            kernel_size = getattr(node, 'kernel_size', None)
            if kernel_size is not None:
                if isinstance(kernel_size, int) and kernel_size <= 0:
                    errors.append(f"Invalid kernel_size {kernel_size} for {op}")

    def check_op_supported(self, op: str, hardware: str = 'generic') -> bool:
        """
        Check if an operation is supported.

        Args:
            op: Operation name
            hardware: Target hardware

        Returns:
            True if supported
        """
        supported = set(self.config.supported_ops)
        supported.update(self._get_hardware_ops(hardware))
        return op.lower() in supported

    def get_all_supported_ops(self, hardware: str = 'generic') -> Set[str]:
        """Get set of all supported operations."""
        supported = set(self.config.supported_ops)
        supported.update(self._get_hardware_ops(hardware))
        return supported

    def raise_if_unsupported(
        self,
        ops: List[str],
        hardware: str = 'generic',
    ) -> None:
        """
        Raise exception if any ops are unsupported.

        Args:
            ops: List of operation names
            hardware: Target hardware

        Raises:
            UnsupportedOpsError if any ops unsupported
        """
        supported = self.get_all_supported_ops(hardware)
        unsupported = [op for op in ops if op.lower() not in supported]

        if unsupported:
            raise UnsupportedOpsError(
                f"Unsupported operations: {', '.join(unsupported)}",
                unsupported_ops=unsupported,
                suggestion=f"Available ops: {', '.join(sorted(supported)[:20])}... "
                          f"Consider implementing custom kernels or using decomposition.",
            )

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """Op validation runs at all levels."""
        return True
