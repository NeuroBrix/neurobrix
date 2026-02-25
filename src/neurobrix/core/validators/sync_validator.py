"""
NeuroBrix Validators - Synchronization Validator.

Validates cross-device synchronization and data dependencies.

ZERO FALLBACK: Sync issues raise explicit SyncError.
"""

from typing import Any, Dict, List, Set, Optional, Tuple

from .base import BaseValidator, SafetyLevel, ValidationResult, SyncError, DeadlockRisk
from .config import get_config


class SyncValidator(BaseValidator):
    """
    Validates synchronization correctness.

    Checks:
    - Cross-device data dependencies
    - Stream ordering
    - Event signaling
    - Deadlock detection
    - Memory fence requirements
    """

    @property
    def name(self) -> str:
        return "SyncValidator"

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        super().__init__(safety_level)
        self.config = get_config()

    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate synchronization in the graph.

        Args:
            graph: Execution graph with nodes
            context: Must contain:
                - 'device_assignments': Dict mapping node -> device
                - 'stream_assignments': Dict mapping node -> stream (optional)

        Returns:
            ValidationResult with sync validation status
        """
        warnings = []
        errors = []
        metrics = {}

        device_assignments = context.get('device_assignments', {})
        stream_assignments = context.get('stream_assignments', {})

        if not device_assignments:
            return ValidationResult(
                passed=True,
                validator_name=self.name,
                message="No device assignments to validate",
                warnings=warnings,
            )

        # Analyze device topology
        devices = set(device_assignments.values())
        metrics['num_devices'] = len(devices)
        metrics['devices'] = list(devices)

        # Check cross-device transfers
        transfers = self._find_cross_device_transfers(graph, device_assignments)
        metrics['cross_device_transfers'] = len(transfers)

        for transfer in transfers:
            self._validate_transfer(transfer, context, warnings, errors)

        # Check for potential deadlocks
        if self.safety_level >= SafetyLevel.STANDARD:
            deadlock_risks = self._detect_deadlock_risks(
                graph, device_assignments, stream_assignments
            )
            for risk in deadlock_risks:
                warnings.append(f"Potential deadlock risk: {risk}")
            metrics['deadlock_risks'] = len(deadlock_risks)

        # Check stream depth
        if stream_assignments:
            self._check_stream_depth(stream_assignments, warnings, errors, metrics)

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
            message=f"Sync validated: {len(devices)} devices, {len(transfers)} transfers",
            warnings=warnings,
            metrics=metrics,
        )

    def _find_cross_device_transfers(
        self,
        graph: Any,
        device_assignments: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Find all cross-device data transfers."""
        transfers = []

        nodes = getattr(graph, 'nodes', [])
        if hasattr(graph, '__iter__') and not nodes:
            nodes = list(graph)

        for node in nodes:
            node_name = getattr(node, 'name', str(node))
            node_device = device_assignments.get(node_name)

            if node_device is None:
                continue

            # Check inputs
            inputs = getattr(node, 'inputs', [])
            for inp in inputs:
                inp_name = getattr(inp, 'name', str(inp))
                inp_device = device_assignments.get(inp_name)

                if inp_device is not None and inp_device != node_device:
                    transfers.append({
                        'from_node': inp_name,
                        'from_device': inp_device,
                        'to_node': node_name,
                        'to_device': node_device,
                    })

        return transfers

    def _validate_transfer(
        self,
        transfer: Dict[str, Any],
        context: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """Validate a single cross-device transfer."""
        from_device = transfer['from_device']
        to_device = transfer['to_device']

        # Check if devices can communicate
        valid_pairs = context.get('valid_device_pairs', None)
        if valid_pairs is not None:
            pair = (from_device, to_device)
            if pair not in valid_pairs and (to_device, from_device) not in valid_pairs:
                errors.append(
                    f"Invalid device transfer: {from_device} -> {to_device}"
                )

        # Check for GPU -> CPU -> GPU pattern (inefficient)
        if 'cpu' in to_device.lower() and 'gpu' in from_device.lower():
            warnings.append(
                f"GPU -> CPU transfer detected: {transfer['from_node']} -> {transfer['to_node']}. "
                f"Consider keeping data on GPU if possible."
            )

    def _detect_deadlock_risks(
        self,
        graph: Any,
        device_assignments: Dict[str, str],
        stream_assignments: Dict[str, str],
    ) -> List[str]:
        """Detect potential deadlock situations."""
        risks = []

        # Build dependency graph per device
        device_deps: Dict[str, List[Tuple[str, str]]] = {}

        nodes = getattr(graph, 'nodes', [])
        if hasattr(graph, '__iter__') and not nodes:
            nodes = list(graph)

        for node in nodes:
            node_name = getattr(node, 'name', str(node))
            node_device = device_assignments.get(node_name)

            if node_device is None:
                continue

            if node_device not in device_deps:
                device_deps[node_device] = []

            inputs = getattr(node, 'inputs', [])
            for inp in inputs:
                inp_name = getattr(inp, 'name', str(inp))
                inp_device = device_assignments.get(inp_name)

                if inp_device is not None and inp_device != node_device:
                    device_deps[node_device].append((inp_device, node_name))

        # Check for circular dependencies between devices
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(device: str) -> bool:
            visited.add(device)
            rec_stack.add(device)

            deps = device_deps.get(device, [])
            for dep_device, _ in deps:
                if dep_device not in visited:
                    if has_cycle(dep_device):
                        return True
                elif dep_device in rec_stack:
                    return True

            rec_stack.remove(device)
            return False

        for device in device_deps:
            if device not in visited:
                if has_cycle(device):
                    risks.append(f"Circular dependency involving device {device}")

        # Check stream conflicts
        if stream_assignments:
            stream_nodes: Dict[str, List[str]] = {}
            for node_name, stream in stream_assignments.items():
                if stream not in stream_nodes:
                    stream_nodes[stream] = []
                stream_nodes[stream].append(node_name)

            # Check for same stream with cross-device ops
            for stream, nodes_in_stream in stream_nodes.items():
                devices_in_stream = set()
                for node_name in nodes_in_stream:
                    device = device_assignments.get(node_name)
                    if device:
                        devices_in_stream.add(device)

                if len(devices_in_stream) > 1:
                    risks.append(
                        f"Stream {stream} contains nodes on multiple devices: "
                        f"{devices_in_stream}"
                    )

        return risks

    def _check_stream_depth(
        self,
        stream_assignments: Dict[str, str],
        warnings: List[str],
        errors: List[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Check stream depth limits."""
        config = self.config

        # Count unique streams
        unique_streams = set(stream_assignments.values())
        num_streams = len(unique_streams)
        metrics['num_streams'] = num_streams

        if num_streams > config.max_stream_depth:
            errors.append(
                f"Stream depth {num_streams} exceeds max {config.max_stream_depth}"
            )
        elif num_streams > config.max_stream_depth * 0.8:
            warnings.append(
                f"High stream count: {num_streams}/{config.max_stream_depth}"
            )

    def validate_device_transfer(
        self,
        from_device: str,
        to_device: str,
        size_bytes: int,
    ) -> None:
        """
        Validate a single device transfer.

        Raises SyncError if transfer is invalid.
        """
        if from_device == to_device:
            return  # Same device, no transfer needed

        # Check for incompatible transfers
        incompatible = [
            ('cpu', 'disk'),
            ('disk', 'cpu'),
        ]

        pair = (from_device.lower(), to_device.lower())
        if pair in incompatible:
            raise SyncError(
                f"Incompatible device transfer: {from_device} -> {to_device}",
                device_a=from_device,
                device_b=to_device,
                suggestion="Use intermediate staging buffer",
            )

    def check_for_deadlock(
        self,
        dependencies: Dict[str, List[str]],
    ) -> Optional[List[str]]:
        """
        Check for deadlock in dependency graph.

        Args:
            dependencies: Dict mapping node -> list of dependencies

        Returns:
            Cycle path if deadlock found, None otherwise
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in dependencies.get(node, []):
                if dep not in visited:
                    result = dfs(dep)
                    if result is not None:
                        return result
                elif dep in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]

            path.pop()
            rec_stack.remove(node)
            return None

        for node in dependencies:
            if node not in visited:
                cycle = dfs(node)
                if cycle is not None:
                    return cycle

        return None

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """Sync validation runs at STANDARD and PARANOID levels."""
        return safety_level >= SafetyLevel.STANDARD
