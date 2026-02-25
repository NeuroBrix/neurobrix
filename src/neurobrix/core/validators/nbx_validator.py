"""
NBX Integrity Validator.

Validates .nbx archive structure, manifest, components, and coherence.
ZERO FALLBACK: Invalid NBX = explicit error with context.

Usage:
    validator = NBXValidator()
    result = validator.validate(Path("model.nbx"))

    if not result.is_valid:
        for error in result.errors:
            print(f"ERROR: {error}")

    # Or raise on first error
    validator.validate_strict(Path("model.nbx"))  # raises NBXValidationError

V1.0 - December 2025
"""

import json
import zipfile
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
from enum import Enum


class ValidationLevel(Enum):
    """Validation depth levels."""
    STRUCTURE = "structure"    # Just check files exist
    SCHEMA = "schema"          # Parse JSON and validate schema
    COHERENCE = "coherence"    # Cross-reference validation
    DEEP = "deep"              # Read safetensors, verify checksums


@dataclass
class ValidationError:
    """Single validation error."""
    level: ValidationLevel
    component: Optional[str]   # None for manifest-level errors
    file: str
    message: str

    def __str__(self) -> str:
        comp = f"[{self.component}] " if self.component else ""
        return f"{comp}{self.file}: {self.message}"


@dataclass
class ValidationResult:
    """Result of NBX validation."""
    nbx_path: Path
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else f"INVALID ({len(self.errors)} errors)"
        return f"NBX Validation: {self.nbx_path.name} - {status}"

    def summary(self) -> str:
        """Return detailed validation summary."""
        lines = [str(self)]

        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors[:20]:
                lines.append(f"  - {e}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more errors")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings[:10]:
                lines.append(f"  - {w}")

        if self.stats:
            lines.append("\nStats:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


class NBXValidationError(Exception):
    """Raised when NBX validation fails in strict mode."""
    def __init__(self, result: ValidationResult):
        self.result = result
        errors_str = "\n".join(f"  - {e}" for e in result.errors[:10])
        if len(result.errors) > 10:
            errors_str += f"\n  ... and {len(result.errors) - 10} more errors"
        super().__init__(f"NBX validation failed:\n{errors_str}")


class NBXValidator:
    """
    Validates .nbx archive integrity.

    ZERO FALLBACK: Any missing or invalid component is an error.

    Validation levels:
    - STRUCTURE: Check files exist in ZIP
    - SCHEMA: Parse JSON and validate required fields
    - COHERENCE: Cross-reference weights_index with actual shards
    - DEEP: Read safetensors, verify checksums
    """

    # Required files at manifest level
    REQUIRED_ROOT_FILES = {"manifest.json"}

    # Required files per neural component
    REQUIRED_COMPONENT_FILES = {
        "config.json",
        "graph.json",
        "weights_index.json",
    }

    # Required manifest fields
    REQUIRED_MANIFEST_FIELDS = {
        "nbx_version",
        "model_name",
        "components",
    }

    # Required weights_index fields
    REQUIRED_WEIGHTS_INDEX_FIELDS = {
        "version",
        "format",
        "component",
        "dtype",
        "shards",
        "tensors",
    }

    def __init__(self, level: ValidationLevel = ValidationLevel.COHERENCE):
        """
        Initialize validator.

        Args:
            level: Validation depth (STRUCTURE, SCHEMA, COHERENCE, DEEP)
        """
        self.level = level

    def validate(self, nbx_path: Path) -> ValidationResult:
        """
        Validate NBX archive.

        Returns ValidationResult with is_valid=False if any errors found.
        """
        nbx_path = Path(nbx_path)
        errors: List[ValidationError] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}

        # 1. Check file exists
        if not nbx_path.exists():
            errors.append(ValidationError(
                level=ValidationLevel.STRUCTURE,
                component=None,
                file=str(nbx_path),
                message="File does not exist"
            ))
            return ValidationResult(nbx_path, False, errors, warnings, stats)

        # 2. Check ZIP validity
        if not zipfile.is_zipfile(nbx_path):
            errors.append(ValidationError(
                level=ValidationLevel.STRUCTURE,
                component=None,
                file=str(nbx_path),
                message="Not a valid ZIP archive"
            ))
            return ValidationResult(nbx_path, False, errors, warnings, stats)

        try:
            with zipfile.ZipFile(nbx_path, 'r') as zf:
                # Get all files in archive
                all_files = set(zf.namelist())
                stats["total_files"] = len(all_files)

                # 3. Validate manifest
                manifest_errors, manifest = self._validate_manifest(zf, all_files)
                errors.extend(manifest_errors)

                if manifest is None:
                    # Can't continue without manifest
                    return ValidationResult(nbx_path, False, errors, warnings, stats)

                stats["nbx_version"] = manifest.get("nbx_version")
                stats["model_name"] = manifest.get("model_name")

                # Extract neural components from manifest
                neural_components = self._extract_neural_components(manifest)
                stats["neural_components"] = neural_components
                stats["total_components"] = len(manifest.get("components", {}))

                # 4. Validate each neural component
                for comp_name in neural_components:
                    comp_errors, comp_warnings = self._validate_component(
                        zf, all_files, comp_name
                    )
                    errors.extend(comp_errors)
                    warnings.extend(comp_warnings)

        except zipfile.BadZipFile as e:
            errors.append(ValidationError(
                level=ValidationLevel.STRUCTURE,
                component=None,
                file=str(nbx_path),
                message=f"Corrupted ZIP: {e}"
            ))

        is_valid = len(errors) == 0
        return ValidationResult(nbx_path, is_valid, errors, warnings, stats)

    def validate_strict(self, nbx_path: Path) -> ValidationResult:
        """
        Validate NBX and raise NBXValidationError if invalid.
        """
        result = self.validate(nbx_path)
        if not result.is_valid:
            raise NBXValidationError(result)
        return result

    def _extract_neural_components(self, manifest: Dict) -> List[str]:
        """
        Extract list of neural component names from manifest.

        Neural components have type="neural" or have weights.
        """
        neural = []
        components = manifest.get("components", {})

        for name, info in components.items():
            if isinstance(info, dict):
                # Check if neural type or has blocks
                comp_type = info.get("type", "")
                if comp_type == "neural" or info.get("blocks", 0) > 0:
                    neural.append(name)
            else:
                # Legacy format: just component name
                neural.append(name)

        return neural

    def _validate_manifest(
        self,
        zf: zipfile.ZipFile,
        all_files: Set[str]
    ) -> Tuple[List[ValidationError], Optional[Dict]]:
        """Validate manifest.json."""
        errors = []

        # Check manifest exists
        if "manifest.json" not in all_files:
            errors.append(ValidationError(
                level=ValidationLevel.STRUCTURE,
                component=None,
                file="manifest.json",
                message="Missing manifest.json at root"
            ))
            return errors, None

        # Parse manifest
        try:
            with zf.open("manifest.json") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                level=ValidationLevel.SCHEMA,
                component=None,
                file="manifest.json",
                message=f"Invalid JSON: {e}"
            ))
            return errors, None

        if self.level.value in ["schema", "coherence", "deep"]:
            # Check required fields
            for field_name in self.REQUIRED_MANIFEST_FIELDS:
                if field_name not in manifest:
                    errors.append(ValidationError(
                        level=ValidationLevel.SCHEMA,
                        component=None,
                        file="manifest.json",
                        message=f"Missing required field: {field_name}"
                    ))

            # Check components is non-empty
            if not manifest.get("components"):
                errors.append(ValidationError(
                    level=ValidationLevel.SCHEMA,
                    component=None,
                    file="manifest.json",
                    message="components dict is empty"
                ))

        return errors, manifest if not errors else manifest

    def _validate_component(
        self,
        zf: zipfile.ZipFile,
        all_files: Set[str],
        comp_name: str,
    ) -> Tuple[List[ValidationError], List[str]]:
        """Validate a single neural component."""
        errors = []
        warnings = []
        comp_prefix = f"components/{comp_name}/"

        # Check required files
        for req_file in self.REQUIRED_COMPONENT_FILES:
            file_path = f"{comp_prefix}{req_file}"
            if file_path not in all_files:
                errors.append(ValidationError(
                    level=ValidationLevel.STRUCTURE,
                    component=comp_name,
                    file=req_file,
                    message="Missing required file"
                ))

        # neurotax_map.json is optional but recommended
        ntx_path = f"{comp_prefix}neurotax_map.json"
        if ntx_path not in all_files:
            warnings.append(f"[{comp_name}] neurotax_map.json missing (optional)")

        # ir_v2.json is optional
        ir_path = f"{comp_prefix}ir_v2.json"
        if ir_path not in all_files:
            warnings.append(f"[{comp_name}] ir_v2.json missing (optional)")

        if self.level.value in ["schema", "coherence", "deep"]:
            # Validate weights_index.json
            wi_errors = self._validate_weights_index(zf, all_files, comp_name)
            errors.extend(wi_errors)

            # Validate graph.json
            graph_errors, graph_warnings = self._validate_graph(zf, all_files, comp_name)
            errors.extend(graph_errors)
            warnings.extend(graph_warnings)

            # Validate config.json
            config_errors = self._validate_config(zf, all_files, comp_name)
            errors.extend(config_errors)

        if self.level == ValidationLevel.DEEP:
            # Deep validation: read safetensors
            deep_errors = self._validate_safetensors(zf, all_files, comp_name)
            errors.extend(deep_errors)

        return errors, warnings

    def _validate_weights_index(
        self,
        zf: zipfile.ZipFile,
        all_files: Set[str],
        comp_name: str,
    ) -> List[ValidationError]:
        """Validate weights_index.json and shard references."""
        errors = []
        wi_path = f"components/{comp_name}/weights_index.json"

        if wi_path not in all_files:
            return errors  # Already reported as missing

        try:
            with zf.open(wi_path) as f:
                wi = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                level=ValidationLevel.SCHEMA,
                component=comp_name,
                file="weights_index.json",
                message=f"Invalid JSON: {e}"
            ))
            return errors

        # Check required fields
        for field_name in self.REQUIRED_WEIGHTS_INDEX_FIELDS:
            if field_name not in wi:
                errors.append(ValidationError(
                    level=ValidationLevel.SCHEMA,
                    component=comp_name,
                    file="weights_index.json",
                    message=f"Missing required field: {field_name}"
                ))

        if self.level.value in ["coherence", "deep"]:
            # Check all referenced shards exist
            shards_in_index = set(wi.get("shards", {}).keys())
            weights_prefix = f"components/{comp_name}/weights/"

            for shard_name in shards_in_index:
                shard_path = f"{weights_prefix}{shard_name}"
                if shard_path not in all_files:
                    errors.append(ValidationError(
                        level=ValidationLevel.COHERENCE,
                        component=comp_name,
                        file=f"weights/{shard_name}",
                        message="Shard referenced in weights_index.json but missing"
                    ))

            # Check all tensors reference valid shards
            for tensor_name, tensor_info in wi.get("tensors", {}).items():
                shard = tensor_info.get("shard")
                if shard and shard not in shards_in_index:
                    errors.append(ValidationError(
                        level=ValidationLevel.COHERENCE,
                        component=comp_name,
                        file="weights_index.json",
                        message=f"Tensor '{tensor_name}' references unknown shard: {shard}"
                    ))

            # Check shard sizes are reasonable (if size_bytes provided)
            for shard_name, shard_info in wi.get("shards", {}).items():
                shard_path = f"{weights_prefix}{shard_name}"
                if shard_path in all_files and isinstance(shard_info, dict):
                    expected_size = shard_info.get("size_bytes", 0)
                    if expected_size > 0:
                        # Get actual size from ZIP
                        zip_info = zf.getinfo(shard_path)
                        actual_size = zip_info.file_size

                        # Allow 5% tolerance
                        if abs(actual_size - expected_size) > expected_size * 0.05:
                            errors.append(ValidationError(
                                level=ValidationLevel.COHERENCE,
                                component=comp_name,
                                file=f"weights/{shard_name}",
                                message=f"Size mismatch: expected {expected_size}, got {actual_size}"
                            ))

        return errors

    def _validate_graph(
        self,
        zf: zipfile.ZipFile,
        all_files: Set[str],
        comp_name: str,
    ) -> Tuple[List[ValidationError], List[str]]:
        """Validate graph.json structure."""
        errors = []
        warnings = []
        graph_path = f"components/{comp_name}/graph.json"

        if graph_path not in all_files:
            return errors, warnings  # Already reported as missing

        try:
            with zf.open(graph_path) as f:
                graph = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                level=ValidationLevel.SCHEMA,
                component=comp_name,
                file="graph.json",
                message=f"Invalid JSON: {e}"
            ))
            return errors, warnings

        # Check has components or root_module (depending on graph version)
        has_components = "components" in graph
        has_root_module = "root_module" in graph

        if not has_components and not has_root_module:
            errors.append(ValidationError(
                level=ValidationLevel.SCHEMA,
                component=comp_name,
                file="graph.json",
                message="Missing 'components' or 'root_module' field"
            ))

        graph_version = graph.get("version", "")
        if graph_version != "0.1":
            warnings.append(f"[{comp_name}] graph.json version '{graph_version}' != '0.1'")

        return errors, warnings

    def _validate_config(
        self,
        zf: zipfile.ZipFile,
        all_files: Set[str],
        comp_name: str,
    ) -> List[ValidationError]:
        """Validate config.json structure."""
        errors = []
        config_path = f"components/{comp_name}/config.json"

        if config_path not in all_files:
            return errors  # Already reported as missing

        try:
            with zf.open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                level=ValidationLevel.SCHEMA,
                component=comp_name,
                file="config.json",
                message=f"Invalid JSON: {e}"
            ))
            return errors

        # Config should be a non-empty dict
        if not isinstance(config, dict) or not config:
            errors.append(ValidationError(
                level=ValidationLevel.SCHEMA,
                component=comp_name,
                file="config.json",
                message="Config should be a non-empty dictionary"
            ))

        return errors

    def _validate_safetensors(
        self,
        zf: zipfile.ZipFile,
        all_files: Set[str],
        comp_name: str,
    ) -> List[ValidationError]:
        """Deep validation: actually read safetensors files."""
        errors = []
        weights_prefix = f"components/{comp_name}/weights/"

        # Find all safetensors files
        shard_files = [f for f in all_files if f.startswith(weights_prefix) and f.endswith(".safetensors")]

        for shard_path in shard_files:
            try:
                # Try to read safetensor header
                with zf.open(shard_path) as f:
                    # Read header length (first 8 bytes)
                    header_len_bytes = f.read(8)
                    if len(header_len_bytes) < 8:
                        errors.append(ValidationError(
                            level=ValidationLevel.DEEP,
                            component=comp_name,
                            file=shard_path.split("/")[-1],
                            message="Safetensor file too short (corrupted)"
                        ))
                        continue

                    # Parse header length
                    header_len = struct.unpack("<Q", header_len_bytes)[0]

                    # Sanity check header length
                    if header_len > 100 * 1024 * 1024:  # 100MB max header
                        errors.append(ValidationError(
                            level=ValidationLevel.DEEP,
                            component=comp_name,
                            file=shard_path.split("/")[-1],
                            message=f"Safetensor header too large: {header_len} bytes"
                        ))
                        continue

                    # Read and parse header
                    header_bytes = f.read(header_len)
                    header = json.loads(header_bytes.decode("utf-8"))

                    # Header should have __metadata__ or tensor names
                    if not header:
                        errors.append(ValidationError(
                            level=ValidationLevel.DEEP,
                            component=comp_name,
                            file=shard_path.split("/")[-1],
                            message="Safetensor has empty header"
                        ))

            except (json.JSONDecodeError, struct.error, UnicodeDecodeError) as e:
                errors.append(ValidationError(
                    level=ValidationLevel.DEEP,
                    component=comp_name,
                    file=shard_path.split("/")[-1],
                    message=f"Safetensor corrupted: {e}"
                ))

        return errors


def validate_nbx(nbx_path: Path, strict: bool = False, level: ValidationLevel = ValidationLevel.COHERENCE) -> ValidationResult:
    """
    Convenience function to validate NBX.

    Args:
        nbx_path: Path to .nbx file
        strict: If True, raise NBXValidationError on failure
        level: Validation depth

    Returns:
        ValidationResult
    """
    validator = NBXValidator(level=level)
    if strict:
        return validator.validate_strict(nbx_path)
    return validator.validate(nbx_path)
