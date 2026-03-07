import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from neurobrix.nbx.cache import ensure_extracted

@dataclass(frozen=True)
class RuntimePackage:
    """
    Immutable NBX v0.1 Runtime Package.
    Contains all raw configurations and component metadata.
    """
    root_path: Path  # Original .nbx file path (for reference/logging)
    cache_path: Path  # Extracted cache directory (where all reads happen)
    manifest: Dict[str, Any]
    topology: Dict[str, Any]  # Unified flow definition (replaces pipeline + execution)
    variables: Dict[str, Any]
    defaults: Dict[str, Any]
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    modules: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class NBXRuntimeLoader:
    """
    Passive loader for NBX v0.1 containers.
    ZERO HEURISTIC: Fails if mandatory artifacts are missing.
    ZERO EXECUTION: No tensors, no devices, no graphs loaded into memory here.

    CACHE ARCHITECTURE:
    - .nbx file is ONLY for transport/packaging
    - Runtime ALWAYS reads from ~/.neurobrix/cache/<model_name>/
    - ensure_extracted() handles extraction on first run
    """

    REQUIRED_CORE_FILES = {
        "manifest": "manifest.json",
        "topology": "topology.json",  # Unified flow (replaces pipeline.json + execution.json)
        "variables": "runtime/variables.json",
        "defaults": "runtime/defaults.json",
    }

    def load(self, nbx_path: str) -> RuntimePackage:
        """
        Loads from cache (extracts .nbx on first run if needed).

        CACHE ARCHITECTURE: Runtime NEVER reads from .nbx directly.
        ensure_extracted() extracts to ~/.neurobrix/cache/<name>/ on first run.
        """
        path = Path(nbx_path)
        if not path.exists():
            raise FileNotFoundError(f"NBX container not found: {nbx_path}")

        # Extract to cache (fast if already cached)
        cache_path = ensure_extracted(path)

        # 1. Load Core Files from cache (ZERO FALLBACK - fail if missing)
        core_data = {}
        for key, rel_path in self.REQUIRED_CORE_FILES.items():
            file_path = cache_path / rel_path
            if not file_path.exists():
                raise RuntimeError(
                    f"NBX v0.1 Violation: Missing mandatory file '{rel_path}'\n"
                    f"  Cache path: {cache_path}\n"
                    f"  FIX: Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
                )
            core_data[key] = self._load_json_from_cache(file_path)

        # 2. Discover and Load Components from cache
        # Components are in 'components/<name>/runtime.json'
        components = {}
        components_dir = cache_path / "components"
        if components_dir.exists():
            for comp_dir in components_dir.iterdir():
                if comp_dir.is_dir():
                    runtime_json = comp_dir / "runtime.json"
                    if runtime_json.exists():
                        components[comp_dir.name] = self._load_json_from_cache(runtime_json)

        # 3. Discover and Load Modules configs from cache
        modules = {}
        manifest_modules = core_data["manifest"].get("modules", {})

        for mod_name, mod_def in manifest_modules.items():
            config_path = mod_def.get("config")

            if config_path:
                config_file = cache_path / config_path
                if config_file.exists():
                    modules[mod_name] = {
                        "type": mod_def.get("type", "generic"),
                        "path": mod_def.get("path", ""),
                        "config": self._load_json_from_cache(config_file)
                    }
                    continue

            # Fallback: search for *config*.json in the module's directory
            mod_path = mod_def.get("path", f"modules/{mod_name}/")
            mod_dir = cache_path / mod_path
            found = False

            if mod_dir.exists():
                for config_file in mod_dir.glob("*config*.json"):
                    modules[mod_name] = {
                        "type": mod_def.get("type", "generic"),
                        "path": mod_path,
                        "config": self._load_json_from_cache(config_file)
                    }
                    found = True
                    break

            if not found:
                # Non-functional modules (e.g. figures, docs) have no config — skip silently
                if mod_def.get("type") == "generic":
                    continue
                raise RuntimeError(
                    f"Module '{mod_name}' declared in manifest but no config found. "
                    f"Expected at: {config_path or mod_path + '*config*.json'}"
                )

        return RuntimePackage(
            root_path=path,
            cache_path=cache_path,
            manifest=core_data["manifest"],
            topology=core_data["topology"],
            variables=core_data["variables"],
            defaults=core_data["defaults"],
            components=components,
            modules=modules
        )

    def _load_json_from_cache(self, file_path: Path) -> Dict[str, Any]:
        """Loads a JSON file from cache directory."""
        try:
            with open(file_path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Malformed JSON in '{file_path}': {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to read '{file_path}': {str(e)}")
