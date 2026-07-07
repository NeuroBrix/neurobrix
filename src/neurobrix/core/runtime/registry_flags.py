"""Runtime-direct read of per-component flags from forge/config/model_registry.yml.

Phase 1 (DtypeEngine triton fix) introduced the per-component
`activations_fp16_safe` flag. Doctrine: changing the YAML must take
effect on next neurobrix run, WITHOUT Forge re-build (R18 immutable
.nbx is preserved — no field added to graph/topology/profile contract).

Lookup precedence at runtime:
  1. env var override (developer iteration / debugging)
  2. forge/config/model_registry.yml when accessible (monorepo / dev)
  3. default value (legitimate for ABSENT registry / model / component /
     flag — the annotations are opt-in)

ZERO FALLBACK boundary (engine audit #2 2026-07-05): an ABSENT registry
is a legitimate deployment state (installed runtime without the build
system co-located) and resolves to defaults. A registry that EXISTS but
cannot be read or parsed RAISES — silently returning defaults would
disable every per-component annotation engine-wide (e.g. the
`activations_fp16_safe` / `requires_fp32_compute` fp32-overflow
protection; graph_executor.py records that exact silent neutralisation
happening once already).

This module ONLY reads. It never writes to the registry. It does not
import any forge code, so it remains decoupled from the build system.
"""

import os
from pathlib import Path
from typing import Any, Optional


_REGISTRY_CACHE: Optional[dict] = None


def _find_registry_yaml() -> Optional[Path]:
    """Locate forge/config/model_registry.yml relative to this file.

    Walk up from the current source location until we find a sibling
    `forge/config/model_registry.yml`. Returns None if not found
    (deployed install without forge co-located).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "forge" / "config" / "model_registry.yml"
        if candidate.exists():
            return candidate
    return None


def _load_registry() -> dict:
    """Load and cache the registry YAML once per process.

    ABSENT registry → {} (legitimate: deployed install without the build
    system co-located; every flag read resolves to its documented
    default). PRESENT-but-unreadable/malformed registry → raise (ZERO
    FALLBACK: it would silently disable every per-component annotation
    engine-wide).
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    path = _find_registry_yaml()
    if path is None:
        _REGISTRY_CACHE = {}
        return _REGISTRY_CACHE
    try:
        import yaml
        with open(path) as f:
            loaded = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(
            f"ZERO FALLBACK: model registry exists at '{path}' but could "
            f"not be read/parsed ({type(e).__name__}: {e}). Silently "
            f"falling back to defaults would disable every per-component "
            f"flag (activations_fp16_safe, requires_fp32_compute, ...) "
            f"engine-wide. Fix the registry YAML."
        ) from e
    if loaded is None:
        loaded = {}  # empty file = empty registry (no annotations)
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"ZERO FALLBACK: model registry at '{path}' must be a YAML "
            f"mapping at top level, got {type(loaded).__name__}. Fix the "
            f"registry YAML."
        )
    _REGISTRY_CACHE = loaded
    return _REGISTRY_CACHE


def get_component_flag(
    model_name: Optional[str],
    component_name: Optional[str],
    flag_name: str,
    default: Any = None,
    env_override: Optional[str] = None,
) -> Any:
    """Return the value of `models.<model_name>.components.<component_name>.<flag_name>`.

    Precedence:
      1. env var (when env_override is provided and set in environment)
      2. registry YAML lookup
      3. default

    Returns default when the registry / model / component / flag is
    ABSENT (annotations are opt-in — legitimate absence). A registry
    file that exists but is unreadable/malformed raises from
    `_load_registry` (ZERO FALLBACK — engine audit #2 2026-07-05; the
    former "never raises" contract silently disabled every annotation
    engine-wide on a bad registry).
    """
    if env_override and env_override in os.environ:
        v = os.environ[env_override].strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off", ""):
            return False
        return v

    if not model_name or not component_name:
        return default

    reg = _load_registry()
    # Registry layout: top-level is keyed by family (llm, vlm, image, audio,
    # tts, stt, audio_llm, multimodal, upscaler, video, ...). Each family
    # maps model_name → entry → components → component_name → flags. We do
    # not require the caller to know the family, so we scan top-level for
    # the model_name. Keys starting with '_' are reserved (templates,
    # defaults) and skipped, as are non-mapping top-level metadata entries.
    for top_key, family_entry in reg.items():
        if str(top_key).startswith("_"):
            continue
        if not isinstance(family_entry, dict):
            continue
        entry = family_entry.get(model_name)
        if not isinstance(entry, dict):
            continue
        comps = entry.get("components", {})
        if not isinstance(comps, dict):
            continue
        comp = comps.get(component_name)
        if not isinstance(comp, dict):
            continue
        return comp.get(flag_name, default)
    return default
