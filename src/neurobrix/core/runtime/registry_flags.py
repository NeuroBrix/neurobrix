"""Per-component runtime flags: the container is the source, the build toolchain's registry a check.

Phase 1 (DtypeEngine triton fix) read per-component flags straight from the
build toolchain's config/model_registry.yml so a YAML edit took effect with no
rebuild. That made a model run differently wherever the registry was not
reachable — the Mac, an installed engine, a worktree without the gitignored
`.nbx_registry` pointer — and it did: Wan2.1-T2V-1.3B rendered a lattice of
16-px cells without `zero_pad_embeddings` (2026-09-20), and on 2026-09-27
fifteen flags of eight video containers still existed only in the registry.
The supervisor's R18 decision of 2026-09-27 02:57: a container carries every
flag the engine reads; the registry serves the build only.

Lookup precedence at runtime:
  1. env var override (developer iteration / debugging)
  2. the container's declaration (nbx/component_flags.py, registered when the
     container is opened from topology.json's extracted_values)
  3. default value (the annotations are opt-in)

Where the registry IS reachable, a flag it declares that the container lacks
or carries with another value is refused by name: a registry edit now takes
effect through the container (Forge's in-place pass or a rebuild), so the
developer's rack and every other machine run the same thing.

ZERO FALLBACK boundary (engine audit #2 2026-07-05): an ABSENT registry
is a legitimate deployment state (installed runtime without the build
system co-located): the container's declarations answer alone. A registry
that EXISTS but cannot be read or parsed RAISES — silently skipping it would
disable the check on every per-component annotation engine-wide (e.g. the
`activations_fp16_safe` / `requires_fp32_compute` fp32-overflow
protection; graph_executor.py records that exact silent neutralisation
happening once already).

This module ONLY reads. It never writes to the registry. It does not
import any build-toolchain code, so it remains decoupled from the build system.
"""

import os
from pathlib import Path
from typing import Any, Optional


_REGISTRY_CACHE: Optional[dict] = None


def _find_registry_yaml() -> Optional[Path]:
    """Locate the build toolchain's config/model_registry.yml.

    Resolution order:
      1. `NBX_MODEL_REGISTRY` env var — absolute path to the YAML. A SET
         path that does not exist RAISES (present-but-broken class,
         ZERO FALLBACK — silently ignoring an explicit setting would
         disable every per-component annotation engine-wide).
      2. `.nbx_registry` pointer file — walk up from this source file;
         the first parent carrying one wins. The pointer holds the
         registry path relative to that parent (one line, gitignored —
         the dev/monorepo hookup). A pointer whose target is missing
         RAISES (same present-but-broken class).
      3. None (deployed install without the build toolchain co-located
         → every flag read resolves to its documented default).
    """
    override = os.environ.get("NBX_MODEL_REGISTRY")
    if override:
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"NBX_MODEL_REGISTRY is set but does not exist: {p} "
            "(ZERO FALLBACK: unset it or fix the path)")
    here = Path(__file__).resolve()
    for parent in here.parents:
        pointer = parent / ".nbx_registry"
        if pointer.exists():
            target = (parent / pointer.read_text().strip()).resolve()
            if target.exists():
                return target
            raise FileNotFoundError(
                f"registry pointer {pointer} targets a missing file: "
                f"{target} (ZERO FALLBACK: fix or remove the pointer)")
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
    """The flag a shipped container declares on a component.

    Precedence (the supervisor's R18 decision of 2026-09-27 02:57 — a
    container carries every flag the engine reads; the developer registry
    serves the build toolchain only, never the runtime):
      1. env var (when env_override is provided and set in environment)
      2. the container's own declaration (nbx/component_flags.py)
      3. default

    The registry is no longer a SOURCE. Where it is reachable (a developer
    checkout with the `.nbx_registry` pointer) it is a CHECK: a flag it
    declares truthy that the container does not carry, or carries with
    another value, is refused by name — the container is stale and runs
    differently on every machine without the registry (the Mac, an installed
    engine, a worktree without the pointer: Wan2.1-T2V rendered a lattice of
    16-px cells that way on 2026-09-20). A registry file that exists but is
    unreadable/malformed raises from `_load_registry` (engine audit #2
    2026-07-05).
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

    from neurobrix.nbx import component_flags
    carried = component_flags.get(model_name, component_name, flag_name, _ABSENT)
    declared = _registry_declaration(model_name, component_name, flag_name)
    # A flag declared false or null is the default and is not carried by the
    # build (importer/runtime_flags.py); anything else must be carried as is.
    if declared is not _ABSENT and carried != declared and not (
            declared in (None, False) and carried is _ABSENT):
        where = "does not carry it" if carried is _ABSENT else f"carries {carried!r}"
        raise RuntimeError(
            f"ZERO FALLBACK: the build toolchain's registry declares "
            f"{model_name}/{component_name}.{flag_name} = {declared!r} and the "
            f"container {where}. A shipped container carries every flag the "
            f"engine reads; this one runs differently on every machine without "
            f"the registry. Write the flag into the container (Forge "
            f"tools/neurotax_rename.py --apply) or rebuild it.")
    return default if carried is _ABSENT else carried


_ABSENT = object()


def _registry_declaration(model_name: str, component_name: str, flag_name: str) -> Any:
    """What the reachable registry declares for this flag, else _ABSENT.

    Registry layout: top-level is keyed by family (llm, vlm, image, audio,
    tts, stt, audio_llm, multimodal, upscaler, video, ...). Each family
    maps model_name -> entry -> components -> component_name -> flags. The
    caller need not know the family, so the top level is scanned for the
    model_name. Keys starting with '_' are reserved (templates, defaults)
    and skipped, as are non-mapping top-level metadata entries.
    """
    for top_key, family_entry in _load_registry().items():
        if str(top_key).startswith("_") or not isinstance(family_entry, dict):
            continue
        entry = family_entry.get(model_name)
        if not isinstance(entry, dict):
            continue
        comps = entry.get("components", {})
        comp = comps.get(component_name) if isinstance(comps, dict) else None
        if isinstance(comp, dict) and flag_name in comp:
            return comp[flag_name]
        return _ABSENT
    return _ABSENT
