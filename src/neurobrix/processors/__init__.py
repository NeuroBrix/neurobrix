"""The processors place: a model's declared non-neural sub-components (a text front end, for one), delivered on
demand and verified by the sha256 its container declares.

One call for the engine: `open_declared(manifest, container_dir, components)` gives the entry processor, bound to
its declaration, or None when the container declares none. A container without the declaration imports nothing
more than this module and its declaration reader: the network, the hash and the lock (store.py) are imported only
in the branch that found a declaration.
"""
from __future__ import annotations

from pathlib import Path

from neurobrix.processors.declaration import declared


def model_name(container_dir) -> str:
    """The name a person types for the model whose cache entry is `container_dir` (its recorded hub name, or the
    entry's own name for one installed before names were recorded: the run door accepts both)."""
    from neurobrix.nbx.names import recorded_name
    return recorded_name(Path(container_dir)) or Path(container_dir).name


def open_declared(manifest: dict, container_dir, components):
    """The entry processor `manifest` declares, present, verified and loaded; None when it declares none.

    `components`: the names of the container's parts (a declaration that names another is refused before any
    fetch)."""
    decls = declared(manifest)
    if not decls:
        return None
    from neurobrix.processors import store
    return store.open_all(decls, components, model_name(container_dir))
