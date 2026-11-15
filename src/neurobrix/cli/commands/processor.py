"""`neurobrix processor`: a model's declared sub-components in the cache, fetched ahead, carried to an offline
machine, listed.

    neurobrix processor fetch <model>                 into the cache, as the model's first run would
    neurobrix processor fetch <model> --to <folder>   every artifact the model declares, written to a folder
    neurobrix processor install <file> [--for <model>]   a carried file into the cache (checked at once with --for)
    neurobrix processor list                          the cache's artifacts
"""
from __future__ import annotations

import json
from pathlib import Path


def _declared_by(asked: str):
    """The declarations of the installed model `asked`, and the name the person types for it."""
    from neurobrix.cli.utils import CACHE_DIR, find_model, resolve_model_arg
    from neurobrix.processors import model_name
    from neurobrix.processors.declaration import declared
    entry = resolve_model_arg(asked)
    find_model(entry)
    manifest = json.loads((CACHE_DIR / entry / "manifest.json").read_text(encoding="utf-8"))
    return declared(manifest), model_name(CACHE_DIR / entry)


def _fetch(args) -> None:
    from neurobrix.processors import messages, store
    from neurobrix.processors.interface import refuse_unspoken
    decls, model = _declared_by(args.model)
    if not decls:
        print(messages.NONE_DECLARED.format(model=model))
        return
    root = Path(args.to).expanduser() if args.to else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    for decl in decls:
        refuse_unspoken(decl)
        store.ensure(decl, model, root=root)
        if root is not None:
            print(root / decl.file)


def _install(args) -> None:
    from neurobrix.processors import store
    decls, model = _declared_by(args.for_model) if args.for_model else (None, "")
    store.install(Path(args.file).expanduser(), decls, model)


def _list(args) -> None:
    from neurobrix.cli.commands.registry import print_table
    from neurobrix.cli.utils import format_size
    from neurobrix.processors import store
    rows = store.listing()
    if not rows:
        print(f"No sub-component in {store.cache_root()}.")
        return
    print_table(["FILE", "SIZE", "SHA256"], [[r["file"], format_size(r["size"]), r["sha256"]] for r in rows])


ACTIONS = {"fetch": _fetch, "install": _install, "list": _list}


def cmd_processor(args) -> int:
    """Dispatch `neurobrix processor <action>`; no action prints the command's help (exit 2)."""
    action = ACTIONS.get(getattr(args, "action", None))
    if action is None:
        args.processor_parser.print_help()
        return 2
    action(args)
    return 0
