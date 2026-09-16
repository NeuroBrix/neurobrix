"""
neurobrix info/inspect — System information and NBX inspection commands.
"""

import sys
from pathlib import Path

from neurobrix import __version__
from neurobrix.cli.utils import PACKAGE_ROOT, CACHE_DIR, STORE_DIR, format_size


def info_record(args) -> dict:
    """What `info` knows, as one record: the engine's version, where things live
    on disk, the installed models, the hardware profiles, the compute stack —
    read from the same sources the human listing reads (Studio request 1:
    machine-readable discovery without loading a model)."""
    rec = {"version": __version__, "package": str(PACKAGE_ROOT), "cache": str(CACHE_DIR), "store": str(STORE_DIR),
           "python": sys.version.split()[0], "models": [], "hardware_profiles": [], "torch": None,
           "cuda_available": False, "gpus": []}
    if CACHE_DIR.exists():
        for model_dir in sorted(CACHE_DIR.iterdir()):
            if model_dir.is_dir() and (model_dir / "manifest.json").exists():
                rec["models"].append({"name": model_dir.name,
                                      "size_bytes": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())})
    hw_dir = PACKAGE_ROOT / "config" / "hardware"
    if hw_dir.exists():
        rec["hardware_profiles"] = [f.stem for f in sorted(hw_dir.glob("*.yml"))]
    try:
        from neurobrix.serving.engine import daemon_identity
        rec.update({k: v for k, v in daemon_identity().items() if k != "engine"})
    except Exception as exc:          # the serving module may not import on a minimal install: said, not hidden
        rec["daemon"] = f"unavailable ({exc})"
    try:
        import torch
        rec["torch"] = torch.__version__
        rec["cuda_available"] = bool(torch.cuda.is_available())
        if rec["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                rec["gpus"].append({"index": i, "name": props.name, "memory_bytes": int(props.total_memory)})
    except ImportError:
        pass
    return rec


def cmd_info(args):
    """Display system information."""
    from neurobrix.cli.json_out import wants_json, emit
    if wants_json(args):
        emit("info", info_record(args))
        return
    print("=" * 70)
    print(f"NeuroBrix v{__version__}")
    print("Universal Deep Learning Inference Engine")
    print("=" * 70)

    show_all = not any([args.models, args.hardware, args.system])

    if args.models or show_all:
        print("\nModels:")
        has_models = False

        if CACHE_DIR.exists():
            registry_models = [
                d for d in sorted(CACHE_DIR.iterdir())
                if d.is_dir() and (d / "manifest.json").exists()
            ]
            if registry_models:
                has_models = True
                for model_dir in registry_models:
                    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    print(f"  - {model_dir.name} ({format_size(total_size)})")

        if not has_models:
            print("  No models found")

    if args.hardware or show_all:
        print("\nHardware Profiles:")
        hw_dir = PACKAGE_ROOT / "config" / "hardware"
        if hw_dir.exists():
            for hw_file in sorted(hw_dir.glob("*.yml")):
                print(f"  - {hw_file.stem}")
        else:
            print("  No hardware profiles found")

    if args.system or show_all:
        print("\nSystem:")
        print(f"  Package: {PACKAGE_ROOT}")
        print(f"  Cache: {CACHE_DIR}")
        print(f"  Store: {STORE_DIR}")
        print(f"  Python: {sys.version.split()[0]}")

        try:
            import torch
            print(f"  PyTorch: {torch.__version__}")
            print(f"  CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"    [{i}] {props.name}: {props.total_memory / (1024**3):.1f} GB")
        except ImportError:
            print("  PyTorch: Not installed")


def cmd_inspect(args):
    """Inspect a .nbx file."""
    from neurobrix.nbx import NBXContainer

    nbx_path = Path(args.nbx_path)
    if not nbx_path.exists():
        # `inspect <model-name>` is the signature people reach for — it was
        # documented that way and it reads naturally — but only a path was
        # accepted, so a model name answered "File not found: <name>" with no
        # hint that a path was wanted (hub walkthrough, 2026-09-03). Resolve a
        # name against the installed models before giving up.
        try:
            from neurobrix.cli.utils import find_model

            nbx_path = Path(find_model(args.nbx_path))
        except FileNotFoundError as exc:
            print(f"ERROR: '{args.nbx_path}' is neither a file nor an "
                  f"installed model.")
            print(f"  {exc}")
            sys.exit(1)

    from neurobrix.cli.json_out import wants_json, emit
    if wants_json(args):
        container = NBXContainer.load(str(nbx_path))
        manifest = container.get_manifest() or {}
        comps = []
        for comp_name in container.list_components():
            comp = container.get_component(comp_name)
            entry = {"name": comp_name, "category": comp.category, "neural": bool(comp.is_neural)}
            if comp.graph:
                nodes = comp.graph.get("nodes", comp.graph.get("operations", comp.graph.get("ops", [])))
                entry["ops"] = len(nodes)
            if getattr(comp, "weight_paths", None):
                entry["shards"] = len(comp.weight_paths)
            comps.append(entry)
        emit("inspect", {"path": str(nbx_path), "model": manifest.get("model_name"),
                         "nbx_version": manifest.get("nbx_version"), "family": manifest.get("family"),
                         "components": comps})
        return
    print("=" * 70)
    print(f"NBX Inspect: {nbx_path.name}")
    print("=" * 70)

    container = NBXContainer.load(str(nbx_path))
    manifest = container.get_manifest() or {}

    print(f"\nFile: {nbx_path}")
    print(f"Model: {manifest.get('model_name', 'Unknown')}")
    print(f"NBX Version: {manifest.get('nbx_version', 'Unknown')}")
    print(f"Family: {manifest.get('family', 'Unknown')}")

    components = container.list_components()
    print(f"\nComponents ({len(components)}):")
    for comp_name in components:
        comp = container.get_component(comp_name)
        neural_str = "neural" if comp.is_neural else "config"
        print(f"  - {comp_name}: {comp.category} ({neural_str})")

    if args.topology:
        print(f"\nTopology (per neural component):")
        for comp_name in components:
            comp = container.get_component(comp_name)
            if comp.graph:
                nodes = comp.graph.get("nodes", comp.graph.get("operations", []))
                print(f"  {comp_name}: {len(nodes)} nodes")
                for node in nodes[:3]:
                    node_id = node.get("id", node.get("op_id", "?"))
                    op_type = node.get("op", node.get("op_type", "?"))
                    print(f"    - {node_id}: {op_type}")

    if args.weights:
        print(f"\nWeights:")
        for comp_name in components:
            comp = container.get_component(comp_name)
            if comp.weight_paths:
                print(f"  {comp_name}: {len(comp.weight_paths)} shard(s)")
                for wp in comp.weight_paths[:3]:
                    print(f"    - {Path(wp).name}")
