"""
neurobrix info/inspect — System information and NBX inspection commands.
"""

import sys
from pathlib import Path

from neurobrix import __version__
from neurobrix.cli.utils import PACKAGE_ROOT, CACHE_DIR, STORE_DIR, format_size


def cmd_info(args):
    """Display system information."""
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
        print(f"ERROR: File not found: {nbx_path}")
        sys.exit(1)

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
